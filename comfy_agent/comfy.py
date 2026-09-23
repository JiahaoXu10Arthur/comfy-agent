"""Minimal ComfyUI HTTP client (from v0, websocket path removed) plus the GPU guard."""
import json
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

_TRANSPORT_ERRORS = (OSError, ValueError)


class ComfyError(RuntimeError):
    pass


class Comfy:
    def __init__(self, host="127.0.0.1:8188", timeout=600):
        self.host, self.timeout, self.client_id = host, timeout, str(uuid.uuid4())

    def _get(self, path):
        with urllib.request.urlopen(f"http://{self.host}{path}", timeout=30) as r:
            return json.load(r)

    def _post(self, path, payload):
        req = urllib.request.Request(f"http://{self.host}{path}", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read()
            return json.loads(body) if body else {}

    def system_stats(self):
        return self._get("/system_stats")

    def alive(self):
        try:
            self._get("/system_stats"); return True
        except Exception:
            return False

    def busy(self):
        try:
            q = self._get("/queue")
            return bool(q.get("queue_running") or q.get("queue_pending"))
        except Exception:
            return False

    def wait_idle(self, timeout=1800, poll=5.0):
        deadline = time.time() + timeout
        while self.busy():
            if time.time() > deadline:
                return False
            time.sleep(poll)
        return True

    def free(self):
        try:
            self._post("/free", {"unload_models": True, "free_memory": True}); return True
        except Exception:
            return False

    def _submit(self, graph):
        try:
            out = self._post("/prompt", {"prompt": graph, "client_id": self.client_id})
        except _TRANSPORT_ERRORS as e:
            raise ComfyError(f"ComfyUI unreachable or refused: {e}") from e
        if "prompt_id" not in out:
            raise ComfyError(f"ComfyUI rejected the graph: {out}")
        return out["prompt_id"]

    def _wait_history(self, pid):
        deadline = time.time() + self.timeout
        hist = {}
        while time.time() < deadline:
            try:
                hist = self._get(f"/history/{pid}").get(pid)
            except _TRANSPORT_ERRORS as e:
                raise ComfyError(f"ComfyUI unreachable or refused: {e}") from e
            if hist:
                break
            time.sleep(1.0)
        if not hist:
            raise ComfyError(f"timed out after {self.timeout}s waiting for {pid}")
        status = hist.get("status", {})
        if status.get("status_str") == "error":
            raise ComfyError(f"execution failed: {status.get('messages')}")
        return hist

    def run(self, graph):
        pid = self._submit(graph)
        hist = self._wait_history(pid)
        for node_out in hist.get("outputs", {}).values():
            for img in node_out.get("images", []):
                if img.get("type") == "temp":
                    continue
                q = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img.get("subfolder", ""),
                                            "type": img.get("type", "output")})
                try:
                    with urllib.request.urlopen(f"http://{self.host}/view?{q}", timeout=120) as r:
                        return img["filename"], r.read()
                except _TRANSPORT_ERRORS as e:
                    raise ComfyError(f"ComfyUI unreachable or refused: {e}") from e
        raise ComfyError(f"run {pid} produced no images; status={hist.get('status', {})}")

    def run_for_outputs(self, graph):
        """Submit a graph that produces no image (e.g. the tagger graph) and return its outputs dict,
        scoped to this prompt's own /history/{prompt_id} entry."""
        pid = self._submit(graph)
        return self._wait_history(pid).get("outputs", {})

    def upload_image(self, path):
        """POST /upload/image (multipart) and return the name ComfyUI stored it under."""
        boundary = uuid.uuid4().hex
        data = Path(path).read_bytes()
        filename = Path(path).name
        body = (f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
                f"Content-Type: image/png\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(f"http://{self.host}/upload/image", data=body,
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.loads(r.read())
        except _TRANSPORT_ERRORS as e:
            raise ComfyError(f"ComfyUI unreachable or refused: {e}") from e
        if "name" not in out:
            raise ComfyError(f"ComfyUI rejected the upload: {out}")
        return out["name"]


def _nvidia_smi():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    return subprocess.run([exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, timeout=20).stdout


def gpu_used_mb():
    """Total VRAM used across the GPU(s), summed from `nvidia-smi --query-gpu=memory.used`.
    Returns None (never 0) when nvidia-smi is missing or nothing parses, so callers fail closed."""
    text = _nvidia_smi()
    if not text:
        return None
    total, saw_line = 0, False
    for line in text.splitlines():
        line = line.strip()
        if not line.isdigit():
            continue
        total += int(line)
        saw_line = True
    return total if saw_line else None


def gpu_ok(comfy, limit_mb=10240):
    """Someone else's usage: total GPU memory used minus ComfyUI's own share
    (GET /system_stats -> devices[0].torch_vram_total, bytes -> MB). Fails closed (False)
    whenever nvidia-smi or system_stats cannot be read."""
    used = gpu_used_mb()
    if used is None:
        return False
    try:
        comfy_share_mb = comfy.system_stats()["devices"][0]["torch_vram_total"] / (1024 * 1024)
    except Exception:
        return False
    return (used - comfy_share_mb) <= limit_mb
