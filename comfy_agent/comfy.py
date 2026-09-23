"""Minimal ComfyUI HTTP client (from v0, websocket path removed) plus the GPU guard."""
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

_TRANSPORT_ERRORS = (urllib.error.HTTPError, urllib.error.URLError, TimeoutError)


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
    return subprocess.run([exe, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, timeout=20).stdout


def gpu_used_mb(exclude_pids=frozenset()):
    text = _nvidia_smi()
    total = 0
    for line in (text or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        if int(parts[0]) in exclude_pids:
            continue
        total += int(parts[1])
    return total


def gpu_ok(limit_mb=10240, comfy_pid=None):
    return gpu_used_mb({comfy_pid} if comfy_pid else frozenset()) <= limit_mb
