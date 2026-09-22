"""Minimal ComfyUI HTTP client (from v0, websocket path removed) plus the GPU guard."""
import json
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
import uuid


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

    def run(self, graph):
        out = self._post("/prompt", {"prompt": graph, "client_id": self.client_id})
        if "prompt_id" not in out:
            raise ComfyError(f"ComfyUI rejected the graph: {out}")
        pid = out["prompt_id"]
        deadline = time.time() + self.timeout
        hist = {}
        while time.time() < deadline:
            hist = self._get(f"/history/{pid}").get(pid)
            if hist:
                break
            time.sleep(1.0)
        if not hist:
            raise ComfyError(f"timed out after {self.timeout}s waiting for {pid}")
        status = hist.get("status", {})
        if status.get("status_str") == "error":
            raise ComfyError(f"execution failed: {status.get('messages')}")
        for node_out in hist.get("outputs", {}).values():
            for img in node_out.get("images", []):
                if img.get("type") == "temp":
                    continue
                q = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img.get("subfolder", ""),
                                            "type": img.get("type", "output")})
                with urllib.request.urlopen(f"http://{self.host}/view?{q}", timeout=120) as r:
                    return img["filename"], r.read()
        raise ComfyError(f"run {pid} produced no images; status={status}")


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
