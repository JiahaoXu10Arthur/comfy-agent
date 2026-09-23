import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import pytest
from comfy_agent import comfy as cm


class Fake(BaseHTTPRequestHandler):
    state = {"queued": [], "done": {}, "fail": False}

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        s = Fake.state
        if self.path == "/system_stats":
            return self._json({})
        if self.path == "/queue":
            return self._json({"queue_running": s["queued"], "queue_pending": []})
        if self.path.startswith("/history/"):
            pid = self.path.rsplit("/", 1)[1]
            if pid in s["done"]:
                return self._json({pid: s["done"][pid]})
            s["done"][pid] = {"status": {"status_str": "error" if s["fail"] else "success", "messages": ["boom"] if s["fail"] else []},
                              "outputs": {"9": {"images": [{"filename": "out_00001_.png", "subfolder": "SDXL", "type": "output"}]}}}
            s["queued"] = []
            return self._json({})
        if self.path.startswith("/view"):
            self.send_response(200); self.end_headers(); self.wfile.write(b"\x89PNGfake")
            return
        self._json({}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); self.rfile.read(n)
        if self.path == "/prompt":
            Fake.state["queued"] = ["x"]
            return self._json({"prompt_id": "p1"})
        if self.path == "/free":
            return self._json({})
        self._json({}, 404)


@pytest.fixture
def server():
    Fake.state = {"queued": [], "done": {}, "fail": False}
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    yield f"127.0.0.1:{srv.server_port}"
    srv.shutdown()
    srv.server_close()
    t.join(timeout=5)


def test_run_returns_first_image(server):
    c = cm.Comfy(server, timeout=5)
    assert c.alive()
    name, data = c.run({"any": "graph"})
    assert name == "out_00001_.png" and data.startswith(b"\x89PNG")


def test_system_stats_hits_the_endpoint(server):
    c = cm.Comfy(server, timeout=5)
    assert c.system_stats() == {}


def test_run_raises_on_execution_error(server):
    Fake.state["fail"] = True
    with pytest.raises(cm.ComfyError, match="boom"):
        cm.Comfy(server, timeout=5).run({})


class FakeStatsComfy:
    def __init__(self, torch_vram_total_bytes):
        self.torch_vram_total_bytes = torch_vram_total_bytes

    def system_stats(self):
        return {"devices": [{"torch_vram_total": self.torch_vram_total_bytes}]}


def test_gpu_used_mb_sums_nvidia_smi_lines(monkeypatch):
    monkeypatch.setattr(cm, "_nvidia_smi", lambda: "8000\n3000\n")
    assert cm.gpu_used_mb() == 11000


def test_gpu_used_mb_none_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(cm, "_nvidia_smi", lambda: None)
    assert cm.gpu_used_mb() is None


def test_gpu_used_mb_none_on_unparseable_output(monkeypatch):
    monkeypatch.setattr(cm, "_nvidia_smi", lambda: "[N/A]\n[N/A]\n")
    assert cm.gpu_used_mb() is None


def test_gpu_used_mb_none_on_nvidia_smi_timeout(monkeypatch):
    monkeypatch.setattr(cm.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    def boom(*a, **k):
        raise cm.subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=20)
    monkeypatch.setattr(cm.subprocess, "run", boom)
    assert cm.gpu_used_mb() is None


def test_gpu_ok_passes_under_limit_only_after_subtracting_comfy_share(monkeypatch):
    monkeypatch.setattr(cm, "_nvidia_smi", lambda: "14000\n")
    assert cm.gpu_ok(FakeStatsComfy(6000 * 1024 * 1024), limit_mb=10240) is True


def test_gpu_ok_fails_over_limit(monkeypatch):
    monkeypatch.setattr(cm, "_nvidia_smi", lambda: "20000\n")
    assert cm.gpu_ok(FakeStatsComfy(1024 * 1024 * 1024), limit_mb=10240) is False


def test_gpu_ok_fails_closed_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(cm, "_nvidia_smi", lambda: None)
    assert cm.gpu_ok(FakeStatsComfy(0)) is False


def test_gpu_ok_fails_closed_on_garbage_nvidia_smi_output(monkeypatch):
    monkeypatch.setattr(cm, "_nvidia_smi", lambda: "not,a,number\n")
    assert cm.gpu_ok(FakeStatsComfy(0)) is False


def test_gpu_ok_fails_closed_when_system_stats_unreadable(monkeypatch):
    monkeypatch.setattr(cm, "_nvidia_smi", lambda: "9000\n")
    class Broken:
        def system_stats(self):
            raise cm.ComfyError("unreachable")
    assert cm.gpu_ok(Broken()) is False


def test_run_raises_comfy_error_on_transport_failure():
    with pytest.raises(cm.ComfyError):
        cm.Comfy("127.0.0.1:1", timeout=1).run({})


def test_run_wraps_oserror_not_just_url_error(monkeypatch):
    def boom(*a, **k):
        raise ConnectionResetError("reset by peer")
    monkeypatch.setattr(cm.urllib.request, "urlopen", boom)
    with pytest.raises(cm.ComfyError):
        cm.Comfy("127.0.0.1:1", timeout=1).run({})


def test_run_wraps_bad_json_from_history():
    class Handler(Fake):
        def do_GET(self):
            if self.path.startswith("/history/"):
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.end_headers(); self.wfile.write(b"not json")
                return
            super().do_GET()
    Fake.state = {"queued": [], "done": {}, "fail": False}
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    try:
        with pytest.raises(cm.ComfyError):
            cm.Comfy(f"127.0.0.1:{srv.server_port}", timeout=2).run({})
    finally:
        srv.shutdown(); srv.server_close(); t.join(timeout=5)
