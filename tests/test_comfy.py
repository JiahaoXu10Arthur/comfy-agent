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


class FakeVramComfy:
    """Fake Comfy client for the GPU guard: free()/wait_idle() succeed (unless told to raise),
    and system_stats() reports CUDA device-wide usage in bytes (or omits "devices" entirely)."""

    def __init__(self, vram_total_mb=None, vram_free_mb=None, free_raises=False, has_devices=True):
        self.vram_total_mb, self.vram_free_mb = vram_total_mb, vram_free_mb
        self.free_raises, self.has_devices = free_raises, has_devices
        self.freed = self.waited = False

    def free(self):
        if self.free_raises:
            raise cm.ComfyError("free failed")
        self.freed = True

    def wait_idle(self):
        self.waited = True
        return True

    def system_stats(self):
        if not self.has_devices:
            return {}
        return {"devices": [{"vram_total": self.vram_total_mb * 1024 * 1024,
                             "vram_free": self.vram_free_mb * 1024 * 1024}]}


def test_gpu_used_mb_reads_device_usage_after_freeing():
    c = FakeVramComfy(vram_total_mb=12466, vram_free_mb=12466 - 1755)
    assert cm.gpu_used_mb(c) == pytest.approx(1755, abs=0.01)
    assert c.freed and c.waited


def test_gpu_ok_passes_under_limit():
    c = FakeVramComfy(vram_total_mb=12466, vram_free_mb=12466 - 1755)
    assert cm.gpu_ok(c, limit_mb=10240) is True


def test_gpu_ok_fails_over_limit():
    c = FakeVramComfy(vram_total_mb=32000, vram_free_mb=500)
    assert cm.gpu_ok(c, limit_mb=10240) is False


def test_gpu_ok_fails_closed_when_free_raises():
    c = FakeVramComfy(vram_total_mb=1, vram_free_mb=1, free_raises=True)
    assert cm.gpu_ok(c) is False


def test_gpu_ok_fails_closed_when_devices_missing():
    c = FakeVramComfy(has_devices=False)
    assert cm.gpu_ok(c) is False


def test_gpu_busy_message_includes_the_measured_number():
    c = FakeVramComfy(vram_total_mb=32000, vram_free_mb=500)
    msg = cm.gpu_busy_message(c, limit_mb=10240)
    assert "31500" in msg and "10240" in msg


def test_gpu_busy_message_falls_back_when_measurement_fails():
    c = FakeVramComfy(has_devices=False)
    msg = cm.gpu_busy_message(c, limit_mb=10240)
    assert "GPU busy" in msg and "10240" in msg


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
