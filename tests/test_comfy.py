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


def test_run_raises_on_execution_error(server):
    Fake.state["fail"] = True
    with pytest.raises(cm.ComfyError, match="boom"):
        cm.Comfy(server, timeout=5).run({})


def test_gpu_used_parses_nvidia_smi(monkeypatch):
    monkeypatch.setattr(cm, "_nvidia_smi", lambda: "123, 8000\n456, 3000\n")
    assert cm.gpu_used_mb() == 11000
    assert cm.gpu_used_mb(exclude_pids={456}) == 8000
    assert cm.gpu_ok(limit_mb=10240, comfy_pid=456) is True
    assert cm.gpu_ok(limit_mb=10240) is False


def test_gpu_used_zero_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(cm, "_nvidia_smi", lambda: None)
    assert cm.gpu_used_mb() == 0


def test_run_raises_comfy_error_on_transport_failure():
    with pytest.raises(cm.ComfyError):
        cm.Comfy("127.0.0.1:1", timeout=1).run({})
