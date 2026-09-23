import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from comfy_agent import comfy as cm
from comfy_agent.tagger import ComfyBackend, Tagger
from comfy_agent.tasks import load_vocabulary


class FakeBackend:
    def __init__(self, detected):
        self.detected = detected

    def tag(self, path):
        return self.detected


def test_judge_maps_each_must_tag(vocab_path):
    v = load_vocabulary(vocab_path)
    t = Tagger(FakeBackend({"1girl": 0.99, "on_side": 0.12, "smile": 0.8}), 0.35, v)
    out = t.judge("x.png", ["1girl", "on side", "smile"])
    assert out["1girl"] == ("LANDED", 0.99)
    assert out["on side"] == ("MISSING", 0.12)
    assert out["smile"] == ("LANDED", 0.8)


def test_judge_missing_tag_not_in_output_is_missing_with_zero(vocab_path):
    v = load_vocabulary(vocab_path)
    t = Tagger(FakeBackend({"1girl": 0.99}), 0.35, v)
    assert t.judge("x.png", ["1girl", "smile"])["smile"] == ("MISSING", 0.0)


def test_judge_keys_output_by_the_original_must_string(vocab_path):
    v = load_vocabulary(vocab_path)
    t = Tagger(FakeBackend({"1girl": 0.99, "on_side": 0.12}), 0.35, v)
    out = t.judge("x.png", ["1girl", "On_Side"])
    assert out["On_Side"] == ("MISSING", 0.12)
    assert "on side" not in out


def test_judge_structural_exclusion_is_not_judgeable(vocab_path):
    v = load_vocabulary(vocab_path)
    t = Tagger(FakeBackend({"1girl": 0.99}), 0.35, v)
    out = t.judge("x.png", ["1girl", "sensitive"])
    assert out["sensitive"] == ("NOT_JUDGEABLE", 0.0)
    assert out["1girl"] == ("LANDED", 0.99)


class FakeComfyServer(BaseHTTPRequestHandler):
    """Keeps every finished prompt in /history, like a real ComfyUI, so a backend that scans
    the whole history instead of its own prompt_id (tagassert 0.1.0's bug, C2) gets caught."""
    state = {"history": {}}

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/history/"):
            pid = self.path.rsplit("/", 1)[1]
            entry = FakeComfyServer.state["history"].get(pid)
            return self._json({pid: entry} if entry else {})
        self._json({}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        if self.path == "/upload/image":
            text = body.decode("latin-1")
            name = text.split('filename="', 1)[1].split('"', 1)[0]
            staged = f"staged_{uuid.uuid4().hex[:8]}_{name}"
            return self._json({"name": staged, "subfolder": "", "type": "input"})
        if self.path == "/prompt":
            payload = json.loads(body)
            image = payload["prompt"]["1"]["inputs"]["image"]
            pid = uuid.uuid4().hex
            FakeComfyServer.state["history"][pid] = {
                "status": {"status_str": "success", "messages": []},
                "outputs": {"2": {"text": [f"tag_for_{image}"]}},
            }
            return self._json({"prompt_id": pid})
        self._json({}, 404)


@pytest.fixture
def comfy_server():
    FakeComfyServer.state = {"history": {}}
    srv = HTTPServer(("127.0.0.1", 0), FakeComfyServer)
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    yield f"127.0.0.1:{srv.server_port}"
    srv.shutdown(); srv.server_close(); t.join(timeout=5)


def test_comfy_backend_scores_each_image_independently(comfy_server, tmp_path):
    comfy = cm.Comfy(comfy_server, timeout=5)
    backend = ComfyBackend(comfy, "wd-eva02-large-tagger-v3")
    results = []
    for name in ("first.png", "second.png", "third.png"):
        p = tmp_path / name
        p.write_bytes(b"\x89PNGfake")
        results.append(backend.tag(p))
    assert results[0] != results[1]
    assert results[1] != results[2]
    assert results[0] != results[2]
    for r, name in zip(results, ("first.png", "second.png", "third.png")):
        assert any(name in tag for tag in r)
