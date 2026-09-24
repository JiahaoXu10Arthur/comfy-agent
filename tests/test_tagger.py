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
            if FakeComfyServer.state.get("unparseable"):
                outputs = {"2": {"text": ["not json"]}}
            elif FakeComfyServer.state.get("outputs_override") is not None:
                outputs = FakeComfyServer.state["outputs_override"]
            else:
                outputs = {"2": {"text": [json.dumps({f"tag_for_{image}": 0.5})]}}
            FakeComfyServer.state["history"][pid] = {
                "status": {"status_str": "success", "messages": []},
                "outputs": outputs,
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
        assert any("staged_" in tag for tag in r)


def test_backend_wraps_an_unparseable_tagger_response_as_comfy_error(comfy_server, tmp_path):
    comfy = cm.Comfy(comfy_server, timeout=5)
    backend = ComfyBackend(comfy, "wd-eva02-large-tagger-v3")
    p = tmp_path / "x.png"
    p.write_bytes(b"\x89PNGfake")
    FakeComfyServer.state["unparseable"] = True
    with pytest.raises(cm.ComfyError):
        backend.tag(p)


def test_judge_lands_a_must_tag_from_the_node_s_unescaped_json_key(comfy_server, tmp_path, vocab_path):
    v = load_vocabulary(vocab_path)
    comfy = cm.Comfy(comfy_server, timeout=5)
    backend = ComfyBackend(comfy, "wd-eva02-large-tagger-v3")
    t = Tagger(backend, 0.35, v)
    p = tmp_path / "x.png"
    p.write_bytes(b"\x89PNGfake")
    FakeComfyServer.state["outputs_override"] = {
        "2": {"text": [json.dumps({"2b (nier:automata)": 0.95})]}}
    out = t.judge(p, ["2b (nier:automata)"])
    assert out["2b (nier:automata)"] == ("LANDED", 0.95)


def test_judge_reports_real_fractional_scores_not_flattened_to_one(comfy_server, tmp_path, vocab_path):
    v = load_vocabulary(vocab_path)
    comfy = cm.Comfy(comfy_server, timeout=5)
    backend = ComfyBackend(comfy, "wd-eva02-large-tagger-v3")
    t = Tagger(backend, 0.35, v)
    p = tmp_path / "x.png"
    p.write_bytes(b"\x89PNGfake")
    FakeComfyServer.state["outputs_override"] = {
        "2": {"text": [json.dumps({"on_side": 0.12, "1girl": 0.97})]}}
    out = t.judge(p, ["on side", "1girl"])
    assert out["on side"] == ("MISSING", 0.12)
    assert out["1girl"] == ("LANDED", 0.97)
