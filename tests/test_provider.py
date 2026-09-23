import json
import pytest
from comfy_agent import provider as pv


class FakeCompletions:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def create(self, **kw):
        self.calls += 1
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content, self.tool_calls = content, tool_calls


class Call:
    def __init__(self, name, args):
        self.function = type("F", (), {"name": name, "arguments": json.dumps(args)})()


class Resp:
    def __init__(self, msg, pt=10, ct=5):
        self.choices = [type("C", (), {"message": msg})()]
        self.usage = type("U", (), {"prompt_tokens": pt, "completion_tokens": ct})()

    def model_dump(self):
        return {"dumped": True}


def make(replies, cassette=None):
    p = pv.Provider("deepseek", "http://x", "k", "m", cassette or pv.Cassette(None, "off"))
    p.client = type("Cl", (), {"chat": type("Ch", (), {"completions": FakeCompletions(replies)})()})()
    return p


def test_chat_returns_tool_call():
    p = make([Resp(Msg(tool_calls=[Call("render", {"seed": 1})]))])
    out = p.chat([{"role": "user", "content": "go"}], tools=[])
    assert out["tool_call"] == {"name": "render", "arguments": {"seed": 1}} and out["usage"]["prompt_tokens"] == 10


def test_chat_returns_text_when_no_tool_call():
    p = make([Resp(Msg(content="hello"))])
    out = p.chat([], [])
    assert out["tool_call"] is None and out["text"] == "hello"


def test_rate_limit_retries_then_raises(monkeypatch):
    import httpx2
    import openai
    monkeypatch.setattr(pv.time, "sleep", lambda s: None)
    response = httpx2.Response(429, request=httpx2.Request("POST", "http://x"))
    err = openai.RateLimitError("slow", response=response, body=None)
    p = make([err, err, err, err])
    with pytest.raises(pv.ProviderError):
        p.chat([], [])
    assert p.client.chat.completions.calls == 4


def test_cassette_replays_without_calling_model(tmp_path):
    cas = pv.Cassette(tmp_path, "record")
    p = make([Resp(Msg(content="first"))], cas)
    msgs = [{"role": "user", "content": "q"}]
    assert p.chat(msgs, [])["text"] == "first"
    p2 = make([], pv.Cassette(tmp_path, "replay"))
    assert p2.chat(msgs, [])["text"] == "first"
    assert p2.client.chat.completions.calls == 0


def test_cassette_replay_miss_raises_without_calling_model(tmp_path):
    cas = pv.Cassette(tmp_path, "replay")
    p = make([], cas)
    with pytest.raises(pv.ProviderError):
        p.chat([{"role": "user", "content": "q"}], [])
    assert p.client.chat.completions.calls == 0


def test_connection_error_raises_provider_error_without_retry():
    import httpx2
    import openai
    req = httpx2.Request("POST", "http://x")
    p = make([openai.APIConnectionError(request=req)])
    with pytest.raises(pv.ProviderError):
        p.chat([], [])
    assert p.client.chat.completions.calls == 1


def test_empty_choices_raises_provider_error():
    class Empty:
        choices = []
    p = make([Empty()])
    with pytest.raises(pv.ProviderError, match="no choices"):
        p.chat([], [])


def test_from_env_deepseek(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    p = pv.from_env("deepseek")
    assert p.model == "deepseek-chat"
    assert p.name == "deepseek"
    assert str(p.client.base_url).startswith("https://api.deepseek.com")


def test_from_env_qwen():
    p = pv.from_env("qwen")
    assert p.model == "qwen3.5:27b"
    assert str(p.client.base_url).startswith("http://127.0.0.1:11434")


def test_chat_handles_unparseable_tool_arguments():
    call = type("Call", (), {"function": type("F", (), {"name": "render", "arguments": "{not json"})()})()
    p = make([Resp(Msg(tool_calls=[call]))])
    out = p.chat([], [])
    assert out["tool_call"]["arguments"] == {"_unparseable": "{not json"}
