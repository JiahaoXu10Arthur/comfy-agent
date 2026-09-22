"""One OpenAI-compatible chat call with tools, recorded to a cassette for replay tests."""
import hashlib
import json
import os
import time
from pathlib import Path

import openai


class ProviderError(RuntimeError):
    pass


class Cassette:
    def __init__(self, directory, mode="off"):
        self.dir = Path(directory) if directory else None
        self.mode = mode
        if self.dir and mode != "off":
            self.dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(messages, tools):
        return hashlib.sha256(json.dumps([messages, tools], sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def get(self, key):
        if self.mode != "replay":
            return None
        p = self.dir / f"{key}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def put(self, key, response):
        if self.mode == "record":
            (self.dir / f"{key}.json").write_text(json.dumps(response, ensure_ascii=False), encoding="utf-8")


class Provider:
    def __init__(self, name, base_url, api_key, model, cassette):
        self.name, self.model, self.cassette = name, model, cassette
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)

    def chat(self, messages, tools):
        key = Cassette.key(messages, tools)
        cached = self.cassette.get(key)
        if cached is not None:
            return cached
        if self.cassette.mode == "replay":
            raise ProviderError(f"cassette miss in replay mode: {key[:12]}")
        delay = 1
        for attempt in range(4):
            try:
                resp = self.client.chat.completions.create(model=self.model, messages=messages,
                                                           tools=tools or None, tool_choice="auto" if tools else None)
                break
            except openai.RateLimitError as e:
                if attempt == 3:
                    raise ProviderError(f"rate limited 4 times: {e}") from e
                time.sleep(delay); delay *= 4
        msg = resp.choices[0].message
        call = None
        if msg.tool_calls:
            tc = msg.tool_calls[0].function
            try:
                args = json.loads(tc.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_unparseable": tc.arguments}
            call = {"name": tc.name, "arguments": args}
        out = {"tool_call": call, "text": msg.content or "",
               "usage": {"prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
                         "completion_tokens": getattr(resp.usage, "completion_tokens", 0)},
               "raw": resp.model_dump() if hasattr(resp, "model_dump") else {}}
        self.cassette.put(key, out)
        return out


PRESETS = {
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY", "deepseek-chat"),
    "qwen": ("http://127.0.0.1:11434/v1", None, "qwen3.5:27b"),
}


def from_env(name, cassette=None):
    base, env, model = PRESETS[name]
    key = os.environ[env] if env else "ollama"
    return Provider(name, base, key, model, cassette or Cassette(None, "off"))
