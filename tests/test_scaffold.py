# tests/test_scaffold.py
import json
from pathlib import Path


def test_graphs_have_named_nodes():
    root = Path(__file__).resolve().parents[1] / "pipelines"
    sdxl = json.loads((root / "sdxl-illustrious.json").read_text(encoding="utf-8"))
    anima = json.loads((root / "anima-clean.json").read_text(encoding="utf-8"))
    for name in ("lora", "pos", "neg", "ks", "save"):
        assert name in sdxl and name in anima
    assert "ckpt" in sdxl and "unet" in anima
    assert sdxl["neg"]["inputs"]["text"] and anima["neg"]["inputs"]["text"]
