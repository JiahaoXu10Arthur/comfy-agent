# tests/test_scaffold.py
import ast
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


def test_every_module_parses_on_python_3_11():
    root = Path(__file__).resolve().parents[1] / "comfy_agent"
    for path in root.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 11))
