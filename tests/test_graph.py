import json
from pathlib import Path
from comfy_agent import graph as g

ROOT = Path(__file__).resolve().parents[1] / "pipelines"


def test_build_prompt_orders_triggers_first_and_formats_weights():
    tags = [{"tag": "1girl", "weight": None}, {"tag": "on side", "weight": 1.3}]
    assert g.build_prompt(tags, ["2b (nier:automata)"]) == "2b (nier:automata), 1girl, (on side:1.3)"


def test_lora_syntax():
    assert g.lora_syntax([{"file": "a", "strength": 1.0}, {"file": "b", "strength": 0.8}]) == "<lora:a:1.00> <lora:b:0.80>"
    assert g.lora_syntax([]) == ""


def test_apply_sets_named_fields_without_mutating_input():
    base = g.load_graph(ROOT / "sdxl-illustrious.json")
    out = g.apply(base, "ckpt", "oneObsession_v24.safetensors", [{"file": "x", "strength": 0.5}],
                  "1girl, smile", seed=7, cfg=7.0, steps=20)
    assert out["ckpt"]["inputs"]["ckpt_name"] == "oneObsession_v24.safetensors"
    assert out["lora"]["inputs"]["lora_syntax"] == "<lora:x:0.50>"
    assert out["pos"]["inputs"]["text"] == "1girl, smile"
    assert out["ks"]["inputs"] ["seed"] == 7 and out["ks"]["inputs"]["cfg"] == 7.0 and out["ks"]["inputs"]["steps"] == 20
    assert base["pos"]["inputs"]["text"] == ""          # the source graph is untouched
    assert out["neg"]["inputs"]["text"] == base["neg"]["inputs"]["text"]   # the negative prompt is untouched


def test_apply_anima_uses_unet_node():
    base = g.load_graph(ROOT / "anima-clean.json")
    out = g.apply(base, "unet", "anima_baseV10.safetensors", [], "1girl", 1, 5.0, 30)
    assert out["unet"]["inputs"]["unet_name"] == "anima_baseV10.safetensors"
