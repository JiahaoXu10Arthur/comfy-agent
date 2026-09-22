# tests/conftest.py
import json
from pathlib import Path
import pytest
import yaml

VOCAB = ["1girl", "2girls", "smile", "on_side", "grey_hair", "white_dress", "double_v",
         "cone_hair_bun", "emilia_(re:zero)", "outdoors", "long_hair", "purple_eyes",
         "looking_at_viewer", "stretching", "from_behind", "2boys", "3girls", "sitting",
         "facing_another", "2b_(nier:automata)", "gotoh_hitori", "bad_anatomy"]


@pytest.fixture
def vocab_path(tmp_path):
    p = tmp_path / "selected_tags.csv"
    rows = ["tag_id,name,category,count"] + [f"{i},{t},{4 if '(' in t else 0},1000" for i, t in enumerate(VOCAB)]
    p.write_text("\n".join(rows), encoding="utf-8")
    return p


@pytest.fixture
def catalog_dict(vocab_path):
    root = Path(__file__).resolve().parents[1]
    return {
        "tagger": "wd-eva02-large-tagger-v3",
        "vocabulary": str(vocab_path),
        "threshold": 0.35,
        "workflows": [
            {"name": "sdxl", "graph": str(root / "pipelines/sdxl-illustrious.json"), "base_node": "ckpt",
             "bases": ["ChenkinNoob-XL-V0.5.safetensors", "oneObsession_v24.safetensors"],
             "defaults": {"steps": 30, "cfg": 6.0}},
            {"name": "anima", "graph": str(root / "pipelines/anima-clean.json"), "base_node": "unet",
             "bases": ["miaomiaoHarem_anima16.safetensors", "anima_baseV10.safetensors"],
             "defaults": {"steps": 30, "cfg": 5.0}},
        ],
        "loras": [
            {"file": "Emilia-all outfits-illu", "workflow": "sdxl", "trigger": ["emilia (re:zero)"], "strength": 1.0, "kind": "char:emilia"},
            {"file": "ReZero_Emilia_IlluXL", "workflow": "sdxl", "trigger": ["emilia (re:zero)", "grey hair"], "strength": 1.0, "kind": "char:emilia"},
            {"file": "nier2b_anima", "workflow": "anima", "trigger": ["2b (nier:automata)"], "strength": 1.0, "kind": "char:2b"},
            {"file": "notrigger_lora", "workflow": "sdxl", "trigger": [], "strength": 1.0, "kind": "char:2b"},
        ],
    }


@pytest.fixture
def catalog_path(tmp_path, catalog_dict):
    p = tmp_path / "catalog.yaml"
    p.write_text(yaml.safe_dump(catalog_dict, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return p


class ScriptedProvider:
    """Returns canned chat() results in order; records what it was asked."""
    def __init__(self, responses):
        self.responses, self.seen = list(responses), []
        self.name = "scripted"

    def chat(self, messages, tools):
        self.seen.append(messages)
        r = self.responses.pop(0)
        return {"tool_call": r.get("tool_call"), "text": r.get("text", ""),
                "usage": {"prompt_tokens": 1, "completion_tokens": 1}, "raw": {}}
