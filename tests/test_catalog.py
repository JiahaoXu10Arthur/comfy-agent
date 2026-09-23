import json
import warnings
from pathlib import Path
import pytest
from comfy_agent import catalog as c

WORKFLOWS = [
    {"name": "sdxl", "graph": "pipelines/sdxl-illustrious.json", "base_node": "ckpt",
     "bases": ["ChenkinNoob-XL-V0.5.safetensors"], "defaults": {"steps": 30, "cfg": 6.0}},
    {"name": "anima", "graph": "pipelines/anima-clean.json", "base_node": "unet",
     "bases": ["miaomiaoHarem_anima16.safetensors"], "defaults": {"steps": 30, "cfg": 5.0}},
]


def meta(tmp_path, name, **over):
    d = {"base_model": "Illustrious", "tags": ["char:emilia"],
         "usage_tips": json.dumps({"strength": 0.8}),
         "civitai": {"trainedWords": "emilia (re:zero), grey hair", "model": {"nsfw": False}}}
    d.update(over)
    (tmp_path / f"{name}.metadata.json").write_text(json.dumps(d), encoding="utf-8")
    (tmp_path / f"{name}.safetensors").write_bytes(b"")


def test_build_reads_fields_and_orders_by_filename(tmp_path):
    meta(tmp_path, "b_lora"); meta(tmp_path, "a_lora", base_model="Anima")
    cat = c.build(tmp_path, WORKFLOWS, "wd-eva02-large-tagger-v3", "v.csv", 0.35)
    assert [l["file"] for l in cat["loras"]] == ["a_lora", "b_lora"]
    assert cat["loras"][0]["workflow"] == "anima" and cat["loras"][1]["workflow"] == "sdxl"
    assert cat["loras"][1]["trigger"] == ["emilia (re:zero)", "grey hair"]
    assert cat["loras"][1]["strength"] == 0.8 and cat["loras"][1]["kind"] == "char:emilia"


def test_build_skips_nsfw_missing_civitai_and_unknown_base(tmp_path):
    meta(tmp_path, "nsfw", civitai={"trainedWords": "x", "model": {"nsfw": True}})
    meta(tmp_path, "nociv"); (tmp_path / "nociv.metadata.json").write_text(json.dumps({"base_model": "Illustrious"}))
    meta(tmp_path, "pony", base_model="Pony")
    meta(tmp_path, "ok")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cat = c.build(tmp_path, WORKFLOWS, "t", "v", 0.35)
    assert [l["file"] for l in cat["loras"]] == ["ok"]
    assert len(w) >= 2


def test_trained_words_list_shape_with_comma_containing_entries(tmp_path):
    meta(tmp_path, "nier2b_anima", base_model="Anima",
         civitai={"trainedWords": ["nier2b,", "red eyes,glowing eyes"], "model": {"nsfw": False}})
    cat = c.build(tmp_path, WORKFLOWS, "t", "v", 0.35)
    assert cat["loras"][0]["trigger"] == ["nier2b", "red eyes", "glowing eyes"]


def test_missing_trained_words_and_strength_defaults(tmp_path):
    meta(tmp_path, "bare", usage_tips="{}", civitai={"model": {"nsfw": False}})
    cat = c.build(tmp_path, WORKFLOWS, "t", "v", 0.35)
    assert cat["loras"][0]["trigger"] == [] and cat["loras"][0]["strength"] == 1.0


def test_build_rejects_over_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "MAX_LORAS", 1)
    meta(tmp_path, "a"); meta(tmp_path, "b")
    with pytest.raises(ValueError):
        c.build(tmp_path, WORKFLOWS, "t", "v", 0.35)


def test_non_dict_usage_tips_defaults_to_strength_1(tmp_path):
    meta(tmp_path, "weird", usage_tips=json.dumps(["not", "a", "dict"]))
    cat = c.build(tmp_path, WORKFLOWS, "t", "v", 0.35)
    assert cat["loras"][0]["strength"] == 1.0


def test_kind_is_first_comma_separated_tag_stripped(tmp_path):
    meta(tmp_path, "str_tags", tags="concept:foot, concept:detail")
    meta(tmp_path, "list_tags", tags=["char:x", "style:y"])
    cat = c.build(tmp_path, WORKFLOWS, "t", "v", 0.35)
    kinds = {l["file"]: l["kind"] for l in cat["loras"]}
    assert kinds["str_tags"] == "concept:foot"
    assert kinds["list_tags"] == "char:x"


def test_match_lora_per_workflow_first_by_order_and_skips_empty_trigger(catalog_dict):
    m = c.match_lora(catalog_dict, "sdxl", ["1girl", "emilia (re:zero)", "white dress"])
    assert m["file"] == "Emilia-all outfits-illu"
    assert c.match_lora(catalog_dict, "anima", ["1girl", "emilia (re:zero)"]) is None
    # kind 匹配：must 里的 "2b (nier:automata)" 和 kind "char:2b" 交集为空，但 trigger 命中
    assert c.match_lora(catalog_dict, "anima", ["2b (nier:automata)"])["file"] == "nier2b_anima"
    # trigger 空的 notrigger_lora 即使 kind 命中也不算
    assert c.match_lora(catalog_dict, "sdxl", ["2b"]) is None


def test_for_model_one_line_per_lora(catalog_dict):
    text = c.for_model(catalog_dict)
    assert text.count("\n") == len(catalog_dict["loras"]) - 1
    assert "nier2b_anima | anima | 2b (nier:automata) | char:2b" in text


def test_write_load_roundtrip(tmp_path, catalog_dict):
    p = tmp_path / "c.yaml"
    c.write(catalog_dict, p)
    assert c.load(p) == catalog_dict
    assert c.workflow(catalog_dict, "anima")["base_node"] == "unet"
