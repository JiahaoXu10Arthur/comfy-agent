# tests/test_factors.py
from comfy_agent import factors as f
from comfy_agent.tasks import load_vocabulary


def arm(**over):
    d = dict(workflow="sdxl", base="ChenkinNoob-XL-V0.5.safetensors", loras=(),
             tags=(("1girl", None), ("on side", None)), seed=1001, cfg=6.0, steps=30)
    d.update(over)
    return f.Arm(**d)


def test_start_only_first_or_restart(catalog_dict, vocab_path):
    v = load_vocabulary(vocab_path)
    assert f.check(None, arm(), "start", catalog_dict, v).ok
    assert not f.check(arm(), arm(seed=2), "start", catalog_dict, v).ok
    assert f.check(arm(), arm(workflow="anima", base="anima_baseV10.safetensors", cfg=5.0), "start", catalog_dict, v).ok
    assert f.check(arm(), arm(base="oneObsession_v24.safetensors"), "start", catalog_dict, v).ok


def test_weight_goes_through_attribution_gate(catalog_dict, vocab_path):
    v = load_vocabulary(vocab_path)
    ok = f.check(arm(), arm(tags=(("1girl", None), ("on side", 1.3))), "weight:on side", catalog_dict, v)
    assert ok.ok
    bad = f.check(arm(), arm(tags=(("1girl", 1.2), ("on side", 1.3))), "weight:on side", catalog_dict, v)
    assert not bad.ok and "MULTIVARIATE" in bad.reason
    wrong = f.check(arm(), arm(tags=(("1girl", None), ("on side", 1.3))), "weight:1girl", catalog_dict, v)
    assert not wrong.ok and "CONSTANT" in wrong.reason
    knob_too = f.check(arm(), arm(tags=(("1girl", None), ("on side", 1.3)), cfg=7.0), "weight:on side", catalog_dict, v)
    assert not knob_too.ok


def test_move_and_reword(catalog_dict, vocab_path):
    v = load_vocabulary(vocab_path)
    assert f.check(arm(), arm(tags=(("on side", None), ("1girl", None))), "move:on side", catalog_dict, v).ok
    # 两个 tag 互换既是「move on side」也是「move 1girl」，两种 claim 都放行
    assert f.check(arm(), arm(tags=(("on side", None), ("1girl", None))), "move:1girl", catalog_dict, v).ok
    three = arm(tags=(("1girl", None), ("on side", None), ("smile", None)))
    assert not f.check(three, arm(tags=(("smile", None), ("on side", None), ("1girl", None))), "move:smile", catalog_dict, v).ok
    ok = f.check(arm(), arm(tags=(("1girl", None), ("grey hair", None))), "reword:on side->grey hair", catalog_dict, v)
    assert ok.ok
    bad = f.check(arm(), arm(tags=(("1girl", None), ("silver hair", None))), "reword:on side->silver hair", catalog_dict, v)
    assert not bad.ok and "judgeable" in bad.reason


def test_lora_is_composite(catalog_dict, vocab_path):
    v = load_vocabulary(vocab_path)
    a = arm(tags=(("1girl", None), ("emilia (re:zero)", None)))
    with_lora = arm(loras=(("Emilia-all outfits-illu", 1.0),), tags=(("1girl", None), ("emilia (re:zero)", None)))
    # trigger 已在 tags 里：允许（多出的 trigger 集合为空是因为它本来就在）
    assert f.check(a, with_lora, "lora:Emilia-all outfits-illu", catalog_dict, v).ok
    b = arm(tags=(("1girl", None),))
    with_lora2 = arm(loras=(("Emilia-all outfits-illu", 1.0),), tags=(("emilia (re:zero)", None), ("1girl", None)))
    assert f.check(b, with_lora2, "lora:Emilia-all outfits-illu", catalog_dict, v).ok
    sneaky = arm(loras=(("Emilia-all outfits-illu", 1.0),), tags=(("emilia (re:zero)", None), ("1girl", None), ("smile", None)))
    assert not f.check(b, sneaky, "lora:Emilia-all outfits-illu", catalog_dict, v).ok
    wrong_wf = arm(loras=(("nier2b_anima", 1.0),), tags=(("2b (nier:automata)", None), ("1girl", None)))
    assert not f.check(b, wrong_wf, "lora:nier2b_anima", catalog_dict, v).ok


def test_lora_claim_rejects_a_weight_change_riding_along(catalog_dict, vocab_path):
    v = load_vocabulary(vocab_path)
    prev = arm(workflow="anima", base="miaomiaoHarem_anima16.safetensors", cfg=5.0, tags=(("1girl", None),))
    sneaky = arm(workflow="anima", base="miaomiaoHarem_anima16.safetensors", cfg=5.0,
                loras=(("nier2b_anima", 1.0),), tags=(("2b (nier:automata)", None), ("1girl", 1.4)))
    bad = f.check(prev, sneaky, "lora:nier2b_anima", catalog_dict, v)
    assert not bad.ok
    honest = arm(workflow="anima", base="miaomiaoHarem_anima16.safetensors", cfg=5.0,
                loras=(("nier2b_anima", 1.0),), tags=(("2b (nier:automata)", None), ("1girl", None)))
    assert f.check(prev, honest, "lora:nier2b_anima", catalog_dict, v).ok


def test_knob_exactly_one(catalog_dict, vocab_path):
    v = load_vocabulary(vocab_path)
    assert f.check(arm(), arm(cfg=7.0), "knob:cfg", catalog_dict, v).ok
    assert not f.check(arm(), arm(cfg=7.0, seed=5), "knob:cfg", catalog_dict, v).ok
    assert not f.check(arm(), arm(cfg=7.0), "knob:seed", catalog_dict, v).ok
    a = arm(loras=(("Emilia-all outfits-illu", 1.0),))
    assert f.check(a, arm(loras=(("Emilia-all outfits-illu", 0.8),)), "knob:strength:Emilia-all outfits-illu", catalog_dict, v).ok


def test_knob_rejects_unknown_names(catalog_dict, vocab_path):
    v = load_vocabulary(vocab_path)
    bad = f.check(arm(), arm(loras=(("Emilia-all outfits-illu", 1.0),)), "knob:loras", catalog_dict, v)
    assert not bad.ok and "knob claim" in bad.reason


def test_reword_rejects_noop_when_old_tag_absent(catalog_dict, vocab_path):
    v = load_vocabulary(vocab_path)
    assert not f.check(arm(), arm(), "reword:nonexistent->grey hair", catalog_dict, v).ok


def test_factor_list_and_arm_from_call():
    a = f.arm_from_call({"workflow": "sdxl", "base": "b", "loras": [{"file": "x", "strength": 0.8}],
                         "tags": [{"tag": "1girl", "weight": None}, {"tag": "on side", "weight": 1.3}],
                         "seed": 1, "cfg": 6.0, "steps": 30})
    assert a.loras == (("x", 0.8),) and a.tags == (("1girl", None), ("on side", 1.3))
    assert f.factor_list(a) == ["workflow:sdxl", "base:b", "lora:x:0.80", "tag:1girl", "tag:(on side:1.3)", "seed:1", "cfg:6.0", "steps:30"]
