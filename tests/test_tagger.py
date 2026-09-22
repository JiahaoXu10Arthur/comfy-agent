from comfy_agent.tagger import Tagger
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
