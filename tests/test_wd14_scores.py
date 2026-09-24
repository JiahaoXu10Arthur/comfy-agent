"""Unit tests for the pure parts of tools/wd14_scores/wd14_scores.py: the csv reader
and select(). Loaded via importlib so the module (and this test) do not need
ComfyUI's folder_paths, numpy, PIL, or onnxruntime installed."""
import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "tools" / "wd14_scores" / "wd14_scores.py"


def _load():
    spec = importlib.util.spec_from_file_location("wd14_scores_under_test", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def wd14():
    return _load()


def test_module_imports_without_comfyui_or_onnxruntime(wd14):
    assert hasattr(wd14, "select") and hasattr(wd14, "read_tags")


def test_read_tags_returns_names_in_row_order(wd14, tmp_path):
    p = tmp_path / "selected_tags.csv"
    p.write_text("tag_id,name,category,count\n0,rating:safe,9,100\n1,1girl,0,200\n2,emilia_(re:zero),4,50\n",
                 encoding="utf-8")
    assert wd14.read_tags(p) == ["rating:safe", "1girl", "emilia_(re:zero)"]


def test_select_drops_tags_below_min_prob(wd14):
    out = wd14.select([0.9, 0.005], ["1girl", "smile"], min_prob=0.01)
    assert out == {"1girl": 0.9}


def test_select_keeps_tags_at_exactly_min_prob(wd14):
    out = wd14.select([0.01], ["smile"], min_prob=0.01)
    assert out == {"smile": 0.01}


def test_select_includes_rating_category_tags(wd14):
    out = wd14.select([0.97], ["rating:safe"], min_prob=0.01)
    assert out == {"rating:safe": 0.97}


def test_select_replaces_underscore_with_space_by_default(wd14):
    out = wd14.select([0.5], ["blue_hair"], min_prob=0.01)
    assert out == {"blue hair": 0.5}


def test_select_keeps_underscore_when_replace_underscore_is_false(wd14):
    out = wd14.select([0.5], ["blue_hair"], min_prob=0.01, replace_underscore=False)
    assert out == {"blue_hair": 0.5}


def test_select_does_not_escape_parentheses(wd14):
    out = wd14.select([0.95], ["2b_(nier:automata)"], min_prob=0.01)
    assert out == {"2b (nier:automata)": 0.95}


def test_select_rounds_probability_to_four_places(wd14):
    out = wd14.select([0.123456789], ["1girl"], min_prob=0.01)
    assert out == {"1girl": 0.1235}
