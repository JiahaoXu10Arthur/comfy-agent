import json

import pytest
import yaml
from comfy_agent import catalog as c
from comfy_agent import tasks as t


def write(tmp_path, name, **d):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(d), encoding="utf-8")
    return p


def test_load_task_ok(tmp_path, vocab_path):
    v = t.load_vocabulary(vocab_path)
    categories = t.load_categories(vocab_path)
    p = write(tmp_path, "007.yaml", id=7, category="pose", must=["1girl", "on side"], seed_hint=1001, budget=12)
    task = t.load_task(p, v, categories)
    assert task.id == 7 and task.must == ["1girl", "on side"] and task.budget == 12


def test_lint_rejects_tag_outside_vocabulary(tmp_path, vocab_path):
    v = t.load_vocabulary(vocab_path)
    categories = t.load_categories(vocab_path)
    p = write(tmp_path, "001.yaml", id=1, category="single", must=["1girl", "silver hair"], seed_hint=1001, budget=12)
    with pytest.raises(t.TaskLintError, match="silver hair"):
        t.load_task(p, v, categories)


def test_lint_rejects_bad_category_and_budget(tmp_path, vocab_path):
    v = t.load_vocabulary(vocab_path)
    categories = t.load_categories(vocab_path)
    p = write(tmp_path, "001.yaml", id=1, category="misc", must=["1girl"], seed_hint=1, budget=12)
    with pytest.raises(t.TaskLintError):
        t.load_task(p, v, categories)
    p = write(tmp_path, "002.yaml", id=2, category="single", must=["1girl"], seed_hint=1, budget=0)
    with pytest.raises(t.TaskLintError):
        t.load_task(p, v, categories)


def test_lint_rejects_missing_id(tmp_path, vocab_path):
    v = t.load_vocabulary(vocab_path)
    categories = t.load_categories(vocab_path)
    p = write(tmp_path, "001.yaml", category="single", must=["1girl"], seed_hint=1, budget=12)
    with pytest.raises(t.TaskLintError, match="id"):
        t.load_task(p, v, categories)


def test_lint_rejects_non_list_must(tmp_path, vocab_path):
    v = t.load_vocabulary(vocab_path)
    categories = t.load_categories(vocab_path)
    p = write(tmp_path, "001.yaml", id=1, category="single", must="1girl", seed_hint=1, budget=12)
    with pytest.raises(t.TaskLintError, match="list"):
        t.load_task(p, v, categories)


def test_lint_rejects_rating_tag(tmp_path, vocab_path):
    v = t.load_vocabulary(vocab_path)
    categories = t.load_categories(vocab_path)
    p = write(tmp_path, "001.yaml", id=1, category="single", must=["1girl", "sensitive"], seed_hint=1, budget=12)
    with pytest.raises(t.TaskLintError, match="rating"):
        t.load_task(p, v, categories)


def test_load_tasks_sorted(tmp_path, vocab_path):
    v = t.load_vocabulary(vocab_path)
    categories = t.load_categories(vocab_path)
    write(tmp_path, "b.yaml", id=2, category="single", must=["1girl"], seed_hint=1, budget=12)
    write(tmp_path, "a.yaml", id=1, category="single", must=["smile"], seed_hint=1, budget=12)
    assert [x.id for x in t.load_tasks(tmp_path, v, categories)] == [1, 2]


def test_load_categories_reads_category_column(vocab_path):
    categories = t.load_categories(vocab_path)
    assert categories["2b (nier:automata)"] == 4
    assert categories["1girl"] == 0


def test_load_task_fills_character_tags(tmp_path, vocab_path):
    v = t.load_vocabulary(vocab_path)
    categories = t.load_categories(vocab_path)
    p = write(tmp_path, "009.yaml", id=9, category="character", must=["1girl", "2b (nier:automata)"], seed_hint=1001, budget=12)
    task = t.load_task(p, v, categories)
    assert task.character_tags == ["2b (nier:automata)"]


def test_character_tags_exclude_generic_tags_that_would_otherwise_match_a_lora(tmp_path, vocab_path, catalog_dict):
    v = t.load_vocabulary(vocab_path)
    categories = t.load_categories(vocab_path)
    cat2 = json.loads(json.dumps(catalog_dict))
    cat2["loras"].append({"file": "generic_1girl_lora", "workflow": "sdxl", "trigger": ["1girl"], "strength": 1.0, "kind": ""})
    p = write(tmp_path, "001.yaml", id=1, category="single", must=["1girl", "smile"], seed_hint=1001, budget=12)
    task = t.load_task(p, v, categories)
    assert task.character_tags == []
    assert c.match_lora(cat2, "sdxl", task.character_tags) is None


def test_character_tag_matches_lora_via_kind_even_when_trigger_is_unrelated(tmp_path, vocab_path, catalog_dict):
    v = t.load_vocabulary(vocab_path)
    categories = t.load_categories(vocab_path)
    cat2 = json.loads(json.dumps(catalog_dict))
    for l in cat2["loras"]:
        if l["file"] == "nier2b_anima":
            l["trigger"] = ["nier2b"]
            l["kind"] = "char:2b (nier:automata)"
    p = write(tmp_path, "009.yaml", id=9, category="character", must=["1girl", "2b (nier:automata)"], seed_hint=1001, budget=12)
    task = t.load_task(p, v, categories)
    assert task.character_tags == ["2b (nier:automata)"]
    assert c.match_lora(cat2, "anima", task.character_tags)["file"] == "nier2b_anima"
