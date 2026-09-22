import pytest
import yaml
from comfy_agent import tasks as t


def write(tmp_path, name, **d):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(d), encoding="utf-8")
    return p


def test_load_task_ok(tmp_path, vocab_path):
    v = t.load_vocabulary(vocab_path)
    p = write(tmp_path, "007.yaml", id=7, category="pose", must=["1girl", "on side"], seed_hint=1001, budget=12)
    task = t.load_task(p, v)
    assert task.id == 7 and task.must == ["1girl", "on side"] and task.budget == 12


def test_lint_rejects_tag_outside_vocabulary(tmp_path, vocab_path):
    v = t.load_vocabulary(vocab_path)
    p = write(tmp_path, "001.yaml", id=1, category="single", must=["1girl", "silver hair"], seed_hint=1001, budget=12)
    with pytest.raises(t.TaskLintError, match="silver hair"):
        t.load_task(p, v)


def test_lint_rejects_bad_category_and_budget(tmp_path, vocab_path):
    v = t.load_vocabulary(vocab_path)
    p = write(tmp_path, "001.yaml", id=1, category="misc", must=["1girl"], seed_hint=1, budget=12)
    with pytest.raises(t.TaskLintError):
        t.load_task(p, v)
    p = write(tmp_path, "002.yaml", id=2, category="single", must=["1girl"], seed_hint=1, budget=0)
    with pytest.raises(t.TaskLintError):
        t.load_task(p, v)


def test_lint_rejects_missing_id(tmp_path, vocab_path):
    v = t.load_vocabulary(vocab_path)
    p = write(tmp_path, "001.yaml", category="single", must=["1girl"], seed_hint=1, budget=12)
    with pytest.raises(t.TaskLintError, match="id"):
        t.load_task(p, v)


def test_lint_rejects_non_list_must(tmp_path, vocab_path):
    v = t.load_vocabulary(vocab_path)
    p = write(tmp_path, "001.yaml", id=1, category="single", must="1girl", seed_hint=1, budget=12)
    with pytest.raises(t.TaskLintError, match="list"):
        t.load_task(p, v)


def test_lint_rejects_rating_tag(tmp_path, vocab_path):
    v = t.load_vocabulary(vocab_path)
    p = write(tmp_path, "001.yaml", id=1, category="single", must=["1girl", "sensitive"], seed_hint=1, budget=12)
    with pytest.raises(t.TaskLintError, match="rating"):
        t.load_task(p, v)


def test_load_tasks_sorted(tmp_path, vocab_path):
    v = t.load_vocabulary(vocab_path)
    write(tmp_path, "b.yaml", id=2, category="single", must=["1girl"], seed_hint=1, budget=12)
    write(tmp_path, "a.yaml", id=1, category="single", must=["smile"], seed_hint=1, budget=12)
    assert [x.id for x in t.load_tasks(tmp_path, v)] == [1, 2]
