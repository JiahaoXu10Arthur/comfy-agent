import sqlite3

import pytest

from comfy_agent.results import Results


def test_render_roundtrip_and_all_landed(tmp_path):
    r = Results(tmp_path / "r.sqlite")
    r.add_render("agent-deepseek-1-7-01", "agent-deepseek", 1, 7, 1, ["workflow:sdxl", "tag:1girl"], "start", "a.png", 4.0)
    r.add_tags("agent-deepseek-1-7-01", {"1girl": ("LANDED", 0.99), "on side": ("MISSING", 0.12)})
    row = r.get("agent-deepseek-1-7-01")
    assert row["factors"] == ["workflow:sdxl", "tag:1girl"] and row["all_landed"] is False and row["min_score"] == 0.12
    r.add_render("agent-deepseek-1-7-02", "agent-deepseek", 1, 7, 2, ["workflow:sdxl"], "knob:seed", "b.png", 4.0)
    r.add_tags("agent-deepseek-1-7-02", {"1girl": ("LANDED", 0.99), "on side": ("LANDED", 0.5)})
    assert r.get("agent-deepseek-1-7-02")["all_landed"] is True
    assert [x["n"] for x in r.renders("agent-deepseek", 1, 7)] == [1, 2]
    r.close()


def test_history_is_scoped_to_same_arm_run_other_tasks(tmp_path):
    r = Results(tmp_path / "r.sqlite")
    r.add_render("a-1-7-01", "agent-deepseek", 1, 7, 1, ["lora:nier2b_anima:1.00"], "start", "", 1)
    r.add_render("a-1-9-01", "agent-deepseek", 1, 9, 1, ["lora:nier2b_anima:1.00"], "start", "", 1)
    r.add_render("a-2-9-01", "agent-deepseek", 2, 9, 1, ["lora:nier2b_anima:1.00"], "start", "", 1)
    r.add_render("g-1-9-01", "grid", 1, 9, 1, ["lora:nier2b_anima:1.00"], "start", "", 1)
    got = r.history("agent-deepseek", 1, exclude_task=7, query="nier2b")
    assert [x["render_id"] for x in got] == ["a-1-9-01"]
    r.close()


def test_conclusion(tmp_path):
    r = Results(tmp_path / "r.sqlite")
    r.add_render("x", "agent-deepseek", 1, 1, 1, [], "start", "", 1)
    r.set_conclusion("x", "NOT_ACHIEVABLE_HERE", "no_progress", "no LoRA for cone hair bun")
    assert r.get("x")["verdict"] == "NOT_ACHIEVABLE_HERE" and r.get("x")["stop_reason"] == "no_progress"
    r.close()


def test_close_closes_the_connection(tmp_path):
    r = Results(tmp_path / "r.sqlite")
    r.add_render("x", "agent-deepseek", 1, 1, 1, [], "start", "", 1)
    r.close()
    with pytest.raises(sqlite3.ProgrammingError):
        r.get("x")
