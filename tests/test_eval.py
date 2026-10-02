import json

import pytest

from comfy_agent import eval as ev
from comfy_agent.loop import Outcome
from comfy_agent.results import Results
from comfy_agent.tagger import Tagger
from comfy_agent.tasks import Task, load_vocabulary
from comfy_agent.tools import Tools


def row(arm, run, task_id, verdict, renders, claim_last="knob:cfg", switched=False, history=False):
    return {"arm": arm, "run": run, "task_id": task_id, "verdict": verdict, "renders": renders,
            "last_claim": claim_last, "switched": switched, "used_history": history, "stop_reason": "model"}


def test_metrics_and_conclusion():
    truth = {1: True, 2: False}
    rows = []
    for r in (1, 2, 3):
        rows += [row("agent-deepseek", r, 1, "ACHIEVED", 3, claim_last="knob:seed", switched=True, history=True),
                 row("agent-deepseek", r, 2, "NOT_ACHIEVABLE_HERE", 5),
                 row("baseline-ladder", r, 1, "ACHIEVED", 6),
                 row("baseline-ladder", r, 2, "BUDGET_EXHAUSTED", 12)]
    rows[0]["verdict"], rows[0]["stop_reason"] = "BLOCKED", "tool_error"
    m = ev.metrics(rows, truth, [1, 2])
    a, b = m["agent-deepseek"], m["baseline-ladder"]
    assert a["blocked"] == 1 and a["correct_rate"] == 1.0 and b["correct_rate"] == 0.5
    assert a["avg_renders"] == 4.2 and b["avg_renders"] == 9.0
    assert a["switched"] == 1 and a["used_history"] == 1 and a["seed_only"] == 2 and b["seed_only"] == 0
    assert a["pass3"] == 1 and b["pass3"] == 1                   # task 2: the ladder is wrong on all three runs; task 1: the agent has one BLOCKED run
    assert a["error_rate"] == 0.0 and b["error_rate"] == 0.5
    assert ev.conclusion_sentence(m) == "The agent beats the ladder on renders and on error rate."


def test_switched_ignores_lora_strength_but_not_file_workflow_base():
    same_file = [{"factors": ["workflow:sdxl", "base:A", "lora:fileA:1.00"]},
                 {"factors": ["workflow:sdxl", "base:A", "lora:fileA:1.30"]}]
    assert ev._switched(same_file) is False

    diff_lora_file = [{"factors": ["workflow:sdxl", "base:A", "lora:fileA:1.00"]},
                       {"factors": ["workflow:sdxl", "base:A", "lora:fileB:1.00"]}]
    assert ev._switched(diff_lora_file) is True

    diff_base = [{"factors": ["workflow:sdxl", "base:A"]},
                 {"factors": ["workflow:sdxl", "base:B"]}]
    assert ev._switched(diff_base) is True

    diff_workflow = [{"factors": ["workflow:sdxl", "base:A"]},
                      {"factors": ["workflow:anima", "base:A"]}]
    assert ev._switched(diff_workflow) is True

    single_render = [{"factors": ["workflow:sdxl", "base:A"]}]
    assert ev._switched(single_render) is False


def test_conclusion_variants():
    base = {"avg_renders": 5.0, "error_rate": 0.2}
    assert "does not need an agent" in ev.conclusion_sentence({"agent-deepseek": {"avg_renders": 5.0, "error_rate": 0.2}, "baseline-ladder": base})
    assert ev.conclusion_sentence({"agent-deepseek": {"avg_renders": 3.9, "error_rate": 0.3}, "baseline-ladder": base}) == "Fewer renders, not more accurate than the ladder."
    assert ev.conclusion_sentence({"agent-deepseek": {"avg_renders": 4.5, "error_rate": 0.1}, "baseline-ladder": base}) == "More accurate, not fewer renders."


class FakeComfy:
    def __init__(self):
        self.graphs = []

    def run(self, g):
        self.graphs.append(g)
        return "o.png", b"\x89PNG"


class FakeBackend:
    def tag(self, path):
        return {"1girl": 0.9}


def _eval_fixture(tmp_path, catalog_dict, vocab_path):
    v = load_vocabulary(vocab_path)
    results = Results(tmp_path / "r.sqlite")
    comfy = FakeComfy()
    tagger = Tagger(FakeBackend(), 0.35, v)

    def make_tools(arm, run, task):
        d = tmp_path / arm / str(run) / f"{task.id:03d}"; d.mkdir(parents=True, exist_ok=True)
        return Tools(catalog_dict, v, comfy, tagger, results, d, arm, run, task)

    tasks = [Task(1, "single", ["1girl"], 1001, 3, []), Task(2, "single", ["1girl"], 1001, 3, [])]
    return tasks, results, make_tools


def test_run_eval_checks_gpu_before_every_task(tmp_path, catalog_dict, vocab_path):
    tasks, results, make_tools = _eval_fixture(tmp_path, catalog_dict, vocab_path)
    calls = []
    ev.run_eval(tasks, catalog_dict, ["baseline-seed"], 1, None, make_tools, results, tmp_path,
               gpu_check=lambda: calls.append(1) or True)
    assert len(calls) == 2
    results.close()


def test_run_eval_raises_when_gpu_busy(tmp_path, catalog_dict, vocab_path):
    tasks, results, make_tools = _eval_fixture(tmp_path, catalog_dict, vocab_path)
    with pytest.raises(RuntimeError, match="GPU busy"):
        ev.run_eval(tasks, catalog_dict, ["baseline-seed"], 1, None, make_tools, results, tmp_path, gpu_check=lambda: False)
    results.close()


def test_run_eval_resumes_without_rerunning_finished_triples(tmp_path, catalog_dict, vocab_path):
    tasks, results, make_tools = _eval_fixture(tmp_path, catalog_dict, vocab_path)
    rows1 = ev.run_eval(tasks, catalog_dict, ["baseline-seed"], 1, None, make_tools, results, tmp_path)
    assert len(rows1) == 2
    total_renders = sum(len(results.renders("baseline-seed", 1, t.id)) for t in tasks)
    rows2 = ev.run_eval(tasks, catalog_dict, ["baseline-seed"], 1, None, make_tools, results, tmp_path)
    assert rows2 == rows1
    assert sum(len(results.renders("baseline-seed", 1, t.id)) for t in tasks) == total_renders
    assert len((tmp_path / "rows.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    results.close()


def test_run_eval_ignores_stale_rows_outside_the_requested_scope(tmp_path, catalog_dict, vocab_path):
    tasks, results, make_tools = _eval_fixture(tmp_path, catalog_dict, vocab_path)
    stale = row("baseline-ladder", 1, 1, "ACHIEVED", 1)
    (tmp_path / "rows.jsonl").write_text(json.dumps(stale) + "\n", encoding="utf-8")
    rows = ev.run_eval(tasks, catalog_dict, ["baseline-seed"], 1, None, make_tools, results, tmp_path)
    assert len(rows) == 2 and all(r["arm"] == "baseline-seed" for r in rows)
    lines = (tmp_path / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["arm"] == "baseline-ladder"
    results.close()


def test_write_report(tmp_path):
    m = {"agent-deepseek": dict(correct_rate=0.9, avg_renders=4.0, switched=3, used_history=2, seed_only=1, error_rate=0.1, pass3=8, blocked=0, n=10),
         "baseline-ladder": dict(correct_rate=0.8, avg_renders=6.0, switched=5, used_history=0, seed_only=4, error_rate=0.2, pass3=7, blocked=0, n=10)}
    ev.write_report(tmp_path / "report.md", m, {"renders": 1200, "minutes": 95.5, "dollars": 0.42})
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "agent-deepseek" in text and "95.5 min" in text and "pass^3" in text and "The agent beats the ladder on renders and on error rate." in text
    assert "deepseek-chat list price" in text
    assert "| 4.00 |" in text                                   # two decimals: 1.07 vs 1.13 must not both print as 1.1
