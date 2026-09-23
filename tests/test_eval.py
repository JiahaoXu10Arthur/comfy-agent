from comfy_agent import eval as ev
from comfy_agent.loop import Outcome


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
    assert a["pass3"] == 1 and b["pass3"] == 1                   # 任务 2 三遍 ladder 都错；任务 1 agent 有一遍 BLOCKED
    assert a["error_rate"] == 0.0 and b["error_rate"] == 0.5
    assert ev.conclusion_sentence(m) == "旋钮档 agent 比写死阶梯省渲染、判得准"


def test_conclusion_variants():
    base = {"avg_renders": 5.0, "error_rate": 0.2}
    assert "不需要 agent" in ev.conclusion_sentence({"agent-deepseek": {"avg_renders": 5.0, "error_rate": 0.2}, "baseline-ladder": base})
    assert ev.conclusion_sentence({"agent-deepseek": {"avg_renders": 3.9, "error_rate": 0.3}, "baseline-ladder": base}) == "省渲染，判得不比阶梯准"
    assert ev.conclusion_sentence({"agent-deepseek": {"avg_renders": 4.5, "error_rate": 0.1}, "baseline-ladder": base}) == "判得准，不省渲染"


def test_write_report(tmp_path):
    m = {"agent-deepseek": dict(correct_rate=0.9, avg_renders=4.0, switched=3, used_history=2, seed_only=1, error_rate=0.1, pass3=8, blocked=0, n=10),
         "baseline-ladder": dict(correct_rate=0.8, avg_renders=6.0, switched=5, used_history=0, seed_only=4, error_rate=0.2, pass3=7, blocked=0, n=10)}
    ev.write_report(tmp_path / "report.md", m, {"renders": 1200, "minutes": 95.5, "dollars": 0.42})
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "agent-deepseek" in text and "总耗时 95.5 分钟" in text and "pass^3" in text and "旋钮档 agent 比写死阶梯省渲染、判得准" in text
