"""Three runs per arm, the metrics table, and the one sentence §1 asks for."""
import dataclasses
import time
from pathlib import Path

from . import baselines, grid as gridmod, loop

ARMS = ("agent-deepseek", "agent-qwen", "baseline-seed", "baseline-ladder")
LADDER, AGENT = "baseline-ladder", "agent-deepseek"


def run_eval(tasks, catalog, arms, runs, make_provider, make_tools, results, out_dir):
    rows = []
    for arm in arms:
        for r in range(1, runs + 1):
            for task in tasks:
                task = dataclasses.replace(task, seed_hint=1000 + r)
                tools = make_tools(arm, r, task)
                run_dir = Path(out_dir) / arm / str(r) / f"{task.id:03d}"
                if arm == "baseline-seed":
                    o = baselines.run_seed_baseline(task, catalog, tools, run_dir)
                elif arm == "baseline-ladder":
                    o = baselines.run_ladder(task, catalog, tools, run_dir)
                else:
                    o = loop.run_task(task, catalog, make_provider(arm), tools, run_dir)
                rends = results.renders(arm, r, task.id)
                claims = [x["claim"] for x in rends]
                switched = any(a["factors"][:2] != b["factors"][:2] or
                               {f for f in a["factors"] if f.startswith("lora:")} != {f for f in b["factors"] if f.startswith("lora:")}
                               for a, b in zip(rends, rends[1:]))
                used_history = any('"tool": "history"' in line for line in
                                   (run_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()) if (run_dir / "steps.jsonl").exists() else False
                rows.append({"arm": arm, "run": r, "task_id": task.id, "verdict": o.verdict, "stop_reason": o.stop_reason,
                             "renders": o.renders, "last_claim": claims[-1] if claims else "", "switched": switched,
                             "used_history": used_history, "prompt_tokens": o.prompt_tokens, "completion_tokens": o.completion_tokens,
                             "seconds": o.seconds})
    return rows


def _correct(row, truth):
    return row["verdict"] == ("ACHIEVED" if truth[row["task_id"]] else "NOT_ACHIEVABLE_HERE")


def metrics(rows, truth, task_ids):
    out = {}
    for arm in sorted({r["arm"] for r in rows}):
        mine = [r for r in rows if r["arm"] == arm]
        live = [r for r in mine if r["verdict"] != "BLOCKED"]
        correct = [r for r in live if _correct(r, truth)]
        by_task = {t: [r for r in live if r["task_id"] == t] for t in task_ids}
        out[arm] = {
            "n": len(task_ids),
            "blocked": len(mine) - len(live),
            "correct_rate": round(len(correct) / len(live), 3) if live else 0.0,
            "avg_renders": round(sum(r["renders"] for r in live) / len(live), 2) if live else 0.0,
            "switched": sum(1 for t in task_ids if any(r["switched"] for r in by_task[t])),
            "used_history": sum(1 for t in task_ids if any(r["used_history"] for r in by_task[t])),
            "seed_only": sum(1 for r in live if r["verdict"] == "ACHIEVED" and r["last_claim"] == "knob:seed"),
            "pass3": sum(1 for t in task_ids if by_task[t] and len(by_task[t]) == len([r for r in mine if r["task_id"] == t])
                         and all(_correct(r, truth) for r in by_task[t])),
        }
        out[arm]["error_rate"] = round(1 - out[arm]["correct_rate"], 3)
    return out


def conclusion_sentence(m):
    a, b = m[AGENT], m[LADDER]
    fewer = a["avg_renders"] <= b["avg_renders"] - 1
    better = a["error_rate"] < b["error_rate"]
    if fewer and better:
        return "旋钮档 agent 比写死阶梯省渲染、判得准"
    if fewer:
        return "省渲染，判得不比阶梯准"
    if better:
        return "判得准，不省渲染"
    return "量过，旋钮档不需要 agent"


def write_report(path, m, totals):
    cols = ["结论正确率", "平均渲染", "换过 workflow/底模/LoRA", "用过 history", "只靠 seed 成功", "判错率", "pass^3", "BLOCKED"]
    lines = ["| arm | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for arm, x in m.items():
        lines.append(f"| {arm} | {x['correct_rate']:.2f} | {x['avg_renders']:.1f} | {x['switched']}/{x['n']} | {x['used_history']}/{x['n']} | "
                     f"{x['seed_only']}/{x['n']*3} | {x['error_rate']:.2f} | {x['pass3']}/{x['n']} | {x['blocked']}/{x['n']*3} |")
    lines += ["", f"总渲染 {totals['renders']} 张，总耗时 {totals['minutes']} 分钟，总花费 ${totals['dollars']}（含网格）", "",
              "pass^3 = 三遍结论都正确的任务数（τ-bench 定义，不是 pass@3）。BLOCKED 的运行剔除后算其它列。n 小，只做定性。", "",
              "结论：" + (conclusion_sentence(m) if AGENT in m and LADDER in m else "（缺 agent-deepseek 或 baseline-ladder，不下结论）")]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
