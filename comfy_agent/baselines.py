"""Two baselines that go through the same tools (same gate, same judge, same conclude rules) but no model."""
import json
import time
from pathlib import Path

from tagassert import normalise

from . import catalog as cat
from .comfy import ComfyError
from .loop import Outcome
from .tools import ToolError, evidence_for

COUNT_TAGS = {"1girl", "2girls", "2boys", "3girls"}


class _Blocked(Exception):
    """Raised by `_step` after a second consecutive ComfyError; caller ends the run BLOCKED."""
    pass


def main_tag(scores, must):
    missing = [(scores[t][1], t) for t in must if t in scores and scores[t][0] == "MISSING"]
    if missing:
        return min(missing)[1]
    return next((t for t in must if t not in COUNT_TAGS), must[0])


def _step(tools, task, args, log, steps_file):
    """render + tag_check; returns (row, all_landed). Retries a ComfyError once for render and
    again for tag_check; a second consecutive ComfyError on either raises _Blocked, which the
    runner catches and turns into a BLOCKED Outcome (stop_reason tool_error). Writes one
    steps.jsonl line per completed render+tag_check."""
    t0 = time.time()
    try:
        r = tools.render(args)
    except ComfyError:
        try:
            r = tools.render(args)
        except ComfyError:
            raise _Blocked()
    except ToolError as e:
        raise RuntimeError(f"baseline produced a render the gate rejected: {e}") from e
    try:
        t = tools.tag_check({"render_id": r["render_id"]})
    except ComfyError:
        try:
            t = tools.tag_check({"render_id": r["render_id"]})
        except ComfyError:
            raise _Blocked()
    log.append(f"{r['n']:02d} | {args['claim']} | min {t['min_score']}")
    row = tools.results.get(r["render_id"])
    steps_file.write(json.dumps({"n": r["n"], "claim": args["claim"], "args": args, "graph": tools.last_graph,
                                 "scores": row["scores"], "seconds": round(time.time() - t0, 3)}, ensure_ascii=False) + "\n")
    return row, t["all_landed"]


def _finish(tools, task, run_dir, log, t0, verdict, reason, rid, missing=""):
    """verdict BLOCKED is code-forced (I10 v3.1: only code may issue it), so it writes
    through results.set_conclusion directly instead of tools.conclude, which now rejects
    BLOCKED as it would a model call."""
    ev = ""
    if rid:
        if verdict == "BLOCKED":
            row = tools.results.get(rid)
            ev = evidence_for(row)
            tools.results.set_conclusion(rid, verdict, reason, missing)
        else:
            out = tools.conclude({"verdict": verdict, "render_id": rid, "missing": missing}, stop_reason=reason)
            ev = out["evidence"]
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "summary.md").write_text(
        f"task {task.id}   must: {', '.join(task.must)}   budget {task.budget}\nmodel none ({tools.arm})\n\n"
        + "\n".join(log) + f"\n\n结论 {verdict} @ {rid}（stop_reason {reason}），{tools.n_renders} 张\n"
        + (f"证据（代码回查）：{ev}\n" if ev else ""), encoding="utf-8")
    return Outcome(verdict, reason, rid, missing, ev, tools.n_renders, 0, 0, 0, 0, round(time.time() - t0, 1))


def _base_args(catalog, wf, tags, seed, cfg=None, loras=None, claim="start"):
    return {"workflow": wf["name"], "base": wf["bases"][0], "loras": loras or [], "tags": tags,
            "seed": seed, "cfg": wf["defaults"]["cfg"] if cfg is None else cfg, "steps": wf["defaults"]["steps"], "claim": claim}


def run_seed_baseline(task, catalog, tools, run_dir):
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    t0, log = time.time(), []
    wf = catalog["workflows"][0]
    tags = [{"tag": t, "weight": None} for t in task.must]
    best = None
    with open(run_dir / "steps.jsonl", "w", encoding="utf-8") as steps_file:
        try:
            for i in range(task.budget):
                args = _base_args(catalog, wf, tags, task.seed_hint + i, claim="start" if i == 0 else "knob:seed")
                row, landed = _step(tools, task, args, log, steps_file)
                if best is None or row["min_score"] > best["min_score"]:
                    best = row
                if landed:
                    return _finish(tools, task, run_dir, log, t0, "ACHIEVED", "model", row["render_id"])
        except _Blocked:
            return _finish(tools, task, run_dir, log, t0, "BLOCKED", "tool_error", best["render_id"] if best else None)
        return _finish(tools, task, run_dir, log, t0, "NOT_ACHIEVABLE_HERE", "budget", best["render_id"],
                       f"最低分 tag {main_tag(best['scores'], task.must)} 最高 {best['min_score']:.2f}")


def run_ladder(task, catalog, tools, run_dir):
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    t0, log = time.time(), []
    best = None
    tags0 = [{"tag": t, "weight": None} for t in task.must]

    def remember(row):
        nonlocal best
        if best is None or row["min_score"] > best["min_score"]:
            best = row

    with open(run_dir / "steps.jsonl", "w", encoding="utf-8") as steps_file:
        try:
            for wf in catalog["workflows"]:
                if tools.n_renders >= task.budget:
                    break
                # 起手
                row, landed = _step(tools, task, _base_args(catalog, wf, tags0, task.seed_hint), log, steps_file); remember(row)
                if landed:
                    return _finish(tools, task, run_dir, log, t0, "ACHIEVED", "model", row["render_id"])
                mt = main_tag(row["scores"], task.must)
                # 主 tag 前移
                moved = [{"tag": mt, "weight": None}] + [x for x in tags0 if x["tag"] != mt]
                if moved != tags0 and tools.n_renders < task.budget:
                    row, landed = _step(tools, task, _base_args(catalog, wf, moved, task.seed_hint, claim=f"move:{mt}"), log, steps_file); remember(row)
                    if landed:
                        return _finish(tools, task, run_dir, log, t0, "ACHIEVED", "model", row["render_id"])
                else:
                    moved = tags0
                # (主 tag:1.3)
                weighted = [{"tag": x["tag"], "weight": 1.3 if x["tag"] == mt else None} for x in moved]
                if tools.n_renders < task.budget:
                    row, landed = _step(tools, task, _base_args(catalog, wf, weighted, task.seed_hint, claim=f"weight:{mt}"), log, steps_file); remember(row)
                    if landed:
                        return _finish(tools, task, run_dir, log, t0, "ACHIEVED", "model", row["render_id"])
                # 匹配到 LoRA 就开
                cur_tags, cur_loras = weighted, []
                lora = cat.match_lora(catalog, wf["name"], task.character_tags)
                if lora and tools.n_renders < task.budget:
                    have = {normalise(x["tag"]) for x in weighted}
                    trig = [{"tag": t, "weight": None} for t in lora["trigger"] if normalise(t) not in have]
                    cur_tags, cur_loras = trig + weighted, [{"file": lora["file"], "strength": lora["strength"]}]
                    row, landed = _step(tools, task, _base_args(catalog, wf, cur_tags, task.seed_hint, loras=cur_loras, claim=f"lora:{lora['file']}"), log, steps_file); remember(row)
                    if landed:
                        return _finish(tools, task, run_dir, log, t0, "ACHIEVED", "model", row["render_id"])
                # cfg + 1
                if tools.n_renders < task.budget:
                    row, landed = _step(tools, task, _base_args(catalog, wf, cur_tags, task.seed_hint, cfg=wf["defaults"]["cfg"] + 1,
                                                                loras=cur_loras, claim="knob:cfg"), log, steps_file); remember(row)
                    if landed:
                        return _finish(tools, task, run_dir, log, t0, "ACHIEVED", "model", row["render_id"])
            # 剩余预算批 seed，沿用最后一条 workflow 的最后一套设置
            last = tools.last_arm
            i = 1
            while tools.n_renders < task.budget:
                args = {"workflow": last.workflow, "base": last.base, "loras": [{"file": f, "strength": s} for f, s in last.loras],
                        "tags": [{"tag": t, "weight": w} for t, w in last.tags], "seed": task.seed_hint + i, "cfg": last.cfg,
                        "steps": last.steps, "claim": "knob:seed"}
                row, landed = _step(tools, task, args, log, steps_file); remember(row)
                if landed:
                    return _finish(tools, task, run_dir, log, t0, "ACHIEVED", "model", row["render_id"])
                i += 1
        except _Blocked:
            return _finish(tools, task, run_dir, log, t0, "BLOCKED", "tool_error", best["render_id"] if best else None)
        return _finish(tools, task, run_dir, log, t0, "NOT_ACHIEVABLE_HERE", "budget", best["render_id"],
                       f"最低分 tag {main_tag(best['scores'], task.must)} 最高 {best['min_score']:.2f}")
