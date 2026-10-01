"""The loop. The model picks the next tool call; the code decides when it is over."""
import json
import time
from dataclasses import dataclass
from pathlib import Path

from . import catalog as cat
from .comfy import ComfyError
from .provider import ProviderError
from .tools import TOOL_SCHEMAS, BadCall, ToolError

STOP_REASONS = ("model", "budget", "turns", "no_progress", "not_judgeable", "tool_error", "model_error", "bad_call")

SYSTEM_PROMPT = """You drive a ComfyUI image pipeline to make every required tag appear in one render.
You never see images. You see the tagger's per-tag score (0-1, threshold {threshold}) for each render.
Rules the code enforces: one factor per render (say which in claim); switching workflow or base is a restart;
a render has no scores until you call tag_check(render_id) — judge every render before the next one;
conclude ACHIEVED only with a render whose tags all LANDED; NOT_ACHIEVABLE_HERE must say what the catalog lacks.
Budget: {budget} renders. Prefer the cheapest change that could move the lowest-scoring tag. Call one tool per turn."""


@dataclass
class Outcome:
    verdict: str
    stop_reason: str
    render_id: str | None
    missing: str
    evidence: str
    renders: int
    turns: int
    calls: int
    prompt_tokens: int
    completion_tokens: int
    seconds: float


def _score_cell(k, v):
    mark = "" if v[0] == "LANDED" else "!"
    return f"{k}={v[1]:.2f}{mark}"


def _table_row(t):
    if not t.get("scores"):                       # rendered, not yet judged
        return f"{t['n']:02d} | {t['claim']} | {' '.join(t['factors'])} | unchecked: call tag_check(\"{t['render_id']}\") | min -"
    return (f"{t['n']:02d} | {t['claim']} | {' '.join(t['factors'])} | "
            f"{' '.join(_score_cell(k, v) for k, v in t['scores'].items())} | min {t['min_score']}")


def build_messages(task, catalog, tried, pending, last, rejection, n_renders):
    sys_msg = SYSTEM_PROMPT.format(threshold=catalog["threshold"], budget=task.budget)
    wfs = "\n".join(f"{w['name']}: bases {w['bases']}; defaults {w['defaults']}" for w in catalog["workflows"])
    table = "\n".join(_table_row(t) for t in sorted(tried + list(pending.values()), key=lambda t: t["n"])) or "(none yet)"
    user = (f"Task {task.id} ({task.category}). must: {', '.join(task.must)}. budget {task.budget}, used {n_renders}. "
            f"seed_hint {task.seed_hint} (start there; seed is a knob you may change).\n"
            f"Workflows:\n{wfs}\nLoRAs (file | workflow | trigger | kind):\n{cat.for_model(catalog)}\n"
            f"Tried so far (n | claim | factors | scores, ! = MISSING | min):\n{table}\n")
    if last is not None:
        user += f"Last tool result: {json.dumps(last, ensure_ascii=False)}\n"
    if rejection:
        user += f"Your last call was rejected: {rejection}\n"
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]


def run_task(task, catalog, provider, tools, run_dir, max_turns=24, no_progress=3, epsilon=0.02):
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    steps = open(run_dir / "steps.jsonl", "w", encoding="utf-8")
    t0 = time.time()
    tried, checked_ids, last, rejection = [], set(), None, None
    pending = {}                                    # render_id -> row, rendered but not tag_checked
    turns = calls = pt = ct = 0
    bad_calls = tool_errors = 0
    best_m, stale = None, 0                        # per-branch: resets on every restart
    task_best_m, task_best_rid = None, None         # task-wide: highest m ever seen, never resets
    final = None

    def finish(verdict, reason, rid, missing, evidence=""):
        return Outcome(verdict, reason, rid, missing, evidence, tools.n_renders, turns, calls, pt, ct, round(time.time() - t0, 1))

    def forced(verdict, reason):
        row = tools.results.get(task_best_rid) if task_best_rid else None
        missing = ""
        if verdict == "NOT_ACHIEVABLE_HERE" and row:
            tag = min(row["scores"], key=lambda k: row["scores"][k][1])
            missing = f"最低分 tag {tag} 最高 {row['scores'][tag][1]:.2f}（render {task_best_rid}）"
        if row:
            tools.results.set_conclusion(task_best_rid, verdict, reason, missing)
        return finish(verdict, reason, task_best_rid, missing)

    def write_step(record):
        line = {"turn": record["turn"], "tool": record["tool"], "args": record.get("args"),
                "result": record.get("result"), "error": record.get("error"),
                "model_text": record.get("model_text", ""), "usage": record.get("usage"),
                "seconds": round(time.time() - turn_t0, 3)}
        if record["tool"] == "render" and record.get("dispatched"):
            line["graph"] = tools.last_graph
        steps.write(json.dumps(line, ensure_ascii=False) + "\n")

    try:
        while final is None:
            if turns >= max_turns:
                final = forced("BUDGET_EXHAUSTED", "turns"); break
            messages = build_messages(task, catalog, tried, pending, last, rejection, tools.n_renders)
            turn_t0 = time.time()
            try:
                resp = provider.chat(messages, TOOL_SCHEMAS)
            except ProviderError:
                final = forced("BLOCKED", "model_error"); break
            turns += 1; calls += 1
            pt += resp["usage"]["prompt_tokens"]; ct += resp["usage"]["completion_tokens"]
            call = resp.get("tool_call")
            record = {"turn": turns, "tool": call["name"] if call else None, "args": call["arguments"] if call else None,
                      "model_text": resp.get("text", ""), "usage": resp["usage"]}
            if not call or not isinstance(call.get("arguments"), dict) or "_unparseable" in call["arguments"]:
                bad_calls += 1
                rejection = "reply with exactly one tool call in the tools format"
                record["error"] = rejection; write_step(record)
                if bad_calls >= 3:
                    final = forced("BLOCKED", "bad_call")
                continue
            if call["name"] == "render" and tools.n_renders >= task.budget:
                rejection = "budget exhausted"
                record["error"] = rejection; write_step(record)
                final = forced("BUDGET_EXHAUSTED", "budget"); continue
            record["dispatched"] = True
            try:
                result = tools.dispatch(call["name"], call["arguments"])
                rejection = None
                tool_errors = bad_calls = 0
            except BadCall as e:
                rejection = str(e); result = None
                tool_errors = 0
                bad_calls += 1
                if bad_calls >= 3:
                    record["error"] = rejection; write_step(record)
                    final = forced("BLOCKED", "bad_call"); continue
            except ToolError as e:
                rejection = str(e); result = None
                tool_errors = bad_calls = 0
            except ComfyError as e:
                tool_errors += 1
                verb = "tag_check" if call["name"] == "tag_check" else "render"
                rejection = f"{verb} failed: {e}"; result = None
                bad_calls = 0
                if tool_errors >= 2:
                    record["error"] = rejection; write_step(record)
                    final = forced("BLOCKED", "tool_error"); continue
            record["result"] = result; record["error"] = rejection
            write_step(record)
            last = result
            if result is None:
                continue
            if call["name"] == "render":
                if call["arguments"].get("claim") == "start" and tried:
                    best_m, stale = None, 0            # restart: new branch, new baseline
                row = tools.results.get(result["render_id"])
                pending[row["render_id"]] = {"n": row["n"], "claim": row["claim"], "factors": row["factors"],
                                             "render_id": row["render_id"]}
            if call["name"] == "tag_check":
                row = tools.results.get(result["render_id"])
                if any(v[0] == "NOT_JUDGEABLE" for v in row["scores"].values()):
                    final = forced("BLOCKED", "not_judgeable"); continue
                m = row["min_score"]
                if task_best_m is None or m > task_best_m:
                    task_best_m, task_best_rid = m, result["render_id"]
                if result["render_id"] not in checked_ids:
                    checked_ids.add(result["render_id"])
                    pending.pop(result["render_id"], None)
                    tried.append({"n": row["n"], "claim": row["claim"], "factors": row["factors"],
                                  "scores": row["scores"], "min_score": row["min_score"]})
                    if best_m is None or m >= best_m + epsilon:
                        best_m, stale = m, 0
                    else:
                        stale += 1
                    if stale >= no_progress:
                        final = forced("NOT_ACHIEVABLE_HERE", "no_progress"); continue
            if call["name"] == "conclude":
                final = finish(result["verdict"], "model", result["render_id"], result["missing"], result["evidence"]); continue
            if call["name"] == "tag_check" and tools.n_renders >= task.budget and not row["all_landed"]:
                final = forced("BUDGET_EXHAUSTED", "budget"); continue
    finally:
        steps.close()
        if final is not None:
            _summary(run_dir, task, provider, catalog, tried, pending, final)
    return final


def _summary(run_dir, task, provider, catalog, tried, pending, out):
    lines = [f"task {task.id}   must: {', '.join(task.must)}   budget {task.budget}",
             f"model {getattr(provider, 'name', '?')}   tagger {catalog['tagger']} @{catalog['threshold']}", "",
             "#  | claim | factors | scores (! = MISSING) | min"]
    for t in sorted(tried + list(pending.values()), key=lambda t: t["n"]):
        lines.append(_table_row(t))
    lines += ["", f"结论 {out.verdict} @ {out.render_id}（stop_reason {out.stop_reason}），{out.renders} 张，model 调用 {out.calls} 次，"
              f"{out.prompt_tokens}+{out.completion_tokens} tokens，{out.seconds} 秒"]
    if out.evidence:
        lines.append(f"证据（代码回查）：{out.evidence}")
    if out.missing:
        lines.append(f"缺什么：{out.missing}")
    (Path(run_dir) / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
