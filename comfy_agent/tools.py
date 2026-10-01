"""The four tools the model can call. Everything the model must not decide is decided here."""
import time
from pathlib import Path

from tagassert import normalise

from . import catalog as cat
from . import factors, graph

VERDICTS = ("ACHIEVED", "NOT_ACHIEVABLE_HERE", "BUDGET_EXHAUSTED", "BLOCKED")
MODEL_VERDICTS = ("ACHIEVED", "NOT_ACHIEVABLE_HERE", "BUDGET_EXHAUSTED")

_TAGS = {"type": "array", "items": {"type": "object", "properties": {
    "tag": {"type": "string"}, "weight": {"type": ["number", "null"]}}, "required": ["tag"]}}
_LORAS = {"type": "array", "items": {"type": "object", "properties": {
    "file": {"type": "string"}, "strength": {"type": "number"}}, "required": ["file", "strength"]}}

TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "render", "description":
        "Render one image. Exactly one factor may differ from the previous render; say which in claim "
        "(start | weight:<tag> | move:<tag> | reword:<old>-><new> | lora:<file> | knob:<seed|cfg|steps|strength:<file>>). "
        "Changing workflow or base is a restart: claim start.",
        "parameters": {"type": "object", "properties": {
            "workflow": {"type": "string"}, "base": {"type": "string"}, "loras": _LORAS, "tags": _TAGS,
            "seed": {"type": "integer"}, "cfg": {"type": "number"}, "steps": {"type": "integer"}, "claim": {"type": "string"}},
            "required": ["workflow", "base", "loras", "tags", "seed", "cfg", "steps", "claim"]}}},
    {"type": "function", "function": {"name": "tag_check", "description": "Judge a render with the tagger. Returns per-tag verdict and score.",
        "parameters": {"type": "object", "properties": {"render_id": {"type": "string"}}, "required": ["render_id"]}}},
    {"type": "function", "function": {"name": "history", "description": "Renders from other tasks in this run whose factors or tags contain the query.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "conclude", "description":
        "Stop. verdict ACHIEVED needs a render_id whose tags all LANDED. NOT_ACHIEVABLE_HERE needs missing: what the catalog lacks.",
        "parameters": {"type": "object", "properties": {
            "verdict": {"type": "string", "enum": list(MODEL_VERDICTS)},
            "render_id": {"type": "string", "description": "one of this task's render ids; for NOT_ACHIEVABLE_HERE the best render you have"},
            "missing": {"type": "string"}},
            "required": ["verdict", "render_id", "missing"]}}},
]


class ToolError(Exception):
    pass


class BadCall(ToolError):
    """An argument-shape error: the call could never have succeeded (unknown workflow/base/
    LoRA file, bad render arguments, unknown verdict, a render_id that is missing or belongs
    to another task). Counts toward loop.py's three-consecutive-bad-calls BLOCK; a gate
    rejection (single_factor, conclude refusal) does not."""
    pass


def evidence_for(row):
    return f"render {row['render_id']}: " + ", ".join(f"{t} {v} {s:.2f}" for t, (v, s) in row["scores"].items())


class Tools:
    def __init__(self, catalog, vocabulary, comfy, tagger, results, out_dir, arm, run, task):
        self.catalog, self.vocabulary, self.comfy, self.tagger, self.results = catalog, vocabulary, comfy, tagger, results
        self.out_dir, self.arm, self.run, self.task = Path(out_dir), arm, run, task
        self.last_arm, self.n_renders, self.last_graph = None, 0, None
        self._graphs = {w["name"]: graph.load_graph(w["graph"]) for w in catalog["workflows"]}
        self.results.delete_scope(arm, run, task.id)

    def _render_id(self, n):
        return f"{self.arm}-{self.run}-{self.task.id}-{n:02d}"

    def render(self, args):
        self.last_graph = None
        try:
            wf = cat.workflow(self.catalog, args["workflow"])
        except KeyError:
            raise BadCall(f"unknown workflow {args.get('workflow')!r}")
        try:
            base = args["base"]
        except KeyError as e:
            raise BadCall(f"bad render arguments: {e}") from e
        if base not in wf["bases"]:
            raise BadCall(f"base {base!r} not in catalog for {wf['name']}: {wf['bases']}")
        try:
            arm = factors.arm_from_call(args)
        except (KeyError, TypeError, ValueError) as e:
            raise BadCall(f"bad render arguments: {e}")
        c = factors.check(self.last_arm, arm, args.get("claim", ""), self.catalog, self.vocabulary)
        if not c.ok:
            raise ToolError(f"single_factor rejected: {c.reason}")
        catalog_loras = {l["file"]: l for l in self.catalog["loras"] if l["workflow"] == wf["name"]}
        triggers = []
        for f, _ in arm.loras:
            entry = catalog_loras.get(f)
            if entry is None:
                raise BadCall(f"LoRA {f!r} is not in the catalog for workflow {wf['name']}")
            triggers += entry["trigger"]
        # trigger 已在 tags 里的不重复
        tag_names = {normalise(t) for t, _ in arm.tags}
        triggers = [t for t in triggers if normalise(t) not in tag_names]
        prompt = graph.build_prompt([{"tag": t, "weight": w} for t, w in arm.tags], triggers)
        g = graph.apply(self._graphs[wf["name"]], wf["base_node"], arm.base,
                        [{"file": f, "strength": s} for f, s in arm.loras], prompt, arm.seed, arm.cfg, arm.steps)
        self.last_graph = g
        t0 = time.time()
        _, data = self.comfy.run(g)
        seconds = round(time.time() - t0, 1)
        n = self.n_renders + 1
        rid = self._render_id(n)
        png = self.out_dir / f"{rid}.png"
        png.write_bytes(data)
        self.results.add_render(rid, self.arm, self.run, self.task.id, n, factors.factor_list(arm),
                                args.get("claim", ""), str(png), seconds)
        self.last_arm, self.n_renders = arm, n
        return {"render_id": rid, "png": str(png), "seconds": seconds, "n": n}

    def tag_check(self, args):
        row = self._own(args.get("render_id"))
        scores = self.tagger.judge(row["png"], self.task.must)
        self.results.add_tags(row["render_id"], scores)
        row = self.results.get(row["render_id"])
        return {"render_id": row["render_id"], "scores": {k: list(v) for k, v in row["scores"].items()},
                "all_landed": row["all_landed"], "min_score": row["min_score"]}

    def history(self, args):
        rows = self.results.history(self.arm, self.run, self.task.id, str(args.get("query", "")))
        return {"rows": [{"task_id": r["task_id"], "factors": r["factors"],
                          "scores": {k: list(v) for k, v in r["scores"].items()}} for r in rows]}

    def _own(self, render_id):
        row = self.results.get(render_id) if render_id else None
        if row is None or (row["arm"], row["run"], row["task_id"]) != (self.arm, self.run, self.task.id):
            own = self.results.renders(self.arm, self.run, self.task.id)
            hint = f"latest: {own[-1]['render_id']}" if own else "none yet: render first"
            raise BadCall(f"render_id {render_id!r} does not belong to this task; pass one of this task's renders ({hint})")
        return row

    def conclude(self, args, stop_reason="model"):
        verdict = args.get("verdict")
        if verdict not in MODEL_VERDICTS:
            raise BadCall(f"verdict must be one of {MODEL_VERDICTS}")
        missing = (args.get("missing") or "").strip()
        rid = args.get("render_id")
        row = self._own(rid)
        if verdict == "ACHIEVED" and not row["all_landed"]:
            raise ToolError("ACHIEVED refused: that render does not have every tag LANDED (run tag_check first)")
        if verdict == "NOT_ACHIEVABLE_HERE" and not (0 < len(missing) <= 200):
            raise ToolError("NOT_ACHIEVABLE_HERE needs missing: one sentence, 1-200 chars")
        evidence = evidence_for(row)
        self.results.set_conclusion(row["render_id"], verdict, stop_reason, missing)
        return {"verdict": verdict, "render_id": rid, "evidence": evidence, "missing": missing}

    def dispatch(self, name, args):
        fn = {"render": self.render, "tag_check": self.tag_check, "history": self.history, "conclude": self.conclude}.get(name)
        if fn is None:
            raise BadCall(f"no tool named {name!r}; tools are render, tag_check, history, conclude")
        return fn(args)
