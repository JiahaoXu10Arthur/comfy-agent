"""The four tools the model can call. Everything the model must not decide is decided here."""
import time
from pathlib import Path

from . import catalog as cat
from . import factors, graph

VERDICTS = ("ACHIEVED", "NOT_ACHIEVABLE_HERE", "BUDGET_EXHAUSTED", "BLOCKED")

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
            "verdict": {"type": "string", "enum": list(VERDICTS)}, "render_id": {"type": ["string", "null"]}, "missing": {"type": "string"}},
            "required": ["verdict", "render_id", "missing"]}}},
]


class ToolError(Exception):
    pass


class Tools:
    def __init__(self, catalog, vocabulary, comfy, tagger, results, out_dir, arm, run, task):
        self.catalog, self.vocabulary, self.comfy, self.tagger, self.results = catalog, vocabulary, comfy, tagger, results
        self.out_dir, self.arm, self.run, self.task = Path(out_dir), arm, run, task
        self.last_arm, self.n_renders = None, 0
        self._graphs = {w["name"]: graph.load_graph(w["graph"]) for w in catalog["workflows"]}

    def _render_id(self, n):
        return f"{self.arm}-{self.run}-{self.task.id}-{n:02d}"

    def render(self, args):
        try:
            wf = cat.workflow(self.catalog, args["workflow"])
        except KeyError:
            raise ToolError(f"unknown workflow {args.get('workflow')!r}")
        if args["base"] not in wf["bases"]:
            raise ToolError(f"base {args['base']!r} not in catalog for {wf['name']}: {wf['bases']}")
        try:
            arm = factors.arm_from_call(args)
        except (KeyError, TypeError, ValueError) as e:
            raise ToolError(f"bad render arguments: {e}")
        c = factors.check(self.last_arm, arm, args.get("claim", ""), self.catalog, self.vocabulary)
        if not c.ok:
            raise ToolError(f"single_factor rejected: {c.reason}")
        triggers = []
        for f, _ in arm.loras:
            entry = next(l for l in self.catalog["loras"] if l["file"] == f)
            triggers += entry["trigger"]
        # trigger 已在 tags 里的不重复
        tag_names = {t for t, _ in arm.tags}
        triggers = [t for t in triggers if t not in tag_names]
        prompt = graph.build_prompt([{"tag": t, "weight": w} for t, w in arm.tags], triggers)
        g = graph.apply(self._graphs[wf["name"]], wf["base_node"], arm.base,
                        [{"file": f, "strength": s} for f, s in arm.loras], prompt, arm.seed, arm.cfg, arm.steps)
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
            raise ToolError(f"render_id {render_id!r} does not belong to this task")
        return row

    def conclude(self, args):
        verdict = args.get("verdict")
        if verdict not in VERDICTS:
            raise ToolError(f"verdict must be one of {VERDICTS}")
        missing = (args.get("missing") or "").strip()
        rid = args.get("render_id")
        row = None
        if verdict != "BLOCKED" or rid:
            row = self._own(rid)
        if verdict == "ACHIEVED" and not (row and row["all_landed"]):
            raise ToolError("ACHIEVED refused: that render does not have every tag LANDED (run tag_check first)")
        if verdict == "NOT_ACHIEVABLE_HERE" and not (0 < len(missing) <= 200):
            raise ToolError("NOT_ACHIEVABLE_HERE needs missing: one sentence, 1-200 chars")
        evidence = ""
        if row:
            evidence = f"render {row['render_id']}: " + ", ".join(
                f"{t} {v} {s:.2f}" for t, (v, s) in row["scores"].items())
            self.results.set_conclusion(row["render_id"], verdict, args.get("stop_reason", "model"), missing)
        return {"verdict": verdict, "render_id": rid, "evidence": evidence, "missing": missing}

    def dispatch(self, name, args):
        fn = {"render": self.render, "tag_check": self.tag_check, "history": self.history, "conclude": self.conclude}.get(name)
        if fn is None:
            raise ToolError(f"no tool named {name!r}; tools are render, tag_check, history, conclude")
        return fn(args)
