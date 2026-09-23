# comfy_agent/factors.py
"""One factor per render, enforced in code.

Weight changes are adjudicated by attribution-gate (the one thing it models
well). Moves, rewordings, LoRA opens and knob turns are checked here, because
attribution-gate reads a moved or reworded tag as a removal plus an addition.
"""
from dataclasses import dataclass

from attribution_gate import adjudicate
from tagassert import judgeable, normalise

CLAIM_KINDS = ("start", "weight", "move", "reword", "lora", "knob")


@dataclass(frozen=True)
class Arm:
    workflow: str
    base: str
    loras: tuple
    tags: tuple
    seed: int
    cfg: float
    steps: int


@dataclass
class Check:
    ok: bool
    reason: str


def arm_from_call(a):
    return Arm(a["workflow"], a["base"],
               tuple((l["file"], float(l["strength"])) for l in a.get("loras", [])),
               tuple((t["tag"], (float(t["weight"]) if t.get("weight") is not None else None)) for t in a["tags"]),
               int(a["seed"]), float(a["cfg"]), int(a["steps"]))


def gate_arm(tags):
    return [f"({t}:{w})" if w is not None else t for t, w in tags]


def factor_list(arm):
    out = [f"workflow:{arm.workflow}", f"base:{arm.base}"]
    out += [f"lora:{f}:{s:.2f}" for f, s in arm.loras]
    out += [f"tag:{x}" for x in gate_arm(arm.tags)]
    out += [f"seed:{arm.seed}", f"cfg:{arm.cfg}", f"steps:{arm.steps}"]
    return out


def _knobs(arm):
    return {"workflow": arm.workflow, "base": arm.base, "loras": arm.loras,
            "seed": arm.seed, "cfg": arm.cfg, "steps": arm.steps}


def _lora_entry(catalog, file):
    for l in catalog["loras"]:
        if l["file"] == file:
            return l
    return None


def check(prev, cur, claim, catalog, vocabulary):
    kind, _, rest = claim.partition(":")
    if kind not in CLAIM_KINDS:
        return Check(False, f"unknown claim kind {kind!r}; use one of {CLAIM_KINDS}")
    if kind == "start":
        if prev is None or cur.workflow != prev.workflow or cur.base != prev.base:
            return Check(True, "start")
        return Check(False, "claim start is only for the first render or a workflow/base restart")
    if prev is None:
        return Check(False, "first render must claim start")
    if cur.workflow != prev.workflow or cur.base != prev.base:
        return Check(False, "workflow/base changed: that is a restart, claim start")
    pk, ck = _knobs(prev), _knobs(cur)
    knobs_changed = [k for k in pk if pk[k] != ck[k]]
    tags_same = cur.tags == prev.tags

    if kind == "weight":
        if knobs_changed:
            return Check(False, f"weight claim but knobs changed: {knobs_changed}")
        v = adjudicate(gate_arm(prev.tags), gate_arm(cur.tags), rest)
        return Check(v.ok, f"{v.code}: {v.reason}")
    if kind == "move":
        if knobs_changed or sorted(prev.tags) != sorted(cur.tags):
            return Check(False, "move claim but tag set or knobs changed")
        others_prev = [t for t in prev.tags if t[0] != rest]
        others_cur = [t for t in cur.tags if t[0] != rest]
        if others_prev != others_cur or prev.tags == cur.tags:
            return Check(False, f"move claim but the change is not a single move of {rest!r}")
        return Check(True, "move")
    if kind == "reword":
        old, _, new = rest.partition("->")
        if knobs_changed:
            return Check(False, "reword claim but knobs changed")
        if old not in [t for t, _ in prev.tags]:
            return Check(False, f"reword claim: {old!r} is not in the previous tags")
        exp = tuple((new if t == old else t, w) for t, w in prev.tags)
        if exp != cur.tags or old == new:
            return Check(False, f"reword claim but the change is not exactly {old!r}->{new!r}")
        if not judgeable(new, vocabulary):
            return Check(False, f"{new!r} is not judgeable by this tagger")
        return Check(True, "reword")
    if kind == "lora":
        entry = _lora_entry(catalog, rest)
        if entry is None or entry["workflow"] != cur.workflow:
            return Check(False, f"{rest!r} is not a catalog LoRA for workflow {cur.workflow}")
        if knobs_changed != ["loras"] or set(prev.loras) | {(rest, float(entry["strength"]))} != set(cur.loras) \
                or len(cur.loras) != len(prev.loras) + 1:
            return Check(False, "lora claim must add exactly that LoRA at its catalog strength")
        trig_norm = [normalise(t) for t in entry["trigger"]]
        prev_norm = [normalise(t) for t, _ in prev.tags]
        added_norm = [t for t in trig_norm if t not in prev_norm]
        if len(cur.tags) != len(added_norm) + len(prev.tags):
            return Check(False, "lora claim: tags must be prev tags with the LoRA's triggers prepended, nothing else")
        added_cur, rest_cur = cur.tags[:len(added_norm)], cur.tags[len(added_norm):]
        if [normalise(t) for t, _ in added_cur] != added_norm or rest_cur != prev.tags:
            return Check(False, "lora claim: tags must be prev tags with the LoRA's triggers prepended, nothing else")
        return Check(True, "lora")
    if kind == "knob":
        if not tags_same:
            return Check(False, "knob claim but tags changed")
        if rest.startswith("strength:"):
            file = rest.split(":", 1)[1]
            pl, cl = dict(prev.loras), dict(cur.loras)
            if knobs_changed != ["loras"] or set(pl) != set(cl) or \
                    [f for f in pl if pl[f] != cl[f]] != [file]:
                return Check(False, f"strength claim must change only {file!r}")
            return Check(True, "strength")
        if rest not in ("seed", "cfg", "steps"):
            return Check(False, f"knob claim must be one of seed, cfg, steps, strength:<file>; got {rest!r}")
        if knobs_changed != [rest]:
            return Check(False, f"knob claim {rest!r} but changed: {knobs_changed}")
        return Check(True, rest)
    return Check(False, "unreachable")
