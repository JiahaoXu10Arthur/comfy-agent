# comfy_agent/grid.py
"""The ground truth: does any cell land every tag. Grouped by workflow and base so the model switches once."""
from itertools import groupby
from pathlib import Path

from tagassert import normalise

from . import catalog as cat, comfy as cm, graph
from .baselines import COUNT_TAGS
from .factors import factor_list, arm_from_call

SEEDS = 6
WEIGHTS = (None, 1.4)


def _main(must):
    return next((t for t in must if t not in COUNT_TAGS), must[0])


def cells(task, catalog):
    out = []
    mt = _main(task.must)
    must_norm = {normalise(t) for t in task.must}
    for wf in catalog["workflows"]:
        lora = cat.match_lora(catalog, wf["name"], task.character_tags)
        lora_axis = [[]] + ([[{"file": lora["file"], "strength": lora["strength"]}]] if lora else [])
        for base in wf["bases"]:
            for loras in lora_axis:
                for w in WEIGHTS:
                    tags = [{"tag": t, "weight": (w if t == mt else None)} for t in task.must]
                    if loras:
                        trig = [{"tag": t, "weight": None} for t in lora["trigger"] if normalise(t) not in must_norm]
                        tags = trig + tags
                    for s in range(SEEDS):
                        out.append({"workflow": wf["name"], "base": base, "loras": loras, "tags": tags,
                                    "seed": task.seed_hint + s, "cfg": wf["defaults"]["cfg"], "steps": wf["defaults"]["steps"],
                                    "claim": "grid"})
    return out


def run_grid(tasks, catalog, comfy, tagger, results, out_dir, gpu_check=None):
    """Cells for every task, sorted globally by (workflow order, base order) so the base
    switches only 3 times across the whole grid (spec §8), not once per task (I8)."""
    gpu_check = gpu_check or (lambda: cm.gpu_ok(comfy))
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    graphs = {w["name"]: graph.load_graph(w["graph"]) for w in catalog["workflows"]}
    wf_order = {w["name"]: i for i, w in enumerate(catalog["workflows"])}
    base_order = {w["name"]: {b: i for i, b in enumerate(w["bases"])} for w in catalog["workflows"]}

    all_cells = [(task, c) for task in tasks for c in cells(task, catalog)]
    all_cells.sort(key=lambda tc: (wf_order[tc[1]["workflow"]], base_order[tc[1]["workflow"]][tc[1]["base"]]))

    n_by_task = {task.id: 0 for task in tasks}
    for (wfname, base), group in groupby(all_cells, key=lambda tc: (tc[1]["workflow"], tc[1]["base"])):
        if not gpu_check():
            raise RuntimeError("GPU busy: someone else is using more than the limit")
        wf = cat.workflow(catalog, wfname)
        for task, c in group:
            must_norm = {normalise(t) for t in task.must}
            n_by_task[task.id] += 1
            n = n_by_task[task.id]
            arm = arm_from_call(c)
            triggers = [t["tag"] for t in c["tags"] if normalise(t["tag"]) not in must_norm]
            prompt = graph.build_prompt([t for t in c["tags"] if normalise(t["tag"]) in must_norm], triggers)
            g = graph.apply(graphs[wfname], wf["base_node"], base, c["loras"], prompt, c["seed"], c["cfg"], c["steps"])
            _, data = comfy.run(g)
            rid = f"grid-0-{task.id}-{n:02d}"
            png = out_dir / f"{rid}.png"; png.write_bytes(data)
            results.add_render(rid, "grid", 0, task.id, n, factor_list(arm), "grid", str(png), 0.0)
            results.add_tags(rid, tagger.judge(str(png), task.must))
    return {task.id: achievable(results, task.id) for task in tasks}


def achievable(results, task_id):
    return any(r["all_landed"] for r in results.renders("grid", 0, task_id))
