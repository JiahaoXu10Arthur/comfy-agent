"""python -m comfy_agent <catalog|run|baseline|grid|eval>. Wiring only; no logic lives here."""
import argparse
import json
import time
from pathlib import Path

from . import catalog as cat, comfy as cm, eval as ev, grid as gridmod, loop, baselines
from .provider import Cassette, from_env
from .results import Results
from .tagger import Tagger, comfy_backend
from .tasks import load_categories, load_task, load_tasks, load_vocabulary
from .tools import Tools

DEFAULT_WORKFLOWS = [
    {"name": "sdxl", "graph": "pipelines/sdxl-illustrious.json", "base_node": "ckpt",
     "bases": ["ChenkinNoob-XL-V0.5.safetensors", "oneObsession_v24.safetensors"],
     "defaults": {"steps": 30, "cfg": 6.0, "sampler": "euler_ancestral", "scheduler": "normal", "width": 1024, "height": 1024}},
    {"name": "anima", "graph": "pipelines/anima-clean.json", "base_node": "unet",
     "bases": ["miaomiaoHarem_anima16.safetensors", "anima_baseV10.safetensors"],
     "defaults": {"steps": 30, "cfg": 5.0, "sampler": "euler_ancestral", "scheduler": "normal", "width": 1024, "height": 1024}},
]


def _ctx(a):
    catalog = cat.load(a.catalog)
    vocab = load_vocabulary(catalog["vocabulary"])
    categories = load_categories(catalog["vocabulary"])
    comfy = cm.Comfy(a.comfy)
    tagger = Tagger(comfy_backend(comfy, catalog["tagger"]), catalog["threshold"], vocab)
    results = Results(Path(a.out) / "results.sqlite")
    return catalog, vocab, categories, comfy, tagger, results


def main(argv=None):
    p = argparse.ArgumentParser(prog="comfy-agent")
    p.add_argument("--catalog", default="catalog.yaml"); p.add_argument("--comfy", default="127.0.0.1:8188"); p.add_argument("--out", default="runs")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("catalog"); c.add_argument("action", choices=["build"]); c.add_argument("--loras", required=True)
    c.add_argument("--vocabulary", required=True); c.add_argument("--tagger", default="wd-eva02-large-tagger-v3"); c.add_argument("--threshold", type=float, default=0.35)
    r = sub.add_parser("run"); r.add_argument("--task", required=True); r.add_argument("--arm", default="agent-deepseek"); r.add_argument("--run", type=int, default=1)
    r.add_argument("--cassette", default=None); r.add_argument("--cassette-mode", default="off")
    b = sub.add_parser("baseline"); b.add_argument("--tasks", required=True); b.add_argument("--arm", choices=["baseline-seed", "baseline-ladder"], required=True); b.add_argument("--run", type=int, default=1)
    g = sub.add_parser("grid"); g.add_argument("--tasks", required=True)
    e = sub.add_parser("eval"); e.add_argument("--tasks", required=True); e.add_argument("--arms", default="agent-deepseek,baseline-seed,baseline-ladder"); e.add_argument("--runs", type=int, default=3)
    a = p.parse_args(argv)

    if a.cmd == "catalog":
        cat.write(cat.build(a.loras, DEFAULT_WORKFLOWS, a.tagger, a.vocabulary, a.threshold), a.catalog)
        print(f"wrote {a.catalog}"); return 0
    catalog, vocab, categories, comfy, tagger, results = _ctx(a)
    gpu_check = lambda: cm.gpu_ok(comfy)
    try:
        def make_tools(arm, run, task):
            d = Path(a.out) / arm / str(run) / f"{task.id:03d}"; d.mkdir(parents=True, exist_ok=True)
            return Tools(catalog, vocab, comfy, tagger, results, d, arm, run, task)

        def make_provider(arm):
            name = arm.split("-", 1)[1]
            return from_env(name, Cassette(getattr(a, "cassette", None), getattr(a, "cassette_mode", "off")))

        if a.cmd == "run":
            task = load_task(a.task, vocab, categories)
            if not gpu_check():
                raise RuntimeError("GPU busy: someone else is using more than the limit")
            o = loop.run_task(task, catalog, make_provider(a.arm), make_tools(a.arm, a.run, task), Path(a.out) / a.arm / str(a.run) / f"{task.id:03d}")
            print(json.dumps(o.__dict__, ensure_ascii=False)); return 0
        tasks = load_tasks(a.tasks, vocab, categories)
        if a.cmd == "baseline":
            fn = baselines.run_seed_baseline if a.arm == "baseline-seed" else baselines.run_ladder
            for t in tasks:
                if not gpu_check():
                    raise RuntimeError("GPU busy: someone else is using more than the limit")
                o = fn(t, catalog, make_tools(a.arm, a.run, t), Path(a.out) / a.arm / str(a.run) / f"{t.id:03d}")
                print(t.id, o.verdict, o.renders)
            return 0
        if a.cmd == "grid":
            truth = gridmod.run_grid(tasks, catalog, comfy, tagger, results, Path(a.out) / "grid", gpu_check=gpu_check)
            (Path(a.out) / "truth.json").write_text(json.dumps(truth), encoding="utf-8"); print(truth); return 0
        if a.cmd == "eval":
            t0 = time.time()
            truth = {int(k): v for k, v in json.loads((Path(a.out) / "truth.json").read_text(encoding="utf-8")).items()}
            rows = ev.run_eval(tasks, catalog, a.arms.split(","), a.runs, make_provider, make_tools, results, a.out, gpu_check=gpu_check)
            m = ev.metrics(rows, truth, [t.id for t in tasks])
            n_grid = sum(len(results.renders("grid", 0, t.id)) for t in tasks)
            totals = {"renders": sum(r["renders"] for r in rows) + n_grid, "minutes": round((time.time() - t0) / 60, 1),
                      "dollars": round(sum(r["prompt_tokens"] * 0.27e-6 + r["completion_tokens"] * 1.1e-6
                                           for r in rows if r["arm"] == ev.AGENT), 3)}
            ev.write_report(Path(a.out) / "report.md", m, totals)
            (Path(a.out) / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
            print(ev.conclusion_sentence(m) if ev.AGENT in m and ev.LADDER in m else m); return 0
    finally:
        results.close()
