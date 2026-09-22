"""Read a pipeline graph and set its fields by node name.

The graphs are checked-in files, never scraped from an output PNG (v0 learned
that a template read back from an image carries every past experiment's edits).
"""
import copy
import json
from pathlib import Path

BASE_FIELD = {"ckpt": "ckpt_name", "unet": "unet_name"}


def load_graph(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lora_syntax(loras):
    return " ".join(f"<lora:{l['file']}:{float(l['strength']):.2f}>" for l in loras)


def build_prompt(tags, triggers):
    parts = list(triggers)
    for t in tags:
        w = t.get("weight")
        parts.append(f"({t['tag']}:{w})" if w is not None else t["tag"])
    return ", ".join(parts)


def apply(graph, base_node, base, loras, prompt, seed, cfg, steps):
    g = copy.deepcopy(graph)
    g[base_node]["inputs"][BASE_FIELD[base_node]] = base
    g["lora"]["inputs"]["lora_syntax"] = lora_syntax(loras)
    g["pos"]["inputs"]["text"] = prompt
    ks = g["ks"]["inputs"]
    ks["seed"], ks["cfg"], ks["steps"] = int(seed), float(cfg), int(steps)
    return g
