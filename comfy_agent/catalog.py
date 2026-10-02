"""The whitelist the model chooses from. Built from LoRA Manager metadata; people delete, never add."""
import json
import warnings
from pathlib import Path

import yaml
from tagassert import normalise

MAX_LORAS = 300
BASE_TO_WORKFLOW = {"Illustrious": "sdxl", "Anima": "anima"}


def _norm(tag):
    return normalise(tag)


def read_metadata(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    name = Path(path).name.replace(".metadata.json", "")
    civ = d.get("civitai")
    if not isinstance(civ, dict):
        warnings.warn(f"{name}: no civitai block, nsfw unknown, skipped")
        return None
    if (civ.get("model") or {}).get("nsfw"):
        warnings.warn(f"{name}: nsfw, skipped")
        return None
    wf = BASE_TO_WORKFLOW.get(d.get("base_model"))
    if wf is None:
        warnings.warn(f"{name}: base_model {d.get('base_model')!r} not supported, skipped")
        return None
    words = civ.get("trainedWords") or ""
    if isinstance(words, list):
        words = ",".join(words)
    trigger, seen = [], set()
    for w in words.split(","):
        n = normalise(w)
        if n and n not in seen:
            seen.add(n)
            trigger.append(n)
    try:
        parsed = json.loads(d.get("usage_tips") or "{}")
        strength = float(parsed.get("strength", 1.0)) if isinstance(parsed, dict) else 1.0
    except (ValueError, TypeError):
        strength = 1.0
    tags = d.get("tags") or []
    if isinstance(tags, str):
        kind = tags.split(",")[0].strip()
    else:
        kind = next((t.strip() for t in tags if ":" in t), tags[0].strip() if tags else "")
    return {"file": name, "workflow": wf, "trigger": trigger, "strength": strength, "kind": kind}


def build(loras_dir, workflows, tagger, vocabulary, threshold):
    loras = []
    for p in sorted(Path(loras_dir).rglob("*.metadata.json"), key=lambda p: p.name):
        entry = read_metadata(p)
        if entry:
            loras.append(entry)
    if len(loras) > MAX_LORAS:
        raise ValueError(f"{len(loras)} LoRAs > {MAX_LORAS}; delete some")
    return {"tagger": tagger, "vocabulary": str(vocabulary), "threshold": float(threshold),
            "workflows": workflows, "loras": loras}


def write(catalog, path):
    Path(path).write_text(yaml.safe_dump(catalog, allow_unicode=True, sort_keys=False), encoding="utf-8")


def load(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def workflow(catalog, name):
    for w in catalog["workflows"]:
        if w["name"] == name:
            return w
    raise KeyError(name)


def match_lora(catalog, workflow_name, must):
    want = {_norm(t) for t in must}
    for l in catalog["loras"]:
        if l["workflow"] != workflow_name or not l["trigger"]:
            continue
        have = {_norm(t) for t in l["trigger"]}
        kind = l.get("kind") or ""
        if ":" in kind:
            have.add(_norm(kind.split(":", 1)[1]))
        if want & have:
            return l
    return None


def for_model(catalog):
    return "\n".join(f"{l['file']} | {l['workflow']} | {l['strength']} | {', '.join(l['trigger'])} | {l.get('kind', '')}"
                     for l in catalog["loras"])
