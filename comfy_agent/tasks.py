"""Task files and the load-time lint that keeps unjudgeable tags out of the eval."""
from dataclasses import dataclass
from pathlib import Path

import yaml
from tagassert import judgeable, normalise, STRUCTURAL_EXCLUSIONS
from tagassert.__main__ import read_vocabulary

CATEGORIES = ("single", "multi", "count", "pose", "character")


class TaskLintError(ValueError):
    pass


@dataclass
class Task:
    id: int
    category: str
    must: list
    seed_hint: int
    budget: int


def load_vocabulary(csv_path):
    return read_vocabulary(csv_path)


def load_task(path, vocabulary):
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(d.get("id"), int):
        raise TaskLintError(f"{path}: id must be an int")
    if d.get("category") not in CATEGORIES:
        raise TaskLintError(f"{path}: category {d.get('category')!r} not in {CATEGORIES}")
    if not isinstance(d.get("budget"), int) or d["budget"] < 1:
        raise TaskLintError(f"{path}: budget must be a positive int")
    must = d.get("must")
    if not isinstance(must, list) or not all(isinstance(tag, str) for tag in must):
        raise TaskLintError(f"{path}: must must be a list of tags")
    if not must:
        raise TaskLintError(f"{path}: must is empty")
    rating = [tag for tag in must if normalise(tag) in STRUCTURAL_EXCLUSIONS]
    if rating:
        raise TaskLintError(f"{path}: rating tags cannot be required: {rating}")
    bad = [tag for tag in must if not judgeable(tag, vocabulary)]
    if bad:
        raise TaskLintError(f"{path}: not judgeable by this tagger: {bad}")
    return Task(int(d["id"]), d["category"], must, int(d.get("seed_hint", 1001)), d["budget"])


def load_tasks(directory, vocabulary):
    return sorted((load_task(p, vocabulary) for p in Path(directory).glob("*.yaml")), key=lambda x: x.id)
