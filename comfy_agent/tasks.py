"""Task files and the load-time lint that keeps unjudgeable tags out of the eval."""
import csv
import io
from dataclasses import dataclass
from pathlib import Path

import yaml
from tagassert import judgeable, normalise, STRUCTURAL_EXCLUSIONS
from tagassert.__main__ import read_vocabulary

CATEGORIES = ("single", "multi", "count", "pose", "character")
CHARACTER_CATEGORIES = (3, 4)  # WD14 selected_tags.csv: 3 = copyright, 4 = character (spec §4 v3.1)


class TaskLintError(ValueError):
    pass


@dataclass
class Task:
    id: int
    category: str
    must: list
    seed_hint: int
    budget: int
    character_tags: list


def load_vocabulary(csv_path):
    return read_vocabulary(csv_path)


def load_categories(csv_path):
    """normalised tag name -> selected_tags.csv category (int). tagassert's own
    read_vocabulary drops this column, so C4's LoRA-matching rule reads it directly."""
    rows = list(csv.reader(io.StringIO(Path(csv_path).read_text(encoding="utf-8"))))
    if not rows or "name" not in rows[0] or "category" not in rows[0]:
        raise ValueError(f"{csv_path}: expected a header with name and category columns")
    ni, ci = rows[0].index("name"), rows[0].index("category")
    return {normalise(r[ni]): int(r[ci]) for r in rows[1:] if len(r) > max(ni, ci)}


def load_task(path, vocabulary, categories):
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
    character_tags = [tag for tag in must if categories.get(normalise(tag)) in CHARACTER_CATEGORIES]
    return Task(int(d["id"]), d["category"], must, int(d.get("seed_hint", 1001)), d["budget"], character_tags)


def load_tasks(directory, vocabulary, categories):
    return sorted((load_task(p, vocabulary, categories) for p in Path(directory).glob("*.yaml")), key=lambda x: x.id)
