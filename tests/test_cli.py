import subprocess, sys

import yaml

from comfy_agent import cli
from comfy_agent.results import Results
from comfy_agent.tagger import Tagger
from comfy_agent.tasks import load_categories, load_vocabulary


def test_cli_help_lists_commands():
    out = subprocess.run([sys.executable, "-m", "comfy_agent", "--help"], capture_output=True, text=True)
    assert out.returncode == 0
    for c in ("catalog", "run", "baseline", "grid", "eval"):
        assert c in out.stdout


class FakeComfy:
    def __init__(self):
        self.graphs = []

    def run(self, g):
        self.graphs.append(g)
        return "o.png", b"\x89PNG"


class FakeBackend:
    def tag(self, path):
        return {"1girl": 0.99, "on_side": 0.99}


def test_cli_end_to_end(tmp_path, monkeypatch, catalog_dict, vocab_path):
    """T14 (I11): grid, then baseline, then eval twice into the same --out, with a fake
    Comfy and tagger backend injected through a monkeypatched cli._ctx. Covers C5's
    rerun-overwrite (baseline and eval both write baseline-ladder/1/7) and eval's resume
    (the second eval call renders nothing new)."""
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    (tasks_dir / "007.yaml").write_text(
        yaml.safe_dump({"id": 7, "category": "pose", "must": ["1girl", "on side"], "seed_hint": 1001, "budget": 2}),
        encoding="utf-8")
    out_dir = tmp_path / "runs"
    db_path = tmp_path / "results.sqlite"

    comfy = FakeComfy()
    vocab = load_vocabulary(vocab_path)
    categories = load_categories(vocab_path)
    tagger = Tagger(FakeBackend(), 0.35, vocab)

    def fake_ctx(a):
        return catalog_dict, vocab, categories, comfy, tagger, Results(db_path)

    def from_env_must_not_be_called(*a, **kw):
        raise AssertionError("make_provider must not be called for baseline arms")

    monkeypatch.setattr(cli, "_ctx", fake_ctx)
    monkeypatch.setattr(cli.cm, "gpu_ok", lambda comfy: True)
    monkeypatch.setattr(cli, "from_env", from_env_must_not_be_called)

    assert cli.main(["--out", str(out_dir), "grid", "--tasks", str(tasks_dir)]) == 0
    assert cli.main(["--out", str(out_dir), "baseline", "--tasks", str(tasks_dir), "--arm", "baseline-ladder"]) == 0
    assert cli.main(["--out", str(out_dir), "eval", "--tasks", str(tasks_dir),
                     "--arms", "baseline-seed,baseline-ladder", "--runs", "1"]) == 0

    n_after_first_eval = len(comfy.graphs)
    assert (out_dir / "report.md").exists()
    assert (out_dir / "rows.json").exists()

    assert cli.main(["--out", str(out_dir), "eval", "--tasks", str(tasks_dir),
                     "--arms", "baseline-seed,baseline-ladder", "--runs", "1"]) == 0
    assert len(comfy.graphs) == n_after_first_eval
