import json
import pytest
from comfy_agent import baselines as bl
from comfy_agent.comfy import ComfyError
from comfy_agent.tools import Tools
from comfy_agent.results import Results
from comfy_agent.tasks import Task, load_vocabulary
from comfy_agent.tagger import Tagger


class FakeComfy:
    def __init__(self):
        self.graphs = []

    def run(self, graph):
        self.graphs.append(graph)
        return "o.png", b"\x89PNG"


class FlakyComfy:
    """Like FakeComfy, but raises ComfyError from `run` on the 1-indexed call numbers in `fail_on`."""
    def __init__(self, fail_on=()):
        self.graphs = []
        self.calls = 0
        self.fail_on = set(fail_on)

    def run(self, graph):
        self.calls += 1
        if self.calls in self.fail_on:
            raise ComfyError("boom")
        self.graphs.append(graph)
        return "o.png", b"\x89PNG"


class Backend:
    """The second must tag's score is decided by `rule(graph)`; 1girl always lands."""
    def __init__(self, comfy, rule, key="on_side"):
        self.comfy, self.rule, self.key = comfy, rule, key

    def tag(self, path):
        g = self.comfy.graphs[-1]
        return {"1girl": 0.99, self.key: self.rule(g)}


@pytest.fixture
def make(tmp_path, catalog_dict, vocab_path):
    opened = []

    def _make(rule, must=("1girl", "on side"), budget=12, arm="baseline-ladder", key="on_side", fail_on=()):
        v = load_vocabulary(vocab_path)
        comfy = FlakyComfy(fail_on) if fail_on else FakeComfy()
        task = Task(7, "pose", list(must), 1001, budget, [m for m in must if "(" in m])
        results = Results(tmp_path / f"{arm}.sqlite")
        opened.append(results)
        tools = Tools(catalog_dict, v, comfy, Tagger(Backend(comfy, rule, key), 0.35, v), results,
                      tmp_path, arm, 1, task)
        return task, tools, comfy
    yield _make
    for r in opened:
        r.close()


def test_seed_baseline_stops_when_landed(make, tmp_path, catalog_dict):
    task, tools, comfy = make(lambda g: 0.9 if g["ks"]["inputs"]["seed"] == 1003 else 0.1, arm="baseline-seed")
    out = bl.run_seed_baseline(task, catalog_dict, tools, tmp_path / "s")
    assert out.verdict == "ACHIEVED" and out.renders == 3
    assert [g["ks"]["inputs"]["seed"] for g in comfy.graphs] == [1001, 1002, 1003]
    assert all(g["ckpt"]["inputs"]["ckpt_name"] == "ChenkinNoob-XL-V0.5.safetensors" for g in comfy.graphs)


def test_seed_baseline_writes_steps_jsonl(make, tmp_path, catalog_dict):
    task, tools, comfy = make(lambda g: 0.9 if g["ks"]["inputs"]["seed"] == 1003 else 0.1, arm="baseline-seed")
    run_dir = tmp_path / "s"
    bl.run_seed_baseline(task, catalog_dict, tools, run_dir)
    lines = [json.loads(l) for l in (run_dir / "steps.jsonl").read_text().splitlines()]
    assert len(lines) == 3
    for line in lines:
        assert set(line) == {"n", "claim", "args", "graph", "scores", "seconds"}
        assert line["graph"]["ckpt"]["inputs"]["ckpt_name"] == "ChenkinNoob-XL-V0.5.safetensors"
    assert [l["n"] for l in lines] == [1, 2, 3]
    assert lines[-1]["scores"]["on side"][0] == "LANDED"


def test_seed_baseline_exhausts(make, tmp_path, catalog_dict):
    task, tools, _ = make(lambda g: 0.1, budget=4, arm="baseline-seed")
    out = bl.run_seed_baseline(task, catalog_dict, tools, tmp_path / "s")
    assert out.verdict == "NOT_ACHIEVABLE_HERE" and out.renders == 4 and out.stop_reason == "budget"


def test_ladder_order_and_early_stop(make, tmp_path, catalog_dict):
    task, tools, comfy = make(lambda g: 0.9 if "(on side:1.3)" in g["pos"]["inputs"]["text"] else 0.1)
    out = bl.run_ladder(task, catalog_dict, tools, tmp_path / "l")
    assert out.verdict == "ACHIEVED" and out.renders == 3
    texts = [g["pos"]["inputs"]["text"] for g in comfy.graphs]
    assert texts == ["1girl, on side", "on side, 1girl", "(on side:1.3), 1girl"]


def test_ladder_walks_both_workflows_then_seeds(make, tmp_path, catalog_dict):
    task, tools, comfy = make(lambda g: 0.1)
    out = bl.run_ladder(task, catalog_dict, tools, tmp_path / "l")
    assert out.verdict == "NOT_ACHIEVABLE_HERE" and out.renders == 12
    # sdxl: start, move, weight, cfg+1 (无 LoRA 匹配) = 4；anima 同样 4；剩 4 张批 seed
    bases = [g["ckpt"]["inputs"]["ckpt_name"] if "ckpt" in g else g["unet"]["inputs"]["unet_name"] for g in comfy.graphs]
    assert bases[:4] == ["ChenkinNoob-XL-V0.5.safetensors"] * 4 and bases[4:8] == ["miaomiaoHarem_anima16.safetensors"] * 4
    assert comfy.graphs[3]["ks"]["inputs"]["cfg"] == 7.0 and comfy.graphs[7]["ks"]["inputs"]["cfg"] == 6.0
    assert [g["ks"]["inputs"]["seed"] for g in comfy.graphs[8:]] == [1002, 1003, 1004, 1005]


def test_ladder_opens_matching_lora(make, tmp_path, catalog_dict):
    task, tools, comfy = make(lambda g: 0.9 if "<lora:nier2b_anima" in g["lora"]["inputs"]["lora_syntax"] else 0.1,
                              must=("1girl", "2b (nier:automata)"), key="2b_(nier:automata)")
    out = bl.run_ladder(task, catalog_dict, tools, tmp_path / "l")
    assert out.verdict == "ACHIEVED"
    g = comfy.graphs[-1]
    assert g["lora"]["inputs"]["lora_syntax"] == "<lora:nier2b_anima:1.00>" and g["pos"]["inputs"]["text"].startswith("(2b (nier:automata):1.3)")


def test_ladder_lora_step_tolerates_trigger_case_mismatch(tmp_path, catalog_dict, vocab_path):
    v = load_vocabulary(vocab_path)
    cat2 = json.loads(json.dumps(catalog_dict))
    for l in cat2["loras"]:
        if l["file"] == "nier2b_anima":
            l["trigger"] = ["2B (NieR:Automata)"]
    task = Task(9, "character", ["1girl", "2b (nier:automata)"], 1001, 12, ["2b (nier:automata)"])
    comfy = FakeComfy()
    rule = lambda g: 0.9 if "<lora:nier2b_anima" in g["lora"]["inputs"]["lora_syntax"] else 0.1
    tools = Tools(cat2, v, comfy, Tagger(Backend(comfy, rule, key="2b (nier:automata)"), 0.35, v),
                  Results(tmp_path / "case.sqlite"), tmp_path, "baseline-ladder", 1, task)
    out = bl.run_ladder(task, cat2, tools, tmp_path / "l")
    assert out.verdict == "ACHIEVED"
    tools.results.close()


def test_seed_baseline_retries_once_on_comfy_error(make, tmp_path, catalog_dict):
    task, tools, comfy = make(lambda g: 0.9 if g["ks"]["inputs"]["seed"] == 1003 else 0.1,
                               arm="baseline-seed-retry", fail_on=(1,))
    out = bl.run_seed_baseline(task, catalog_dict, tools, tmp_path / "s")
    assert out.verdict == "ACHIEVED" and out.renders == 3
    assert [g["ks"]["inputs"]["seed"] for g in comfy.graphs] == [1001, 1002, 1003]


def test_seed_baseline_blocked_after_two_comfy_errors(make, tmp_path, catalog_dict):
    task, tools, comfy = make(lambda g: 0.9, arm="baseline-seed-blocked", fail_on=(1, 2))
    run_dir = tmp_path / "s"
    out = bl.run_seed_baseline(task, catalog_dict, tools, run_dir)
    assert out.verdict == "BLOCKED" and out.stop_reason == "tool_error" and out.render_id is None
    assert out.renders == 0
    assert (run_dir / "summary.md").exists()


def test_ladder_retries_once_on_comfy_error(make, tmp_path, catalog_dict):
    task, tools, comfy = make(lambda g: 0.9 if "(on side:1.3)" in g["pos"]["inputs"]["text"] else 0.1,
                               arm="baseline-ladder-retry", fail_on=(1,))
    out = bl.run_ladder(task, catalog_dict, tools, tmp_path / "l")
    assert out.verdict == "ACHIEVED" and out.renders == 3
    texts = [g["pos"]["inputs"]["text"] for g in comfy.graphs]
    assert texts == ["1girl, on side", "on side, 1girl", "(on side:1.3), 1girl"]


def test_ladder_blocked_after_two_comfy_errors_with_prior_render(make, tmp_path, catalog_dict):
    task, tools, comfy = make(lambda g: 0.1, arm="baseline-ladder-blocked2", fail_on=(2, 3))
    run_dir = tmp_path / "l"
    out = bl.run_ladder(task, catalog_dict, tools, run_dir)
    assert out.verdict == "BLOCKED" and out.stop_reason == "tool_error"
    assert out.render_id == "baseline-ladder-blocked2-1-7-01"
    assert out.renders == 1
    assert (run_dir / "summary.md").exists()


def test_ladder_blocked_writes_conclusion_to_results_without_going_through_conclude(make, tmp_path, catalog_dict):
    task, tools, comfy = make(lambda g: 0.1, arm="baseline-ladder-blocked3", fail_on=(2, 3))
    out = bl.run_ladder(task, catalog_dict, tools, tmp_path / "l")
    assert out.verdict == "BLOCKED" and out.stop_reason == "tool_error" and out.render_id
    row = tools.results.get(out.render_id)
    assert row["verdict"] == "BLOCKED" and row["stop_reason"] == "tool_error"


def test_step_reraises_tool_error_as_runtime_error(make, tmp_path, catalog_dict):
    task, tools, comfy = make(lambda g: 0.9)
    bad_args = bl._base_args(catalog_dict, catalog_dict["workflows"][0],
                             [{"tag": t, "weight": None} for t in task.must], task.seed_hint)
    bad_args["workflow"] = "not-a-workflow"
    with pytest.raises(RuntimeError, match="baseline produced a render the gate rejected"):
        bl._step(tools, task, bad_args, [], None)
    assert comfy.graphs == []


def test_main_tag():
    assert bl.main_tag({"1girl": ("LANDED", 0.9), "on side": ("MISSING", 0.1), "smile": ("MISSING", 0.2)}, ["1girl", "on side", "smile"]) == "on side"
    assert bl.main_tag({"1girl": ("LANDED", 0.9), "on side": ("LANDED", 0.5)}, ["1girl", "on side"]) == "on side"
