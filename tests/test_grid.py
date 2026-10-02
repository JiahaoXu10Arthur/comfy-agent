import pytest
from comfy_agent import grid
from comfy_agent.results import Results
from comfy_agent.tasks import Task, load_vocabulary
from comfy_agent.tagger import Tagger


class FakeComfy:
    def __init__(self):
        self.graphs = []

    def run(self, g):
        self.graphs.append(g); return "o.png", b"\x89PNG"


class Backend:
    def __init__(self, comfy, rule):
        self.comfy, self.rule = comfy, rule

    def tag(self, path):
        return self.rule(self.comfy.graphs[-1])


def test_cells_counts_per_lora_match(catalog_dict):
    t9 = Task(9, "character", ["1girl", "2b (nier:automata)"], 1001, 12, ["2b (nier:automata)"])
    c = grid.cells(t9, catalog_dict)
    assert len(c) == 24 + 48                    # sdxl has no LoRA: 2×1×2×6; anima has one: 2×2×2×6
    assert [x["workflow"] for x in c[:24]] == ["sdxl"] * 24 and c[24]["workflow"] == "anima"
    assert {x["seed"] for x in c} == set(range(1001, 1007))
    t1 = Task(1, "single", ["1girl", "cone hair bun"], 1001, 12, [])
    assert len(grid.cells(t1, catalog_dict)) == 48
    weights = {x["tags"][-1]["weight"] for x in grid.cells(t1, catalog_dict)}
    assert weights == {None, 1.4}


def test_cells_does_not_duplicate_a_lora_trigger_that_differs_from_a_must_tag_by_case(catalog_dict):
    t9 = Task(9, "character", ["1girl", "2B_(NieR:Automata)"], 1001, 12, ["2B_(NieR:Automata)"])
    cells = grid.cells(t9, catalog_dict)
    anima_with_lora = next(c for c in cells if c["workflow"] == "anima" and c["loras"])
    assert [t["tag"] for t in anima_with_lora["tags"]] == ["1girl", "2B_(NieR:Automata)"]


def test_run_grid_prompt_does_not_duplicate_a_lora_trigger_that_differs_from_a_must_tag_by_case(tmp_path, catalog_dict, vocab_path):
    v = load_vocabulary(vocab_path)
    comfy = FakeComfy()
    tagger = Tagger(Backend(comfy, lambda g: {"1girl": 0.9}), 0.35, v)
    res = Results(tmp_path / "r2.sqlite")
    task = Task(9, "character", ["1girl", "2B_(NieR:Automata)"], 1001, 12, ["2B_(NieR:Automata)"])
    grid.run_grid([task], catalog_dict, comfy, tagger, res, tmp_path, gpu_check=lambda: True)
    lora_texts = [g["pos"]["inputs"]["text"] for g in comfy.graphs
                  if "unet" in g and "<lora:nier2b_anima" in g["lora"]["inputs"]["lora_syntax"]]
    assert lora_texts
    for text in lora_texts:
        assert "2B_(NieR:Automata)" in text
        assert "2b (nier:automata)" not in text
    res.close()


def test_cells_uses_character_tags_not_must_for_lora_matching(catalog_dict):
    t_no_char = Task(9, "character", ["1girl", "2b (nier:automata)"], 1001, 12, [])
    cells_no_char = grid.cells(t_no_char, catalog_dict)
    assert not any(c["loras"] for c in cells_no_char if c["workflow"] == "anima")

    t_char = Task(9, "character", ["1girl", "2b (nier:automata)"], 1001, 12, ["2b (nier:automata)"])
    cells_char = grid.cells(t_char, catalog_dict)
    assert any(c["loras"] for c in cells_char if c["workflow"] == "anima")


def test_run_grid_marks_achievable_and_checks_gpu(tmp_path, catalog_dict, vocab_path):
    v = load_vocabulary(vocab_path)
    comfy = FakeComfy()
    rule = lambda g: {"1girl": 0.99, "on_side": 0.9 if "oneObsession" in str(g.get("ckpt", {})) else 0.1}
    tagger = Tagger(Backend(comfy, rule), 0.35, v)
    res = Results(tmp_path / "r.sqlite")
    tasks = [Task(7, "pose", ["1girl", "on side"], 1001, 12, []), Task(8, "pose", ["1girl", "stretching"], 1001, 12, [])]
    checks = []
    out = grid.run_grid(tasks, catalog_dict, comfy, tagger, res, tmp_path, gpu_check=lambda: checks.append(1) or True)
    assert out == {7: True, 8: False}
    assert len(comfy.graphs) == 96
    assert len(checks) == 4      # globally 2 workflows x 2 bases, not per task
    bases = []
    for g in comfy.graphs:
        b = g["ckpt"]["inputs"]["ckpt_name"] if "ckpt" in g else g["unet"]["inputs"]["unet_name"]
        if not bases or bases[-1] != b:
            bases.append(b)
    assert bases == ["ChenkinNoob-XL-V0.5.safetensors", "oneObsession_v24.safetensors",
                     "miaomiaoHarem_anima16.safetensors", "anima_baseV10.safetensors"]
    assert grid.achievable(res, 7) is True and grid.achievable(res, 8) is False
    res.close()


def test_run_grid_resumes_without_rerendering_finished_tasks(tmp_path, catalog_dict, vocab_path):
    v = load_vocabulary(vocab_path)
    comfy = FakeComfy()
    rule = lambda g: {"1girl": 0.99, "on_side": 0.9 if "oneObsession" in str(g.get("ckpt", {})) else 0.1}
    tagger = Tagger(Backend(comfy, rule), 0.35, v)
    res = Results(tmp_path / "r.sqlite")
    tasks = [Task(7, "pose", ["1girl", "on side"], 1001, 12, []), Task(8, "pose", ["1girl", "stretching"], 1001, 12, [])]
    out1 = grid.run_grid(tasks, catalog_dict, comfy, tagger, res, tmp_path, gpu_check=lambda: True)
    n_after_first = len(comfy.graphs)
    out2 = grid.run_grid(tasks, catalog_dict, comfy, tagger, res, tmp_path, gpu_check=lambda: True)
    assert out2 == out1
    assert len(comfy.graphs) == n_after_first
    res.close()


def test_run_grid_resumes_by_rerunning_an_unfinished_task_in_full(tmp_path, catalog_dict, vocab_path):
    v = load_vocabulary(vocab_path)
    comfy = FakeComfy()
    rule = lambda g: {"1girl": 0.99, "on_side": 0.9 if "oneObsession" in str(g.get("ckpt", {})) else 0.1,
                      "stretching": 0.1}
    tagger = Tagger(Backend(comfy, rule), 0.35, v)
    res = Results(tmp_path / "r.sqlite")
    t7 = Task(7, "pose", ["1girl", "on side"], 1001, 12, [])
    t8 = Task(8, "pose", ["1girl", "stretching"], 1001, 12, [])
    n_cells_7, n_cells_8 = len(grid.cells(t7, catalog_dict)), len(grid.cells(t8, catalog_dict))

    grid.run_grid([t7], catalog_dict, comfy, tagger, res, tmp_path, gpu_check=lambda: True)
    assert len(res.renders("grid", 0, 7)) == n_cells_7
    n_after_t7 = len(comfy.graphs)

    calls = {"n": 0}

    def flaky_gpu():
        calls["n"] += 1
        return calls["n"] <= 1
    with pytest.raises(RuntimeError, match="GPU busy"):
        grid.run_grid([t7, t8], catalog_dict, comfy, tagger, res, tmp_path, gpu_check=flaky_gpu)
    n_after_interrupt = len(comfy.graphs)
    assert n_after_interrupt > n_after_t7
    n_task8_partial = len(res.renders("grid", 0, 8))
    assert 0 < n_task8_partial < n_cells_8
    assert len(res.renders("grid", 0, 7)) == n_cells_7   # task 7 untouched by the interruption

    out = grid.run_grid([t7, t8], catalog_dict, comfy, tagger, res, tmp_path, gpu_check=lambda: True)
    assert len(comfy.graphs) == n_after_interrupt + n_cells_8   # task 7 not re-rendered
    assert len(res.renders("grid", 0, 7)) == n_cells_7
    assert len(res.renders("grid", 0, 8)) == n_cells_8

    res2 = Results(tmp_path / "r_full.sqlite")
    comfy2 = FakeComfy()
    tagger2 = Tagger(Backend(comfy2, rule), 0.35, v)
    full = grid.run_grid([t7, t8], catalog_dict, comfy2, tagger2, res2, tmp_path / "full", gpu_check=lambda: True)
    assert out == full
    res.close()
    res2.close()


def test_run_grid_stops_when_gpu_busy(tmp_path, catalog_dict, vocab_path):
    v = load_vocabulary(vocab_path)
    comfy = FakeComfy()
    tagger = Tagger(Backend(comfy, lambda g: {"1girl": 0.9}), 0.35, v)
    res = Results(tmp_path / "r.sqlite")
    with pytest.raises(RuntimeError):
        grid.run_grid([Task(7, "pose", ["1girl"], 1001, 12, [])], catalog_dict, comfy, tagger, res, tmp_path, gpu_check=lambda: False)
    res.close()
