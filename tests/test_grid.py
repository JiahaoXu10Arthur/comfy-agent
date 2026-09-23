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
    assert len(c) == 24 + 48                    # sdxl 无 LoRA：2×1×2×6；anima 有：2×2×2×6
    assert [x["workflow"] for x in c[:24]] == ["sdxl"] * 24 and c[24]["workflow"] == "anima"
    assert {x["seed"] for x in c} == set(range(1001, 1007))
    t1 = Task(1, "single", ["1girl", "cone hair bun"], 1001, 12, [])
    assert len(grid.cells(t1, catalog_dict)) == 48
    weights = {x["tags"][-1]["weight"] for x in grid.cells(t1, catalog_dict)}
    assert weights == {None, 1.4}


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


def test_run_grid_stops_when_gpu_busy(tmp_path, catalog_dict, vocab_path):
    v = load_vocabulary(vocab_path)
    comfy = FakeComfy()
    tagger = Tagger(Backend(comfy, lambda g: {"1girl": 0.9}), 0.35, v)
    res = Results(tmp_path / "r.sqlite")
    with pytest.raises(RuntimeError):
        grid.run_grid([Task(7, "pose", ["1girl"], 1001, 12, [])], catalog_dict, comfy, tagger, res, tmp_path, gpu_check=lambda: False)
    res.close()
