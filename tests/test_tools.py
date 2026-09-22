import pytest
from comfy_agent import tools as tl
from comfy_agent.results import Results
from comfy_agent.tasks import Task, load_vocabulary
from comfy_agent.tagger import Tagger


class FakeComfy:
    def __init__(self):
        self.graphs = []

    def run(self, graph):
        self.graphs.append(graph)
        return "out.png", b"\x89PNGfake"


class FakeBackend:
    def __init__(self):
        self.detected = {"1girl": 0.99, "on_side": 0.1}

    def tag(self, path):
        return self.detected


@pytest.fixture
def tools(tmp_path, catalog_dict, vocab_path):
    v = load_vocabulary(vocab_path)
    task = Task(7, "pose", ["1girl", "on side"], 1001, 12)
    comfy, backend = FakeComfy(), FakeBackend()
    t = tl.Tools(catalog_dict, v, comfy, Tagger(backend, 0.35, v), Results(tmp_path / "r.sqlite"),
                 tmp_path, "agent-deepseek", 1, task)
    t._comfy, t._backend = comfy, backend
    return t


def call(seed=1001, cfg=6.0, tags=None, claim="start", loras=None, workflow="sdxl", base="ChenkinNoob-XL-V0.5.safetensors"):
    return {"workflow": workflow, "base": base, "loras": loras or [],
            "tags": tags or [{"tag": "1girl"}, {"tag": "on side"}], "seed": seed, "cfg": cfg, "steps": 30, "claim": claim}


def test_render_then_tag_check(tools):
    r = tools.render(call())
    assert r["render_id"] == "agent-deepseek-1-7-01" and r["n"] == 1
    assert tools._comfy.graphs[0]["pos"]["inputs"]["text"] == "1girl, on side"
    t = tools.tag_check({"render_id": r["render_id"]})
    assert t["all_landed"] is False and t["min_score"] == 0.1 and t["scores"]["on side"][0] == "MISSING"


def test_render_rejects_second_factor_without_rendering(tools):
    tools.render(call())
    with pytest.raises(tl.ToolError, match="MULTIVARIATE"):
        tools.render(call(tags=[{"tag": "1girl", "weight": 1.2}, {"tag": "on side", "weight": 1.3}], claim="weight:on side"))
    assert len(tools._comfy.graphs) == 1 and tools.n_renders == 1


def test_render_unknown_workflow_or_base_is_tool_error(tools):
    with pytest.raises(tl.ToolError):
        tools.render(call(workflow="flux"))
    with pytest.raises(tl.ToolError):
        tools.render(call(base="nope.safetensors"))


def test_conclude_rules(tools):
    r = tools.render(call())
    tools.tag_check({"render_id": r["render_id"]})
    with pytest.raises(tl.ToolError, match="LANDED"):
        tools.conclude({"verdict": "ACHIEVED", "render_id": r["render_id"], "missing": ""})
    with pytest.raises(tl.ToolError, match="missing"):
        tools.conclude({"verdict": "NOT_ACHIEVABLE_HERE", "render_id": r["render_id"], "missing": ""})
    with pytest.raises(tl.ToolError, match="belong"):
        tools.conclude({"verdict": "NOT_ACHIEVABLE_HERE", "render_id": "other-1-1-01", "missing": "x"})
    tools._backend.detected["on_side"] = 0.9
    r2 = tools.render(call(seed=1002, claim="knob:seed"))
    tools.tag_check({"render_id": r2["render_id"]})
    out = tools.conclude({"verdict": "ACHIEVED", "render_id": r2["render_id"], "missing": ""})
    assert out["evidence"].startswith("render agent-deepseek-1-7-02:") and "on side LANDED 0.90" in out["evidence"]
    assert tools.results.get(r2["render_id"])["verdict"] == "ACHIEVED"


def test_history_excludes_own_task(tools):
    tools.results.add_render("agent-deepseek-1-9-01", "agent-deepseek", 1, 9, 1, ["lora:nier2b_anima:1.00"], "start", "", 1)
    tools.results.add_render("agent-deepseek-1-7-01", "agent-deepseek", 1, 7, 1, ["lora:nier2b_anima:1.00"], "start", "", 1)
    rows = tools.history({"query": "nier2b"})["rows"]
    assert [r["task_id"] for r in rows] == [9]


def test_dispatch_unknown_tool(tools):
    with pytest.raises(tl.ToolError):
        tools.dispatch("paint", {})
