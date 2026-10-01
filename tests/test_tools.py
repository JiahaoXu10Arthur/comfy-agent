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
    task = Task(7, "pose", ["1girl", "on side"], 1001, 12, [])
    comfy, backend = FakeComfy(), FakeBackend()
    t = tl.Tools(catalog_dict, v, comfy, Tagger(backend, 0.35, v), Results(tmp_path / "r.sqlite"),
                 tmp_path, "agent-deepseek", 1, task)
    t._comfy, t._backend = comfy, backend
    yield t
    t.results.close()


def call(seed=1001, cfg=6.0, tags=None, claim="start", loras=None, workflow="sdxl", base="ChenkinNoob-XL-V0.5.safetensors"):
    return {"workflow": workflow, "base": base, "loras": loras or [],
            "tags": tags or [{"tag": "1girl"}, {"tag": "on side"}], "seed": seed, "cfg": cfg, "steps": 30, "claim": claim}


def test_render_then_tag_check(tools):
    r = tools.render(call())
    assert r["render_id"] == "agent-deepseek-1-7-01" and r["n"] == 1
    assert tools._comfy.graphs[0]["pos"]["inputs"]["text"] == "1girl, on side"
    t = tools.tag_check({"render_id": r["render_id"]})
    assert t["all_landed"] is False and t["min_score"] == 0.1 and t["scores"]["on side"][0] == "MISSING"


def test_render_with_lora_builds_prompt_and_lora_syntax(tmp_path, catalog_dict, vocab_path):
    v = load_vocabulary(vocab_path)
    task = Task(9, "character", ["1girl", "2b (nier:automata)"], 1001, 12, ["2b (nier:automata)"])
    comfy = FakeComfy()
    t = tl.Tools(catalog_dict, v, comfy, Tagger(FakeBackend(), 0.35, v), Results(tmp_path / "r.sqlite"),
                tmp_path, "agent-deepseek", 1, task)
    args = {"workflow": "anima", "base": "miaomiaoHarem_anima16.safetensors",
            "loras": [{"file": "nier2b_anima", "strength": 1.0}],
            "tags": [{"tag": "1girl"}], "seed": 1001, "cfg": 5.0, "steps": 30, "claim": "start"}
    r = t.render(args)
    assert r["render_id"] == "agent-deepseek-1-9-01"
    g = comfy.graphs[0]
    assert g["lora"]["inputs"]["lora_syntax"] == "<lora:nier2b_anima:1.00>"
    assert g["pos"]["inputs"]["text"] == "2b (nier:automata), 1girl"
    t.results.close()


def test_render_does_not_duplicate_a_lora_trigger_that_differs_from_a_must_tag_by_case(tmp_path, catalog_dict, vocab_path):
    v = load_vocabulary(vocab_path)
    task = Task(9, "character", ["1girl", "2b (nier:automata)"], 1001, 12, ["2b (nier:automata)"])
    comfy = FakeComfy()
    t = tl.Tools(catalog_dict, v, comfy, Tagger(FakeBackend(), 0.35, v), Results(tmp_path / "dup.sqlite"),
                tmp_path, "agent-deepseek", 1, task)
    args = {"workflow": "anima", "base": "miaomiaoHarem_anima16.safetensors",
            "loras": [{"file": "nier2b_anima", "strength": 1.0}],
            "tags": [{"tag": "1girl"}, {"tag": "2B_(NieR:Automata)"}], "seed": 1001, "cfg": 5.0, "steps": 30, "claim": "start"}
    r = t.render(args)
    assert r["render_id"] == "agent-deepseek-1-9-01"
    assert comfy.graphs[0]["pos"]["inputs"]["text"] == "1girl, 2B_(NieR:Automata)"
    t.results.close()


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


def test_conclude_rejects_a_model_supplied_blocked_verdict(tools):
    r = tools.render(call())
    tools.tag_check({"render_id": r["render_id"]})
    with pytest.raises(tl.BadCall, match="verdict must be one of"):
        tools.conclude({"verdict": "BLOCKED", "render_id": r["render_id"], "missing": ""})
    assert tools.results.get(r["render_id"])["verdict"] is None


def test_conclude_ignores_a_model_supplied_stop_reason(tools):
    r = tools.render(call())
    tools.tag_check({"render_id": r["render_id"]})
    tools.conclude({"verdict": "NOT_ACHIEVABLE_HERE", "render_id": r["render_id"], "missing": "x",
                    "stop_reason": "budget"})
    assert tools.results.get(r["render_id"])["stop_reason"] == "model"


def test_history_excludes_own_task(tools):
    tools.results.add_render("agent-deepseek-1-9-01", "agent-deepseek", 1, 9, 1, ["lora:nier2b_anima:1.00"], "start", "", 1)
    tools.results.add_render("agent-deepseek-1-7-01", "agent-deepseek", 1, 7, 1, ["lora:nier2b_anima:1.00"], "start", "", 1)
    rows = tools.history({"query": "nier2b"})["rows"]
    assert [r["task_id"] for r in rows] == [9]


def test_dispatch_unknown_tool(tools):
    with pytest.raises(tl.ToolError):
        tools.dispatch("paint", {})


def test_render_missing_arguments_is_tool_error(tools):
    with pytest.raises(tl.ToolError):
        tools.render({"workflow": "sdxl"})


def test_rerun_same_scope_overwrites_instead_of_crashing(tmp_path, catalog_dict, vocab_path):
    v = load_vocabulary(vocab_path)
    task = Task(7, "pose", ["1girl", "on side"], 1001, 12, [])
    results = Results(tmp_path / "r.sqlite")
    comfy, backend = FakeComfy(), FakeBackend()
    t1 = tl.Tools(catalog_dict, v, comfy, Tagger(backend, 0.35, v), results, tmp_path, "agent-deepseek", 1, task)
    t1.render(call())
    t1.render(call(seed=1002, claim="knob:seed"))
    assert len(results.renders("agent-deepseek", 1, 7)) == 2

    t2 = tl.Tools(catalog_dict, v, comfy, Tagger(backend, 0.35, v), results, tmp_path, "agent-deepseek", 1, task)
    r = t2.render(call())
    rows = results.renders("agent-deepseek", 1, 7)
    assert [x["render_id"] for x in rows] == [r["render_id"]]
    results.close()


def test_render_unknown_lora_is_tool_error(tools):
    with pytest.raises(tl.ToolError, match="nope"):
        tools.render(call(loras=[{"file": "nope", "strength": 1.0}]))
    assert tools._comfy.graphs == []


def test_conclude_with_null_render_id_names_this_tasks_latest_render(tools):
    # deepseek 2026-10-02: concluded NOT_ACHIEVABLE_HERE three times with render_id null and was refused each time
    conclude = next(s for s in tl.TOOL_SCHEMAS if s["function"]["name"] == "conclude")
    assert conclude["function"]["parameters"]["properties"]["render_id"]["type"] == "string"
    with pytest.raises(tl.ToolError, match="render first"):
        tools.conclude({"verdict": "NOT_ACHIEVABLE_HERE", "render_id": None, "missing": "x"})
    tools.render(call())
    with pytest.raises(tl.ToolError, match=r"one of this task's renders.*agent-deepseek-1-7-01"):
        tools.conclude({"verdict": "NOT_ACHIEVABLE_HERE", "render_id": None, "missing": "x"})


def test_conclude_truncates_a_long_missing_instead_of_refusing(tools):
    # deepseek 2026-10-02: listed all four bases in `missing` (300+ chars) and was refused twice
    r = tools.render(call())
    tools.tag_check({"render_id": r["render_id"]})
    out = tools.conclude({"verdict": "NOT_ACHIEVABLE_HERE", "render_id": r["render_id"], "missing": "x" * 300})
    assert out["verdict"] == "NOT_ACHIEVABLE_HERE"
    assert tools.results.get(r["render_id"])["missing"] == "x" * 200


def test_render_id_may_be_the_row_number_from_the_table(tools):
    # deepseek 2026-10-02: concluded with render_id "03" / "2" — the n column of the table it is shown
    r1 = tools.render(call()); tools.tag_check({"render_id": r1["render_id"]})
    tools._backend.detected["on_side"] = 0.9
    r2 = tools.render(call(seed=1002, claim="knob:seed"))
    t = tools.tag_check({"render_id": "02"})
    assert t["render_id"] == r2["render_id"] and t["all_landed"]
    out = tools.conclude({"verdict": "ACHIEVED", "render_id": "2", "missing": ""})
    assert out["render_id"] == r2["render_id"]
    with pytest.raises(tl.ToolError, match="belong"):
        tools.conclude({"verdict": "ACHIEVED", "render_id": "7", "missing": ""})
