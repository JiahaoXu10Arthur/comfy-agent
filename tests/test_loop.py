import json
import pytest
from comfy_agent import loop as lp
from comfy_agent.tools import Tools, ToolError
from comfy_agent.results import Results
from comfy_agent.tasks import Task, load_vocabulary
from comfy_agent.tagger import Tagger
from comfy_agent.comfy import ComfyError
from conftest import ScriptedProvider


class FakeComfy:
    def __init__(self, fail_times=0):
        self.fail_times, self.calls = fail_times, 0

    def run(self, graph):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ComfyError("execution failed: out of memory")
        return "o.png", b"\x89PNG"


class FakeBackend:
    def __init__(self, seq):
        self.seq = list(seq)

    def tag(self, path):
        s = self.seq.pop(0) if len(self.seq) > 1 else self.seq[0]
        return {"1girl": 0.99, "on_side": s}


def render_call(seed, claim, w=None):
    tags = [{"tag": "1girl"}, {"tag": "on side", "weight": w}]
    return {"tool_call": {"name": "render", "arguments": {"workflow": "sdxl", "base": "ChenkinNoob-XL-V0.5.safetensors",
            "loras": [], "tags": tags, "seed": seed, "cfg": 6.0, "steps": 30, "claim": claim}}}


def tc(name, **args):
    return {"tool_call": {"name": name, "arguments": args}}


@pytest.fixture
def make(tmp_path, catalog_dict, vocab_path):
    def _make(scores, responses, comfy=None, budget=12):
        v = load_vocabulary(vocab_path)
        task = Task(7, "pose", ["1girl", "on side"], 1001, budget)
        comfy = comfy or FakeComfy()
        tools = Tools(catalog_dict, v, comfy, Tagger(FakeBackend(scores), 0.35, v), Results(tmp_path / "r.sqlite"),
                      tmp_path, "agent-deepseek", 1, task)
        return task, ScriptedProvider(responses), tools
    return _make


def test_achieves_and_writes_summary(make, tmp_path, catalog_dict):
    task, prov, tools = make([0.1, 0.9], [
        render_call(1001, "start"), tc("tag_check", render_id="agent-deepseek-1-7-01"),
        render_call(1001, "weight:on side", w=1.3), tc("tag_check", render_id="agent-deepseek-1-7-02"),
        tc("conclude", verdict="ACHIEVED", render_id="agent-deepseek-1-7-02", missing="")])
    out = lp.run_task(task, catalog_dict, prov, tools, tmp_path / "run")
    assert out.verdict == "ACHIEVED" and out.renders == 2 and out.turns == 5 and out.stop_reason == "model"
    lines = (tmp_path / "run/steps.jsonl").read_text().splitlines()
    assert len(lines) == 5 and json.loads(lines[0])["tool"] == "render"
    summ = (tmp_path / "run/summary.md").read_text()
    assert "ACHIEVED" in summ and "weight:on side" in summ
    # 第二轮的输入里有已试过的表
    assert "on side" in json.dumps(prov.seen[2], ensure_ascii=False)


def test_no_progress_forces_not_achievable(make, tmp_path):
    task, prov, tools = make([0.10, 0.11, 0.115, 0.118], [
        render_call(1001, "start"), tc("tag_check", render_id="agent-deepseek-1-7-01"),
        render_call(1002, "knob:seed"), tc("tag_check", render_id="agent-deepseek-1-7-02"),
        render_call(1003, "knob:seed"), tc("tag_check", render_id="agent-deepseek-1-7-03"),
        render_call(1004, "knob:seed"), tc("tag_check", render_id="agent-deepseek-1-7-04"),
        tc("conclude", verdict="ACHIEVED", render_id="x", missing="")])
    out = lp.run_task(task, tools.catalog, prov, tools, tmp_path / "run")
    assert out.verdict == "NOT_ACHIEVABLE_HERE" and out.stop_reason == "no_progress" and out.renders == 4
    assert out.missing.startswith("最低分 tag on side")


def test_budget_exhausted(make, tmp_path):
    resp = []
    for i in range(3):
        resp += [render_call(1001 + i, "start" if i == 0 else "knob:seed"), tc("tag_check", render_id=f"agent-deepseek-1-7-0{i+1}")]
    resp += [render_call(1010, "knob:seed")]
    task, prov, tools = make([0.1, 0.5, 0.1, 0.6], resp, budget=3)
    out = lp.run_task(task, tools.catalog, prov, tools, tmp_path / "run")
    assert out.verdict == "BUDGET_EXHAUSTED" and out.stop_reason == "budget" and out.renders == 3


def test_rejections_count_as_turns_and_are_fed_back(make, tmp_path):
    task, prov, tools = make([0.1], [
        render_call(1001, "start"),
        render_call(1002, "weight:on side"),          # 假 claim：其实改的是 seed → 拒
        tc("conclude", verdict="NOT_ACHIEVABLE_HERE", render_id="agent-deepseek-1-7-01", missing="no LoRA")])
    out = lp.run_task(task, tools.catalog, prov, tools, tmp_path / "run")
    assert out.turns == 3 and out.renders == 1 and out.verdict == "NOT_ACHIEVABLE_HERE"
    assert "single_factor rejected" in json.dumps(prov.seen[2], ensure_ascii=False)


def test_bad_calls_three_times_blocked(make, tmp_path):
    task, prov, tools = make([0.1], [{"text": "I think..."}, tc("paint", x=1), {"text": "?"}])
    out = lp.run_task(task, tools.catalog, prov, tools, tmp_path / "run")
    assert out.verdict == "BLOCKED" and out.stop_reason == "bad_call" and out.turns == 3


def test_tool_error_twice_blocked(make, tmp_path):
    task, prov, tools = make([0.1], [render_call(1001, "start"), render_call(1001, "start")], comfy=FakeComfy(fail_times=2))
    out = lp.run_task(task, tools.catalog, prov, tools, tmp_path / "run")
    assert out.verdict == "BLOCKED" and out.stop_reason == "tool_error" and out.render_id is None


def test_wrong_render_arguments_count_as_bad_calls(make, tmp_path):
    bad = {"tool_call": {"name": "render", "arguments": {"workflow": "flux", "base": "x", "loras": [],
            "tags": [{"tag": "1girl"}], "seed": 1, "cfg": 6.0, "steps": 30, "claim": "start"}}}
    task, prov, tools = make([0.1], [bad, bad, bad])
    out = lp.run_task(task, tools.catalog, prov, tools, tmp_path / "run")
    assert out.verdict == "BLOCKED" and out.stop_reason == "bad_call"


def test_bad_calls_spread_out_do_not_block(make, tmp_path):
    task, prov, tools = make([0.9], [
        render_call(1001, "start"), tc("tag_check", render_id="agent-deepseek-1-7-01"),
        tc("paint", x=1),
        render_call(1002, "knob:seed"), tc("tag_check", render_id="agent-deepseek-1-7-02"),
        tc("paint", x=1),
        tc("conclude", verdict="ACHIEVED", render_id="agent-deepseek-1-7-02", missing="")])
    out = lp.run_task(task, tools.catalog, prov, tools, tmp_path / "run")
    assert out.verdict == "ACHIEVED"


def test_gate_rejections_never_count_as_bad_calls(make, tmp_path):
    task, prov, tools = make([0.9], [
        render_call(1001, "start"),
        render_call(1002, "weight:on side"), render_call(1003, "weight:on side"), render_call(1004, "weight:on side"),
        tc("tag_check", render_id="agent-deepseek-1-7-01"),
        tc("conclude", verdict="ACHIEVED", render_id="agent-deepseek-1-7-01", missing="")])
    out = lp.run_task(task, tools.catalog, prov, tools, tmp_path / "run")
    assert out.verdict == "ACHIEVED"


def test_restart_resets_no_progress_counter(make, tmp_path):
    task, prov, tools = make([0.10, 0.10, 0.10, 0.10, 0.9], [
        render_call(1001, "start"), tc("tag_check", render_id="agent-deepseek-1-7-01"),
        render_call(1002, "knob:seed"), tc("tag_check", render_id="agent-deepseek-1-7-02"),
        render_call(1003, "knob:seed"), tc("tag_check", render_id="agent-deepseek-1-7-03"),
        {"tool_call": {"name": "render", "arguments": {"workflow": "sdxl", "base": "oneObsession_v24.safetensors", "loras": [],
            "tags": [{"tag": "1girl"}, {"tag": "on side"}], "seed": 1001, "cfg": 6.0, "steps": 30, "claim": "start"}}},
        tc("tag_check", render_id="agent-deepseek-1-7-04"),
        {"tool_call": {"name": "render", "arguments": {"workflow": "sdxl", "base": "oneObsession_v24.safetensors", "loras": [],
            "tags": [{"tag": "1girl"}, {"tag": "on side"}], "seed": 1002, "cfg": 6.0, "steps": 30, "claim": "knob:seed"}}},
        tc("tag_check", render_id="agent-deepseek-1-7-05"),
        tc("conclude", verdict="ACHIEVED", render_id="agent-deepseek-1-7-05", missing="")])
    # 前三张 0.10 会触发「不涨」计数到 2（第 1 张是基准），restart 后清零，所以能走到第 5 张
    out = lp.run_task(task, tools.catalog, prov, tools, tmp_path / "run")
    assert out.verdict == "ACHIEVED" and out.renders == 5


def test_best_render_tracked_by_min_score_not_all_landed_override(make, tmp_path):
    task, prov, tools = make([0.90, 0.40, 0.40, 0.40], [
        render_call(1001, "start"), tc("tag_check", render_id="agent-deepseek-1-7-01"),
        render_call(1002, "knob:seed"), tc("tag_check", render_id="agent-deepseek-1-7-02"),
        render_call(1003, "knob:seed"), tc("tag_check", render_id="agent-deepseek-1-7-03"),
        render_call(1004, "knob:seed"), tc("tag_check", render_id="agent-deepseek-1-7-04")])
    out = lp.run_task(task, tools.catalog, prov, tools, tmp_path / "run")
    assert out.verdict == "NOT_ACHIEVABLE_HERE" and out.stop_reason == "no_progress"
    assert out.render_id == "agent-deepseek-1-7-01"
    assert "0.90" in out.missing


def test_budget_enforced_before_render_dispatch(make, tmp_path):
    task, prov, tools = make([0.1], [render_call(1001, "start"), render_call(1002, "knob:seed")], budget=1)
    out = lp.run_task(task, tools.catalog, prov, tools, tmp_path / "run")
    assert out.verdict == "BUDGET_EXHAUSTED" and out.stop_reason == "budget" and out.renders == 1


def test_tool_error_streak_resets_on_intervening_tool_error(make, tmp_path):
    bad_base = {"tool_call": {"name": "render", "arguments": {"workflow": "sdxl", "base": "not-a-real-base.safetensors",
                "loras": [], "tags": [{"tag": "1girl"}, {"tag": "on side"}], "seed": 1002, "cfg": 6.0, "steps": 30, "claim": "start"}}}
    task, prov, tools = make([0.9], [
        render_call(1001, "start"), bad_base, render_call(1001, "start"), render_call(1001, "start"),
        tc("tag_check", render_id="agent-deepseek-1-7-01"),
        tc("conclude", verdict="ACHIEVED", render_id="agent-deepseek-1-7-01", missing="")],
        comfy=FakeComfy(fail_times=2))
    out = lp.run_task(task, tools.catalog, prov, tools, tmp_path / "run")
    assert not (out.verdict == "BLOCKED" and out.stop_reason == "tool_error")
    assert out.verdict == "ACHIEVED"
