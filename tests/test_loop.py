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
    opened = []

    def _make(scores, responses, comfy=None, budget=12):
        v = load_vocabulary(vocab_path)
        task = Task(7, "pose", ["1girl", "on side"], 1001, budget, [])
        comfy = comfy or FakeComfy()
        results = Results(tmp_path / "r.sqlite")
        opened.append(results)
        tools = Tools(catalog_dict, v, comfy, Tagger(FakeBackend(scores), 0.35, v), results,
                      tmp_path, "agent-deepseek", 1, task)
        return task, ScriptedProvider(responses), tools
    yield _make
    for r in opened:
        r.close()


def test_achieves_and_writes_summary(make, tmp_path, catalog_dict):
    task, prov, tools = make([0.1, 0.9], [
        render_call(1001, "start"), tc("tag_check", render_id="agent-deepseek-1-7-01"),
        render_call(1001, "weight:on side", w=1.3), tc("tag_check", render_id="agent-deepseek-1-7-02"),
        tc("conclude", verdict="ACHIEVED", render_id="agent-deepseek-1-7-02", missing="")])
    out = lp.run_task(task, catalog_dict, prov, tools, tmp_path / "run")
    assert out.verdict == "ACHIEVED" and out.renders == 2 and out.turns == 5 and out.stop_reason == "model"
    lines = (tmp_path / "run/steps.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5 and json.loads(lines[0])["tool"] == "render"
    summ = (tmp_path / "run/summary.md").read_text(encoding="utf-8")
    assert "ACHIEVED" in summ and "weight:on side" in summ
    # 第二轮的输入里有已试过的表
    assert "on side" in json.dumps(prov.seen[2], ensure_ascii=False)


def test_steps_jsonl_carries_the_full_record(make, tmp_path):
    task, prov, tools = make([0.9], [
        render_call(1001, "start"), tc("tag_check", render_id="agent-deepseek-1-7-01"),
        tc("conclude", verdict="ACHIEVED", render_id="agent-deepseek-1-7-01", missing="")])
    lp.run_task(task, tools.catalog, prov, tools, tmp_path / "run")
    lines = [json.loads(l) for l in (tmp_path / "run/steps.jsonl").read_text(encoding="utf-8").splitlines()]
    common_keys = {"turn", "tool", "args", "result", "error", "model_text", "usage", "seconds"}
    for line in lines:
        assert common_keys <= set(line)
        assert set(line["usage"]) == {"prompt_tokens", "completion_tokens"}
    render_line, tag_check_line = lines[0], lines[1]
    assert render_line["tool"] == "render" and "graph" in render_line
    assert render_line["graph"]["pos"]["inputs"]["text"] == "1girl, on side"
    assert tag_check_line["tool"] == "tag_check" and "graph" not in tag_check_line


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


class FailingBackend:
    def tag(self, path):
        raise ComfyError("boom")


def test_tag_check_comfy_error_message_names_tag_check_not_render(tmp_path, catalog_dict, vocab_path):
    v = load_vocabulary(vocab_path)
    task = Task(7, "pose", ["1girl", "on side"], 1001, 12, [])
    results = Results(tmp_path / "tagfail.sqlite")
    tools = Tools(catalog_dict, v, FakeComfy(), Tagger(FailingBackend(), 0.35, v), results, tmp_path, "agent-deepseek", 1, task)
    prov = ScriptedProvider([
        render_call(1001, "start"),
        tc("tag_check", render_id="agent-deepseek-1-7-01"),
        tc("tag_check", render_id="agent-deepseek-1-7-01")])
    out = lp.run_task(task, tools.catalog, prov, tools, tmp_path / "run")
    assert out.verdict == "BLOCKED" and out.stop_reason == "tool_error"
    lines = [json.loads(l) for l in (tmp_path / "run/steps.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines[-1]["error"].startswith("tag_check failed:")
    results.close()


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
        render_call(1003, "knob:seed"), tc("tag_check", render_id="agent-deepseek-1-7-03"),
        tc("paint", x=1),
        tc("conclude", verdict="ACHIEVED", render_id="agent-deepseek-1-7-03", missing="")])
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


def _restart_call(seed, claim, base="oneObsession_v24.safetensors"):
    return {"tool_call": {"name": "render", "arguments": {"workflow": "sdxl", "base": base, "loras": [],
            "tags": [{"tag": "1girl"}, {"tag": "on side"}], "seed": seed, "cfg": 6.0, "steps": 30, "claim": claim}}}


def test_forced_conclusion_after_restart_uses_task_wide_best(make, tmp_path):
    task, prov, tools = make([0.30, 0.05, 0.05, 0.05, 0.05], [
        render_call(1001, "start"), tc("tag_check", render_id="agent-deepseek-1-7-01"),
        _restart_call(1001, "start"), tc("tag_check", render_id="agent-deepseek-1-7-02"),
        _restart_call(1002, "knob:seed"), tc("tag_check", render_id="agent-deepseek-1-7-03"),
        _restart_call(1003, "knob:seed"), tc("tag_check", render_id="agent-deepseek-1-7-04"),
        _restart_call(1004, "knob:seed"), tc("tag_check", render_id="agent-deepseek-1-7-05")])
    out = lp.run_task(task, tools.catalog, prov, tools, tmp_path / "run")
    assert out.verdict == "NOT_ACHIEVABLE_HERE" and out.stop_reason == "no_progress"
    assert out.render_id == "agent-deepseek-1-7-01"


def test_rechecking_one_render_does_not_advance_no_progress(make, tmp_path):
    task, prov, tools = make([0.30], [render_call(1001, "start")]
                              + [tc("tag_check", render_id="agent-deepseek-1-7-01")] * 4
                              + [tc("conclude", verdict="NOT_ACHIEVABLE_HERE", render_id="agent-deepseek-1-7-01", missing="no LoRA")])
    out = lp.run_task(task, tools.catalog, prov, tools, tmp_path / "run")
    assert out.renders == 1 and out.verdict == "NOT_ACHIEVABLE_HERE" and out.stop_reason == "model"
    assert "used 1." in json.dumps(prov.seen[-1], ensure_ascii=False)


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


def test_unchecked_renders_are_listed_and_prompt_demands_tag_check(make, tmp_path):
    # deepseek 2026-09-30: rendered 12 times without one tag_check because unchecked renders vanished from the table
    task, provider, tools = make([0.1], [
        render_call(1001, "start"),
        tc("history", query="on side"),
        tc("tag_check", render_id="agent-deepseek-1-7-01"),
        tc("history", query="on side"),
        tc("conclude", verdict="NOT_ACHIEVABLE_HERE", render_id="agent-deepseek-1-7-01", missing="x")])
    lp.run_task(task, tools.catalog, provider, tools, tmp_path / "run")
    assert "tag_check" in provider.seen[0][0]["content"]                      # system prompt says to judge every render
    assert "seed_hint 1001" in provider.seen[0][1]["content"]                 # §9: every arm starts a run from the same seed
    after_render = provider.seen[1][1]["content"]
    assert '01 | start' in after_render and 'unchecked' in after_render and 'tag_check("agent-deepseek-1-7-01")' in after_render
    after_check = provider.seen[3][1]["content"]
    assert "unchecked" not in after_check and "on side=0.10!" in after_check


def test_summary_lists_renders_left_unchecked(make, tmp_path):
    task, provider, tools = make([0.1], [render_call(1001, "start")] + [{"text": "no call"}] * 3)
    out = lp.run_task(task, tools.catalog, provider, tools, tmp_path / "run")
    assert out.stop_reason == "bad_call" and out.renders == 1
    summ = (tmp_path / "run/summary.md").read_text(encoding="utf-8")
    assert "01 | start" in summ and "unchecked" in summ


def test_identical_repeated_call_is_rejected_without_dispatch(make, tmp_path):
    # deepseek 2026-10-01: 20 of 24 turns were history("uzuki sayaka") → [] again and again
    task, provider, tools = make([0.1], [
        tc("history", query="on side"), tc("history", query="on side"), tc("history", query="on side"),
        tc("history", query="on side"), tc("history", query="on side"),
        render_call(1001, "start"), tc("tag_check", render_id="agent-deepseek-1-7-01"),
        tc("conclude", verdict="NOT_ACHIEVABLE_HERE", render_id="agent-deepseek-1-7-01", missing="x")])
    out = lp.run_task(task, tools.catalog, provider, tools, tmp_path / "run")
    assert out.verdict == "NOT_ACHIEVABLE_HERE" and out.stop_reason == "model" and out.turns == 8   # repeats are turns, not BLOCKED
    lines = [json.loads(l) for l in (tmp_path / "run/steps.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines[0]["error"] is None and lines[1]["error"].startswith("identical to turn 1") and lines[4]["error"].startswith("identical to turn 1")
    assert "the answer will not change" in provider.seen[2][1]["content"]


def test_prompt_carries_history_queries_already_asked(make, tmp_path):
    task, provider, tools = make([0.1], [
        tc("history", query="on side"), tc("history", query="lying"),
        render_call(1001, "start"), tc("tag_check", render_id="agent-deepseek-1-7-01"),
        tc("conclude", verdict="NOT_ACHIEVABLE_HERE", render_id="agent-deepseek-1-7-01", missing="x")])
    lp.run_task(task, tools.catalog, provider, tools, tmp_path / "run")
    assert "history already asked" not in provider.seen[0][1]["content"]
    assert 'history already asked: "on side" (0 rows), "lying" (0 rows)' in provider.seen[2][1]["content"]


def test_system_prompt_states_the_restart_and_zero_score_rules():
    s = lp.SYSTEM_PROMPT
    assert "previous render's workflow and base" in s and "near 0" in s and "do not ask again" in s


def test_refused_repeat_resets_the_bad_call_streak(make, tmp_path):
    task, provider, tools = make([0.1], [
        tc("history", query="q"), {"text": "?"}, {"text": "?"}, tc("history", query="q"), {"text": "?"},
        render_call(1001, "start"), tc("tag_check", render_id="agent-deepseek-1-7-01"),
        tc("conclude", verdict="NOT_ACHIEVABLE_HERE", render_id="agent-deepseek-1-7-01", missing="x")])
    out = lp.run_task(task, tools.catalog, provider, tools, tmp_path / "run")
    assert out.stop_reason == "model" and out.turns == 8          # 3 bad calls were not consecutive


def test_repeat_separated_by_another_call_is_dispatched(make, tmp_path):
    task, provider, tools = make([0.1], [
        tc("history", query="q"), tc("history", query="z"), tc("history", query="q"),
        render_call(1001, "start"), tc("tag_check", render_id="agent-deepseek-1-7-01"),
        tc("conclude", verdict="NOT_ACHIEVABLE_HERE", render_id="agent-deepseek-1-7-01", missing="x")])
    lp.run_task(task, tools.catalog, provider, tools, tmp_path / "run")
    lines = [json.loads(l) for l in (tmp_path / "run/steps.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [l["error"] for l in lines[:3]] == [None, None, None]
