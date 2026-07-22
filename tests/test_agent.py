"""Tests for the agent: planning, tool use, and (the focus) failure recovery."""

from __future__ import annotations

import json

from agent import Agent, Calculator, FlakyAPI, ScriptedLLM, Search
from agent.tools import ToolError


def _say(action: str, action_input: str = "", thought: str = "") -> str:
    return json.dumps({"thought": thought, "action": action,
                       "action_input": action_input})


def _tools():
    return [Calculator(), Search(), FlakyAPI()]


# --- tools in isolation ---------------------------------------------------- #

def test_calculator_basic():
    assert Calculator().run("3 * (4 + 5)") == "27"


def test_calculator_rejects_code():
    import pytest
    with pytest.raises(ToolError):
        Calculator().run("__import__('os').system('echo hi')")


def test_calculator_division_by_zero_raises_toolerror():
    import pytest
    with pytest.raises(ToolError):
        Calculator().run("1/0")


def test_calculator_rejects_bool_operands():
    import pytest
    with pytest.raises(ToolError):
        Calculator().run("True + 1")


def test_search_hit_and_miss():
    s = Search()
    assert "299,792,458" in s.run("speed of light")
    assert s.run("flux capacitor") == "No results found."


# --- happy path ------------------------------------------------------------ #

def test_single_tool_then_finish():
    llm = ScriptedLLM([_say("calculator", "2 + 2"), _say("finish", "4")])
    result = Agent(llm, _tools()).run("What is 2 + 2?")
    assert result.succeeded
    assert result.answer == "4"
    assert result.steps[0].observation == "4"


def test_multi_tool_plan():
    llm = ScriptedLLM([
        _say("search", "speed of light"),
        _say("calculator", "299792458 / 1000"),
        _say("finish", "299792.458 km/s"),
    ])
    result = Agent(llm, _tools()).run("Speed of light in km/s?")
    assert result.succeeded
    assert [s.action for s in result.steps] == ["search", "calculator", "finish"]


# --- failure recovery (the 'Done when') ------------------------------------ #

def test_recovers_from_tool_exception():
    # First the model calls a bad expression (tool raises), then it corrects.
    llm = ScriptedLLM([
        _say("calculator", "1/0"),     # -> ERROR observation, not a crash
        _say("calculator", "10 / 2"),  # recovery
        _say("finish", "5"),
    ])
    result = Agent(llm, _tools()).run("Compute 10/2")
    assert result.succeeded and result.answer == "5"
    assert result.steps[0].observation.startswith("ERROR")  # caught, surfaced


def test_recovers_from_garbage_tool_result():
    # The flaky API returns an EMPTY payload on the first call.
    flaky = FlakyAPI(fail_times=1, mode="garbage")
    llm = ScriptedLLM([
        _say("exchange_rate", "EUR"),  # garbage -> ERROR empty result
        _say("exchange_rate", "EUR"),  # now succeeds
        _say("finish", "1 USD = 0.92 EUR"),
    ])
    result = Agent(llm, [Calculator(), Search(), flaky]).run("USD to EUR?")
    assert result.succeeded
    assert result.steps[0].observation == "ERROR: tool returned an empty result."
    assert flaky.calls == 2  # it actually retried


def test_recovers_from_raised_api_failure():
    flaky = FlakyAPI(fail_times=2, mode="raise")  # two transient 503s
    llm = ScriptedLLM([
        _say("exchange_rate", "GBP"),
        _say("exchange_rate", "GBP"),
        _say("exchange_rate", "GBP"),
        _say("finish", "1 USD = 0.79 GBP"),
    ])
    result = Agent(llm, [flaky]).run("USD to GBP?")
    assert result.succeeded
    assert sum(1 for s in result.steps if s.observation.startswith("ERROR")) == 2


def test_unknown_tool_is_reported_not_crashed():
    llm = ScriptedLLM([_say("frobnicate", "x"), _say("finish", "ok")])
    result = Agent(llm, _tools()).run("...")
    assert result.steps[0].observation.startswith("ERROR: no such tool")
    assert result.succeeded


def test_recovers_from_malformed_llm_json():
    llm = ScriptedLLM(["this is not json at all", _say("finish", "recovered")])
    result = Agent(llm, _tools(), max_parse_retries=2).run("...")
    assert result.succeeded and result.answer == "recovered"


def test_gives_up_after_too_many_bad_json():
    llm = ScriptedLLM(["nope"])  # always invalid
    result = Agent(llm, _tools(), max_parse_retries=2).run("...")
    assert not result.succeeded
    assert "valid JSON" in result.reason


def test_does_not_loop_forever():
    # The model never finishes — the agent must stop at max_steps, not hang.
    llm = ScriptedLLM([_say("calculator", "1 + 1")])  # repeats this forever
    result = Agent(llm, _tools(), max_steps=4).run("...")
    assert not result.succeeded
    assert "within 4 steps" in result.reason
    assert len(result.steps) == 4


def test_tool_that_crashes_unexpectedly_is_contained():
    class Bomb(Calculator):
        name = "bomb"
        def run(self, query: str) -> str:
            raise RuntimeError("boom")  # not a ToolError

    llm = ScriptedLLM([_say("bomb", "x"), _say("finish", "safe")])
    result = Agent(llm, [Bomb()]).run("...")
    assert result.steps[0].observation.startswith("ERROR: tool 'bomb' crashed")
    assert result.succeeded


# --- a reactive end-to-end run (LLM policy reads observations) ------------- #

def test_reactive_policy_end_to_end():
    """A scripted *policy* that reacts to the transcript, like a real LLM would:
    search for a fact, then compute with it, then finish."""
    def policy(system: str, transcript: str) -> str:
        if "Observation:" not in transcript:
            return _say("search", "seconds in a day")
        if "86,400" in transcript and "Action: calculator" not in transcript:
            return _say("calculator", "86400 * 2")
        if "172800" in transcript:
            return _say("finish", "172800")
        return _say("finish", "unsure")

    result = Agent(ScriptedLLM(policy), _tools()).run("How many seconds in two days?")
    assert result.succeeded and result.answer == "172800"
    assert [s.action for s in result.steps] == ["search", "calculator", "finish"]
