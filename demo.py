"""Demo: watch the agent recover from a flaky tool, with no API key.

Uses a scripted LLM "policy" so the run is deterministic and offline. To run
against a real model instead, construct the agent with
`get_llm("anthropic", api_key=...)` and give it a free-form question.

    python demo.py
"""

from __future__ import annotations

from agent import Agent, Calculator, FlakyAPI, ScriptedLLM, Search


def scripted_policy(system: str, transcript: str) -> str:
    """A tiny hand-written policy that reacts to observations like an LLM would."""
    import json

    def say(action, action_input, thought=""):
        return json.dumps({"thought": thought, "action": action,
                           "action_input": action_input})

    have_rate = "0.92" in transcript          # the exchange_rate succeeded
    did_convert = "46" in transcript          # the calculator ran

    if did_convert:
        return say("finish", "50 USD is about 46 EUR.")
    if have_rate:
        return say("calculator", "50 * 0.92", "Convert 50 USD to EUR.")
    if "Action: exchange_rate" not in transcript:
        return say("exchange_rate", "EUR", "I need the USD->EUR rate.")
    # exchange_rate was tried but we still have no rate -> a transient failure;
    # retry it instead of giving up.
    return say("exchange_rate", "EUR", "Transient failure — retrying.")


def main() -> int:
    tools = [Calculator(), Search(), FlakyAPI(fail_times=1, mode="garbage")]
    agent = Agent(ScriptedLLM(scripted_policy), tools, max_steps=8)
    result = agent.run("How many euros is 50 US dollars?")

    print("=== Agent trace ===")
    for i, step in enumerate(result.steps, 1):
        if step.action == "finish":
            print(f"{i}. FINISH -> {step.action_input}")
        else:
            print(f"{i}. {step.action}({step.action_input!r}) -> {step.observation}")
    print("\nSucceeded:", result.succeeded, "| reason:", result.reason)
    print("Answer:", result.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
