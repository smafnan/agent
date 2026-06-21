"""The agent: plan -> act -> observe, looping until it answers or gives up.

Control flow (a ReAct-style loop):

    1. Show the LLM the question, the available tools, and the running transcript.
    2. The LLM replies with ONE JSON object:
         {"thought": "...", "action": "<tool>|finish", "action_input": "..."}
    3. If ``action`` is a tool, run it and append the observation; if it's
       ``finish``, return the answer.
    4. Repeat up to ``max_steps``.

Failure handling is the point of this project — every failure mode is caught and
turned into an *observation the agent can react to*, never a crash:

  * Tool raises          -> "ERROR: <message>"   (agent can retry / switch tools)
  * Tool returns garbage -> "ERROR: empty result"
  * Unknown tool name    -> "ERROR: no such tool"
  * Malformed LLM JSON   -> a correction note; bounded re-prompts
  * Runs too long        -> stop at ``max_steps`` and report, don't loop forever
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .providers import LLM
from .tools.base import Tool, ToolError

SYSTEM_TEMPLATE = """\
You are a problem-solving agent. Work in steps. At each step respond with EXACTLY
one JSON object and nothing else:
  {{"thought": "<your reasoning>", "action": "<action>", "action_input": "<input>"}}

`action` is either one of the tool names below, or the literal "finish" to give
the final answer (put the answer in `action_input`).

Tools:
{tools}

Rules:
- Use a tool when you need a fact or a computation; do not guess numbers.
- If an observation says ERROR or "No results", adapt: fix your input, try a
  different tool, or retry — do not give up after a single failure.
- When you have the answer, use action "finish".
"""


@dataclass
class Step:
    thought: str
    action: str
    action_input: str
    observation: str


@dataclass
class AgentResult:
    answer: str | None
    succeeded: bool
    steps: list[Step] = field(default_factory=list)
    reason: str = ""  # why it stopped (finished / max_steps / parse failure)


def _extract_json(text: str) -> str | None:
    """First brace-balanced {...} object in the text (tolerates surrounding prose)."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


class Agent:
    def __init__(
        self, llm: LLM, tools: list[Tool], *, max_steps: int = 8,
        max_parse_retries: int = 2,
    ) -> None:
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps
        self.max_parse_retries = max_parse_retries

    def _system(self) -> str:
        tool_lines = "\n".join(
            f"- {t.name}: {t.description}" for t in self.tools.values()
        )
        return SYSTEM_TEMPLATE.format(tools=tool_lines)

    def run(self, question: str) -> AgentResult:
        system = self._system()
        transcript = f"Question: {question}\n"
        steps: list[Step] = []
        parse_errors = 0

        for _ in range(self.max_steps):
            raw = self.llm.complete(system, transcript + "\nYour JSON response:")
            decision = self._parse(raw)

            if decision is None:
                # The model didn't return usable JSON. Re-prompt with guidance,
                # but bound the number of attempts so we never loop forever.
                parse_errors += 1
                if parse_errors > self.max_parse_retries:
                    return AgentResult(None, False, steps,
                                       reason="LLM failed to produce valid JSON.")
                transcript += (
                    "\nObservation: ERROR your last response was not a single "
                    "valid JSON object. Respond with only the JSON object.\n"
                )
                continue

            action = str(decision.get("action", "")).strip()
            action_input = str(decision.get("action_input", ""))
            thought = str(decision.get("thought", ""))

            if action == "finish":
                steps.append(Step(thought, action, action_input, ""))
                return AgentResult(action_input, True, steps, reason="finished")

            observation = self._run_tool(action, action_input)
            steps.append(Step(thought, action, action_input, observation))
            transcript += (
                f"\nThought: {thought}\nAction: {action}({action_input})\n"
                f"Observation: {observation}\n"
            )

        return AgentResult(None, False, steps,
                           reason=f"Did not finish within {self.max_steps} steps.")

    def _parse(self, raw: str) -> dict | None:
        candidate = _extract_json(raw)
        if candidate is None:
            return None
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None

    def _run_tool(self, name: str, query: str) -> str:
        """Execute a tool, converting every failure into a safe observation."""
        tool = self.tools.get(name)
        if tool is None:
            return (f"ERROR: no such tool '{name}'. "
                    f"Available: {', '.join(self.tools)}.")
        try:
            result = tool.run(query)
        except ToolError as exc:
            return f"ERROR: {exc}"
        except Exception as exc:  # never let an unexpected tool bug crash the agent
            return f"ERROR: tool '{name}' crashed: {exc}"
        if result is None or not str(result).strip():
            return "ERROR: tool returned an empty result."
        return str(result)
