# Multi-Step Agent — Plans, Calls Tools, Recovers From Failure

> **AI Engineer Roadmap — Project 4.2**
> *Teaches: agentic control flow, tool use, state management, failure handling.*
> *Done when: your agent gracefully handles a tool returning garbage instead of crashing.*

A ReAct-style agent that **plans in steps, calls real tools, tracks state across
turns, and — the whole point — recovers when a tool fails** instead of crashing.
It is provider-agnostic (Anthropic / OpenAI), and ships a deterministic
**scripted LLM** so the entire agent loop and every failure path are testable
offline with no API key.

```bash
python -m venv .venv && source .venv/bin/activate   # Win: .\.venv\Scripts\activate
pip install -e ".[dev]"          # core is stdlib-only; this adds pytest
python demo.py                   # watch the agent recover from a flaky tool
pytest -q                        # 15 tests, fully offline

pip install -e ".[anthropic]"    # to drive it with a real model:
#   from agent import Agent, Calculator, Search, FlakyAPI, get_llm
#   Agent(get_llm("anthropic", api_key=...), [Calculator(), Search()]).run("...")
```

---

## The loop

At each step the LLM sees the question, the available tools, and the running
transcript, and replies with **one JSON object**:

```json
{"thought": "...", "action": "<tool name> | finish", "action_input": "..."}
```

The agent runs the tool, appends the observation, and repeats — up to
`max_steps`. `finish` ends the run with the answer. State (the transcript and the
structured step trace) is carried across turns so the agent can build on earlier
observations.

## Failure handling — the "Done when"

Every failure mode is caught and turned into an **observation the agent can react
to**, never an exception that ends the run:

| Failure | What the agent sees | Result |
| --- | --- | --- |
| Tool raises (`1/0`, transient 503) | `ERROR: <message>` | agent retries or switches tools |
| Tool returns **garbage / empty** | `ERROR: tool returned an empty result.` | agent retries |
| Unknown tool name | `ERROR: no such tool 'x'` | agent picks a valid tool |
| Tool crashes unexpectedly (non-`ToolError`) | `ERROR: tool 'x' crashed: ...` | contained, run continues |
| Malformed LLM JSON | a correction note | bounded re-prompts, then gives up cleanly |
| Runs too long | — | stops at `max_steps`, reports; **never loops forever** |

### Recovery in action (`python demo.py`)

The `exchange_rate` tool is configured to return an **empty payload on its first
call** (simulating a flaky API). The agent doesn't crash — it sees the error,
retries, and completes the task:

```
1. exchange_rate('EUR') -> ERROR: tool returned an empty result.
2. exchange_rate('EUR') -> 1 USD = 0.92 EUR
3. calculator('50 * 0.92') -> 46
4. FINISH -> 50 USD is about 46 EUR.

Succeeded: True | reason: finished
```

The 15-test suite pins down each path: recovery from a raised exception, from an
empty/garbage result, from a tool that crashes unexpectedly, from an unknown tool
name, from malformed LLM JSON — plus the guards that the agent **gives up cleanly**
after too many bad responses and **stops at `max_steps`** instead of looping
forever. One end-to-end test drives a reactive policy that reads observations and
chains `search → calculator → finish`.

---

## Tools

| Tool | What it does | Failure modes it exercises |
| --- | --- | --- |
| `calculator` | Safe arithmetic via an **AST walk** (no `eval`) | rejects code injection; raises on `1/0` and bad syntax |
| `search` | Keyword lookup over an in-memory fact base | returns `"No results found."` (agent must read it) |
| `exchange_rate` (`FlakyAPI`) | Mock USD rate API | configurable to raise or return garbage N times, then succeed |

Adding a tool is a subclass of `Tool` with a `run(query) -> str` method.

## Layout

```
src/agent/
├── core.py            # the plan->act->observe loop + all failure handling
├── providers.py       # LLM interface, ScriptedLLM (offline), Anthropic/OpenAI
└── tools/
    ├── base.py        # Tool + ToolError
    ├── calculator.py  # safe AST calculator
    ├── search.py      # in-memory knowledge lookup
    └── flaky_api.py   # deterministically unreliable tool for testing recovery
demo.py                # offline recovery demo
tests/                 # 15 tests, every failure path covered
```

## Design notes

- **The LLM is reduced to one method** (`complete(system, transcript) -> str`),
  which is what makes the scripted/test LLM and real providers interchangeable.
- **`ToolError` vs unexpected exceptions** are both handled, but reported
  distinctly — a contract failure ("division by zero") reads differently from a
  genuine bug ("tool crashed"), which helps the agent *and* the developer.
- **Bounded everything**: parse retries and total steps are capped, so a
  misbehaving model degrades to a clean failure result rather than an infinite
  loop or a stack trace.

## License

MIT.
