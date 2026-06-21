"""A deliberately unreliable API tool — to exercise the agent's failure handling.

Real APIs time out, return 500s, or hand back malformed payloads. This tool
reproduces those failure modes deterministically so we can *test* that the agent
recovers:

  * ``mode="raise"``   -> raises ToolError for the first ``fail_times`` calls.
  * ``mode="garbage"`` -> returns an empty/garbage string the first ``fail_times``.

After the configured number of failures it returns a valid result. This lets a
test assert that the agent retries through transient failure and still succeeds.
"""

from __future__ import annotations

from .base import Tool, ToolError


class FlakyAPI(Tool):
    name = "exchange_rate"
    description = "Get a (mock) USD exchange rate, e.g. 'EUR'."

    _RATES = {"eur": "0.92", "gbp": "0.79", "jpy": "151.4"}

    def __init__(self, fail_times: int = 0, mode: str = "raise") -> None:
        self.fail_times = fail_times
        self.mode = mode
        self.calls = 0

    def run(self, query: str) -> str:
        self.calls += 1
        if self.calls <= self.fail_times:
            if self.mode == "raise":
                raise ToolError("503 Service Unavailable (transient).")
            if self.mode == "garbage":
                return ""  # empty payload — looks like success but isn't usable
        rate = self._RATES.get(query.strip().lower())
        if rate is None:
            return "No results found."
        return f"1 USD = {rate} {query.strip().upper()}"
