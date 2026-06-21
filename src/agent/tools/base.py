"""Tool interface.

A tool takes a single string input and returns a string observation. Tools are
allowed to *fail* — raise an exception or return an empty/garbage result — and it
is the agent's job to cope. Keeping the contract this narrow is what lets the
agent treat every tool uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ToolError(Exception):
    """Raised by a tool when it cannot produce a usable result."""


class Tool(ABC):
    name: str = "tool"
    description: str = ""

    @abstractmethod
    def run(self, query: str) -> str:
        """Execute the tool and return an observation string."""
        raise NotImplementedError
