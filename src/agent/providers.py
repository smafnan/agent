"""LLM providers behind one tiny interface, plus a scripted LLM for tests.

The agent only needs ``complete(system, transcript) -> str``. A real provider
calls an API; the ``ScriptedLLM`` returns a fixed queue of responses, which lets
the entire agent loop — planning, tool calls, and failure recovery — be tested
deterministically with no network.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable


class LLM(ABC):
    name = "base"

    @abstractmethod
    def complete(self, system: str, transcript: str, *, max_tokens: int = 512) -> str:
        raise NotImplementedError


class ScriptedLLM(LLM):
    """Returns canned responses in order (or computes them from the transcript).

    Pass a list of strings, or a callable ``(system, transcript) -> str`` for
    responses that react to what the agent has seen so far.
    """

    name = "scripted"

    def __init__(self, responses: list[str] | Callable[[str, str], str]) -> None:
        self._responses = list(responses) if isinstance(responses, list) else None
        self._callable = responses if callable(responses) else None
        self._i = 0
        self.calls = 0

    def complete(self, system: str, transcript: str, *, max_tokens: int = 512) -> str:
        self.calls += 1
        if self._callable is not None:
            return self._callable(system, transcript)
        idx = min(self._i, len(self._responses) - 1)
        self._i += 1
        return self._responses[idx]


class AnthropicLLM(LLM):  # pragma: no cover - requires network + key
    name = "anthropic"

    def __init__(self, api_key: str, model: str = "claude-opus-4-8") -> None:
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, system: str, transcript: str, *, max_tokens: int = 512) -> str:
        resp = self._client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": transcript}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")


class OpenAILLM(LLM):  # pragma: no cover - requires network + key
    name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self.model = model

    def complete(self, system: str, transcript: str, *, max_tokens: int = 512) -> str:
        resp = self._client.chat.completions.create(
            model=self.model, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": transcript}],
        )
        return resp.choices[0].message.content or ""


def get_llm(name: str, api_key: str | None = None, model: str | None = None) -> LLM:
    """Construct a real provider by name ('anthropic' or 'openai')."""
    name = name.lower()
    if name == "anthropic":
        return AnthropicLLM(api_key, model) if model else AnthropicLLM(api_key)
    if name == "openai":
        return OpenAILLM(api_key, model) if model else OpenAILLM(api_key)
    raise ValueError(f"Unknown provider '{name}'")
