"""A small knowledge-search tool over an in-memory fact base.

Stands in for a real search/retrieval API. Crucially, it returns "No results
found" for unknown queries rather than raising — so the agent must *read* the
observation and decide what to do, exactly like a real search that comes back
empty.
"""

from __future__ import annotations

from .base import Tool

_FACTS = {
    "speed of light": "The speed of light is 299,792,458 metres per second.",
    "earth radius": "Earth's mean radius is about 6,371 kilometres.",
    "moon distance": "The Moon is about 384,400 kilometres from Earth on average.",
    "water boiling point": "Water boils at 100 degrees Celsius at sea level.",
    "seconds in a day": "There are 86,400 seconds in a day.",
    "pi": "Pi is approximately 3.14159.",
}


class Search(Tool):
    name = "search"
    description = "Look up a fact by keyword, e.g. 'speed of light'."

    def run(self, query: str) -> str:
        q = query.strip().lower()
        # Exact, then substring match against the fact keys.
        if q in _FACTS:
            return _FACTS[q]
        for key, value in _FACTS.items():
            if key in q or q in key:
                return value
        return "No results found."
