"""agent - a multi-step tool-using agent with robust failure handling."""

from .core import Agent, AgentResult, Step
from .providers import LLM, ScriptedLLM, get_llm
from .tools import Tool, ToolError, Calculator, Search, FlakyAPI

__all__ = [
    "Agent", "AgentResult", "Step",
    "LLM", "ScriptedLLM", "get_llm",
    "Tool", "ToolError", "Calculator", "Search", "FlakyAPI",
]
__version__ = "1.0.0"
