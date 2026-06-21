"""A safe arithmetic calculator.

Evaluates a math expression without ``eval`` — it walks the Python AST and only
permits numbers and arithmetic operators, so a malicious or malformed input
(``__import__('os')``, ``1/0``, ``2 +``) raises a clean :class:`ToolError`
instead of executing code or crashing the agent.
"""

from __future__ import annotations

import ast
import operator

from .base import Tool, ToolError

# Only these AST operators are allowed.
_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    raise ToolError("Only numbers and + - * / // % ** are allowed.")


class Calculator(Tool):
    name = "calculator"
    description = "Evaluate an arithmetic expression, e.g. '3 * (4 + 5)'."

    def run(self, query: str) -> str:
        try:
            tree = ast.parse(query.strip(), mode="eval")
            result = _eval(tree.body)
        except ToolError:
            raise
        except ZeroDivisionError as exc:
            raise ToolError("Division by zero.") from exc
        except Exception as exc:
            raise ToolError(f"Could not parse expression: {exc}") from exc
        # Render whole numbers without a trailing .0 for tidy answers.
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)
