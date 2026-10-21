"""Safe arithmetic expressions for matrix entry fields.

Supports numbers, ``+ - * / ** %``, parentheses, constants (pi, e, tau) and a
whitelist of math functions. Nothing is evaluated with ``eval``; the AST is
walked explicitly, so attribute access, imports and names are rejected.
"""

from __future__ import annotations

import ast
import math
import operator

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "ln": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "hypot": math.hypot,
    "abs": abs,
    "floor": math.floor,
    "ceil": math.ceil,
    "round": round,
    "radians": math.radians,
    "degrees": math.degrees,
    "min": min,
    "max": max,
    "pow": pow,
}
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}
_MAX = 1e6


def evaluate(text: str) -> float:
    """Evaluate ``text`` as an arithmetic expression, or raise ValueError."""
    if not isinstance(text, str):
        raise ValueError("not a text expression")
    source = text.strip().replace("^", "**")
    if not source:
        raise ValueError("empty expression")
    try:
        value = float(_eval(ast.parse(source, mode="eval").body))
    except (SyntaxError, TypeError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError(str(exc) or "invalid expression") from exc
    if not math.isfinite(value):
        raise ValueError("result is not finite")
    if abs(value) > _MAX:
        raise ValueError(f"value must be within ±{_MAX:g}")
    return value


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if isinstance(node.value, bool):
            raise ValueError("booleans are not numbers")
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS or node.keywords:
            raise ValueError("unsupported function or argument form")
        return _FUNCS[node.func.id](*[_eval(argument) for argument in node.args])
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    raise ValueError("unsupported expression")
