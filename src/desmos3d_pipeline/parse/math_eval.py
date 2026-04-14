from __future__ import annotations

import ast
import math
import re
from typing import Mapping

_ALLOWED_FUNCS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
    "abs": abs,
}
_ALLOWED_CONSTS = {"pi": math.pi, "e": math.e}


class SafeEvalError(ValueError):
    pass


def normalize_symbol_name(name: str) -> str:
    cleaned = name.replace("{", "").replace("}", "")
    return cleaned


def to_python_expr(expr: str, symbol_map: Mapping[str, str] | None = None) -> str:
    text = expr.replace("^", "**")
    text = text.replace("}{", ")*(")
    if symbol_map:
        for src, dst in sorted(symbol_map.items(), key=lambda item: len(item[0]), reverse=True):
            text = text.replace(src, dst)
    text = re.sub(r"([0-9)])([A-Za-z(])", r"\1*\2", text)
    text = re.sub(r"([xyz])([A-Za-z(])", r"\1*\2", text)
    text = re.sub(r"([0-9xyz}])\(", r"\1*(", text)
    text = text.replace("}(", "}*(")
    text = text.replace(")(", ")*(")
    text = text.replace("{", "").replace("}", "")
    return text


def safe_eval(expr: str, variables: Mapping[str, float]) -> float:
    tree = ast.parse(expr, mode="eval")
    return float(_eval_node(tree.body, variables))


def _eval_node(node: ast.AST, variables: Mapping[str, float]) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id in variables:
            return float(variables[node.id])
        if node.id in _ALLOWED_CONSTS:
            return float(_ALLOWED_CONSTS[node.id])
        raise SafeEvalError(f"Unknown symbol: {node.id}")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        val = _eval_node(node.operand, variables)
        return val if isinstance(node.op, ast.UAdd) else -val
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
        left = _eval_node(node.left, variables)
        right = _eval_node(node.right, variables)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        return left ** right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS:
        args = [_eval_node(arg, variables) for arg in node.args]
        return float(_ALLOWED_FUNCS[node.func.id](*args))
    raise SafeEvalError(f"Unsupported expression node: {ast.dump(node)}")
