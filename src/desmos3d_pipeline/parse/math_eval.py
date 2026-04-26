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
    "min": min,
    "max": max,
}
_ALLOWED_CONSTS = {"pi": math.pi, "e": math.e}


class SafeEvalError(ValueError):
    pass


def _replace_vertical_abs_bars(expr: str) -> str:
    """Turn Desmos-style ``|inner|`` (after ``\\left``/``\\right`` stripping) into ``abs(inner)``."""
    out: list[str] = []
    i = 0
    n = len(expr)
    while i < n:
        if expr[i] != "|":
            out.append(expr[i])
            i += 1
            continue
        depth = 0
        j = i + 1
        start = j
        while j < n:
            c = expr[j]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            elif c == "|" and depth == 0:
                inner = expr[start:j]
                inner_done = _replace_vertical_abs_bars(inner)
                out.append("abs(")
                out.append(inner_done)
                out.append(")")
                i = j + 1
                break
            j += 1
        else:
            out.append("|")
            i += 1
    return "".join(out)


def normalize_symbol_name(name: str) -> str:
    cleaned = name.replace("{", "").replace("}", "")
    return cleaned


def to_python_expr(expr: str, symbol_map: Mapping[str, str] | None = None) -> str:
    text = expr.replace("^", "**")
    # Normalized Desmos exports use ``operatorname{abs}(...)`` with backslashes stripped.
    text = re.sub(r"operatorname\{abs\}", "abs", text)
    text = re.sub(r"sqrt\{([^{}]+)\}", r"sqrt(\1)", text)
    text = text.replace("}{", ")*(")
    if symbol_map:
        for src, dst in sorted(symbol_map.items(), key=lambda item: len(item[0]), reverse=True):
            text = text.replace(src, dst)
    # ``sin7x`` / ``cos3y`` (implicit coefficient) before ``|…|`` so ``|sin7x|`` becomes ``abs(sin(7*x))``.
    text = re.sub(
        r"(?<![A-Za-z0-9_])(sin|cos|tan)(\d+)([xyuvt])(?![A-Za-z0-9_])",
        r"\1(\2*\3)",
        text,
        flags=re.IGNORECASE,
    )
    text = _replace_vertical_abs_bars(text)
    text = re.sub(r"([0-9)])([A-Za-z(])", r"\1*\2", text)
    # Normalized LaTeX turns ``\pi v`` into ``piv``; insert a multiplication before param names.
    text = re.sub(r"(?<![A-Za-z0-9_])pi(?=[A-Za-z])", "pi*", text)
    text = re.sub(r"([xyz])([A-Za-z(])", r"\1*\2", text)
    # Implicit multiplication: symbol followed by axis, e.g. ax -> a*x, a_1x -> a_1*x
    text = re.sub(r"([A-Za-z0-9_}])([xyz])", r"\1*\2", text)
    text = re.sub(r"([0-9xyz}])\(", r"\1*(", text)
    # Single-letter parameter before ``(`` (e.g. ``a(sqrt(2)-1)``); exclude common func names.
    text = re.sub(
        r"(?<![A-Za-z0-9_])(?!sqrt|sin|cos|tan|abs|min|max)([a-z])\(",
        r"\1*(",
        text,
        flags=re.IGNORECASE,
    )
    # Desmos implicit multiply: parameter then paren, e.g. ``4*t(1-t)`` must become ``4*t*(1-t)``
    # (otherwise ``t(1-t)`` parses as a function call). Word-boundary avoids ``sqrt(`` -> ``sqrt*(``.
    for name in ("t", "u", "v"):
        text = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(name)}\(", rf"{name}*(", text)
    text = text.replace("}(", "}*(")
    text = text.replace(")(", ")*(")
    text = text.replace("{", "").replace("}", "")
    return text


def safe_eval(expr: str, variables: Mapping[str, float], *, clamp_sqrt: bool = False) -> float:
    tree = ast.parse(expr, mode="eval")
    return float(_eval_node(tree.body, variables, clamp_sqrt))


def _eval_node(node: ast.AST, variables: Mapping[str, float], clamp_sqrt: bool = False) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id in variables:
            return float(variables[node.id])
        if node.id in _ALLOWED_CONSTS:
            return float(_ALLOWED_CONSTS[node.id])
        raise SafeEvalError(f"Unknown symbol: {node.id}")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.List):
        el0 = node.operand.elts[0] if node.operand.elts else None
        if isinstance(el0, ast.Constant) and isinstance(el0.value, (int, float)):
            return float(-el0.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        val = _eval_node(node.operand, variables, clamp_sqrt)
        return val if isinstance(node.op, ast.UAdd) else -val
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        left = _eval_node(node.left, variables, clamp_sqrt)
        right = _eval_node(node.right, variables, clamp_sqrt)
        # Python ``**`` on negative bases with non-integer exponents yields ``complex``; meshing needs a real scalar.
        # Use the real part of the principal complex value (matches common 3D export behavior for slabs/surfaces).
        out = complex(left, 0.0) ** complex(right, 0.0)
        return float(out.real)

    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left = _eval_node(node.left, variables, clamp_sqrt)
        right = _eval_node(node.right, variables, clamp_sqrt)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCS:
        args = [_eval_node(arg, variables, clamp_sqrt) for arg in node.args]
        if clamp_sqrt and node.func.id == "sqrt":
            return float(math.sqrt(max(0.0, args[0])))
        return float(_ALLOWED_FUNCS[node.func.id](*args))
    if isinstance(node, ast.Compare):
        parts = [_eval_node(node.left, variables, clamp_sqrt)]
        for comp in node.comparators:
            parts.append(_eval_node(comp, variables, clamp_sqrt))
        for i, op in enumerate(node.ops):
            a, b = parts[i], parts[i + 1]
            if isinstance(op, ast.Lt) and not (a < b):
                raise SafeEvalError("chained comparison false")
            if isinstance(op, ast.LtE) and not (a <= b):
                raise SafeEvalError("chained comparison false")
            if isinstance(op, ast.Gt) and not (a > b):
                raise SafeEvalError("chained comparison false")
            if isinstance(op, ast.GtE) and not (a >= b):
                raise SafeEvalError("chained comparison false")
            if isinstance(op, ast.Eq) and not (abs(a - b) <= 1e-9):
                raise SafeEvalError("chained comparison false")
        return 1.0
    raise SafeEvalError(f"Unsupported expression node: {ast.dump(node)}")
