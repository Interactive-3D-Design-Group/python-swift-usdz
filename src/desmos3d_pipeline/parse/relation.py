from __future__ import annotations

import re
from dataclasses import dataclass

from desmos3d_pipeline.ir.models import RangeConstraint
from desmos3d_pipeline.parse.math_eval import safe_eval, to_python_expr


@dataclass(slots=True)
class RelationInfo:
    core: str
    operators: list[str]
    has_chained_inequality: bool


def _negate_expr(expr: str) -> str:
    text = expr.strip()
    return f"-({text})"


def detect_relations(core_expr: str) -> RelationInfo:
    ops = re.findall(r"<=|>=|=|<|>", core_expr)
    return RelationInfo(
        core=core_expr,
        operators=ops,
        has_chained_inequality=len(ops) >= 2,
    )


def parse_interval_constraint(expr: str) -> RangeConstraint | None:
    expr = expr.strip()

    # ``abs(axis) < c`` / legacy ``operatorname{abs}(axis) < c`` (normalized).
    m = re.fullmatch(r"(?:abs|operatorname\{abs\})\(([xyz])\)(<=|<)([^<>=]+)", expr)
    if m:
        axis, op, c = m.groups()
        bound = c.strip()
        inc = op == "<="
        return RangeConstraint(
            axis=axis,
            lower=_negate_expr(bound),
            lower_inclusive=inc,
            upper=bound,
            upper_inclusive=inc,
        )

    # Handle chained form with negated axis: lower < -x < upper  =>  -upper < x < -lower
    m = re.fullmatch(r"([^<>=]+)(<=|<)-([xyz])(<=|<)([^<>=]+)", expr)
    if m:
        lower, lop, axis, uop, upper = m.groups()
        return RangeConstraint(
            axis=axis,
            lower=_negate_expr(upper),
            lower_inclusive=uop == "<=",
            upper=_negate_expr(lower),
            upper_inclusive=lop == "<=",
        )

    # Handle chained form with negated axis reversed: upper > -x > lower  =>  -upper < x < -lower
    m = re.fullmatch(r"([^<>=]+)(>=|>)-([xyz])(>=|>)([^<>=]+)", expr)
    if m:
        upper, uop, axis, lop, lower = m.groups()
        return RangeConstraint(
            axis=axis,
            lower=_negate_expr(upper),
            lower_inclusive=uop == ">=",
            upper=_negate_expr(lower),
            upper_inclusive=lop == ">=",
        )

    # Canonical chained form: lower < axis < upper
    m = re.fullmatch(r"([^<>=]+)(<=|<)([xyz])(<=|<)([^<>=]+)", expr)
    if m:
        lower, lop, axis, uop, upper = m.groups()
        return RangeConstraint(
            axis=axis,
            lower=lower,
            lower_inclusive=lop == "<=",
            upper=upper,
            upper_inclusive=uop == "<=",
        )

    # Reversed chained form: upper > axis > lower  (same as lower < axis < upper)
    m = re.fullmatch(r"([^<>=]+)(>=|>)([xyz])(>=|>)([^<>=]+)", expr)
    if m:
        upper, uop, axis, lop, lower = m.groups()
        return RangeConstraint(
            axis=axis,
            lower=lower,
            lower_inclusive=lop == ">=",
            upper=upper,
            upper_inclusive=uop == ">=",
        )

    # Chained form with axis on the left: axis < upper <or> axis > lower
    m = re.fullmatch(r"([xyz])(<=|<)([^<>=]+)", expr)
    if m:
        axis, op, upper = m.groups()
        return RangeConstraint(axis=axis, lower=None, lower_inclusive=False, upper=upper, upper_inclusive=op == "<=")

    m = re.fullmatch(r"-([xyz])(<=|<)([^<>=]+)", expr)
    if m:
        axis, op, upper = m.groups()
        # -x <= c  <=>  x >= -c
        return RangeConstraint(axis=axis, lower=_negate_expr(upper), lower_inclusive=op == "<=", upper=None, upper_inclusive=False)

    m = re.fullmatch(r"([xyz])(>=|>)([^<>=]+)", expr)
    if m:
        axis, op, lower = m.groups()
        return RangeConstraint(axis=axis, lower=lower, lower_inclusive=op == ">=", upper=None, upper_inclusive=False)

    m = re.fullmatch(r"-([xyz])(>=|>)([^<>=]+)", expr)
    if m:
        axis, op, lower = m.groups()
        # -x >= c  <=>  x <= -c
        return RangeConstraint(axis=axis, lower=None, lower_inclusive=False, upper=_negate_expr(lower), upper_inclusive=op == ">=")

    # Reversed chained form with axis in the middle but constant on left and right swapped:
    # axis <= upper <= lower  OR  axis >= lower >= upper
    # These appear in some Desmos exports as: -64 > x > -98.3 and 1.35 >= y >= 1.3 (already covered above),
    # but also as: x < upper < lower (rare). Normalize by detecting axis-first chained sequences.
    m = re.fullmatch(r"([xyz])(<=|<)([^<>=]+)(<=|<)([^<>=]+)", expr)
    if m:
        axis, op1, mid, op2, upper = m.groups()
        # Treat as axis between mid and upper; if values are swapped, downstream resolution will clamp via max/min.
        return RangeConstraint(axis=axis, lower=None, lower_inclusive=False, upper=upper, upper_inclusive=op2 == "<=")

    m = re.fullmatch(r"([^<>=]+)(<=|<)([xyz])", expr)
    if m:
        lower, op, axis = m.groups()
        return RangeConstraint(axis=axis, lower=lower, lower_inclusive=op == "<=", upper=None, upper_inclusive=False)

    m = re.fullmatch(r"([^<>=]+)(<=|<)-([xyz])", expr)
    if m:
        lower, op, axis = m.groups()
        # c <= -x  <=>  x <= -c
        return RangeConstraint(axis=axis, lower=None, lower_inclusive=False, upper=_negate_expr(lower), upper_inclusive=op == "<=")

    m = re.fullmatch(r"([^<>=]+)(>=|>)([xyz])", expr)
    if m:
        upper, op, axis = m.groups()
        return RangeConstraint(axis=axis, lower=None, lower_inclusive=False, upper=upper, upper_inclusive=op == ">=")

    m = re.fullmatch(r"([^<>=]+)(>=|>)-([xyz])", expr)
    if m:
        upper, op, axis = m.groups()
        # c >= -x  <=>  x >= -c
        return RangeConstraint(axis=axis, lower=_negate_expr(upper), lower_inclusive=op == ">=", upper=None, upper_inclusive=False)

    # Degenerate interval: ``z=0`` (Desmos plane / slab at a constant).
    m = re.fullmatch(r"([xyz])=([^<>=]+)", expr)
    if m:
        axis, val = m.groups()
        return RangeConstraint(axis=axis, lower=val, lower_inclusive=True, upper=val, upper_inclusive=True)

    return None


def restriction_chain_evaluable(expr: str) -> bool:
    """True if ``expr`` is an odd-length chain ``a < b < c`` / ``<=`` style evaluable with ``x,y,z``."""
    parts = [t.strip() for t in re.split(r"(<=|>=|<|>)", expr.strip()) if t != ""]
    if len(parts) < 3 or len(parts) % 2 == 0:
        return False
    env = {"x": 0.1, "y": -0.2, "z": 0.3}
    try:
        for i in range(0, len(parts), 2):
            safe_eval(to_python_expr(parts[i], {}), env)
        return True
    except Exception:
        return False


def split_interval_clauses(expr: str) -> list[str]:
    """Split ``a<=z<=b,c<=z<=d``-style compound brace content into separate interval strings."""
    expr = expr.strip()
    if "," not in expr:
        return [expr]
    depth = 0
    start = 0
    parts: list[str] = []
    for i, ch in enumerate(expr):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(expr[start:i].strip())
            start = i + 1
    parts.append(expr[start:].strip())
    if len(parts) < 2:
        return [expr]
    if not all(re.search(r"[<>=]", p) for p in parts):
        return [expr]
    return parts


def restriction_axis_interval_ok(expr: str) -> bool:
    """True if ``expr`` is one or more comma-separated pieces each matching ``parse_interval_constraint``."""
    for clause in split_interval_clauses(expr):
        if parse_interval_constraint(clause.strip()) is None:
            return False
    return True


def surface_domain_meshable(restrictions: list[str], viewport: dict[str, float] | None) -> bool:
    """Domain suitable for sampled/plane meshing: axis intervals and/or evaluable chains with viewport."""
    if not restrictions:
        return True
    vp = viewport or {}
    vp_xy = all(k in vp for k in ("xmin", "xmax", "ymin", "ymax"))
    for r in restrictions:
        if restriction_axis_interval_ok(r):
            continue
        if vp_xy and restriction_chain_evaluable(r):
            continue
        return False
    return True
