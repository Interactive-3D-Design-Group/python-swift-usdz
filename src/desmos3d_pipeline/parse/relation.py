from __future__ import annotations

import re
from dataclasses import dataclass

from desmos3d_pipeline.ir.models import RangeConstraint


@dataclass(slots=True)
class RelationInfo:
    core: str
    operators: list[str]
    has_chained_inequality: bool


def detect_relations(core_expr: str) -> RelationInfo:
    ops = re.findall(r"<=|>=|=|<|>", core_expr)
    return RelationInfo(
        core=core_expr,
        operators=ops,
        has_chained_inequality=len(ops) >= 2,
    )


def parse_interval_constraint(expr: str) -> RangeConstraint | None:
    expr = expr.strip()

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

    m = re.fullmatch(r"([xyz])(>=|>)([^<>=]+)", expr)
    if m:
        axis, op, lower = m.groups()
        return RangeConstraint(axis=axis, lower=lower, lower_inclusive=op == ">=", upper=None, upper_inclusive=False)

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

    m = re.fullmatch(r"([^<>=]+)(>=|>)([xyz])", expr)
    if m:
        upper, op, axis = m.groups()
        return RangeConstraint(axis=axis, lower=None, lower_inclusive=False, upper=upper, upper_inclusive=op == ">=")

    return None
