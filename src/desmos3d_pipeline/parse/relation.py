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

    m = re.fullmatch(r"([xyz])(<=|<)([^<>=]+)", expr)
    if m:
        axis, op, upper = m.groups()
        return RangeConstraint(axis=axis, lower=None, lower_inclusive=False, upper=upper, upper_inclusive=op == "<=")

    m = re.fullmatch(r"([xyz])(>=|>)([^<>=]+)", expr)
    if m:
        axis, op, lower = m.groups()
        return RangeConstraint(axis=axis, lower=lower, lower_inclusive=op == ">=", upper=None, upper_inclusive=False)

    m = re.fullmatch(r"([^<>=]+)(<=|<)([xyz])", expr)
    if m:
        lower, op, axis = m.groups()
        return RangeConstraint(axis=axis, lower=lower, lower_inclusive=op == "<=", upper=None, upper_inclusive=False)

    m = re.fullmatch(r"([^<>=]+)(>=|>)([xyz])", expr)
    if m:
        upper, op, axis = m.groups()
        return RangeConstraint(axis=axis, lower=None, lower_inclusive=False, upper=upper, upper_inclusive=op == ">=")

    return None
