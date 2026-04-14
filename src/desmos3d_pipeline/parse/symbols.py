from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class SymbolAssignment:
    name: str
    expr: str


@dataclass(slots=True)
class PointDefinition:
    name: str
    x: str
    y: str
    z: str


def parse_assignment(core_expr: str) -> SymbolAssignment | None:
    m = re.fullmatch(r"([A-Za-z](?:_[A-Za-z0-9]+|\{[^{}]+\})?)(?:=)(.+)", core_expr)
    if not m:
        return None
    name, rhs = m.groups()
    if name in {"x", "y", "z"}:
        return None
    if rhs.startswith("(") and rhs.endswith(")"):
        return None
    return SymbolAssignment(name=name, expr=rhs)


def parse_point_definition(core_expr: str) -> PointDefinition | None:
    m = re.fullmatch(
        r"([A-Za-z](?:(?:_[A-Za-z0-9]+)|(?:_\{[^{}]+\})|(?:\{[^{}]+\}))?)=\(([^,]+),([^,]+),([^\)]+)\)",
        core_expr,
    )
    if not m:
        return None
    name, x, y, z = m.groups()
    return PointDefinition(name=name, x=x, y=y, z=z)
