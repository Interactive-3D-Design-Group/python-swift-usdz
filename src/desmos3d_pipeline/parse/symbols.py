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
    m = re.fullmatch(
        r"([A-Za-z](?:(?:_[A-Za-z0-9]+)|(?:_\{[^{}]+\})|(?:\{[^{}]+\}))?)(?:=)(.+)",
        core_expr,
    )
    if not m:
        return None
    name, rhs = m.groups()
    name = re.sub(r"_\{([^{}]+)\}", r"_\1", name)
    if name in {"x", "y", "z"}:
        return None
    if rhs.startswith("(") and rhs.endswith(")"):
        return None
    return SymbolAssignment(name=name, expr=rhs)


def parse_point_definition(core_expr: str) -> PointDefinition | None:
    c = core_expr.strip()
    for _ in range(4):
        mwrap = re.fullmatch(
            r"([A-Za-z](?:(?:_[A-Za-z0-9]+)|(?:_\{[^{}]+\})|(?:\{[^{}]+\}))?)=\(\(([^,]+),([^,]+),([^\)]+)\)\)",
            c,
        )
        if not mwrap:
            break
        c = f"{mwrap.group(1)}=({mwrap.group(2)},{mwrap.group(3)},{mwrap.group(4)})"
    m = re.fullmatch(
        r"([A-Za-z](?:(?:_[A-Za-z0-9]+)|(?:_\{[^{}]+\})|(?:\{[^{}]+\}))?)=\(([^,]+),([^,]+),([^\)]+)\)",
        c,
    )
    if not m:
        return None
    name, x, y, z = m.groups()
    return PointDefinition(name=name, x=x, y=y, z=z)
