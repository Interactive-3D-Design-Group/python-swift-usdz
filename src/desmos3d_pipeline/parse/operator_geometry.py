"""Parse Desmos ``operatorname{triangle|polygon|segment}(...)`` argument lists into inline coordinate triples."""

from __future__ import annotations

import re

from desmos3d_pipeline.normalize.latex import extract_brace_restrictions
from desmos3d_pipeline.parse.math_eval import safe_eval, to_python_expr
from desmos3d_pipeline.parse.parametric import (
    _split_first_paren_block,
    _split_three_commas,
    _strip_redundant_outer_parens,
)

VERTEX_POINT_REF_MARKER = "__ref__"
VertexSpec = tuple[str, str, str]


def strip_operatorname_args_prefix(normalized_latex: str, op_name: str) -> str | None:
    """Return text after ``operatorname{op}`` (or legacy ``op(`` for ``triangle``/``polygon``/``segment``)."""
    n = normalized_latex.strip()
    pref = f"operatorname{{{op_name}}}"
    if n.startswith(pref):
        return n[len(pref) :]
    if n.startswith(f"{op_name}("):
        return n[len(op_name) + 1 : -1] if n.endswith(")") else n[len(op_name) + 1 :]
    return None


def _split_commas_depth_zero(s: str) -> list[str]:
    depth = 0
    start = 0
    parts: list[str] = []
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(s[start:i].strip())
            start = i + 1
    parts.append(s[start:].strip())
    return [p for p in parts if p]


def _parse_vertex_chunk(chunk: str) -> VertexSpec | None:
    tb = _split_first_paren_block(chunk.strip())
    if tb is None:
        return None
    tri, rest = tb
    if rest.strip():
        return None
    inner = _strip_redundant_outer_parens(tri.strip())
    parts = _split_three_commas(inner)
    if parts is not None:
        return parts
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", inner):
        return (VERTEX_POINT_REF_MARKER, inner, "")
    return None


def parse_parenthesized_xyz_tuple_list(core: str) -> list[VertexSpec] | None:
    """Parse ``(x,y,z)`` / ``(A)`` vertices with optional outer ``((...),...)`` wrapper from Desmos."""
    t = core.strip().replace(" ", "")
    if not t:
        return None
    inner: str
    if t.startswith("("):
        fb = _split_first_paren_block(t)
        if fb is None:
            return None
        inner_first, tail = fb
        if not tail.strip():
            if "(" in inner_first:
                inner = inner_first
            else:
                one = _parse_vertex_chunk(f"({inner_first})")
                if one is None:
                    return None
                return [one]
        else:
            inner = t
    else:
        inner = t
    chunks = _split_commas_depth_zero(inner)
    if not chunks:
        return None
    out: list[VertexSpec] = []
    for ch in chunks:
        spec = _parse_vertex_chunk(ch)
        if spec is None:
            return None
        out.append(spec)
    return out


def triple_is_numeric_literal(triple: tuple[str, str, str], symbol_env: dict[str, float]) -> bool:
    """True if each component evaluates to a float with no unresolved identifiers (beyond ``symbol_env``)."""
    py_map: dict[str, str] = {}
    allowed = set(symbol_env.keys())
    for comp in triple:
        c = comp.strip()
        if re.fullmatch(r"[-+]?[0-9]+(?:\.[0-9]*)?(?:[eE][-+]?[0-9]+)?", c):
            continue
        if re.search(r"[A-Za-z]", c):
            toks = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", c))
            if not toks.issubset(allowed):
                return False
        try:
            safe_eval(to_python_expr(c, py_map), symbol_env)
        except Exception:
            return False
    return True


def vertex_specs_numeric_or_pointrefs(specs: list[VertexSpec]) -> bool:
    """Each vertex is either a numeric ``(x,y,z)`` literal triple or a single point label ``(A)``."""
    for s in specs:
        if s[0] == VERTEX_POINT_REF_MARKER and s[2] == "":
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", s[1]):
                return False
            continue
        if not triple_is_numeric_literal(s, {}):
            return False
    return True


def point_list_is_numeric(points: list[tuple[str, str, str]], symbol_env: dict[str, float]) -> bool:
    return all(triple_is_numeric_literal(p, symbol_env) for p in points)


def resolve_vertex_specs_to_triples(
    specs: list[VertexSpec],
    point_xyz: dict[str, tuple[str, str, str]],
) -> list[tuple[str, str, str]] | None:
    out: list[tuple[str, str, str]] = []
    for s in specs:
        if s[0] == VERTEX_POINT_REF_MARKER and s[2] == "":
            hit = point_xyz.get(s[1])
            if hit is None:
                return None
            out.append(hit)
        else:
            out.append(s)
    return out


def operator_call_core_and_restrictions(normalized_latex: str, op_name: str) -> tuple[str, list[str]] | None:
    """Strip ``operatorname{op}`` / ``op(``, then ``extract_brace_restrictions`` on the suffix."""
    suf = strip_operatorname_args_prefix(normalized_latex, op_name)
    if suf is None:
        return None
    return extract_brace_restrictions(suf)
