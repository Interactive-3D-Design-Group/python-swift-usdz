from __future__ import annotations

import re
from typing import Literal, TypeAlias

PointRefOrTriple: TypeAlias = tuple[str, str, str] | str


def _split_first_paren_block(s: str) -> tuple[str, str] | None:
    """If ``s`` starts with ``(``, return ``(inner, rest)`` after the matching close paren."""
    if not s.startswith("("):
        return None
    depth = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return s[1:i], s[i + 1 :]
    return None


def _split_three_commas(inner: str) -> tuple[str, str, str] | None:
    depth = 0
    start = 0
    parts: list[str] = []
    for i, ch in enumerate(inner):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(inner[start:i].strip())
            start = i + 1
    parts.append(inner[start:].strip())
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def _strip_redundant_outer_parens(s: str) -> str:
    t = s.strip()
    while len(t) >= 2 and t.startswith("(") and t.endswith(")"):
        inner = t[1:-1]
        if _balanced_paren(inner):
            t = inner
        else:
            break
    return t


def _balanced_paren(s: str) -> bool:
    depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _split_diff_at_depth_zero(lhs_rhs: str) -> tuple[str, str] | None:
    """Split ``(A)-(B)`` at the minus between two top-level groups."""
    depth = 0
    for i, ch in enumerate(lhs_rhs):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "-" and depth == 0 and i > 0:
            return lhs_rhs[:i].strip(), lhs_rhs[i + 1 :].strip()
    return None


def _parse_point_ref_or_xyz_inner(inner: str) -> PointRefOrTriple | None:
    t = _strip_redundant_outer_parens(inner.strip())
    trip = _split_three_commas(t)
    if trip is not None:
        return trip
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", t):
        return t
    return None


def parse_parametric_line_point_and_q(core: str) -> tuple[PointRefOrTriple, PointRefOrTriple] | None:
    """Parse ``P+t*(Q-P)``; each endpoint is an ``(x,y,z)`` triple or a single point label."""
    text = core.replace(" ", "")
    while text.endswith(")") and text.count("(") < text.count(")"):
        text = text[:-1]
    if "+t" not in text:
        return None
    first = _split_first_paren_block(text)
    if first is None:
        return None
    _p_inner, rest = first
    if not rest.startswith("+t"):
        return None
    rest = rest[2:]
    if rest.startswith("*"):
        rest = rest[1:]
    vec = _strip_redundant_outer_parens(rest)
    vec = _strip_redundant_outer_parens(vec)
    diff = _split_diff_at_depth_zero(vec)
    if diff is None:
        return None
    left_tup, right_tup = diff
    l_in = _split_first_paren_block(left_tup)
    r_in = _split_first_paren_block(right_tup)
    if l_in is None or r_in is None or l_in[1] != "" or r_in[1] != "":
        return None
    qv = _parse_point_ref_or_xyz_inner(l_in[0])
    pv = _parse_point_ref_or_xyz_inner(r_in[0])
    if qv is None or pv is None:
        return None
    return (pv, qv)


def try_parse_parametric_line_point_t_vector(core: str) -> tuple[str, str, str] | None:
    """Parse ``(px,py,pz)+t*((qx,qy,qz)-(px,py,pz))`` (and ``+t(`` without ``*``).

    Returns component expressions ``px+t*(qx-px)`` suitable for ``safe_eval`` with parameter ``t``.
    """
    pq = parse_parametric_line_point_and_q(core)
    if pq is None:
        return None
    p, q = pq
    if isinstance(p, str) or isinstance(q, str):
        return None
    px, py, pz = p
    qx, qy, qz = q
    return (
        f"({px})+t*(({qx})-({px}))",
        f"({py})+t*(({qy})-({py}))",
        f"({pz})+t*(({qz})-({pz}))",
    )


def split_xyz_parametric_tuple(core: str) -> tuple[str, str, str] | None:
    """Split a top-level ``(xexpr,yexpr,zexpr)`` triple (normalized Desmos core)."""
    text = core.strip()
    if not (text.startswith("(") and text.endswith(")")):
        return None
    inner = text[1:-1]
    depth = 0
    start = 0
    parts: list[str] = []
    for i, ch in enumerate(inner):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(inner[start:i].strip())
            start = i + 1
    parts.append(inner[start:].strip())
    if len(parts) != 3:
        return None
    if any("+t" in p for p in parts):
        return None
    if any("+u" in p or "+v" in p for p in parts):
        return None
    return parts[0], parts[1], parts[2]


def try_parse_parametric_uv_point_u_v_vectors(core: str) -> tuple[str, str, str] | None:
    """Parse ``(px,py,pz)+u((ax,ay,az)-(bx,by,bz))+v((cx,cy,cz)-(dx,dy,dz))`` (Desmos plane patch)."""
    text = core.replace(" ", "")
    first = _split_first_paren_block(text)
    if first is None:
        return None
    p_inner, rest = first
    p_parts = _split_three_commas(p_inner)
    if p_parts is None:
        return None
    px, py, pz = p_parts
    if not rest.startswith("+u"):
        return None
    rest = rest[2:]
    if rest.startswith("*"):
        rest = rest[1:]
    u_block, rest = _split_first_paren_block(rest)
    if u_block is None or rest is None:
        return None
    u_diff = _split_diff_at_depth_zero(u_block)
    if u_diff is None:
        return None
    ua, ub = u_diff
    if not rest.startswith("+v"):
        return None
    rest = rest[2:]
    if rest.startswith("*"):
        rest = rest[1:]
    v_block, tail = _split_first_paren_block(rest)
    if v_block is None or tail.strip():
        return None
    v_diff = _split_diff_at_depth_zero(v_block)
    if v_diff is None:
        return None
    va, vb = v_diff

    def _comps(t: str) -> tuple[str, str, str] | None:
        t = t.strip()
        blk = _split_first_paren_block(t)
        if blk is None or blk[1] != "":
            return None
        return _split_three_commas(blk[0])

    u_parts_a = _comps(ua)
    u_parts_b = _comps(ub)
    v_parts_a = _comps(va)
    v_parts_b = _comps(vb)
    if not u_parts_a or not u_parts_b or not v_parts_a or not v_parts_b:
        return None
    uax, uay, uaz = u_parts_a
    ubx, uby, ubz = u_parts_b
    vax, vay, vaz = v_parts_a
    vbx, vby, vbz = v_parts_b
    xe = f"({px})+u*(({uax})-({ubx}))+v*(({vax})-({vbx}))"
    ye = f"({py})+u*(({uay})-({uby}))+v*(({vay})-({vby}))"
    ze = f"({pz})+u*(({uaz})-({ubz}))+v*(({vaz})-({vbz}))"
    return xe, ye, ze


def _identifiers(expr: str) -> set[str]:
    return set(re.findall(r"(?<![A-Za-z0-9_])([A-Za-z][A-Za-z0-9_]*)(?![A-Za-z0-9_])", expr))


def _uses_param(name: str, text: str) -> bool:
    """True if ``name`` appears as a Desmos parameter (allows ``16u`` style implicit multiply)."""
    return re.search(rf"(?<![A-Za-z_]){re.escape(name)}(?![A-Za-z0-9_])", text) is not None


_ALLOWED_NAMES = {"sin", "cos", "tan", "sqrt", "abs", "pi", "e"}


def infer_parametric_kind(x_expr: str, y_expr: str, z_expr: str) -> Literal["uv", "t"] | None:
    """Return ``uv`` for Desmos-style ``(f(u,v),g(u,v),h(u,v))``, ``t`` for ``(f(t),g(t),h(t))``.

    Rejects expressions that reference graph coordinates ``x``, ``y``, ``z`` or unknown symbols.
    """
    combined = f"{x_expr}{y_expr}{z_expr}"
    names = _identifiers(combined)
    if names & {"x", "y", "z"}:
        return None
    unknown = names - _ALLOWED_NAMES - {"u", "v", "t"}
    if unknown:
        return None
    has_u = _uses_param("u", combined)
    has_v = _uses_param("v", combined)
    has_t = _uses_param("t", combined)
    if has_t and not has_u and not has_v:
        return "t"
    if has_u or has_v:
        return "uv"
    return None
