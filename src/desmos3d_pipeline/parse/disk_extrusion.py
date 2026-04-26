from __future__ import annotations

import math
import re
from dataclasses import dataclass

from desmos3d_pipeline.parse.math_eval import safe_eval, to_python_expr
from desmos3d_pipeline.parse.parametric import _split_first_paren_block, _split_three_commas
from desmos3d_pipeline.parse.relation import parse_interval_constraint


@dataclass(slots=True)
class DiskExtrusionSpec:
    """Solid (u-cu)^2+(v-cv)^2<=radius_sq with u,v distinct axes in {x,y,z} and constant radius_sq.

    For cylinder *equality*, ``stretch_u`` / ``stretch_v`` are |m| when a term is ``(m*axis+const)^2``
    (defaults ``1`` for plain ``(axis-center)^2``).
    """

    axis_u: str
    axis_v: str
    center_u: float
    center_v: float
    radius_sq: float
    stretch_u: float = 1.0
    stretch_v: float = 1.0

    @property
    def extrusion_axis(self) -> str:
        for a in ("x", "y", "z"):
            if a not in (self.axis_u, self.axis_v):
                return a
        raise RuntimeError("invalid disk axes")


def _parse_axis_linear_center(inner: str) -> tuple[str, float] | None:
    """Parse ``x``, ``x-25``, ``x+62`` (normalized, no spaces) -> (axis, center)."""
    inner = inner.strip()
    m = re.fullmatch(r"([xyz])([+-]\d+(?:\.\d+)?)?", inner)
    if not m:
        return None
    axis, tail = m.group(1), m.group(2)
    if not tail:
        return axis, 0.0
    # (axis + tail) is coordinate; center c where (coord - c)^2 matches: y+8 -> c=-8
    return axis, -float(tail)


def _parse_axis_affine_center(inner: str) -> tuple[str, float, float] | None:
    """Parse ``m*axis+const`` (one of x,y,z) into ``(axis, center, |m|)`` for squared ``(m*(axis-center))^2``.

    Accepts ``0.35z-7``, ``x+3.5``, ``-x+2``, ``x``, ``x-25`` (same centers as :func:`_parse_axis_linear_center`).
    """
    inner = inner.strip().replace(" ", "")
    if not inner:
        return None

    m = re.fullmatch(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))([xyz])([+-]\d+(?:\.\d+)?)?$", inner)
    if m:
        coeff = float(m.group(1))
        axis = m.group(2)
        tail = m.group(3)
        const = float(tail) if tail else 0.0
        if abs(coeff) < 1e-15:
            return None
        center = -const / coeff
        return axis, center, abs(coeff)

    m2 = re.fullmatch(r"^(-)([xyz])([+-]\d+(?:\.\d+)?)?$", inner)
    if m2:
        axis = m2.group(2)
        coeff = -1.0
        tail = m2.group(3)
        const = float(tail) if tail else 0.0
        center = -const / coeff
        return axis, center, abs(coeff)

    lin = _parse_axis_linear_center(inner)
    if lin is None:
        return None
    axis, center = lin
    return axis, center, 1.0


def _parse_two_square_terms(text: str) -> tuple[str, str, float] | None:
    """Return (inner1, inner2, radius_sq) for ``inner1^2+inner2^2 <= rhs`` (already oriented)."""
    m = re.fullmatch(r"\((.+?)\)\^2\+\((.+?)\)\^2(<=|<)([^<>=]+)", text)
    if not m:
        return None
    a, b, op, rhs = m.groups()
    if op not in {"<=", "<"}:
        return None
    try:
        r2 = float(rhs)
    except ValueError:
        try:
            r2 = float(safe_eval(to_python_expr(rhs, {}), {}))
        except Exception:
            return None
    if r2 <= 0:
        return None
    return a, b, r2


def try_parse_axis_aligned_disk_inequality(core: str) -> DiskExtrusionSpec | None:
    """Recognize (u..)^2+(v..)^2<=K or K>=(u..)^2+(v..)^2 with linear u,v terms only."""
    text = core.replace(" ", "")

    oriented: tuple[str, str, float] | None = None
    t = _parse_two_square_terms(text)
    if t:
        oriented = t
    else:
        m = re.fullmatch(r"([^<>=]+)(>=)(\(.+?\)\^2\+\(.+?\)\^2)", text)
        if m:
            lhs, op, rhs = m.groups()
            if op != ">=":
                return None
            t2 = _parse_two_square_terms(rhs + "<=" + lhs)
            if t2:
                oriented = t2

    if not oriented:
        m = re.fullmatch(r"([xyz])\^2\+([xyz])\^2(<=|<)([^<>=]+)", text)
        if m:
            ax1, ax2, op, rhs = m.groups()
            if op not in {"<=", "<"} or ax1 == ax2:
                return None
            try:
                r2 = float(rhs)
            except ValueError:
                try:
                    r2 = float(safe_eval(to_python_expr(rhs, {}), {}))
                except Exception:
                    return None
            if r2 <= 0:
                return None
            inner1, inner2 = ax1, ax2
            p1 = _parse_axis_linear_center(inner1)
            p2 = _parse_axis_linear_center(inner2)
            if not p1 or not p2:
                return None
            oriented = (inner1, inner2, r2)
        else:
            return None

    inner1, inner2, r2 = oriented
    p1 = _parse_axis_linear_center(inner1)
    p2 = _parse_axis_linear_center(inner2)
    if not p1 or not p2:
        return None
    ax1, c1 = p1
    ax2, c2 = p2
    if ax1 == ax2:
        return None
    return DiskExtrusionSpec(axis_u=ax1, axis_v=ax2, center_u=c1, center_v=c2, radius_sq=r2)


def try_parse_vertical_cylinder_equality(core: str) -> DiskExtrusionSpec | None:
    """Parse ``(x-cx)^2+(y-cy)^2=K`` or ``x^2+y^2=K`` as a vertical cylinder *surface* along the remaining axis."""
    text = core.replace(" ", "")

    def _rhs_r2(rhs: str) -> float | None:
        try:
            r2 = float(rhs)
        except ValueError:
            try:
                r2 = float(safe_eval(to_python_expr(rhs, {}), {}))
            except Exception:
                return None
        if r2 <= 0:
            return None
        return r2

    m = re.fullmatch(r"\((.+?)\)\^2\+\((.+?)\)\^2=([^<>=]+)", text)
    if m:
        inner1, inner2, rhs = m.groups()
        r2 = _rhs_r2(rhs)
        if r2 is None:
            return None
        p1 = _parse_axis_affine_center(inner1)
        p2 = _parse_axis_affine_center(inner2)
        if not p1 or not p2:
            return None
        ax1, c1, s1 = p1
        ax2, c2, s2 = p2
        if ax1 == ax2:
            return None
        return DiskExtrusionSpec(
            axis_u=ax1,
            axis_v=ax2,
            center_u=c1,
            center_v=c2,
            radius_sq=r2,
            stretch_u=s1,
            stretch_v=s2,
        )

    m = re.fullmatch(r"([xyz])\^2\+([xyz])\^2=([^<>=]+)", text)
    if not m:
        return None
    ax1, ax2, rhs = m.groups()
    if ax1 == ax2:
        return None
    r2 = _rhs_r2(rhs)
    if r2 is None:
        return None
    p1 = _parse_axis_linear_center(ax1)
    p2 = _parse_axis_linear_center(ax2)
    if not p1 or not p2:
        return None
    c1, c2 = p1[1], p2[1]
    return DiskExtrusionSpec(axis_u=ax1, axis_v=ax2, center_u=c1, center_v=c2, radius_sq=r2)


def _eval_const_bound(s: str | None) -> float | None:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        try:
            return float(safe_eval(s, {}))
        except Exception:
            return None


def _merge_range_into(axis: str, rc, acc: dict[str, dict[str, float | None]]) -> bool:
    cur = acc[axis]
    lo = _eval_const_bound(rc.lower) if rc.lower is not None else None
    hi = _eval_const_bound(rc.upper) if rc.upper is not None else None
    if lo is not None:
        cur["lo"] = lo if cur["lo"] is None else max(cur["lo"], lo)
    if hi is not None:
        cur["hi"] = hi if cur["hi"] is None else min(cur["hi"], hi)
    if cur["lo"] is not None and cur["hi"] is not None and cur["lo"] > cur["hi"]:
        return False
    return True


def _viewport_axis_extent(viewport: dict[str, float], axis: str) -> tuple[float, float] | None:
    """Return finite (min, max) for ``axis`` from graph viewport keys ``{axis}min`` / ``{axis}max``."""
    lo_k, hi_k = f"{axis}min", f"{axis}max"
    if lo_k not in viewport or hi_k not in viewport:
        return None
    lo, hi = float(viewport[lo_k]), float(viewport[hi_k])
    if lo >= hi:
        return None
    return lo, hi


def try_disk_extrusion_world_bbox(
    restrictions: list[str],
    spec: DiskExtrusionSpec,
    viewport: dict[str, float],
) -> tuple[float, float, float, float, float, float] | None:
    """Merge brace intervals into an axis-aligned bbox; pad disk plane axes when open.

    If the inequality is a 2D disk in (axis_u, axis_v) with no brace bounds on the extrusion
    axis, fill that axis from the graph ``viewport`` so the solid is finite ("promoted" slab).
    """
    acc = {a: {"lo": None, "hi": None} for a in ("x", "y", "z")}
    for r in restrictions:
        rc = parse_interval_constraint(r)
        if rc is None:
            return None
        if not _merge_range_into(rc.axis, rc, acc):
            return None

    ext = spec.extrusion_axis
    ext_vp = _viewport_axis_extent(viewport, ext)
    if acc[ext]["lo"] is None:
        if ext_vp is None:
            return None
        acc[ext]["lo"] = ext_vp[0]
    if acc[ext]["hi"] is None:
        if ext_vp is None:
            return None
        acc[ext]["hi"] = ext_vp[1]

    pad = math.sqrt(max(spec.radius_sq, 0.0)) + 1e-5
    for ax in (spec.axis_u, spec.axis_v):
        c = spec.center_u if ax == spec.axis_u else spec.center_v
        if acc[ax]["lo"] is None:
            acc[ax]["lo"] = c - pad
        if acc[ax]["hi"] is None:
            acc[ax]["hi"] = c + pad

    for a in ("x", "y", "z"):
        if acc[a]["lo"] is None or acc[a]["hi"] is None:
            return None

    xmin, xmax = acc["x"]["lo"], acc["x"]["hi"]
    ymin, ymax = acc["y"]["lo"], acc["y"]["hi"]
    zmin, zmax = acc["z"]["lo"], acc["z"]["hi"]
    if xmin >= xmax:
        mid = (xmin + xmax) / 2.0
        pad = max(1e-4, abs(mid) * 1e-6 + 1e-4)
        xmin, xmax = mid - pad, mid + pad
    if ymin >= ymax:
        mid = (ymin + ymax) / 2.0
        pad = max(1e-4, abs(mid) * 1e-6 + 1e-4)
        ymin, ymax = mid - pad, mid + pad
    if zmin >= zmax:
        mid = (zmin + zmax) / 2.0
        pad = max(1e-4, abs(mid) * 1e-6 + 1e-4)
        zmin, zmax = mid - pad, mid + pad

    return (float(xmin), float(xmax), float(ymin), float(ymax), float(zmin), float(zmax))


@dataclass(slots=True)
class SphereSolidSpec:
    """Solid ``(x-cx)^2+(y-cy)^2+(z-cz)^2<=radius_sq`` (axis-aligned ball, ``radius_sq`` is RHS of inequality)."""

    center_x: float
    center_y: float
    center_z: float
    radius_sq: float


def _sphere_radius_sq(rhs: str) -> float | None:
    rhs = rhs.strip()
    try:
        r2 = float(rhs)
    except ValueError:
        try:
            r2 = float(safe_eval(to_python_expr(rhs, {}), {}))
        except Exception:
            return None
    if r2 <= 0:
        return None
    return r2


def try_parse_sphere_solid_inequality(core: str) -> SphereSolidSpec | None:
    """Parse ``(x-cx)^2+(y-cy)^2+(z-cz)^2<r^2`` (or ``<=``), including ``x^2+y^2+z^2`` forms."""
    text = core.replace(" ", "")

    def _from_centers(i1: str, i2: str, i3: str, op: str, rhs: str) -> SphereSolidSpec | None:
        if op not in {"<", "<="}:
            return None
        r2 = _sphere_radius_sq(rhs)
        if r2 is None:
            return None
        p1 = _parse_axis_linear_center(i1)
        p2 = _parse_axis_linear_center(i2)
        p3 = _parse_axis_linear_center(i3)
        if not p1 or not p2 or not p3:
            return None
        ax1, c1 = p1
        ax2, c2 = p2
        ax3, c3 = p3
        if {ax1, ax2, ax3} != {"x", "y", "z"}:
            return None
        centers = {ax1: c1, ax2: c2, ax3: c3}
        return SphereSolidSpec(center_x=centers["x"], center_y=centers["y"], center_z=centers["z"], radius_sq=r2)

    m = re.fullmatch(r"\((.+?)\)\^2\+\((.+?)\)\^2\+\((.+?)\)\^2(<|<=)([^<>=]+)", text)
    if m:
        spec = _from_centers(*m.groups())
        if spec is not None:
            return spec

    m2 = re.fullmatch(r"([xyz])\^2\+([xyz])\^2\+([xyz])\^2(<|<=)([^<>=]+)", text)
    if m2:
        a1, a2, a3, op, rhs = m2.groups()
        if {a1, a2, a3} != {"x", "y", "z"}:
            return None
        r2 = _sphere_radius_sq(rhs)
        if r2 is None:
            return None
        p1 = _parse_axis_linear_center(a1)
        p2 = _parse_axis_linear_center(a2)
        p3 = _parse_axis_linear_center(a3)
        if not p1 or not p2 or not p3:
            return None
        centers = {p1[0]: p1[1], p2[0]: p2[1], p3[0]: p3[1]}
        return SphereSolidSpec(center_x=centers["x"], center_y=centers["y"], center_z=centers["z"], radius_sq=r2)

    return None


def _numeric_literal_coord(s: str) -> float | None:
    """Parse a numeric literal for sphere centers / radius; reject bare identifiers (e.g. slider names)."""
    s = s.strip()
    while s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    if re.search(r"[A-Za-z]", s):
        return None
    try:
        return float(safe_eval(to_python_expr(s, {}), {}))
    except Exception:
        return None


def try_parse_operatorname_sphere_tuple_core(core: str) -> SphereSolidSpec | None:
    """Parse Desmos ``operatorname{sphere}`` argument core ``((cx,cy,cz),r)`` or ``(cx,cy,cz),r`` (normalized)."""
    t = core.strip().replace(" ", "")
    outer = _split_first_paren_block(t)
    if outer is None:
        return None
    inner_pair, rest_outer = outer
    inner_pair = inner_pair.strip()
    if inner_pair.startswith("("):
        tb = _split_first_paren_block(inner_pair)
        if tb is None:
            return None
        tri_s, after_triple = tb
        tail = after_triple.strip()
        if not tail.startswith(","):
            return None
        rad_s = tail[1:].strip().rstrip(")")
    else:
        tri_s = inner_pair
        tail = rest_outer.strip()
        if not tail.startswith(","):
            return None
        rad_s = tail[1:].strip().rstrip(")")
    parts = _split_three_commas(tri_s)
    if parts is None:
        return None
    sx, sy, sz = parts
    cx = _numeric_literal_coord(sx)
    cy = _numeric_literal_coord(sy)
    cz = _numeric_literal_coord(sz)
    r = _numeric_literal_coord(rad_s)
    if cx is None or cy is None or cz is None or r is None or r <= 0:
        return None
    r2 = r * r
    return SphereSolidSpec(center_x=cx, center_y=cy, center_z=cz, radius_sq=r2)


def try_sphere_solid_world_bbox(
    restrictions: list[str],
    spec: SphereSolidSpec,
    _viewport: dict[str, float],
) -> tuple[float, float, float, float, float, float] | None:
    """Merge brace intervals; pad missing axes with the ball radius; clip to the ball's axis-aligned bounds."""
    acc = {a: {"lo": None, "hi": None} for a in ("x", "y", "z")}
    for r in restrictions:
        rc = parse_interval_constraint(r)
        if rc is None:
            return None
        if not _merge_range_into(rc.axis, rc, acc):
            return None

    rpad = math.sqrt(max(spec.radius_sq, 0.0)) + 1e-5
    for ax, c in zip(("x", "y", "z"), (spec.center_x, spec.center_y, spec.center_z)):
        if acc[ax]["lo"] is None:
            acc[ax]["lo"] = c - rpad
        if acc[ax]["hi"] is None:
            acc[ax]["hi"] = c + rpad

    for ax, c in zip(("x", "y", "z"), (spec.center_x, spec.center_y, spec.center_z)):
        slo, shi = c - rpad, c + rpad
        acc[ax]["lo"] = max(float(acc[ax]["lo"]), slo)
        acc[ax]["hi"] = min(float(acc[ax]["hi"]), shi)

    xmin, xmax = acc["x"]["lo"], acc["x"]["hi"]
    ymin, ymax = acc["y"]["lo"], acc["y"]["hi"]
    zmin, zmax = acc["z"]["lo"], acc["z"]["hi"]
    if xmin >= xmax:
        mid = (xmin + xmax) / 2.0
        pad = max(1e-4, abs(mid) * 1e-6 + 1e-4)
        xmin, xmax = mid - pad, mid + pad
    if ymin >= ymax:
        mid = (ymin + ymax) / 2.0
        pad = max(1e-4, abs(mid) * 1e-6 + 1e-4)
        ymin, ymax = mid - pad, mid + pad
    if zmin >= zmax:
        mid = (zmin + zmax) / 2.0
        pad = max(1e-4, abs(mid) * 1e-6 + 1e-4)
        zmin, zmax = mid - pad, mid + pad

    return (float(xmin), float(xmax), float(ymin), float(ymax), float(zmin), float(zmax))
