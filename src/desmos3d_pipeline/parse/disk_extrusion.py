from __future__ import annotations

import math
import re
from dataclasses import dataclass

from desmos3d_pipeline.parse.math_eval import safe_eval
from desmos3d_pipeline.parse.relation import parse_interval_constraint


@dataclass(slots=True)
class DiskExtrusionSpec:
    """Solid (u-cu)^2+(v-cv)^2<=radius_sq with u,v distinct axes in {x,y,z} and constant radius_sq."""

    axis_u: str
    axis_v: str
    center_u: float
    center_v: float
    radius_sq: float

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


def _parse_two_square_terms(text: str) -> tuple[str, str, float] | None:
    """Return (inner1, inner2, radius_sq) for ``inner1^2+inner2^2 <= rhs`` (already oriented)."""
    m = re.fullmatch(r"\((.+?)\)\^2\+\((.+?)\)\^2(<=)([^<>=]+)", text)
    if not m:
        return None
    a, b, op, rhs = m.groups()
    if op != "<=":
        return None
    try:
        r2 = float(rhs)
    except ValueError:
        try:
            r2 = float(safe_eval(rhs, {}))
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
        m = re.fullmatch(r"([xyz])\^2\+([xyz])\^2(<=)([^<>=]+)", text)
        if m:
            ax1, ax2, op, rhs = m.groups()
            if op != "<=" or ax1 == ax2:
                return None
            try:
                r2 = float(rhs)
            except ValueError:
                try:
                    r2 = float(safe_eval(rhs, {}))
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


def try_disk_extrusion_world_bbox(
    restrictions: list[str],
    spec: DiskExtrusionSpec,
    _viewport: dict[str, float],
) -> tuple[float, float, float, float, float, float] | None:
    """Merge brace intervals into an axis-aligned bbox; pad disk plane axes when open."""
    acc = {a: {"lo": None, "hi": None} for a in ("x", "y", "z")}
    for r in restrictions:
        rc = parse_interval_constraint(r)
        if rc is None:
            return None
        if not _merge_range_into(rc.axis, rc, acc):
            return None

    ext = spec.extrusion_axis
    if acc[ext]["lo"] is None or acc[ext]["hi"] is None:
        return None

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
    if xmin >= xmax or ymin >= ymax or zmin >= zmax:
        return None

    return (float(xmin), float(xmax), float(ymin), float(ymax), float(zmin), float(zmax))
