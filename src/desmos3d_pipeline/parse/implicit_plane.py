"""Detect linear implicit planes ``F(x,y,z)=0`` with a single ``=`` and solve to ``z=f(x,y)``."""

from __future__ import annotations

import random

from desmos3d_pipeline.parse.math_eval import SafeEvalError, safe_eval, to_python_expr


def try_linear_implicit_plane_z_rhs(core: str) -> str | None:
    """If ``core`` is a linear implicit equality, return RHS string for ``z=...`` (mesh phase is ``z=f(x,y)`` only)."""
    c = core.strip().replace(" ", "")
    if c.count("=") != 1:
        return None
    if any(op in c for op in ("<=", ">=", "<", ">")):
        return None
    lhs, rhs = core.strip().split("=", 1)
    lhs, rhs = lhs.strip(), rhs.strip()
    combined = f"({lhs})-({rhs})"
    try:
        py = to_python_expr(combined, {})
    except Exception:
        return None

    def ev(x: float, y: float, z: float) -> float:
        return float(safe_eval(py, {"x": x, "y": y, "z": z}))

    try:
        d = ev(0.0, 0.0, 0.0)
        a = ev(1.0, 0.0, 0.0) - d
        b = ev(0.0, 1.0, 0.0) - d
        ccoef = ev(0.0, 0.0, 1.0) - d
    except (SafeEvalError, ZeroDivisionError, ValueError, TypeError):
        return None

    rng = random.Random(42)
    for _ in range(10):
        x = rng.uniform(-2.7, 3.1)
        y = rng.uniform(-1.9, 2.4)
        z = rng.uniform(-2.2, 2.8)
        try:
            got = ev(x, y, z)
        except (SafeEvalError, ZeroDivisionError, ValueError, TypeError):
            return None
        if abs(got - (a * x + b * y + ccoef * z + d)) > 5e-4 * max(1.0, abs(got), abs(a * x + b * y + ccoef * z + d)):
            return None

    if abs(ccoef) < 1e-9:
        return None

    parts: list[str] = []
    if abs(a) >= 1e-12:
        parts.append(f"({repr(a)})*x")
    if abs(b) >= 1e-12:
        parts.append(f"({repr(b)})*y")
    if abs(d) >= 1e-12:
        parts.append(f"({repr(d)})")
    numerator = "+".join(parts) if parts else "0"
    return f"-({numerator})/({repr(ccoef)})"
