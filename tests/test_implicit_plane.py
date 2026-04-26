from __future__ import annotations

from desmos3d_pipeline.classify.rules import classify_expression
from desmos3d_pipeline.ir.models import ClassificationStatus, ExpressionFamily
from desmos3d_pipeline.parse.implicit_plane import try_linear_implicit_plane_z_rhs
from desmos3d_pipeline.parse.math_eval import safe_eval, to_python_expr


def test_try_linear_implicit_plane_z_rhs_desmos_style() -> None:
    rhs = try_linear_implicit_plane_z_rhs("2.8x-1.25z+8.4=0")
    assert rhs is not None
    py = to_python_expr(rhs, {})
    x, y = 1.0, -0.5
    z = safe_eval(py, {"x": x, "y": y, "z": 0.0})
    assert abs((2.8 * x - 1.25 * z + 8.4)) < 1e-5


def test_linear_implicit_plane_classify_supported_with_viewport() -> None:
    vp = {"xmin": -5.0, "xmax": 5.0, "ymin": -5.0, "ymax": 5.0, "zmin": -5.0, "zmax": 5.0}
    r = classify_expression("2.8x-1.25z+8.4=0{-2<x<2}{-2<y<2}", "expression", vp)
    assert r.family == ExpressionFamily.LINEAR_SURFACE_PATCH
    assert r.status == ClassificationStatus.SUPPORTED


def test_z_surface_x_plus_y_domain_supported_with_viewport() -> None:
    vp = {"xmin": -5.0, "xmax": 5.0, "ymin": -5.0, "ymax": 5.0, "zmin": -5.0, "zmax": 5.0}
    r = classify_expression("z=x+y{-1<x+y<1}", "expression", vp)
    assert r.family == ExpressionFamily.LINEAR_SURFACE_PATCH
    assert r.status == ClassificationStatus.SUPPORTED


def test_z_surface_x_plus_y_domain_requires_viewport() -> None:
    r = classify_expression("z=x+y{-1<x+y<1}", "expression", {})
    assert r.family == ExpressionFamily.LINEAR_SURFACE_PATCH
    assert r.status == ClassificationStatus.GEOMETRY_INELIGIBLE
