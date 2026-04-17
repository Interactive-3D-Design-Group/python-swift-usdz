from __future__ import annotations

from desmos3d_pipeline.classify.rules import classify_expression
from desmos3d_pipeline.parse.disk_extrusion import try_disk_extrusion_world_bbox, try_parse_axis_aligned_disk_inequality
from desmos3d_pipeline.ir.builder import _first_scalar_from_desmos_bracket_list, _resolve_symbol_table
from desmos3d_pipeline.ir.models import ClassificationStatus, ExpressionFamily
from desmos3d_pipeline.normalize.latex import extract_brace_restrictions, normalize_latex
from desmos3d_pipeline.parse.relation import parse_interval_constraint


def test_normalize_latex_desmos_tokens() -> None:
    raw = r"z\le\frac{x}{2}\left\{-3<x<3\right\}"
    normalized = normalize_latex(raw)
    assert "<=" in normalized
    assert "(x)/(2)" in normalized
    assert "{" in normalized and "}" in normalized


def test_extract_brace_restrictions() -> None:
    core, restrictions = extract_brace_restrictions("z=7{-3<x<3}{-26<y<26}")
    assert core == "z=7"
    assert restrictions == ["-3<x<3", "-26<y<26"]


def test_parse_interval_constraint_chained() -> None:
    constraint = parse_interval_constraint("-5<=x<=5")
    assert constraint is not None
    assert constraint.axis == "x"
    assert constraint.lower == "-5"
    assert constraint.upper == "5"


def test_classify_constant_plane_supported() -> None:
    result = classify_expression("z=80{100>x>60}{20>y>-20}", "expression")
    assert result.family == ExpressionFamily.CONSTANT_PLANE
    assert result.status == ClassificationStatus.SUPPORTED


def test_classify_box_supported() -> None:
    result = classify_expression("-5<=x<=5{-80<=z<=171}{20<=y<=25}", "expression")
    assert result.family == ExpressionFamily.BOX_BOUNDED_REGION
    assert result.status == ClassificationStatus.SUPPORTED


def test_classify_unknown() -> None:
    result = classify_expression("thisIsNotValid", "expression")
    assert result.status == ClassificationStatus.UNRECOGNIZED


def test_first_scalar_from_desmos_bracket_list() -> None:
    assert _first_scalar_from_desmos_bracket_list("[-12,-10.8,-9.6]") == -12.0
    assert _first_scalar_from_desmos_bracket_list("[0.95,2.15]") == 0.95
    assert _first_scalar_from_desmos_bracket_list("0.5") is None
    assert _first_scalar_from_desmos_bracket_list("[a,1]") is None


def test_resolve_symbol_table_slider_list_literals() -> None:
    out = _resolve_symbol_table(
        {
            "a": "[-12,-10.8]",
            "b": "[-11.75,-10.55]",
            "x": "1",
        }
    )
    assert out["a"] == -12.0
    assert out["b"] == -11.75
    assert out["x"] == 1.0


def test_disk_extrusion_parse_vertical_cylinder() -> None:
    spec = try_parse_axis_aligned_disk_inequality("(x-25)^2+(y+8)^2<=2")
    assert spec is not None
    assert spec.axis_u == "x" and spec.axis_v == "y"
    assert spec.center_u == 25.0 and spec.center_v == -8.0
    assert spec.radius_sq == 2.0
    bbox = try_disk_extrusion_world_bbox(["9<z<43"], spec, {})
    assert bbox is not None
    assert bbox[4] == 9.0 and bbox[5] == 43.0


def test_disk_extrusion_parse_implicit_squares() -> None:
    spec = try_parse_axis_aligned_disk_inequality("x^2+y^2<=2000")
    assert spec is not None
    assert spec.radius_sq == 2000.0


def test_classify_disk_extrusion_supported() -> None:
    r = classify_expression("(x+1)^2+(y-2)^2<=3{0<z<1}", "expression")
    assert r.family == ExpressionFamily.DISK_EXTRUSION_SOLID
    assert r.status == ClassificationStatus.SUPPORTED


def test_partial_axis_slabs_not_promoted_with_viewport() -> None:
    """Viewport must not complete missing axis: would mesh a scene-sized slab (JSONLondon regression)."""
    vp = {"xmin": -200.0, "xmax": 200.0, "ymin": -50.0, "ymax": 50.0, "zmin": -139.0, "zmax": 139.0}
    r = classify_expression("20>y>-20{-225<x<225}", "expression", vp)
    assert r.family == ExpressionFamily.INEQUALITY_REGION
    assert r.status == ClassificationStatus.RECOGNIZED_UNSUPPORTED


def test_classify_param_assignment_geometry_ineligible() -> None:
    r = classify_expression("a=[1,2,3]", "expression")
    assert r.family == ExpressionFamily.PARAM_ASSIGNMENT
    assert r.status == ClassificationStatus.GEOMETRY_INELIGIBLE
