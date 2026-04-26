from __future__ import annotations

from desmos3d_pipeline.classify.rules import classify_expression
from desmos3d_pipeline.parse.disk_extrusion import (
    try_disk_extrusion_world_bbox,
    try_parse_axis_aligned_disk_inequality,
    try_parse_operatorname_sphere_tuple_core,
    try_parse_sphere_solid_inequality,
    try_parse_vertical_cylinder_equality,
    try_sphere_solid_world_bbox,
)
from desmos3d_pipeline.ir.builder import _first_scalar_from_desmos_bracket_list, _resolve_symbol_table
from desmos3d_pipeline.ir.models import (
    ClassificationStatus,
    ExpressionFamily,
    SourceRef,
    SphereSolidNode,
    VerticalCylinderSurfaceNode,
)
from desmos3d_pipeline.normalize.latex import extract_brace_restrictions, normalize_latex
from desmos3d_pipeline.parse.relation import parse_interval_constraint
from desmos3d_pipeline.parse.math_eval import SafeEvalError, safe_eval, to_python_expr
from desmos3d_pipeline.parse.parametric import (
    split_xyz_parametric_tuple,
    try_parse_parametric_line_point_t_vector,
    try_parse_parametric_uv_point_u_v_vectors,
)
from desmos3d_pipeline.parse.symbols import parse_assignment, parse_point_definition
from desmos3d_pipeline.mesh.meshers import mesh_sphere_solid, mesh_vertical_cylinder_surface


def test_normalize_latex_desmos_tokens() -> None:
    raw = r"z\le\frac{x}{2}\left\{-3<x<3\right\}"
    normalized = normalize_latex(raw)
    assert "<=" in normalized
    assert "(x)/(2)" in normalized
    assert "{" in normalized and "}" in normalized


def test_normalize_operatorname_abs_in_domain_becomes_chained_compare() -> None:
    nl = normalize_latex(r"z=0.1{\operatorname{abs}(x)<3.5}{y<-5}")
    assert "operatorname{abs}" not in nl
    assert "-(3.5)<x<3.5" in nl


def test_constant_plane_with_operatorname_abs_domain_supported() -> None:
    nl = normalize_latex(r"z=0.1{\operatorname{abs}(x)<3.5}{y<-5}{y>-19}")
    vp = {"xmin": -10.0, "xmax": 10.0, "ymin": -25.0, "ymax": 10.0, "zmin": 0.0, "zmax": 10.0}
    r = classify_expression(nl, "expression", vp)
    assert r.family == ExpressionFamily.CONSTANT_PLANE
    assert r.status == ClassificationStatus.SUPPORTED


def test_extract_brace_restrictions() -> None:
    core, restrictions = extract_brace_restrictions("z=7{-3<x<3}{-26<y<26}")
    assert core == "z=7"
    assert restrictions == ["-3<x<3", "-26<y<26"]


def test_extract_brace_restrictions_nested_domain() -> None:
    core, restrictions = extract_brace_restrictions("-2<x<-1.8{0<z<23{-2<y<2}}")
    assert core == "-2<x<-1.8"
    assert restrictions == ["0<z<23", "-2<y<2"]


def test_extract_brace_does_not_strip_sqrt_brace() -> None:
    core, restrictions = extract_brace_restrictions("z=2a(sqrt{2}-1){0<x<1}")
    assert "sqrt(2)" in core
    assert restrictions == ["0<x<1"]


def test_parse_assignment_subscript_braces() -> None:
    a = parse_assignment("p_{pattern}=[0.1,0.2,0.3]")
    assert a is not None
    assert a.name == "p_pattern"
    assert a.expr == "[0.1,0.2,0.3]"


def test_normalize_latex_pi_times_param() -> None:
    assert normalize_latex("cos(2piu)") == "cos(2*pi*u)"


def test_normalize_latex_trig_pi_times_param() -> None:
    assert normalize_latex("cos(piu)") == "cos(pi*u)"
    assert normalize_latex("sin(piv)") == "sin(pi*v)"


def test_safe_eval_chained_compare_min_list_neg() -> None:
    assert abs(safe_eval("1<2<3", {}) - 1.0) < 1e-9
    assert abs(safe_eval("1<x<3", {"x": 2.0}) - 1.0) < 1e-9
    assert abs(safe_eval("min(7-2,5+1)", {}) - 5.0) < 1e-9
    assert abs(safe_eval("-[4.25,3.25]", {}) - (-4.25)) < 1e-9


def test_safe_eval_fractional_pow_negative_base_real_part() -> None:
    py = to_python_expr("(-5)**1.15", {})
    v = safe_eval(py, {})
    assert isinstance(v, float)
    assert abs(v - (-5.671479792853096)) < 1e-6


def test_safe_eval_chained_compare_requires_all_parts() -> None:
    try:
        safe_eval("1<x<0", {"x": 0.5})
    except SafeEvalError:
        return
    raise AssertionError("expected chained compare to fail")


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


def test_vertical_cylinder_equality_affine_in_axis() -> None:
    spec = try_parse_vertical_cylinder_equality("(0.35z-7)^2+(x+3.5)^2=0.3")
    assert spec is not None
    assert spec.axis_u == "z" and spec.axis_v == "x"
    assert abs(spec.center_u - (7.0 / 0.35)) < 1e-6
    assert abs(spec.center_v - (-3.5)) < 1e-9
    assert abs(spec.stretch_u - 0.35) < 1e-9
    assert abs(spec.stretch_v - 1.0) < 1e-9


def test_classify_affine_vertical_cylinder_supported() -> None:
    vp = {"xmin": -10.0, "xmax": 10.0, "ymin": -10.0, "ymax": 10.0, "zmin": -5.0, "zmax": 25.0}
    r = classify_expression("(0.35z-7)^2+(x+3.5)^2=0.3{-2<y<2}", "expression", vp)
    assert r.family == ExpressionFamily.VERTICAL_CYLINDER_SURFACE
    assert r.status == ClassificationStatus.SUPPORTED


def test_parametric_line_extra_closing_paren_classifies() -> None:
    s = "(-15,8,9)+t*((-15,8,40)-(-15,8,9)))"
    r = classify_expression(s, "expression", {})
    assert r.family == ExpressionFamily.PARAMETRIC_T_CURVE
    assert r.status == ClassificationStatus.SUPPORTED


def test_mesh_vertical_cylinder_with_stretch() -> None:
    ref = SourceRef("t.json", "1", None, None, 0)
    node = VerticalCylinderSurfaceNode(
        node_type="vertical_cylinder_surface",
        source_ref=ref,
        family=ExpressionFamily.VERTICAL_CYLINDER_SURFACE,
        status=ClassificationStatus.SUPPORTED,
        original_latex="",
        normalized_latex="",
        color=None,
        hidden=False,
        metadata={},
        axis_u="z",
        axis_v="x",
        center_u=20.0,
        center_v=-3.5,
        radius_sq=0.3,
        stretch_u=0.35,
        stretch_v=1.0,
        extrusion_axis="y",
        z_min=-2.0,
        z_max=2.0,
        theta_segments=20,
        z_segments=4,
    )
    mesh = mesh_vertical_cylinder_surface(node)
    assert mesh.faces
    assert mesh.vertices


def test_classify_disk_extrusion_supported() -> None:
    r = classify_expression("(x+1)^2+(y-2)^2<=3{0<z<1}", "expression")
    assert r.family == ExpressionFamily.DISK_EXTRUSION_SOLID
    assert r.status == ClassificationStatus.SUPPORTED


def test_disk_extrusion_promotes_missing_extrusion_from_viewport() -> None:
    spec = try_parse_axis_aligned_disk_inequality("x^2+y^2<=4")
    assert spec is not None
    vp = {"xmin": -100.0, "xmax": 100.0, "ymin": -100.0, "ymax": 100.0, "zmin": -3.0, "zmax": 7.0}
    bbox = try_disk_extrusion_world_bbox(["-1<x<2", "-2<y<3"], spec, vp)
    assert bbox is not None
    assert bbox[4] == -3.0 and bbox[5] == 7.0


def test_classify_disk_solid_no_z_brace_uses_viewport() -> None:
    vp = {"xmin": -10.0, "xmax": 10.0, "ymin": -10.0, "ymax": 10.0, "zmin": 0.0, "zmax": 4.0}
    r = classify_expression("(x)^2+(y)^2<=1{-2<x<2}{-2<y<2}", "expression", vp)
    assert r.family == ExpressionFamily.DISK_EXTRUSION_SOLID
    assert r.status == ClassificationStatus.SUPPORTED


def test_classify_disk_without_extrusion_viewport_stays_general_inequality() -> None:
    r = classify_expression("(x)^2+(y)^2<=1{-2<x<2}{-2<y<2}", "expression", {})
    assert r.family == ExpressionFamily.INEQUALITY_REGION
    assert r.status == ClassificationStatus.GEOMETRY_INELIGIBLE


def test_disk_xz_plane_promotes_y_from_viewport() -> None:
    spec = try_parse_axis_aligned_disk_inequality("(x)^2+(z-1)^2<=1")
    assert spec is not None
    assert spec.extrusion_axis == "y"
    vp = {"xmin": -5.0, "xmax": 5.0, "ymin": -2.5, "ymax": 2.5, "zmin": 0.0, "zmax": 2.0}
    bbox = try_disk_extrusion_world_bbox(["0<z<2", "-1<x<1"], spec, vp)
    assert bbox is not None
    assert abs(bbox[2] - (-2.5)) < 1e-9 and abs(bbox[3] - 2.5) < 1e-9


def test_parse_operatorname_sphere_tuple_core_double_and_single_wrap() -> None:
    s1 = try_parse_operatorname_sphere_tuple_core("((0,0,37.5),1.7)")
    assert s1 is not None
    assert abs(s1.center_z - 37.5) < 1e-9 and abs(s1.radius_sq - 1.7 * 1.7) < 1e-9
    s2 = try_parse_operatorname_sphere_tuple_core("(3.35,-4,8.5),0.8")
    assert s2 is not None
    assert abs(s2.center_x - 3.35) < 1e-9 and abs(s2.radius_sq - 0.64) < 1e-9


def test_operatorname_sphere_primitive_classifies_supported() -> None:
    s = normalize_latex(r"\operatorname{sphere}((0,0,2),0.5)")
    r = classify_expression(s, "expression", {})
    assert r.family == ExpressionFamily.SPHERE_SOLID
    assert r.status == ClassificationStatus.SUPPORTED


def test_operatorname_sphere_symbolic_center_stays_unrecognized() -> None:
    r = classify_expression("operatorname{sphere}((s,d,p),o)", "expression", {})
    assert r.family == ExpressionFamily.UNKNOWN
    assert r.status == ClassificationStatus.UNRECOGNIZED


def test_try_parse_sphere_solid_inequality() -> None:
    s = try_parse_sphere_solid_inequality("(x+7.6)^2+(y-1.3)^2+(z-5)^2<0.5^2")
    assert s is not None
    assert abs(s.center_x - (-7.6)) < 1e-9
    assert abs(s.center_y - 1.3) < 1e-9
    assert abs(s.center_z - 5.0) < 1e-9
    assert abs(s.radius_sq - 0.25) < 1e-9


def test_try_parse_sphere_unit_ball_xyz() -> None:
    s = try_parse_sphere_solid_inequality("x^2+y^2+z^2<=1")
    assert s is not None
    assert s.center_x == s.center_y == s.center_z == 0.0
    assert abs(s.radius_sq - 1.0) < 1e-9


def test_classify_sphere_solid_with_half_space_z() -> None:
    r = classify_expression("(x+7.6)^2+(y-1.3)^2+(z-5)^2<0.5^2{z<5}", "expression")
    assert r.family == ExpressionFamily.SPHERE_SOLID
    assert r.status == ClassificationStatus.SUPPORTED


def test_sphere_solid_world_bbox_clips_to_ball() -> None:
    s = try_parse_sphere_solid_inequality("(x)^2+(y)^2+(z)^2<4")  # centers 0
    assert s is not None
    bbox = try_sphere_solid_world_bbox(["z<10"], s, {})
    assert bbox is not None
    zmin, zmax = bbox[4], bbox[5]
    assert zmax <= 2.0 + 0.01
    assert zmin >= -2.0 - 0.01


def test_mesh_sphere_solid_unit_ball() -> None:
    ref = SourceRef("t.json", "1", None, None, 0)
    node = SphereSolidNode(
        node_type="sphere_solid",
        source_ref=ref,
        family=ExpressionFamily.SPHERE_SOLID,
        status=ClassificationStatus.SUPPORTED,
        original_latex="",
        normalized_latex="",
        color=None,
        hidden=False,
        metadata={},
        center_x=0.0,
        center_y=0.0,
        center_z=0.0,
        radius_sq=1.0,
        x_min=-1.0,
        x_max=1.0,
        y_min=-1.0,
        y_max=1.0,
        z_min=-1.0,
        z_max=1.0,
        voxel_resolution=24,
    )
    mesh = mesh_sphere_solid(node)
    assert mesh.faces
    assert mesh.vertices


def test_partial_axis_slabs_not_promoted_with_viewport() -> None:
    """Viewport must not complete missing axis: would mesh a scene-sized slab (JSONLondon regression)."""
    vp = {"xmin": -200.0, "xmax": 200.0, "ymin": -50.0, "ymax": 50.0, "zmin": -139.0, "zmax": 139.0}
    r = classify_expression("20>y>-20{-225<x<225}", "expression", vp)
    assert r.family == ExpressionFamily.INEQUALITY_REGION
    assert r.status == ClassificationStatus.GEOMETRY_INELIGIBLE


def test_abs_x_band_with_y_z_braces_not_y_slab() -> None:
    """S2-01 Group C expr 28: chained-x core with y/z bands must not match Y_SLAB (wrong mesher)."""
    raw = (
        r"\operatorname{abs}(x)<1.5\left\{\operatorname{abs}(x)>1.4\right\}"
        r"\left\{y>-4\right\}\left\{y<-3\right\}\left\{0.5<z<2.5\right\}"
    )
    nl = normalize_latex(raw)
    vp = {"xmin": -5.0, "xmax": 5.0, "ymin": -5.0, "ymax": 5.0, "zmin": -1.0, "zmax": 5.0}
    r = classify_expression(nl, "expression", vp)
    assert r.family != ExpressionFamily.Y_SLAB_REGION


def test_triangle_polygon_calls_numeric_supported() -> None:
    r = classify_expression("triangle((0,0,0),(1,0,0),(0,1,0))", "expression")
    assert r.family == ExpressionFamily.TRIANGLE_CALL
    assert r.status == ClassificationStatus.SUPPORTED
    r2 = classify_expression("polygon((0,0,0),(1,0,0),(0,1,0))", "expression")
    assert r2.family == ExpressionFamily.POLYGON_CALL
    assert r2.status == ClassificationStatus.SUPPORTED


def test_operatorname_triangle_polygon_and_segment_numeric_supported() -> None:
    tri = normalize_latex(r"\operatorname{triangle}((0,0,0),(1,0,0),(0,1,0))")
    assert tri.startswith("operatorname{triangle}")
    r = classify_expression(tri, "expression")
    assert r.family == ExpressionFamily.TRIANGLE_CALL
    assert r.status == ClassificationStatus.SUPPORTED
    poly = normalize_latex(r"\operatorname{polygon}((0,0,0),(1,0,0),(0,1,0),(0,0,1))")
    r2 = classify_expression(poly, "expression")
    assert r2.family == ExpressionFamily.POLYGON_CALL
    assert r.status == ClassificationStatus.SUPPORTED
    seg = normalize_latex(r"\operatorname{segment}((0,0,0),(1,1,1))")
    r3 = classify_expression(seg, "expression")
    assert r3.family == ExpressionFamily.SEGMENT_CALL
    assert r3.status == ClassificationStatus.SUPPORTED


def test_operatorname_triangle_with_domain_braces_supported() -> None:
    s = normalize_latex(r"\operatorname{triangle}((0,0,0),(1,0,0),(0,1,0)){-1<x<1}")
    r = classify_expression(s, "expression")
    assert r.family == ExpressionFamily.TRIANGLE_CALL
    assert r.status == ClassificationStatus.SUPPORTED


def test_operator_polygon_wrong_arity_geometry_ineligible() -> None:
    r = classify_expression("triangle((0,0,0),(1,0,0))", "expression")
    assert r.family == ExpressionFamily.TRIANGLE_CALL
    assert r.status == ClassificationStatus.GEOMETRY_INELIGIBLE
    r2 = classify_expression("segment((0,0,0),(1,1,1),(2,2,2))", "expression")
    assert r2.family == ExpressionFamily.SEGMENT_CALL
    assert r2.status == ClassificationStatus.GEOMETRY_INELIGIBLE


def test_operator_polygon_non_numeric_geometry_ineligible() -> None:
    r = classify_expression("triangle((a,0,0),(1,0,0),(0,1,0))", "expression")
    assert r.family == ExpressionFamily.TRIANGLE_CALL
    assert r.status == ClassificationStatus.GEOMETRY_INELIGIBLE


def test_normalize_double_wrapped_point_triple() -> None:
    assert normalize_latex("G=((20,20,0))") == "G=(20,20,0)"


def test_parse_point_definition_accepts_double_wrapped_triple() -> None:
    p = parse_point_definition("G=((20,20,0))")
    assert p is not None
    assert p.name == "G"
    assert p.x == "20" and p.y == "20" and p.z == "0"


def test_normalize_two_point_bracket_to_segment() -> None:
    nl = normalize_latex(r"\left[A,B\right]")
    assert nl == "operatorname{segment}((A),(B))"


def test_triangle_operatorname_labeled_vertices_supported() -> None:
    r = classify_expression("operatorname{triangle}((A),(B),(C))", "expression")
    assert r.family == ExpressionFamily.TRIANGLE_CALL
    assert r.status == ClassificationStatus.SUPPORTED


def test_parse_interval_comma_split_z_ranges() -> None:
    from desmos3d_pipeline.parse.relation import restriction_axis_interval_ok

    s = "-3.25<=z<=0.5,-10.5<=z<=-6.75"
    assert restriction_axis_interval_ok(s)


def test_empty_expression_body_geometry_ineligible() -> None:
    r = classify_expression("", "expression")
    assert r.family == ExpressionFamily.UNKNOWN
    assert r.status == ClassificationStatus.GEOMETRY_INELIGIBLE
    assert "Empty" in r.reason


def test_general_inequality_region_geometry_ineligible() -> None:
    r = classify_expression("x+y+z<5", "expression")
    assert r.family == ExpressionFamily.INEQUALITY_REGION
    assert r.status == ClassificationStatus.GEOMETRY_INELIGIBLE


def test_classify_param_assignment_geometry_ineligible() -> None:
    r = classify_expression("a=[1,2,3]", "expression")
    assert r.family == ExpressionFamily.PARAM_ASSIGNMENT
    assert r.status == ClassificationStatus.GEOMETRY_INELIGIBLE


def test_try_parse_parametric_line_point_t_vector() -> None:
    core = "(25,8,40)+t*((15,8,40)-(25,8,40))"
    xyz = try_parse_parametric_line_point_t_vector(core)
    assert xyz is not None
    x, y, z = xyz
    assert x == "(25)+t*((15)-(25))"
    assert y == "(8)+t*((8)-(8))"
    assert z == "(40)+t*((40)-(40))"


def test_split_xyz_parametric_tuple_rejects_line_form() -> None:
    assert split_xyz_parametric_tuple("(25,8,40)+t*((15,8,40)-(25,8,40))") is None


def test_classify_parametric_line_supported() -> None:
    r = classify_expression("(25,8,40)+t*((15,8,40)-(25,8,40)){0<t<1}", "expression")
    assert r.family == ExpressionFamily.PARAMETRIC_T_CURVE
    assert r.status == ClassificationStatus.SUPPORTED


def test_to_python_expr_implicit_mul_param_before_paren() -> None:
    py = to_python_expr("4*t(1-t)", {})
    assert "t*(1-t)" in py
    assert abs(safe_eval(py, {"t": 0.5}) - 1.0) < 1e-9


def test_extract_brace_strip_paren_only_domain_wrapper() -> None:
    raw = r"z=20\ \left\{-23.92<x<-9.92\right\}\ \left(\left\{\ -40<y<60\right\}\right)"
    n = normalize_latex(raw)
    core, restrictions = extract_brace_restrictions(n)
    assert core == "z=20"
    assert restrictions == ["-23.92<x<-9.92", "-40<y<60"]


def test_to_python_expr_min_with_abs_bars() -> None:
    py = to_python_expr("93+min(7-|x|,7-|y|)*(22)/(7)", {})
    assert "abs(x)" in py and "abs(y)" in py
    assert abs(safe_eval(py, {"x": 2.0, "y": 3.0}) - (93 + min(5, 4) * (22) / (7))) < 1e-9


def test_to_python_expr_trig_coef_inside_abs() -> None:
    py = to_python_expr("0.3|sin7x|+3", {})
    assert "abs(sin(7*x))" in py.replace(" ", "")


def test_try_parse_parametric_uv_point_u_v_vectors() -> None:
    core = "(35.75,-16.45,13)+u((-35.75,-16.45,13)-(35.75,-16.45,13))+v((35.75,0,14.52)-(35.75,-16.45,13))"
    t = try_parse_parametric_uv_point_u_v_vectors(core)
    assert t is not None
    xe, ye, ze = t
    env = {"u": 1.0, "v": 0.0}
    assert abs(safe_eval(to_python_expr(xe, {}), env) - (-35.75)) < 1e-6
    assert abs(safe_eval(to_python_expr(ye, {}), env) - (-16.45)) < 1e-6
    assert abs(safe_eval(to_python_expr(ze, {}), env) - 13.0) < 1e-6


def test_split_xyz_parametric_tuple_rejects_u_v_tail() -> None:
    assert split_xyz_parametric_tuple("(1,2,3)+u((0,0,0)-(1,1,1))+v((1,0,0)-(0,0,0))") is None


def test_normalize_latex_desmos_list_ellipsis_to_first() -> None:
    n = normalize_latex("(y-[0.2,0.4...1.8])^{2}\\le0.002")
    assert "(y-(0.2))" in n


def test_safe_eval_clamp_sqrt() -> None:
    assert abs(safe_eval("sqrt(-4)", {}, clamp_sqrt=True) - 0.0) < 1e-9
    assert abs(safe_eval("sqrt(4)", {}, clamp_sqrt=True) - 2.0) < 1e-9


def test_classify_z_slab_with_trailing_x_chain() -> None:
    n = normalize_latex("(y-(0.2))^2<=0.00250<=z<=13.2-0.03<=x<=2.03")
    r = classify_expression(n, "expression")
    assert r.status == ClassificationStatus.SUPPORTED
    assert r.family == ExpressionFamily.Z_SLAB_REGION
