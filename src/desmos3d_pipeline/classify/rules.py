from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from desmos3d_pipeline.ir.models import ClassificationResult, ClassificationStatus, ExpressionFamily
from desmos3d_pipeline.normalize.latex import extract_brace_restrictions
from desmos3d_pipeline.parse.disk_extrusion import (
    try_disk_extrusion_world_bbox,
    try_parse_axis_aligned_disk_inequality,
    try_parse_operatorname_sphere_tuple_core,
    try_parse_sphere_solid_inequality,
    try_parse_vertical_cylinder_equality,
    try_sphere_solid_world_bbox,
)
from desmos3d_pipeline.parse.implicit_plane import try_linear_implicit_plane_z_rhs
from desmos3d_pipeline.parse.operator_geometry import (
    operator_call_core_and_restrictions,
    parse_parenthesized_xyz_tuple_list,
    vertex_specs_numeric_or_pointrefs,
)
from desmos3d_pipeline.parse.parametric import (
    infer_parametric_kind,
    parse_parametric_line_point_and_q,
    split_xyz_parametric_tuple,
    try_parse_parametric_line_point_t_vector,
    try_parse_parametric_uv_point_u_v_vectors,
)
from desmos3d_pipeline.parse.relation import detect_relations, parse_interval_constraint, surface_domain_meshable
from desmos3d_pipeline.parse.symbols import parse_assignment, parse_point_definition


@dataclass(slots=True)
class ClassificationContext:
    core_expr: str
    restrictions: list[str]


def _normalized_desmos_operator_call(core: str, op_name: str) -> bool:
    """True for ``op(...)`` or ``operatorname{op}(...)`` after LaTeX normalization (backslashes stripped)."""
    c = core.strip()
    if c.startswith(f"{op_name}("):
        return True
    return c.startswith(f"operatorname{{{op_name}}}(")


def _strip_leading_operatorname_for_domain_extract(normalized_latex: str, op_name: str) -> str:
    """Remove ``operatorname{op}`` so ``extract_brace_restrictions`` only sees real domain braces."""
    s = normalized_latex.strip()
    prefix = f"operatorname{{{op_name}}}"
    if s.startswith(prefix):
        return s[len(prefix) :]
    return s


def _classify_operator_polygon_meshable(
    nl: str, op_name: str, family: ExpressionFamily
) -> ClassificationResult | None:
    """``operatorname{triangle|polygon|segment}(...)`` with parseable numeric ``(x,y,z)`` tuples."""
    if not _normalized_desmos_operator_call(nl, op_name):
        return None
    packed = operator_call_core_and_restrictions(nl, op_name)
    if packed is None:
        return None
    core, restrictions = packed
    pts = parse_parenthesized_xyz_tuple_list(core)
    if pts is None:
        return ClassificationResult(
            family,
            ClassificationStatus.GEOMETRY_INELIGIBLE,
            f"{op_name}() vertex list not parseable",
            0.9,
            _fingerprint(nl[:512], restrictions),
        )
    n = len(pts)
    if op_name == "triangle" and n != 3:
        return ClassificationResult(
            family,
            ClassificationStatus.GEOMETRY_INELIGIBLE,
            "triangle() requires exactly three vertices",
            0.9,
            _fingerprint(nl[:512], restrictions),
        )
    if op_name == "segment" and n != 2:
        return ClassificationResult(
            family,
            ClassificationStatus.GEOMETRY_INELIGIBLE,
            "segment() requires exactly two endpoints",
            0.9,
            _fingerprint(nl[:512], restrictions),
        )
    if op_name == "polygon" and n < 3:
        return ClassificationResult(
            family,
            ClassificationStatus.GEOMETRY_INELIGIBLE,
            "polygon() requires at least three vertices",
            0.9,
            _fingerprint(nl[:512], restrictions),
        )
    if not vertex_specs_numeric_or_pointrefs(pts):
        return ClassificationResult(
            family,
            ClassificationStatus.GEOMETRY_INELIGIBLE,
            f"{op_name}() vertex coordinates must be numeric literals or point labels",
            0.88,
            _fingerprint(nl[:512], restrictions),
        )
    return ClassificationResult(
        family,
        ClassificationStatus.SUPPORTED,
        f"Desmos {op_name}() face/segment with numeric vertices",
        0.87,
        _fingerprint(nl[:512], restrictions),
    )


def _fingerprint(core: str, restrictions: list[str]) -> str:
    relation_sig = ",".join(re.findall(r"<=|>=|=|<|>", core))
    vars_sig = "".join(sorted(set(re.findall(r"[xyz]", core))))
    fn_sig = "|".join(sorted(set(re.findall(r"[A-Za-z]+(?=\()", core))))
    payload = f"core:{core}|rel:{relation_sig}|vars:{vars_sig}|fns:{fn_sig}|r:{'|'.join(restrictions)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def classify_expression(
    normalized_latex: str,
    expression_type: str,
    viewport: dict[str, float] | None = None,
) -> ClassificationResult:
    if expression_type in {"folder", "text"}:
        return ClassificationResult(
            family=ExpressionFamily.TEXT_OR_FOLDER,
            status=ClassificationStatus.GEOMETRY_INELIGIBLE,
            reason="Non-geometry Desmos item",
            confidence=1.0,
            fingerprint=_fingerprint(expression_type, []),
        )

    nl = normalized_latex.strip()
    if not nl:
        return ClassificationResult(
            ExpressionFamily.UNKNOWN,
            ClassificationStatus.GEOMETRY_INELIGIBLE,
            "Empty expression body",
            0.99,
            _fingerprint("", []),
        )

    # Desmos ``\\operatorname{triangle|polygon|segment}`` — strip operator prefix before brace extraction.
    tri = _classify_operator_polygon_meshable(nl, "triangle", ExpressionFamily.TRIANGLE_CALL)
    if tri is not None:
        return tri
    poly = _classify_operator_polygon_meshable(nl, "polygon", ExpressionFamily.POLYGON_CALL)
    if poly is not None:
        return poly
    seg = _classify_operator_polygon_meshable(nl, "segment", ExpressionFamily.SEGMENT_CALL)
    if seg is not None:
        return seg

    if _normalized_desmos_operator_call(nl, "sphere"):
        suf = _strip_leading_operatorname_for_domain_extract(normalized_latex, "sphere")
        core_sp, restrictions = extract_brace_restrictions(suf)
        sp = try_parse_operatorname_sphere_tuple_core(core_sp)
        if sp is not None:
            bbox = try_sphere_solid_world_bbox(restrictions, sp, viewport or {})
            if bbox is not None:
                return ClassificationResult(
                    ExpressionFamily.SPHERE_SOLID,
                    ClassificationStatus.SUPPORTED,
                    "Desmos sphere((cx,cy,cz),r) primitive (filled ball)",
                    0.86,
                    _fingerprint(nl[:512], restrictions),
                )

    core, restrictions = extract_brace_restrictions(normalized_latex)
    context = ClassificationContext(core_expr=core, restrictions=restrictions)

    if not core.strip():
        return ClassificationResult(
            ExpressionFamily.UNKNOWN,
            ClassificationStatus.GEOMETRY_INELIGIBLE,
            "Empty expression body",
            0.99,
            _fingerprint(core, restrictions),
        )

    point = parse_point_definition(context.core_expr)
    if point:
        return ClassificationResult(ExpressionFamily.POINT_DEFINITION, ClassificationStatus.SUPPORTED, "3D point definition", 0.98, _fingerprint(core, restrictions))

    assign = parse_assignment(context.core_expr)
    if assign:
        return ClassificationResult(
            ExpressionFamily.PARAM_ASSIGNMENT,
            ClassificationStatus.GEOMETRY_INELIGIBLE,
            "Parameter or color assignment (not meshed)",
            0.95,
            _fingerprint(core, restrictions),
        )

    if try_parse_parametric_line_point_t_vector(context.core_expr) is not None:
        return ClassificationResult(
            ExpressionFamily.PARAMETRIC_T_CURVE,
            ClassificationStatus.SUPPORTED,
            "Parametric line (point + t times vector)",
            0.9,
            _fingerprint(core, restrictions),
        )
    if parse_parametric_line_point_and_q(context.core_expr) is not None:
        return ClassificationResult(
            ExpressionFamily.PARAMETRIC_T_CURVE,
            ClassificationStatus.SUPPORTED,
            "Parametric line with labeled point endpoints",
            0.88,
            _fingerprint(core, restrictions),
        )

    uv_bilinear = try_parse_parametric_uv_point_u_v_vectors(context.core_expr)
    if uv_bilinear is not None:
        xe, ye, ze = uv_bilinear
        kind = infer_parametric_kind(xe, ye, ze)
        if kind == "uv":
            return ClassificationResult(
                ExpressionFamily.PARAMETRIC_UV_SURFACE,
                ClassificationStatus.SUPPORTED,
                "Parametric u,v surface (point + u,v vectors)",
                0.89,
                _fingerprint(core, restrictions),
            )

    triple = split_xyz_parametric_tuple(context.core_expr)
    if triple is not None:
        xe, ye, ze = triple
        kind = infer_parametric_kind(xe, ye, ze)
        if kind == "uv":
            return ClassificationResult(
                ExpressionFamily.PARAMETRIC_UV_SURFACE,
                ClassificationStatus.SUPPORTED,
                "Parametric u,v surface",
                0.88,
                _fingerprint(core, restrictions),
            )
        if kind == "t":
            return ClassificationResult(
                ExpressionFamily.PARAMETRIC_T_CURVE,
                ClassificationStatus.SUPPORTED,
                "Parametric t curve",
                0.86,
                _fingerprint(core, restrictions),
            )

    rel = detect_relations(context.core_expr)

    def _is_constant_bound_expr(s: str | None) -> bool:
        if s is None:
            return False
        return re.search(r"[A-Za-z]", s) is None

    def _has_const_interval(intervals, axis: str) -> bool:
        for iv in intervals:
            if iv is None or iv.axis != axis:
                continue
            if _is_constant_bound_expr(iv.lower) and _is_constant_bound_expr(iv.upper):
                return True
        return False

    # X slab: x bounded between two functions/constants with finite y/z bounds.
    # Example (after normalization): -x<=(z)/(1.8)-479 { -x>=(z)/(1.8)-482 } { z-bounds } { y-bounds }
    if rel.operators and any(op in rel.operators for op in ["<", ">", "<=", ">="]):
        parts = [context.core_expr] + list(context.restrictions)
        intervals = [parse_interval_constraint(p) for p in parts]
        by_axis = {}
        for iv in intervals:
            if iv is None:
                continue
            by_axis.setdefault(iv.axis, 0)
            by_axis[iv.axis] += 1

        # Must have finite y and z bounds that do NOT depend on variables.
        has_yz_intervals = _has_const_interval(intervals, "y") and _has_const_interval(intervals, "z")

        def _parse_x_ineq(s: str):
            m = re.fullmatch(r"(-?)x(<=|>=|<|>)(.+)", s)
            if not m:
                return None
            sign, op, rhs = m.groups()
            rhs = rhs.strip()
            # normalize to x >= lower or x <= upper
            if sign == "-":
                # -x <= rhs  -> x >= -rhs ;  -x >= rhs -> x <= -rhs
                if op in {"<", "<="}:
                    return ("lower", f"-({rhs})")
                return ("upper", f"-({rhs})")
            if op in {"<", "<="}:
                return ("upper", rhs)
            return ("lower", rhs)

        x_lower = None
        x_upper = None
        for p in parts:
            parsed = _parse_x_ineq(p)
            if parsed is None:
                continue
            kind, expr = parsed
            if kind == "lower":
                x_lower = expr
            else:
                x_upper = expr

        # Only accept if both bounds exist and (a) depend on z or are numeric.
        # Require the **core** to anchor on ``x`` (avoid promoting ``abs(x)<c`` rewrites whose core is an ``x`` chain
        # but whose braces describe a different slab family).
        core_iv0 = parse_interval_constraint(context.core_expr.replace(" ", ""))
        core_anchors_x = _parse_x_ineq(context.core_expr.replace(" ", "")) is not None or (
            core_iv0 is not None and core_iv0.axis == "x"
        )
        if has_yz_intervals and x_lower and x_upper and core_anchors_x:
            def _safe_expr(e: str) -> bool:
                return re.fullmatch(r"[-+0-9.*/()z]+", e.replace(" ", "")) is not None
            if _safe_expr(x_lower) and _safe_expr(x_upper):
                return ClassificationResult(ExpressionFamily.X_SLAB_REGION, ClassificationStatus.SUPPORTED, "x bounded between two functions/constants", 0.9, _fingerprint(context.core_expr, context.restrictions))

    # Y slab: y bounded between two functions/constants of z with finite x bounds and a z range
    # (explicit z interval preferred; viewport inference allowed downstream if x bounds exist).
    if rel.operators and any(op in rel.operators for op in ["<", ">", "<=", ">="]):
        parts = [context.core_expr] + list(context.restrictions)
        intervals = [parse_interval_constraint(p) for p in parts]

        def _parse_y_ineq(s: str):
            m = re.fullmatch(r"(-?)y(<=|>=|<|>)(.+)", s.replace(" ", ""))
            if not m:
                return None
            sign, op, rhs = m.groups()
            rhs = rhs.strip()
            if sign == "-":
                # -y <= rhs -> y >= -rhs ; -y >= rhs -> y <= -rhs
                if op in {"<", "<="}:
                    return ("lower", f"-({rhs})")
                return ("upper", f"-({rhs})")
            if op in {"<", "<="}:
                return ("upper", rhs)
            return ("lower", rhs)

        y_lower = None
        y_upper = None
        for p in parts:
            parsed = _parse_y_ineq(p)
            if not parsed:
                continue
            kind, expr = parsed
            if kind == "lower":
                y_lower = expr
            else:
                y_upper = expr

        def _safe_y_expr(e: str) -> bool:
            # Only allow numeric + z dependence (no x/y).
            return re.fullmatch(r"[-+0-9.*/()z]+", e.replace(" ", "")) is not None

        has_x_interval = _has_const_interval(intervals, "x")
        has_z_interval = _has_const_interval(intervals, "z")

        core_iv1 = parse_interval_constraint(context.core_expr.replace(" ", ""))
        core_anchors_y = _parse_y_ineq(context.core_expr.replace(" ", "")) is not None or (
            core_iv1 is not None and core_iv1.axis == "y"
        )
        if (
            has_x_interval
            and y_lower
            and y_upper
            and _safe_y_expr(y_lower)
            and _safe_y_expr(y_upper)
            and core_anchors_y
        ):
            # z interval may be absent; mesher will fall back to viewport zmin/zmax.
            return ClassificationResult(
                ExpressionFamily.Y_SLAB_REGION,
                ClassificationStatus.SUPPORTED,
                "y bounded between two functions/constants",
                0.88 if has_z_interval else 0.78,
                _fingerprint(context.core_expr, context.restrictions),
            )

    cyl_eq = try_parse_vertical_cylinder_equality(context.core_expr)
    if cyl_eq is not None:
        bbox = try_disk_extrusion_world_bbox(list(context.restrictions), cyl_eq, viewport or {})
        if bbox is not None:
            return ClassificationResult(
                ExpressionFamily.VERTICAL_CYLINDER_SURFACE,
                ClassificationStatus.SUPPORTED,
                "Vertical cylinder surface (quadratic equality)",
                0.84,
                _fingerprint(core, restrictions),
            )

    # Chained inequality slab: f(x,y,...) >= z >= g(x,y,...)
    mslab = re.fullmatch(r"(.+?)(<=|>=|<|>)z(<=|>=|<|>)(.+)", context.core_expr.replace(" ", ""))
    if mslab:
        left, op1, op2, right = mslab.groups()
        # Accept both directions (>= z >=) or (<= z <=) as a "between" slab.
        ok = (op1 in {">", ">="} and op2 in {">", ">="}) or (op1 in {"<", "<="} and op2 in {"<", "<="})
        has_intervals = all(parse_interval_constraint(r) is not None for r in restrictions)
        if ok and has_intervals:
            return ClassificationResult(ExpressionFamily.Z_SLAB_REGION, ClassificationStatus.SUPPORTED, "z bounded between two functions", 0.9, _fingerprint(core, restrictions))

    # Explicit plane/function surfaces: axis = expression
    m = re.fullmatch(r"([xyz])=(.+)", context.core_expr)
    if m:
        axis, rhs = m.groups()
        domain_ok = surface_domain_meshable(restrictions, viewport)
        is_constant_numeric = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", rhs) is not None
        # Also treat simple arithmetic numeric expressions (e.g. -90-21, 32+7, 90+42) as constants.
        is_constant_arith = re.fullmatch(r"[-+0-9.*/()]+", rhs) is not None and not re.search(r"[A-Za-z]", rhs)
        if axis in {"x", "y", "z"} and (is_constant_numeric or is_constant_arith):
            status = ClassificationStatus.SUPPORTED if domain_ok else ClassificationStatus.GEOMETRY_INELIGIBLE
            reason = "Constant plane" if domain_ok else "Constant plane without domain braces (unbounded)"
            return ClassificationResult(ExpressionFamily.CONSTANT_PLANE, status, reason, 0.95, _fingerprint(core, restrictions))
        if axis == "z" and ("x" in rhs or "y" in rhs):
            has_quadratic = "^2" in rhs
            family = ExpressionFamily.QUADRATIC_SURFACE_PATCH if has_quadratic else ExpressionFamily.LINEAR_SURFACE_PATCH
            status = ClassificationStatus.SUPPORTED if domain_ok else ClassificationStatus.GEOMETRY_INELIGIBLE
            reason = "Explicit z=f(x,y) surface" if domain_ok else "Explicit z=f(x,y) without domain braces (unbounded)"
            return ClassificationResult(family, status, reason, 0.92, _fingerprint(core, restrictions))
        return ClassificationResult(
            ExpressionFamily.INEQUALITY_REGION,
            ClassificationStatus.GEOMETRY_INELIGIBLE,
            "Implicit equation (x/y form not meshed here)",
            0.75,
            _fingerprint(core, restrictions),
        )

    implicit_z = try_linear_implicit_plane_z_rhs(context.core_expr)
    if implicit_z is not None:
        domain_ok = surface_domain_meshable(restrictions, viewport)
        status = ClassificationStatus.SUPPORTED if domain_ok else ClassificationStatus.GEOMETRY_INELIGIBLE
        reason = (
            "Linear implicit plane solved as z=f(x,y)"
            if domain_ok
            else "Linear implicit plane without meshable domain (add braces or viewport)"
        )
        return ClassificationResult(
            ExpressionFamily.LINEAR_SURFACE_PATCH,
            status,
            reason,
            0.9,
            _fingerprint(core, restrictions),
        )

    # Box-like chained interval core expression
    interval_core = parse_interval_constraint(context.core_expr)
    if interval_core:
        all_parts = [interval_core] + [parse_interval_constraint(r) for r in restrictions]
        if all(p is not None for p in all_parts):
            axes = {p.axis for p in all_parts if p is not None}
            if axes.issuperset({"x", "y", "z"}):
                return ClassificationResult(ExpressionFamily.BOX_BOUNDED_REGION, ClassificationStatus.SUPPORTED, "Box/prism bounded region", 0.93, _fingerprint(core, restrictions))
            # Do not infer a missing axis from graph viewport: it produces huge solids that do not
            # match Desmos 3D for slab-like inequalities (e.g. y-band + x-band without explicit z).
            return ClassificationResult(
                ExpressionFamily.INEQUALITY_REGION,
                ClassificationStatus.GEOMETRY_INELIGIBLE,
                "Partial axis bounds only (no full box)",
                0.8,
                _fingerprint(core, restrictions),
            )

    # Solid ball (three squared linear terms) with brace bounds (or inferred AABB from radius).
    if rel.operators and any(op in rel.operators for op in ["<", ">", "<=", ">="]):
        sphere = try_parse_sphere_solid_inequality(context.core_expr)
        if sphere is not None:
            bbox = try_sphere_solid_world_bbox(list(context.restrictions), sphere, viewport or {})
            if bbox is not None:
                return ClassificationResult(
                    ExpressionFamily.SPHERE_SOLID,
                    ClassificationStatus.SUPPORTED,
                    "Solid sphere (quadratic inequality, axis-aligned clip)",
                    0.84,
                    _fingerprint(core, restrictions),
                )

    # Axis-aligned disk (two squared linear terms) extruded along the third axis, with brace bounds.
    if rel.operators and any(op in rel.operators for op in ["<", ">", "<=", ">="]):
        disk = try_parse_axis_aligned_disk_inequality(context.core_expr)
        if disk is not None:
            bbox = try_disk_extrusion_world_bbox(list(context.restrictions), disk, viewport or {})
            if bbox is not None:
                return ClassificationResult(
                    ExpressionFamily.DISK_EXTRUSION_SOLID,
                    ClassificationStatus.SUPPORTED,
                    "Axis-aligned disk extrusion (quadratic inequality solid)",
                    0.82,
                    _fingerprint(core, restrictions),
                )

    # General inequalities: classified for diagnostics but not meshed in this pipeline.
    if rel.operators and any(op in rel.operators for op in ["<", ">", "<=", ">="]):
        return ClassificationResult(
            ExpressionFamily.INEQUALITY_REGION,
            ClassificationStatus.GEOMETRY_INELIGIBLE,
            "General inequality region (not meshed in this pipeline)",
            0.7,
            _fingerprint(core, restrictions),
        )

    return ClassificationResult(ExpressionFamily.UNKNOWN, ClassificationStatus.UNRECOGNIZED, "Unrecognized expression form", 0.2, _fingerprint(core, restrictions))
