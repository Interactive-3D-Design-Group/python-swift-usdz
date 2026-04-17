from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from desmos3d_pipeline.ir.models import ClassificationResult, ClassificationStatus, ExpressionFamily
from desmos3d_pipeline.normalize.latex import extract_brace_restrictions
from desmos3d_pipeline.parse.disk_extrusion import try_disk_extrusion_world_bbox, try_parse_axis_aligned_disk_inequality
from desmos3d_pipeline.parse.relation import detect_relations, parse_interval_constraint
from desmos3d_pipeline.parse.symbols import parse_assignment, parse_point_definition


@dataclass(slots=True)
class ClassificationContext:
    core_expr: str
    restrictions: list[str]


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
            status=ClassificationStatus.RECOGNIZED_UNSUPPORTED,
            reason="Non-geometry Desmos item",
            confidence=1.0,
            fingerprint=_fingerprint(expression_type, []),
        )

    core, restrictions = extract_brace_restrictions(normalized_latex)
    context = ClassificationContext(core_expr=core, restrictions=restrictions)

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

    if context.core_expr.startswith("triangle("):
        return ClassificationResult(ExpressionFamily.TRIANGLE_CALL, ClassificationStatus.RECOGNIZED_UNSUPPORTED, "Triangle function recognized", 0.95, _fingerprint(core, restrictions))

    if context.core_expr.startswith("polygon("):
        return ClassificationResult(ExpressionFamily.POLYGON_CALL, ClassificationStatus.RECOGNIZED_UNSUPPORTED, "Polygon function recognized", 0.95, _fingerprint(core, restrictions))

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
        if has_yz_intervals and x_lower and x_upper:
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

        if has_x_interval and y_lower and y_upper and _safe_y_expr(y_lower) and _safe_y_expr(y_upper):
            # z interval may be absent; mesher will fall back to viewport zmin/zmax.
            return ClassificationResult(
                ExpressionFamily.Y_SLAB_REGION,
                ClassificationStatus.SUPPORTED,
                "y bounded between two functions/constants",
                0.88 if has_z_interval else 0.78,
                _fingerprint(context.core_expr, context.restrictions),
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
        has_intervals = all(parse_interval_constraint(r) is not None for r in restrictions)
        is_constant_numeric = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", rhs) is not None
        # Also treat simple arithmetic numeric expressions (e.g. -90-21, 32+7, 90+42) as constants.
        is_constant_arith = re.fullmatch(r"[-+0-9.*/()]+", rhs) is not None and not re.search(r"[A-Za-z]", rhs)
        if axis in {"x", "y", "z"} and (is_constant_numeric or is_constant_arith):
            status = ClassificationStatus.SUPPORTED if has_intervals else ClassificationStatus.RECOGNIZED_UNSUPPORTED
            return ClassificationResult(ExpressionFamily.CONSTANT_PLANE, status, "Constant plane", 0.95, _fingerprint(core, restrictions))
        if axis == "z" and ("x" in rhs or "y" in rhs):
            has_quadratic = "^2" in rhs
            family = ExpressionFamily.QUADRATIC_SURFACE_PATCH if has_quadratic else ExpressionFamily.LINEAR_SURFACE_PATCH
            status = ClassificationStatus.SUPPORTED if has_intervals else ClassificationStatus.RECOGNIZED_UNSUPPORTED
            return ClassificationResult(family, status, "Explicit z=f(x,y) surface", 0.92, _fingerprint(core, restrictions))
        return ClassificationResult(ExpressionFamily.INEQUALITY_REGION, ClassificationStatus.RECOGNIZED_UNSUPPORTED, "Equation recognized but not yet meshable", 0.75, _fingerprint(core, restrictions))

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
            return ClassificationResult(ExpressionFamily.INEQUALITY_REGION, ClassificationStatus.RECOGNIZED_UNSUPPORTED, "Partial axis bounds only", 0.8, _fingerprint(core, restrictions))

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

    # General inequalities are recognized but unsupported in first meshing phase.
    if rel.operators and any(op in rel.operators for op in ["<", ">", "<=", ">="]):
        return ClassificationResult(ExpressionFamily.INEQUALITY_REGION, ClassificationStatus.RECOGNIZED_UNSUPPORTED, "Inequality region recognized", 0.7, _fingerprint(core, restrictions))

    return ClassificationResult(ExpressionFamily.UNKNOWN, ClassificationStatus.UNRECOGNIZED, "Unrecognized expression form", 0.2, _fingerprint(core, restrictions))
