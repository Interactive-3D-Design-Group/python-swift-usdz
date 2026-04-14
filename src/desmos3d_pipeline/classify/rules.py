from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from desmos3d_pipeline.ir.models import ClassificationResult, ClassificationStatus, ExpressionFamily
from desmos3d_pipeline.normalize.latex import extract_brace_restrictions
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


def classify_expression(normalized_latex: str, expression_type: str) -> ClassificationResult:
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
        return ClassificationResult(ExpressionFamily.PARAM_ASSIGNMENT, ClassificationStatus.RECOGNIZED_UNSUPPORTED, "Parameter assignment captured", 0.95, _fingerprint(core, restrictions))

    if context.core_expr.startswith("triangle("):
        return ClassificationResult(ExpressionFamily.TRIANGLE_CALL, ClassificationStatus.RECOGNIZED_UNSUPPORTED, "Triangle function recognized", 0.95, _fingerprint(core, restrictions))

    if context.core_expr.startswith("polygon("):
        return ClassificationResult(ExpressionFamily.POLYGON_CALL, ClassificationStatus.RECOGNIZED_UNSUPPORTED, "Polygon function recognized", 0.95, _fingerprint(core, restrictions))

    rel = detect_relations(context.core_expr)

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
        if axis in {"x", "y", "z"} and re.fullmatch(r"[-+]?\d+(?:\.\d+)?", rhs):
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
            return ClassificationResult(ExpressionFamily.INEQUALITY_REGION, ClassificationStatus.RECOGNIZED_UNSUPPORTED, "Partial axis bounds only", 0.8, _fingerprint(core, restrictions))

    # General inequalities are recognized but unsupported in first meshing phase.
    if rel.operators and any(op in rel.operators for op in ["<", ">", "<=", ">="]):
        return ClassificationResult(ExpressionFamily.INEQUALITY_REGION, ClassificationStatus.RECOGNIZED_UNSUPPORTED, "Inequality region recognized", 0.7, _fingerprint(core, restrictions))

    return ClassificationResult(ExpressionFamily.UNKNOWN, ClassificationStatus.UNRECOGNIZED, "Unrecognized expression form", 0.2, _fingerprint(core, restrictions))
