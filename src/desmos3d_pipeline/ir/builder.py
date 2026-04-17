from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any

from desmos3d_pipeline.classify.rules import classify_expression
from desmos3d_pipeline.io.desmos_json import extract_expression_list, extract_viewport, load_desmos_json
from desmos3d_pipeline.ir.models import (
    BoxVolumeNode,
    ClassificationStatus,
    Diagnostic,
    DiskExtrusionSolidNode,
    ExpressionFamily,
    ExpressionRecord,
    GeometryNode,
    XSlabNode,
    YSlabNode,
    PlanePatchNode,
    PointNode,
    ZSlabNode,
    SampledSurfaceNode,
    Severity,
    SourceRef,
)
from desmos3d_pipeline.normalize.latex import extract_brace_restrictions, normalize_latex
from desmos3d_pipeline.parse.math_eval import normalize_symbol_name, safe_eval, to_python_expr
from desmos3d_pipeline.parse.disk_extrusion import try_disk_extrusion_world_bbox, try_parse_axis_aligned_disk_inequality
from desmos3d_pipeline.parse.relation import parse_interval_constraint
from desmos3d_pipeline.parse.symbols import parse_assignment, parse_point_definition


@dataclass(slots=True)
class GeometryBuildResult:
    source_file: str
    nodes: list[GeometryNode]
    diagnostics: list[Diagnostic]
    symbol_table: dict[str, str]


def build_geometry_for_file(path: Path) -> GeometryBuildResult:
    desmos_file, diagnostics = load_desmos_json(path)
    if desmos_file is None:
        return GeometryBuildResult(source_file=path.name, nodes=[], diagnostics=diagnostics, symbol_table={})

    items = extract_expression_list(desmos_file.data)
    viewport = extract_viewport(desmos_file.data)
    folders: dict[str, str] = {}
    symbol_table: dict[str, str] = {}
    records: list[tuple[ExpressionRecord, Any]] = []

    for index, item in enumerate(items):
        item_type = str(item.get("type", "expression"))
        if item_type == "folder":
            folder_id = str(item.get("id")) if item.get("id") is not None else None
            if folder_id:
                folders[folder_id] = str(item.get("title", ""))

        expr_id = str(item.get("id")) if item.get("id") is not None else None
        folder_id = str(item.get("folderId")) if item.get("folderId") is not None else None
        source_ref = SourceRef(path.name, expr_id, folder_id, folders.get(folder_id), index)
        raw_latex = str(item.get("latex", ""))
        normalized = normalize_latex(raw_latex)
        record = ExpressionRecord(
            source_ref=source_ref,
            expression_type=item_type,
            raw_latex=raw_latex,
            normalized_latex=normalized,
            color=item.get("color"),
            hidden=bool(item.get("hidden", False)),
            extend_to_3d=bool(item.get("extendTo3D", False)),
            lines=bool(item.get("lines", False)),
        )
        classification = classify_expression(normalized, item_type, viewport)
        records.append((record, classification))

        core, _ = extract_brace_restrictions(normalized)
        assign = parse_assignment(core)
        if assign:
            symbol_table[assign.name] = assign.expr
            symbol_table[normalize_symbol_name(assign.name)] = assign.expr

    python_symbol_map = {name: normalize_symbol_name(name) for name in symbol_table}
    resolved_symbols = _resolve_symbol_table(symbol_table)
    nodes: list[GeometryNode] = []

    for record, classification in records:
        if classification.status != ClassificationStatus.SUPPORTED:
            continue
        try:
            node = _build_node(record, classification.family, resolved_symbols, python_symbol_map, viewport)
            if node is not None:
                nodes.append(node)
        except Exception as exc:
            diagnostics.append(
                Diagnostic(
                    severity=Severity.WARNING,
                    code="GEOMETRY_BUILD_FAILED",
                    message=str(exc),
                    source_ref=record.source_ref,
                    details={"latex": record.raw_latex, "family": classification.family.value},
                )
            )

    diagnostics.append(Diagnostic(severity=Severity.INFO, code="GEOMETRY_BUILD_COMPLETE", message="Geometry IR build complete", details={"file": path.name, "node_count": len(nodes)}))
    return GeometryBuildResult(source_file=path.name, nodes=nodes, diagnostics=diagnostics, symbol_table=symbol_table)


def _first_scalar_from_desmos_bracket_list(expr: str) -> float | None:
    """Desmos exports multi-stop sliders as ``name=[v0,v1,...]`` after normalization.

    We pick the first numeric entry so meshing is deterministic when the JSON
    does not include a separate current slider value.
    """
    expr = expr.strip()
    if not (expr.startswith("[") and expr.endswith("]")):
        return None
    inner = expr[1:-1].strip()
    if not inner:
        return None
    values: list[float] = []
    for part in inner.split(","):
        token = part.strip()
        if not token:
            return None
        try:
            values.append(float(token))
        except ValueError:
            return None
    return values[0] if values else None


def _resolve_symbol_table(symbol_table: dict[str, str]) -> dict[str, float]:
    resolved: dict[str, float] = {}
    python_names = {name: normalize_symbol_name(name) for name in symbol_table}
    pending = dict(symbol_table)
    for _ in range(len(pending) + 2):
        progress = False
        for name, expr in list(pending.items()):
            list_scalar = _first_scalar_from_desmos_bracket_list(expr)
            if list_scalar is not None:
                resolved[normalize_symbol_name(name)] = list_scalar
                pending.pop(name)
                progress = True
                continue
            py_expr = to_python_expr(expr, python_names)
            try:
                value = safe_eval(py_expr, resolved)
            except Exception:
                continue
            resolved[normalize_symbol_name(name)] = value
            pending.pop(name)
            progress = True
        if not pending or not progress:
            break
    return resolved


def _build_node(
    record: ExpressionRecord,
    family: ExpressionFamily,
    resolved_symbols: dict[str, float],
    python_symbol_map: dict[str, str],
    viewport: dict[str, float],
) -> GeometryNode | None:
    core, restrictions = extract_brace_restrictions(record.normalized_latex)
    metadata = {
        "python_symbol_map": python_symbol_map,
        "resolved_symbols": resolved_symbols,
        "viewport": viewport,
        "raw_restrictions": restrictions,
        "core_expr": core,
    }

    if family == ExpressionFamily.POINT_DEFINITION:
        point = parse_point_definition(core)
        if point is None:
            return None
        return PointNode(
            node_type="point",
            source_ref=record.source_ref,
            family=family,
            status=ClassificationStatus.SUPPORTED,
            original_latex=record.raw_latex,
            normalized_latex=record.normalized_latex,
            color=record.color,
            hidden=record.hidden,
            metadata=metadata,
            name=point.name,
            x=point.x,
            y=point.y,
            z=point.z,
        )

    bounds = [parse_interval_constraint(r) for r in restrictions]
    bounds = [b for b in bounds if b is not None]

    if family == ExpressionFamily.CONSTANT_PLANE:
        axis, value = core.split("=", 1)
        plane_metadata = dict(metadata)
        plane_metadata["fixed_axes"] = {axis: value}
        return PlanePatchNode(
            node_type="plane_patch",
            source_ref=record.source_ref,
            family=family,
            status=ClassificationStatus.SUPPORTED,
            original_latex=record.raw_latex,
            normalized_latex=record.normalized_latex,
            color=record.color,
            hidden=record.hidden,
            metadata=plane_metadata,
            axis=axis,
            value=value,
            bounds=bounds,
        )

    if family == ExpressionFamily.BOX_BOUNDED_REGION:
        core_range = parse_interval_constraint(core)
        ranges = ([core_range] if core_range else []) + bounds
        # Some Desmos exports encode "z between two functions with bounded x/y" as chained
        # interval parts, which would otherwise fall into box voxel fallback and lose curve detail.
        x_has_finite = any(r.axis == "x" and r.lower is not None and r.upper is not None for r in ranges)
        y_has_finite = any(r.axis == "y" and r.lower is not None and r.upper is not None for r in ranges)
        z_lowers = [r.lower for r in ranges if r.axis == "z" and r.lower is not None]
        z_uppers = [r.upper for r in ranges if r.axis == "z" and r.upper is not None]
        if x_has_finite and y_has_finite and z_lowers and z_uppers:
            lower_expr = z_lowers[-1]
            upper_expr = z_uppers[-1]
            z_expr = f"{lower_expr} {upper_expr}"
            if re.search(r"[xy]", z_expr):
                return ZSlabNode(
                    node_type="z_slab",
                    source_ref=record.source_ref,
                    family=ExpressionFamily.Z_SLAB_REGION,
                    status=ClassificationStatus.SUPPORTED,
                    original_latex=record.raw_latex,
                    normalized_latex=record.normalized_latex,
                    color=record.color,
                    hidden=record.hidden,
                    metadata=metadata,
                    lower_expr=lower_expr,
                    upper_expr=upper_expr,
                    bounds=ranges,
                    sampling_hint=(160, 20),
                )
        return BoxVolumeNode(
            node_type="box_volume",
            source_ref=record.source_ref,
            family=family,
            status=ClassificationStatus.SUPPORTED,
            original_latex=record.raw_latex,
            normalized_latex=record.normalized_latex,
            color=record.color,
            hidden=record.hidden,
            metadata=metadata,
            ranges=ranges,
        )

    if family == ExpressionFamily.Z_SLAB_REGION:
        core_no_space = core.replace(" ", "")
        mslab = re.fullmatch(r"(.+?)(<=|>=|<|>)z(<=|>=|<|>)(.+)", core_no_space)
        if not mslab:
            return None
        left, op1, op2, right = mslab.groups()
        # Normalize so lower_expr <= z <= upper_expr
        if op1 in {">", ">="} and op2 in {">", ">="}:
            upper_expr, lower_expr = left, right
        else:
            lower_expr, upper_expr = left, right
        return ZSlabNode(
            node_type="z_slab",
            source_ref=record.source_ref,
            family=family,
            status=ClassificationStatus.SUPPORTED,
            original_latex=record.raw_latex,
            normalized_latex=record.normalized_latex,
            color=record.color,
            hidden=record.hidden,
            metadata=metadata,
            lower_expr=lower_expr,
            upper_expr=upper_expr,
            bounds=bounds,
            sampling_hint=(96, 32),
        )

    if family == ExpressionFamily.X_SLAB_REGION:
        parts = [core] + restrictions

        def _parse_x_ineq(s: str):
            m = re.fullmatch(r"(-?)x(<=|>=|<|>)(.+)", s.replace(" ", ""))
            if not m:
                return None
            sign, op, rhs = m.groups()
            if sign == "-":
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
            if not parsed:
                continue
            kind, expr = parsed
            if kind == "lower":
                x_lower = expr
            else:
                x_upper = expr

        if not (x_lower and x_upper):
            return None

        return XSlabNode(
            node_type="x_slab",
            source_ref=record.source_ref,
            family=family,
            status=ClassificationStatus.SUPPORTED,
            original_latex=record.raw_latex,
            normalized_latex=record.normalized_latex,
            color=record.color,
            hidden=record.hidden,
            metadata=metadata,
            lower_expr=x_lower,
            upper_expr=x_upper,
            bounds=bounds,
            sampling_hint=(48, 48),
        )

    if family == ExpressionFamily.Y_SLAB_REGION:
        parts = [core] + restrictions

        def _parse_y_ineq(s: str):
            m = re.fullmatch(r"(-?)y(<=|>=|<|>)(.+)", s.replace(" ", ""))
            if not m:
                return None
            sign, op, rhs = m.groups()
            if sign == "-":
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

        if not (y_lower and y_upper):
            return None

        return YSlabNode(
            node_type="y_slab",
            source_ref=record.source_ref,
            family=family,
            status=ClassificationStatus.SUPPORTED,
            original_latex=record.raw_latex,
            normalized_latex=record.normalized_latex,
            color=record.color,
            hidden=record.hidden,
            metadata=metadata,
            lower_expr=y_lower,
            upper_expr=y_upper,
            bounds=bounds,
            sampling_hint=(48, 256),
        )

    if family == ExpressionFamily.DISK_EXTRUSION_SOLID:
        spec = try_parse_axis_aligned_disk_inequality(core)
        if spec is None:
            return None
        bbox = try_disk_extrusion_world_bbox(restrictions, spec, viewport)
        if bbox is None:
            return None
        xmin, xmax, ymin, ymax, zmin, zmax = bbox
        return DiskExtrusionSolidNode(
            node_type="disk_extrusion_solid",
            source_ref=record.source_ref,
            family=family,
            status=ClassificationStatus.SUPPORTED,
            original_latex=record.raw_latex,
            normalized_latex=record.normalized_latex,
            color=record.color,
            hidden=record.hidden,
            metadata=metadata,
            axis_u=spec.axis_u,
            axis_v=spec.axis_v,
            center_u=spec.center_u,
            center_v=spec.center_v,
            radius_sq=spec.radius_sq,
            x_min=xmin,
            x_max=xmax,
            y_min=ymin,
            y_max=ymax,
            z_min=zmin,
            z_max=zmax,
            voxel_resolution=30,
        )

    if family in {ExpressionFamily.LINEAR_SURFACE_PATCH, ExpressionFamily.QUADRATIC_SURFACE_PATCH}:
        axis, rhs = core.split("=", 1)
        return SampledSurfaceNode(
            node_type="sampled_surface",
            source_ref=record.source_ref,
            family=family,
            status=ClassificationStatus.SUPPORTED,
            original_latex=record.raw_latex,
            normalized_latex=record.normalized_latex,
            color=record.color,
            hidden=record.hidden,
            metadata=metadata,
            dependent_axis=axis,
            function_expr=rhs,
            bounds=bounds,
            sampling_hint=(48, 48) if family == ExpressionFamily.QUADRATIC_SURFACE_PATCH else (24, 24),
        )

    return None
