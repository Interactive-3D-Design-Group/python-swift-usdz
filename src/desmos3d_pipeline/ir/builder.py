from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from desmos3d_pipeline.classify.rules import classify_expression
from desmos3d_pipeline.io.desmos_json import extract_expression_list, load_desmos_json
from desmos3d_pipeline.ir.models import (
    BoxVolumeNode,
    ClassificationStatus,
    Diagnostic,
    ExpressionFamily,
    ExpressionRecord,
    GeometryNode,
    PlanePatchNode,
    PointNode,
    SampledSurfaceNode,
    Severity,
    SourceRef,
)
from desmos3d_pipeline.normalize.latex import extract_brace_restrictions, normalize_latex
from desmos3d_pipeline.parse.math_eval import normalize_symbol_name, safe_eval, to_python_expr
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
    viewport = _extract_viewport(desmos_file.data)
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
        classification = classify_expression(normalized, item_type)
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


def _resolve_symbol_table(symbol_table: dict[str, str]) -> dict[str, float]:
    resolved: dict[str, float] = {}
    python_names = {name: normalize_symbol_name(name) for name in symbol_table}
    pending = dict(symbol_table)
    for _ in range(len(pending) + 2):
        progress = False
        for name, expr in list(pending.items()):
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


def _extract_viewport(data: dict[str, Any]) -> dict[str, float]:
    viewport = data.get("graph", {}).get("viewport", {})
    out: dict[str, float] = {}
    for key in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"):
        value = viewport.get(key)
        if isinstance(value, (int, float)):
            out[key] = float(value)
    return out
