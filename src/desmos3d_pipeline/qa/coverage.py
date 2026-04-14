from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from desmos3d_pipeline.classify.rules import classify_expression
from desmos3d_pipeline.io.desmos_json import extract_expression_list, load_desmos_json
from desmos3d_pipeline.ir.builder import build_geometry_for_file
from desmos3d_pipeline.ir.models import ClassificationStatus, Diagnostic, ExpressionRecord, Severity, SourceRef
from desmos3d_pipeline.mesh.meshers import mesh_geometry_nodes
from desmos3d_pipeline.normalize.latex import normalize_latex


@dataclass(slots=True)
class CoverageGroup:
    fingerprint: str
    family: str
    status: str
    reason: str
    count: int
    meshed_count: int
    example_expression_ids: list[str]
    example_latex: list[str]
    example_normalized: list[str]
    flags: dict[str, int]


@dataclass(slots=True)
class CoverageReport:
    source_file: str
    total_items: int
    total_expressions: int
    meshed_expression_count: int
    supported_expression_count: int
    supported_but_not_meshed_count: int
    recognized_unsupported_count: int
    unrecognized_count: int
    groups: list[CoverageGroup]
    diagnostics: list[Diagnostic]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_coverage_for_file(path: Path) -> CoverageReport:
    desmos_file, diagnostics = load_desmos_json(path)
    if desmos_file is None:
        diagnostics.append(
            Diagnostic(
                severity=Severity.ERROR,
                code="COVERAGE_LOAD_FAILED",
                message="Failed to load Desmos JSON for coverage report",
                details={"file": path.name},
            )
        )
        return CoverageReport(
            source_file=path.name,
            total_items=0,
            total_expressions=0,
            meshed_expression_count=0,
            supported_expression_count=0,
            supported_but_not_meshed_count=0,
            recognized_unsupported_count=0,
            unrecognized_count=0,
            groups=[],
            diagnostics=diagnostics,
        )

    items = extract_expression_list(desmos_file.data)

    # Determine what actually got meshed.
    geometry = build_geometry_for_file(path)
    meshes, failures = mesh_geometry_nodes(geometry.nodes)
    meshed_ids = {m.expression_id for m in meshes if m.expression_id is not None}
    if failures:
        diagnostics.append(
            Diagnostic(
                severity=Severity.WARNING,
                code="COVERAGE_MESH_FAILURES",
                message="Some meshes failed during coverage pass (these items may be missing)",
                details={"failed_mesh_count": len(failures), "failures": failures[:10]},
            )
        )

    folders: dict[str, str] = {}
    audited: list[tuple[ExpressionRecord, Any]] = []
    expr_count = 0
    for index, item in enumerate(items):
        item_type = str(item.get("type", "expression"))
        if item_type == "folder":
            folder_id = str(item.get("id")) if item.get("id") is not None else None
            title = str(item.get("title", ""))
            if folder_id:
                folders[folder_id] = title

        expr_id = str(item.get("id")) if item.get("id") is not None else None
        folder_id = str(item.get("folderId")) if item.get("folderId") is not None else None
        folder_name = folders.get(folder_id)

        source_ref = SourceRef(
            source_file=path.name,
            expression_id=expr_id,
            folder_id=folder_id,
            folder_name=folder_name,
            index=index,
        )

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
        if item_type == "expression":
            expr_count += 1
        classification = classify_expression(normalized, item_type)
        audited.append((record, classification))

    supported = [a for a in audited if a[1].status == ClassificationStatus.SUPPORTED and a[0].expression_type == "expression"]
    supported_count = len(supported)
    supported_but_not_meshed = [
        a
        for a in supported
        if (a[0].source_ref.expression_id is not None and a[0].source_ref.expression_id not in meshed_ids and not a[0].hidden)
    ]

    rec_unsup = sum(1 for a in audited if a[1].status == ClassificationStatus.RECOGNIZED_UNSUPPORTED and a[0].expression_type == "expression")
    unrec = sum(1 for a in audited if a[1].status == ClassificationStatus.UNRECOGNIZED and a[0].expression_type == "expression")

    # Build fingerprint-grouped “missing geometry” groups:
    # - expressions only
    # - not hidden
    # - either unsupported/unrecognized, OR supported but still not meshed
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}

    def _touch(group_key: tuple[str, str, str], record: ExpressionRecord, classification) -> None:
        g = groups.setdefault(
            group_key,
            {
                "fingerprint": classification.fingerprint,
                "family": classification.family.value if hasattr(classification.family, "value") else str(classification.family),
                "status": classification.status.value if hasattr(classification.status, "value") else str(classification.status),
                "reason": classification.reason,
                "count": 0,
                "meshed_count": 0,
                "example_expression_ids": [],
                "example_latex": [],
                "example_normalized": [],
                "flags": {"lines": 0, "extendTo3D": 0, "has_color": 0},
            },
        )
        g["count"] += 1
        expr_id = record.source_ref.expression_id or ""
        if record.source_ref.expression_id in meshed_ids:
            g["meshed_count"] += 1
        if len(g["example_expression_ids"]) < 5 and expr_id:
            g["example_expression_ids"].append(expr_id)
            g["example_latex"].append(record.raw_latex[:240])
            g["example_normalized"].append(record.normalized_latex[:240])
        if record.lines:
            g["flags"]["lines"] += 1
        if record.extend_to_3d:
            g["flags"]["extendTo3D"] += 1
        if record.color:
            g["flags"]["has_color"] += 1

    for record, classification in audited:
        if record.expression_type != "expression":
            continue
        if record.hidden:
            continue

        expr_id = record.source_ref.expression_id
        is_meshed = expr_id is not None and expr_id in meshed_ids
        if classification.status != ClassificationStatus.SUPPORTED or (classification.status == ClassificationStatus.SUPPORTED and not is_meshed):
            key = (classification.fingerprint, str(classification.family), str(classification.status))
            _touch(key, record, classification)

    group_objs = [
        CoverageGroup(
            fingerprint=g["fingerprint"],
            family=g["family"],
            status=g["status"],
            reason=g["reason"],
            count=g["count"],
            meshed_count=g["meshed_count"],
            example_expression_ids=g["example_expression_ids"],
            example_latex=g["example_latex"],
            example_normalized=g["example_normalized"],
            flags=g["flags"],
        )
        for g in groups.values()
    ]
    group_objs.sort(key=lambda gg: (-(gg.count - gg.meshed_count), -gg.count, gg.family, gg.fingerprint))

    diagnostics.append(
        Diagnostic(
            severity=Severity.INFO,
            code="COVERAGE_COMPLETE",
            message="Coverage report complete",
            details={
                "file": path.name,
                "meshed_expression_ids": len(meshed_ids),
                "supported_but_not_meshed": len(supported_but_not_meshed),
                "group_count": len(group_objs),
            },
        )
    )

    return CoverageReport(
        source_file=path.name,
        total_items=len(items),
        total_expressions=expr_count,
        meshed_expression_count=len(meshed_ids),
        supported_expression_count=supported_count,
        supported_but_not_meshed_count=len(supported_but_not_meshed),
        recognized_unsupported_count=rec_unsup,
        unrecognized_count=unrec,
        groups=group_objs,
        diagnostics=diagnostics,
    )

