from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from desmos3d_pipeline.classify.rules import classify_expression
from desmos3d_pipeline.io.desmos_json import extract_expression_list, load_desmos_json
from desmos3d_pipeline.ir.models import (
    AuditExpressionItem,
    AuditStatus,
    ClassificationStatus,
    Diagnostic,
    ExpressionRecord,
    FileAuditReport,
    FolderSummary,
    Severity,
    SourceRef,
)
from desmos3d_pipeline.normalize.latex import normalize_latex


def run_audit_for_file(path: Path) -> FileAuditReport:
    desmos_file, diagnostics = load_desmos_json(path)
    if desmos_file is None:
        return FileAuditReport(
            source_file=path.name,
            total_expressions=0,
            supported_count=0,
            recognized_unsupported_count=0,
            unrecognized_count=0,
            per_folder_summary=[],
            unsupported_expressions=[],
            unknown_fingerprints=[],
            diagnostics=diagnostics,
            status=AuditStatus.FAIL,
        )

    items = extract_expression_list(desmos_file.data)
    folders: dict[str, str] = {}
    audited: list[AuditExpressionItem] = []

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

        classification = classify_expression(normalized, item_type)
        audited.append(AuditExpressionItem(record=record, classification=classification))

    supported = sum(1 for a in audited if a.classification.status == ClassificationStatus.SUPPORTED)
    rec_unsup = sum(1 for a in audited if a.classification.status == ClassificationStatus.RECOGNIZED_UNSUPPORTED)
    unrec = sum(1 for a in audited if a.classification.status == ClassificationStatus.UNRECOGNIZED)

    folder_stats: dict[tuple[str | None, str | None], FolderSummary] = {}
    for a in audited:
        key = (a.record.source_ref.folder_id, a.record.source_ref.folder_name)
        if key not in folder_stats:
            folder_stats[key] = FolderSummary(folder_id=key[0], folder_name=key[1])
        fs = folder_stats[key]
        fs.total += 1
        if a.classification.status == ClassificationStatus.SUPPORTED:
            fs.supported += 1
        elif a.classification.status == ClassificationStatus.RECOGNIZED_UNSUPPORTED:
            fs.recognized_unsupported += 1
        else:
            fs.unrecognized += 1

    unsupported = []
    unknown_fingerprints = []
    for a in audited:
        if a.classification.status != ClassificationStatus.SUPPORTED:
            unsupported.append(
                {
                    "expression_id": a.record.source_ref.expression_id,
                    "folder": a.record.source_ref.folder_name,
                    "family": a.classification.family,
                    "status": a.classification.status,
                    "reason": a.classification.reason,
                    "fingerprint": a.classification.fingerprint,
                    "latex": a.record.raw_latex,
                    "normalized_latex": a.record.normalized_latex,
                }
            )
        if a.classification.status == ClassificationStatus.UNRECOGNIZED:
            unknown_fingerprints.append(a.classification.fingerprint)

    if unrec == 0:
        status = AuditStatus.PASS if rec_unsup == 0 else AuditStatus.PARTIAL
    else:
        status = AuditStatus.PARTIAL

    diagnostics.append(
        Diagnostic(
            severity=Severity.INFO,
            code="AUDIT_COMPLETE",
            message="Audit classification complete",
            details={"file": path.name},
        )
    )

    return FileAuditReport(
        source_file=path.name,
        total_expressions=len(audited),
        supported_count=supported,
        recognized_unsupported_count=rec_unsup,
        unrecognized_count=unrec,
        per_folder_summary=sorted(folder_stats.values(), key=lambda x: (x.folder_name or "", x.folder_id or "")),
        unsupported_expressions=unsupported,
        unknown_fingerprints=sorted(set(unknown_fingerprints)),
        diagnostics=diagnostics,
        status=status,
    )


def to_json_safe(report: Any) -> dict[str, Any]:
    payload = report.to_dict() if hasattr(report, "to_dict") else report
    return _enum_to_values(payload)


def _enum_to_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _enum_to_values(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_enum_to_values(v) for v in value]
    if hasattr(value, "value"):
        return value.value
    return value
