from __future__ import annotations

import json
from pathlib import Path

from desmos3d_pipeline.ir.models import BatchAuditSummary
from desmos3d_pipeline.qa.audit import to_json_safe


def write_file_report(out_dir: Path, report_name: str, report_obj: object) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / report_name
    if hasattr(report_obj, "to_dict"):
        payload = to_json_safe(report_obj)  # type: ignore[arg-type]
    else:
        payload = report_obj
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_batch_summary(out_dir: Path, batch: BatchAuditSummary) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = to_json_safe(batch)
    out_path = out_dir / "batch_summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
