from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from desmos3d_pipeline.ir.models import Diagnostic, Severity


@dataclass(slots=True)
class DesmosFile:
    path: Path
    data: dict[str, Any]


def load_desmos_json(path: Path) -> tuple[DesmosFile | None, list[Diagnostic]]:
    diagnostics: list[Diagnostic] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        diagnostics.append(Diagnostic(severity=Severity.ERROR, code="IO_READ_FAILED", message=str(exc)))
        return None, diagnostics

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        diagnostics.append(Diagnostic(severity=Severity.ERROR, code="JSON_PARSE_FAILED", message=str(exc)))
        return None, diagnostics

    if not isinstance(data, dict):
        diagnostics.append(Diagnostic(severity=Severity.ERROR, code="JSON_ROOT_INVALID", message="Expected object at JSON root"))
        return None, diagnostics

    if "expressions" not in data:
        diagnostics.append(Diagnostic(severity=Severity.WARNING, code="MISSING_EXPRESSIONS", message="No expressions section found"))

    return DesmosFile(path=path, data=data), diagnostics


def extract_expression_list(data: dict[str, Any]) -> list[dict[str, Any]]:
    expressions = data.get("expressions", {})
    items = expressions.get("list", []) if isinstance(expressions, dict) else []
    return [it for it in items if isinstance(it, dict)]
