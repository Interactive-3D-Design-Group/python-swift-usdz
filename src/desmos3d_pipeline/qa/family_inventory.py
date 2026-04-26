from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from desmos3d_pipeline.classify.rules import classify_expression
from desmos3d_pipeline.io.desmos_json import extract_expression_list, extract_viewport, load_desmos_json
from desmos3d_pipeline.ir.models import ClassificationStatus
from desmos3d_pipeline.normalize.latex import normalize_latex


def run_family_inventory(paths: list[Path]) -> dict[str, Any]:
    """Classify every non-hidden ``expression`` item and tally families where status is not ``SUPPORTED``.

    Output is meant for prioritizing new geometry families (``UNKNOWN``, ``INEQUALITY_REGION``, etc.).
    """
    pair_counts: Counter[tuple[str, str]] = Counter()
    family_non_supported: Counter[str] = Counter()
    status_totals: Counter[str] = Counter()
    expressions_considered = 0

    for path in paths:
        desmos_file, _ = load_desmos_json(path)
        if desmos_file is None:
            continue
        items = extract_expression_list(desmos_file.data)
        viewport = extract_viewport(desmos_file.data)
        for item in items:
            item_type = str(item.get("type", "expression"))
            if item_type != "expression":
                continue
            if bool(item.get("hidden", False)):
                continue
            raw_latex = str(item.get("latex", ""))
            normalized = normalize_latex(raw_latex)
            expressions_considered += 1
            c = classify_expression(normalized, item_type, viewport)
            if c.status != ClassificationStatus.SUPPORTED:
                fk = c.family.value
                sk = c.status.value
                pair_counts[(fk, sk)] += 1
                family_non_supported[fk] += 1
                status_totals[sk] += 1

    by_pair = [
        {"family": f, "status": s, "count": n}
        for (f, s), n in sorted(pair_counts.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    ]
    by_family = [
        {"family": f, "count": n} for f, n in sorted(family_non_supported.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    by_status = [
        {"status": s, "count": n} for s, n in sorted(status_totals.items(), key=lambda kv: (-kv[1], kv[0]))
    ]

    non_supported = sum(pair_counts.values())
    return {
        "files_scanned": len(paths),
        "expressions_considered": expressions_considered,
        "non_supported_expression_count": non_supported,
        "by_family_and_status": by_pair,
        "non_supported_by_family": by_family,
        "non_supported_by_status": by_status,
    }
