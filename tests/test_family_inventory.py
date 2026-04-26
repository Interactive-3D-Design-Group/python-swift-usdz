from __future__ import annotations

from pathlib import Path

import pytest

from desmos3d_pipeline.qa.family_inventory import run_family_inventory


def test_run_family_inventory_jsonreference_smoke() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "JSONreference.json"
    if not path.is_file():
        pytest.skip("JSONreference.json not in repo checkout")
    out = run_family_inventory([path])
    assert out["files_scanned"] == 1
    assert out["expressions_considered"] > 0
    assert "by_family_and_status" in out
    assert "non_supported_by_family" in out
    assert out["non_supported_expression_count"] <= out["expressions_considered"]
