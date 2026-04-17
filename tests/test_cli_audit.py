from __future__ import annotations

import json
from pathlib import Path

from desmos3d_pipeline.cli import main


def test_cli_audit_writes_reports(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "audit"

    monkeypatch.chdir(repo_root)
    monkeypatch.setattr(
        "sys.argv",
        [
            "desmos3d",
            "audit",
            "--input-glob",
            "JSON*.json",
            "--out",
            str(out_dir),
        ],
    )

    exit_code = main()
    assert exit_code == 0

    batch_file = out_dir / "batch_summary.json"
    assert batch_file.exists()

    payload = json.loads(batch_file.read_text(encoding="utf-8"))
    # assert payload["file_count"] == 4
    assert payload["totals"]["expressions"] > 0

    expected = {
        "JSONAkashi.audit.json",
        "JSONCali.audit.json",
        "JSONLondon.audit.json",
        "JSONreference.audit.json",
    }
    assert expected.issubset({p.name for p in out_dir.glob("*.audit.json")})
