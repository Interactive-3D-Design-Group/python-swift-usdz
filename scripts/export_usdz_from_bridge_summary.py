#!/usr/bin/env python3
"""Run Swift ``usdz-exporter`` for every bridge row with ``mesh_count > 0``.

Reads ``artifacts/<bridge_subdir>/bridge_summary.json`` and writes one ``.usdz``
per entry under ``artifacts/<usdz_subdir>/`` using ASCII-safe filenames.

Usage (from repo root)::

    PYTHONPATH=src python3 -m desmos3d_pipeline.cli export-bridge \\
        --input-glob '[[]4B[]]*.json' --out artifacts/bridge_4b
    python3 scripts/export_usdz_from_bridge_summary.py \\
        --bridge-dir artifacts/bridge_4b --usdz-dir artifacts/usdz_4b
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def usdz_filename_from_manifest_relative(manifest_rel: str) -> str:
    """``[4B] 3D Diagram - S2-01 Group A/manifest.json`` -> ``4B_3D_Diagram_S2-01_Group_A.usdz``."""
    stem = Path(manifest_rel).parent.name
    s = stem.replace("[4B] ", "4B_").replace("[", "").replace("]", "")
    s = re.sub(r"[^\w.-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s + ".usdz"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bridge-dir", type=Path, default=Path("artifacts/bridge_4b"))
    parser.add_argument("--usdz-dir", type=Path, default=Path("artifacts/usdz_4b"))
    args = parser.parse_args()

    repo: Path = args.repo.resolve()
    bridge_dir = (repo / args.bridge_dir).resolve() if not args.bridge_dir.is_absolute() else args.bridge_dir
    usdz_dir = (repo / args.usdz_dir).resolve() if not args.usdz_dir.is_absolute() else args.usdz_dir
    summary_path = bridge_dir / "bridge_summary.json"
    if not summary_path.is_file():
        print(f"Missing {summary_path}", file=sys.stderr)
        return 1

    usdz_dir.mkdir(parents=True, exist_ok=True)
    pkg = repo / "swift-usdz-exporter"
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    eligible = [r for r in rows if int(r.get("mesh_count", 0) or 0) > 0]
    print(f"Exporting {len(eligible)} USDZ file(s) from {bridge_dir} -> {usdz_dir}")

    ok = 0
    for row in eligible:
        manifest = bridge_dir / str(row["manifest"])
        out = usdz_dir / usdz_filename_from_manifest_relative(str(row["manifest"]))
        cmd = [
            "swift",
            "run",
            "usdz-exporter",
            "--manifest",
            str(manifest),
            "--output",
            str(out),
        ]
        r = subprocess.run(cmd, cwd=pkg, check=False)
        if r.returncode != 0:
            print(f"FAILED ({r.returncode}): {manifest} -> {out}", file=sys.stderr)
            return r.returncode or 1
        ok += 1
    print(f"Done: {ok} USDZ file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
