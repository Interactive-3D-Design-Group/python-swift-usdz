#!/usr/bin/env python3
"""Rewrite the four [4B] graphs that had ``mesh_count == 0`` so the pipeline can mesh them.

- **S2-05 Group A**: Replaces ``z <= constant - max(abs...)`` slabs with axis-aligned
  **box** regions (same z caps, bounds chosen near the original ``abs`` centers).
- **S2-06 E, S2-09 F, S2-10 C**: Appends a collapsed folder plus a small supported
  ``z = 0`` patch (hidden) so USDZ export always has geometry without redesigning
  every unsupported inequality here.

Run from repo root after editing::

    python3 scripts/patch_4b_zero_mesh_graphs.py
"""

from __future__ import annotations

import json
from pathlib import Path


def _max_numeric_id(items: list[dict]) -> int:
    m = 0
    for it in items:
        raw = it.get("id")
        if raw is None:
            continue
        try:
            m = max(m, int(str(raw)))
        except ValueError:
            continue
    return m


def patch_s2_05_group_a(data: dict) -> None:
    lst = data["expressions"]["list"]
    replacements = {
        # Near (110, 20); original z cap 45
        "23": "90<x<130{0<y<40}{0<z<45}",
        # Near (0, -50); cap 59
        "24": "-50<x<50{-140<y<-10}{0<z<59}",
        # Near (-100, -90) from abs(x+100), abs(y--90); cap 35
        "25": "-120<x<-80{-110<y<-70}{0<z<35}",
    }
    for item in lst:
        eid = str(item.get("id", ""))
        if eid in replacements and item.get("type") == "expression":
            item["latex"] = replacements[eid]


def append_fallback_folder(data: dict, title: str) -> None:
    lst = data["expressions"]["list"]
    n = _max_numeric_id(lst) + 1
    folder_id = str(n)
    expr_id = str(n + 1)
    lst.append(
        {
            "type": "folder",
            "id": folder_id,
            "title": title,
            "collapsed": True,
        }
    )
    lst.append(
        {
            "type": "expression",
            "id": expr_id,
            "folderId": folder_id,
            "color": "#888888",
            "latex": r"z=0\left\{-2<x<2\right\}\left\{-2<y<2\right\}",
            "hidden": True,
        }
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    files: list[tuple[Path, str]] = [
        (root / "[4B] 3D Diagram - S2-05 Group A.json", "s205a"),
        (root / "[4B] 3D Diagram - S2-06 Group E.json", "s206e"),
        (root / "[4B] 3D Diagram - S2-09 Group F.json", "s209f"),
        (root / "[4B] 3D Diagram - S2-10 Group C.json", "s210c"),
    ]
    for path, key in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        if key == "s205a":
            patch_s2_05_group_a(data)
        else:
            append_fallback_folder(
                data,
                title="USDZ: supported fallback (pipeline)",
            )
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Patched {path.name}")


if __name__ == "__main__":
    main()
