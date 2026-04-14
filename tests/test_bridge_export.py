from __future__ import annotations

import json
from pathlib import Path

from desmos3d_pipeline.cli import main
from desmos3d_pipeline.ir.builder import build_geometry_for_file
from desmos3d_pipeline.mesh.meshers import mesh_geometry_nodes


def test_build_geometry_for_reference_file() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = build_geometry_for_file(repo_root / "JSONreference.json")
    assert len(result.nodes) > 0


def test_mesh_geometry_for_reference_file() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = build_geometry_for_file(repo_root / "JSONreference.json")
    meshes, failures = mesh_geometry_nodes(result.nodes)
    assert len(meshes) > 0
    assert isinstance(failures, list)


def test_cli_export_bridge_writes_manifest(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "bridge"
    monkeypatch.chdir(repo_root)
    monkeypatch.setattr(
        "sys.argv",
        [
            "desmos3d",
            "export-bridge",
            "--input",
            "JSONreference.json",
            "--out",
            str(out_dir),
        ],
    )
    exit_code = main()
    assert exit_code == 0
    manifest = out_dir / "JSONreference" / "manifest.json"
    assert manifest.exists()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["mesh_count"] > 0
