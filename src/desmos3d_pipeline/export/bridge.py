from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from desmos3d_pipeline.mesh.meshers import Mesh


@dataclass(slots=True)
class MeshManifestEntry:
    name: str
    obj_file: str
    color: str | None
    source_file: str
    expression_id: str | None
    family: str
    bounds: dict[str, list[float]] | None


def export_obj_bundle(meshes: list[Mesh], failures: list[dict[str, str]], out_dir: Path) -> Path:
    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    entries: list[MeshManifestEntry] = []
    for mesh in meshes:
        obj_name = f"{mesh.name}.obj"
        _write_obj(mesh_dir / obj_name, mesh)
        entries.append(MeshManifestEntry(name=mesh.name, obj_file=f"meshes/{obj_name}", color=mesh.color, source_file=mesh.source_file, expression_id=mesh.expression_id, family=mesh.family, bounds=mesh.bounds()))
    manifest = {
        "schema_version": 1,
        "mesh_count": len(meshes),
        "failed_mesh_count": len(failures),
        "meshes": [asdict(entry) for entry in entries],
        "failures": failures,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _write_obj(path: Path, mesh: Mesh) -> None:
    lines = [f"o {mesh.name}"]
    for x, y, z in mesh.vertices:
        lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
    for face in mesh.faces:
        lines.append(f"f {face[0]} {face[1]} {face[2]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
