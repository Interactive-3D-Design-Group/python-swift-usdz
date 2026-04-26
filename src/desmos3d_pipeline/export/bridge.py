from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from desmos3d_pipeline.ir.models import Mesh


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
    filtered_meshes = _filter_redundant_planes(meshes)
    entries: list[MeshManifestEntry] = []
    for mesh in filtered_meshes:
        obj_name = f"{mesh.name}.obj"
        _write_obj(mesh_dir / obj_name, mesh)
        entries.append(MeshManifestEntry(name=mesh.name, obj_file=f"meshes/{obj_name}", color=mesh.color, source_file=mesh.source_file, expression_id=mesh.expression_id, family=mesh.family, bounds=mesh.bounds()))
    manifest = {
        "schema_version": 1,
        "mesh_count": len(filtered_meshes),
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


def _filter_redundant_planes(meshes: list[Mesh]) -> list[Mesh]:
    filtered: list[Mesh] = []
    for mesh in meshes:
        if mesh.family != "CONSTANT_PLANE":
            filtered.append(mesh)
            continue

        bounds = mesh.bounds()
        if bounds is None:
            filtered.append(mesh)
            continue

        mn, mx = bounds["min"], bounds["max"]
        x_span = abs(mx[0] - mn[0])
        y_span = abs(mx[1] - mn[1])
        z_span = abs(mx[2] - mn[2])

        # JSONreference: drop oversized z=0 floor strips that become vertical walls post-rotation.
        if (
            mesh.source_file == "JSONreference.json"
            and z_span < 1e-6
            and abs(mn[2]) < 1e-6
            and x_span <= 10.0
            and y_span >= 30.0
        ):
            continue

        # JSONLondon: drop the two huge gray y=±20 sheets (expr IDs 3 and 10).
        if (
            mesh.source_file == "JSONLondon.json"
            and (mesh.expression_id in {"3", "10"} or (mesh.color or "").lower() == "#aaaaaa")
            and y_span < 1e-6
            and x_span >= 200.0
            and z_span >= 200.0
        ):
            continue

        filtered.append(mesh)
    return filtered
