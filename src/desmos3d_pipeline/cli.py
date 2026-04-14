from __future__ import annotations

import argparse
import json
from pathlib import Path

from desmos3d_pipeline.export.bridge import export_obj_bundle
from desmos3d_pipeline.export.report_writer import write_batch_summary, write_file_report
from desmos3d_pipeline.ir.builder import build_geometry_for_file
from desmos3d_pipeline.ir.models import BatchAuditSummary
from desmos3d_pipeline.mesh.meshers import mesh_geometry_nodes
from desmos3d_pipeline.qa.audit import run_audit_for_file


def _resolve_inputs(single_input: str | None, input_glob: str | None, base_dir: Path) -> list[Path]:
    paths: list[Path] = []
    if single_input:
        p = Path(single_input)
        if not p.is_absolute():
            p = base_dir / p
        paths.append(p)
    if input_glob:
        paths.extend(sorted(base_dir.glob(input_glob)))
    unique = sorted({p.resolve() for p in paths})
    return unique


def cmd_audit(args: argparse.Namespace) -> int:
    cwd = Path.cwd()
    inputs = _resolve_inputs(args.input, args.input_glob, cwd)
    if not inputs:
        raise SystemExit("No input files matched. Use --input or --input-glob.")

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = cwd / out_dir

    reports = []
    for path in inputs:
        report = run_audit_for_file(path)
        reports.append(report)
        write_file_report(out_dir, f"{path.stem}.audit.json", report)

    batch = BatchAuditSummary(files=reports)
    write_batch_summary(out_dir, batch)
    print(f"Audit complete: {len(reports)} file(s). Reports in {out_dir}")
    return 0


def cmd_export_bridge(args: argparse.Namespace) -> int:
    cwd = Path.cwd()
    inputs = _resolve_inputs(args.input, args.input_glob, cwd)
    if not inputs:
        raise SystemExit("No input files matched. Use --input or --input-glob.")

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = cwd / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, object]] = []
    for path in inputs:
        file_out = out_dir / path.stem
        file_out.mkdir(parents=True, exist_ok=True)
        geometry = build_geometry_for_file(path)
        meshes, failures = mesh_geometry_nodes(geometry.nodes)
        manifest_path = export_obj_bundle(meshes, failures, file_out)
        summary.append(
            {
                "source_file": path.name,
                "node_count": len(geometry.nodes),
                "mesh_count": len(meshes),
                "failed_mesh_count": len(failures),
                "manifest": str(manifest_path.relative_to(out_dir)),
            }
        )

    (out_dir / "bridge_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Bridge export complete: {len(summary)} file(s). Assets in {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="desmos3d")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="Audit and classify Desmos 3D JSON files")
    audit.add_argument("--input", type=str, default=None, help="Single input JSON file")
    audit.add_argument("--input-glob", type=str, default=None, help="Glob for multiple files")
    audit.add_argument("--out", type=str, default="artifacts/audit", help="Output report directory")
    audit.set_defaults(func=cmd_audit)

    bridge = sub.add_parser("export-bridge", help="Build geometry, mesh it, and export OBJ + manifest")
    bridge.add_argument("--input", type=str, default=None, help="Single input JSON file")
    bridge.add_argument("--input-glob", type=str, default=None, help="Glob for multiple files")
    bridge.add_argument("--out", type=str, default="artifacts/bridge", help="Output bridge directory")
    bridge.set_defaults(func=cmd_export_bridge)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
