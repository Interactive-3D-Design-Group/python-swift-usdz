from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from desmos3d_pipeline.export.bridge import export_obj_bundle
from desmos3d_pipeline.export.report_writer import write_batch_summary, write_file_report
from desmos3d_pipeline.ir.builder import build_geometry_for_file
from desmos3d_pipeline.ir.models import BatchAuditSummary
from desmos3d_pipeline.mesh.meshers import mesh_geometry_nodes
from desmos3d_pipeline.qa.audit import run_audit_for_file
from desmos3d_pipeline.qa.coverage import run_coverage_for_file
from desmos3d_pipeline.qa.family_inventory import run_family_inventory


def _resolve_inputs(
    single_input: str | None, input_glob: str | Sequence[str] | None, base_dir: Path
) -> list[Path]:
    paths: list[Path] = []
    if single_input:
        p = Path(single_input)
        if not p.is_absolute():
            p = base_dir / p
        paths.append(p)
    if input_glob:
        patterns = [input_glob] if isinstance(input_glob, str) else list(input_glob)
        for pattern in patterns:
            paths.extend(sorted(base_dir.glob(pattern)))
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
        geometry = build_geometry_for_file(path, include_hidden=args.include_hidden)
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


def cmd_coverage(args: argparse.Namespace) -> int:
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
        report = run_coverage_for_file(path, include_hidden=args.include_hidden)
        out_path = out_dir / f"{path.stem}.coverage.json"
        out_path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
        # Also write a concise markdown summary for quick review.
        md_path = out_dir / f"{path.stem}.coverage.md"
        top = report.groups[:15]
        lines = [
            f"## Coverage summary: {path.name}",
            "",
            f"- total expressions: **{report.total_expressions}**",
            f"- meshed expression ids: **{report.meshed_expression_count}**",
            f"- supported expressions: **{report.supported_expression_count}**",
            f"- supported but not meshed: **{report.supported_but_not_meshed_count}**",
            f"- recognized unsupported (reserved; classifier uses **0** — see geometry ineligible): **{report.recognized_unsupported_count}**",
            f"- geometry ineligible (non-mesh: params, inequalities we skip, etc.): **{report.geometry_ineligible_count}**",
            f"- unrecognized: **{report.unrecognized_count}**",
            "",
            "## Top missing groups",
            "",
            "| missing | total | family | status | fingerprint | example ids | example normalized |",
            "|---:|---:|---|---|---|---|---|",
        ]
        for g in top:
            missing = g.count - g.meshed_count
            if missing <= 0:
                continue
            ex_ids = ",".join(g.example_expression_ids[:3])
            ex_norm = (g.example_normalized[0] if g.example_normalized else "").replace("|", "\\|")
            ex_norm = (ex_norm[:120] + "…") if len(ex_norm) > 120 else ex_norm
            lines.append(f"| {missing} | {g.count} | {g.family} | {g.status} | `{g.fingerprint}` | `{ex_ids}` | `{ex_norm}` |")
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        summary.append(
            {
                "source_file": path.name,
                "total_expressions": report.total_expressions,
                "meshed_expression_count": report.meshed_expression_count,
                "supported_expression_count": report.supported_expression_count,
                "supported_but_not_meshed_count": report.supported_but_not_meshed_count,
                "recognized_unsupported_count": report.recognized_unsupported_count,
                "geometry_ineligible_count": report.geometry_ineligible_count,
                "unrecognized_count": report.unrecognized_count,
                "group_count": len(report.groups),
                "report": str(out_path.relative_to(out_dir)),
                "markdown": str(md_path.relative_to(out_dir)),
            }
        )

    (out_dir / "coverage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Coverage reports complete: {len(summary)} file(s). Reports in {out_dir}")
    return 0


def cmd_family_inventory(args: argparse.Namespace) -> int:
    cwd = Path.cwd()
    inputs = _resolve_inputs(args.input, args.input_glob, cwd)
    if not inputs:
        raise SystemExit("No input files matched. Use --input or --input-glob.")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = cwd / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = run_family_inventory(inputs)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Family inventory: {payload['files_scanned']} file(s), "
        f"{payload['non_supported_expression_count']} non-supported / "
        f"{payload['expressions_considered']} expressions. Wrote {out_path}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="desmos3d")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="Audit and classify Desmos 3D JSON files")
    audit.add_argument("--input", type=str, default=None, help="Single input JSON file")
    audit.add_argument(
        "--input-glob",
        action="append",
        default=None,
        metavar="PATTERN",
        help="Glob for multiple files (repeat to merge several patterns, e.g. JSON*.json and [4B]*.json)",
    )
    audit.add_argument("--out", type=str, default="artifacts/audit", help="Output report directory")
    audit.set_defaults(func=cmd_audit)

    bridge = sub.add_parser("export-bridge", help="Build geometry, mesh it, and export OBJ + manifest")
    bridge.add_argument("--input", type=str, default=None, help="Single input JSON file")
    bridge.add_argument(
        "--input-glob",
        action="append",
        default=None,
        metavar="PATTERN",
        help="Glob for multiple files (repeat to merge several patterns, e.g. JSON*.json and [4B]*.json)",
    )
    bridge.add_argument("--out", type=str, default="artifacts/bridge", help="Output bridge directory")
    bridge.add_argument(
        "--include-hidden",
        action="store_true",
        help="Mesh expressions marked hidden in Desmos (e.g. reference z=0 tiles); default skips them",
    )
    bridge.set_defaults(func=cmd_export_bridge)

    cov = sub.add_parser("coverage", help="Report what expressions are missing from meshing, grouped by fingerprint")
    cov.add_argument("--input", type=str, default=None, help="Single input JSON file")
    cov.add_argument(
        "--input-glob",
        action="append",
        default=None,
        metavar="PATTERN",
        help="Glob for multiple files (repeat to merge several patterns, e.g. JSON*.json and [4B]*.json)",
    )
    cov.add_argument("--out", type=str, default="artifacts/coverage", help="Output coverage report directory")
    cov.add_argument(
        "--include-hidden",
        action="store_true",
        help="Match export-bridge: count/mesh hidden expressions when measuring coverage",
    )
    cov.set_defaults(func=cmd_coverage)

    inv = sub.add_parser(
        "family-inventory",
        help="Aggregate classifier family+status for non-SUPPORTED expressions (strategy / backlog)",
    )
    inv.add_argument("--input", type=str, default=None, help="Single input JSON file")
    inv.add_argument(
        "--input-glob",
        action="append",
        default=None,
        metavar="PATTERN",
        help="Glob for multiple files (repeat to merge several patterns)",
    )
    inv.add_argument(
        "--out",
        type=str,
        default="artifacts/family_inventory/unsupported_families.json",
        help="Output JSON path",
    )
    inv.set_defaults(func=cmd_family_inventory)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
