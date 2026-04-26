# Desmos 3D to USDZ Pipeline

Compiler-like pipeline to parse Desmos 3D JSON, classify expressions, generate geometry, and export USDZ.

## Current milestone

- Phase 1 audit/classification pipeline in Python (in progress)
- Swift USDZ exporter scaffold created (implementation follows after Python bridge export)

## Run instructions

### 1) Set up Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
```

### 2) Run tests

```bash
python3 -m pytest -q
```

### 3) Run audit on one file

```bash
python3 -m desmos3d_pipeline.cli audit --input JSONreference.json --out artifacts/audit
```

### 4) Run audit on all sample JSON files and `[4B]` curriculum files

Use `--input-glob` twice so `pathlib` matches both `JSON*.json` and names that start with a literal `[4B]` (square brackets must be escaped as `[[]` / `[]]` in the pattern).

```bash
python3 -m desmos3d_pipeline.cli audit --input-glob "JSON*.json" --input-glob '[[]4B[]]*.json' --out artifacts/audit
```

### 5) Output artifacts

Reports are written to `artifacts/audit`:

- `*.audit.json` (per-input audit reports)
- `batch_summary.json` (aggregate summary across all processed files)

### 6) Build mesh bridge artifacts (OBJ + manifest)

```bash
PYTHONPATH=src python3 -m desmos3d_pipeline.cli export-bridge --input-glob "JSON*.json" --input-glob '[[]4B[]]*.json' --out artifacts/bridge
```

Some curriculum graphs only expose a **hidden** reference plane (e.g. `z=0` with a small domain). By default those are skipped so construction geometry stays out of USDZ; add **`--include-hidden`** on `export-bridge` (and the same flag on **`coverage`**) to mesh them anyway.

This writes:

- `artifacts/bridge/<file>/meshes/*.obj`
- `artifacts/bridge/<file>/manifest.json`
- `artifacts/bridge/bridge_summary.json`

### 7) Export USDZ with Swift CLI

```bash
cd swift-usdz-exporter
swift build
swift run usdz-exporter --manifest ../artifacts/bridge/JSONLondon/manifest.json --output ../artifacts/usdz/JSONLondon.usdz
```

For another file, swap `JSONLondon` with `JSONreference`, `JSONCali`, or `JSONAkashi`.

### 7.4) Family inventory (non-supported classifier buckets)

Rolls up `ExpressionFamily` × `ClassificationStatus` for every non-hidden expression that is not `SUPPORTED` (same globs as audit):

```bash
PYTHONPATH=src python3 -m desmos3d_pipeline.cli family-inventory \
  --input-glob "JSON*.json" --input-glob '[[]4B[]]*.json' \
  --out artifacts/family_inventory/unsupported_families.json
```

Output: `artifacts/family_inventory/unsupported_families.json` (`by_family_and_status`, `non_supported_by_family`, `non_supported_by_status`).

### 7.5) Coverage report (what’s still missing)

This generates fingerprint-grouped reports of expressions that are not currently meshed (plus a small markdown summary per file):

```bash
cd /Users/careylai/Desktop/python-swift-usdz && PYTHONPATH=src python3 -m desmos3d_pipeline.cli coverage --input-glob "JSON*.json" --input-glob '[[]4B[]]*.json' --out artifacts/coverage
```

Outputs:

- `artifacts/coverage/<file>.coverage.json`
- `artifacts/coverage/<file>.coverage.md`
- `artifacts/coverage/coverage_summary.json`

### 8) Run full pipeline for all JSON files (single command)

Includes `JSON*.json` samples and `[[]4B[]]*.json` curriculum files (see §4 for why the bracket pattern is written that way).

```bash
cd /Users/careylai/Desktop/python-swift-usdz && \
  PYTHONPATH=src python3 -m pytest -q && \
  PYTHONPATH=src python3 -m desmos3d_pipeline.cli audit \
    --input-glob "JSON*.json" --input-glob '[[]4B[]]*.json' --out artifacts/audit && \
  PYTHONPATH=src python3 -m desmos3d_pipeline.cli family-inventory \
    --input-glob "JSON*.json" --input-glob '[[]4B[]]*.json' --out artifacts/family_inventory/unsupported_families.json && \
  PYTHONPATH=src python3 -m desmos3d_pipeline.cli export-bridge \
    --input-glob "JSON*.json" --input-glob '[[]4B[]]*.json' --out artifacts/bridge && \
  cd swift-usdz-exporter && \
  mkdir -p ../artifacts/usdz && \
  for manifest in ../artifacts/bridge/*/manifest.json; do \
    name="$(basename "$(dirname "$manifest")")"; \
    swift run usdz-exporter --manifest "$manifest" --output "../artifacts/usdz/${name}.usdz"; \
  done && \
  cd .. && \
  PYTHONPATH=src python3 -m desmos3d_pipeline.cli coverage \
    --input-glob "JSON*.json" --input-glob '[[]4B[]]*.json' --out artifacts/coverage
```