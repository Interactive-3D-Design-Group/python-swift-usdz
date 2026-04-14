from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class ClassificationStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    RECOGNIZED_UNSUPPORTED = "RECOGNIZED_UNSUPPORTED"
    UNRECOGNIZED = "UNRECOGNIZED"


class ExpressionFamily(str, Enum):
    CONSTANT_PLANE = "CONSTANT_PLANE"
    BOX_BOUNDED_REGION = "BOX_BOUNDED_REGION"
    Z_SLAB_REGION = "Z_SLAB_REGION"
    X_SLAB_REGION = "X_SLAB_REGION"
    Y_SLAB_REGION = "Y_SLAB_REGION"
    LINEAR_SURFACE_PATCH = "LINEAR_SURFACE_PATCH"
    QUADRATIC_SURFACE_PATCH = "QUADRATIC_SURFACE_PATCH"
    POINT_DEFINITION = "POINT_DEFINITION"
    TRIANGLE_CALL = "TRIANGLE_CALL"
    POLYGON_CALL = "POLYGON_CALL"
    INEQUALITY_REGION = "INEQUALITY_REGION"
    PARAM_ASSIGNMENT = "PARAM_ASSIGNMENT"
    TEXT_OR_FOLDER = "TEXT_OR_FOLDER"
    UNKNOWN = "UNKNOWN"


class AuditStatus(str, Enum):
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"


@dataclass(slots=True)
class SourceRef:
    source_file: str
    expression_id: str | None
    folder_id: str | None
    folder_name: str | None
    index: int


@dataclass(slots=True)
class Diagnostic:
    severity: Severity
    code: str
    message: str
    source_ref: SourceRef | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExpressionRecord:
    source_ref: SourceRef
    expression_type: str
    raw_latex: str
    normalized_latex: str
    color: str | None
    hidden: bool
    extend_to_3d: bool
    lines: bool


@dataclass(slots=True)
class ClassificationResult:
    family: ExpressionFamily
    status: ClassificationStatus
    reason: str
    confidence: float
    fingerprint: str


@dataclass(slots=True)
class RangeConstraint:
    axis: str
    lower: str | None
    lower_inclusive: bool
    upper: str | None
    upper_inclusive: bool


@dataclass(slots=True)
class GeometryNode:
    node_type: str
    source_ref: SourceRef
    family: ExpressionFamily
    status: ClassificationStatus
    original_latex: str
    normalized_latex: str
    color: str | None
    hidden: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PlanePatchNode(GeometryNode):
    axis: str = "z"
    value: str = "0"
    bounds: list[RangeConstraint] = field(default_factory=list)


@dataclass(slots=True)
class BoxVolumeNode(GeometryNode):
    ranges: list[RangeConstraint] = field(default_factory=list)


@dataclass(slots=True)
class ZSlabNode(GeometryNode):
    lower_expr: str = ""
    upper_expr: str = ""
    bounds: list[RangeConstraint] = field(default_factory=list)
    sampling_hint: tuple[int, int] = (72, 24)


@dataclass(slots=True)
class XSlabNode(GeometryNode):
    lower_expr: str = ""
    upper_expr: str = ""
    bounds: list[RangeConstraint] = field(default_factory=list)
    sampling_hint: tuple[int, int] = (48, 48)


@dataclass(slots=True)
class YSlabNode(GeometryNode):
    lower_expr: str = ""
    upper_expr: str = ""
    bounds: list[RangeConstraint] = field(default_factory=list)
    sampling_hint: tuple[int, int] = (48, 256)


@dataclass(slots=True)
class SampledSurfaceNode(GeometryNode):
    dependent_axis: str = "z"
    function_expr: str = ""
    bounds: list[RangeConstraint] = field(default_factory=list)
    sampling_hint: tuple[int, int] = (64, 64)


@dataclass(slots=True)
class PointNode(GeometryNode):
    name: str = ""
    x: str = "0"
    y: str = "0"
    z: str = "0"


@dataclass(slots=True)
class PolygonFaceNode(GeometryNode):
    point_refs: list[str] = field(default_factory=list)
    inline_vertices: list[tuple[str, str, str]] = field(default_factory=list)


@dataclass(slots=True)
class UnsupportedExpressionNode(GeometryNode):
    unsupported_reason: str = ""
    fingerprint: str = ""


@dataclass(slots=True)
class AuditExpressionItem:
    record: ExpressionRecord
    classification: ClassificationResult


@dataclass(slots=True)
class FolderSummary:
    folder_id: str | None
    folder_name: str | None
    total: int = 0
    supported: int = 0
    recognized_unsupported: int = 0
    unrecognized: int = 0


@dataclass(slots=True)
class FileAuditReport:
    source_file: str
    total_expressions: int
    supported_count: int
    recognized_unsupported_count: int
    unrecognized_count: int
    per_folder_summary: list[FolderSummary]
    unsupported_expressions: list[dict[str, Any]]
    unknown_fingerprints: list[str]
    diagnostics: list[Diagnostic]
    status: AuditStatus

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BatchAuditSummary:
    files: list[FileAuditReport]

    def to_dict(self) -> dict[str, Any]:
        reports = [f.to_dict() for f in self.files]
        return {
            "file_count": len(reports),
            "totals": {
                "expressions": sum(r["total_expressions"] for r in reports),
                "supported": sum(r["supported_count"] for r in reports),
                "recognized_unsupported": sum(r["recognized_unsupported_count"] for r in reports),
                "unrecognized": sum(r["unrecognized_count"] for r in reports),
            },
            "blocked_files": [r["source_file"] for r in reports if r["status"] == AuditStatus.FAIL],
            "files": reports,
        }
