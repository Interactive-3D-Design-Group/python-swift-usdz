from __future__ import annotations

"""Geometry IR, classification enums, and triangle-mesh output types.

Concrete ``GeometryNode`` subclasses are produced by ``ir.builder`` and consumed by
``mesh.meshers.mesh_geometry_nodes``. Coverage:

- **PlanePatchNode** — constant ``x|y|z`` planes (``CONSTANT_PLANE``).
- **BoxVolumeNode** — axis-aligned inequalities (``BOX_BOUNDED_REGION``).
- **ZSlabNode / XSlabNode / YSlabNode** — one axis between two surfaces (slab families).
- **SampledSurfaceNode** — ``z=f(x,y)`` from linear/quadratic surface classifications.
- **DiskExtrusionSolidNode** — filled disk extrusion (``DISK_EXTRUSION_SOLID``).
- **SphereSolidNode** — filled ball ``(x-cx)^2+…<=r^2`` (``SPHERE_SOLID``).
- **VerticalCylinderSurfaceNode** — hollow cylinder wall.
- **ParametricUVPatchNode / ParametricTCurveNode** — parametric surfaces and curves.
- **PointNode** — labeled 3D points.
- **PolygonFaceNode** — reserved for future ``polygon(...)`` / face lists (optional mesher).
- **UnsupportedExpressionNode** — placeholder IR for expressions that will not mesh.

``Mesh`` is the exporter-facing triangle soup returned by meshers (not a ``GeometryNode``).
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class ClassificationStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    # Reserved for forms we might classify as known-but-not-exported; the main classifier
    # routes non-meshable expressions to ``GEOMETRY_INELIGIBLE`` so coverage can report zero here.
    RECOGNIZED_UNSUPPORTED = "RECOGNIZED_UNSUPPORTED"
    UNRECOGNIZED = "UNRECOGNIZED"
    # Valid Desmos items that are not meshed by design (parameters, colors, etc.).
    GEOMETRY_INELIGIBLE = "GEOMETRY_INELIGIBLE"


class ExpressionFamily(str, Enum):
    """Classifier output families. Several map to ``SampledSurfaceNode`` (``LINEAR_*`` / ``QUADRATIC_*``)."""

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
    SEGMENT_CALL = "SEGMENT_CALL"
    INEQUALITY_REGION = "INEQUALITY_REGION"
    DISK_EXTRUSION_SOLID = "DISK_EXTRUSION_SOLID"
    SPHERE_SOLID = "SPHERE_SOLID"
    VERTICAL_CYLINDER_SURFACE = "VERTICAL_CYLINDER_SURFACE"
    PARAMETRIC_UV_SURFACE = "PARAMETRIC_UV_SURFACE"
    PARAMETRIC_T_CURVE = "PARAMETRIC_T_CURVE"
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
class Mesh:
    """Triangulated geometry for OBJ / USDZ export (output of meshers, not Desmos IR)."""

    name: str
    color: str | None
    vertices: list[tuple[float, float, float]] = field(default_factory=list)
    faces: list[tuple[int, int, int]] = field(default_factory=list)
    source_file: str = ""
    expression_id: str | None = None
    family: str = ""

    def bounds(self) -> dict[str, list[float]] | None:
        if not self.vertices:
            return None
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        zs = [v[2] for v in self.vertices]
        return {"min": [min(xs), min(ys), min(zs)], "max": [max(xs), max(ys), max(zs)]}


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
class VerticalCylinderSurfaceNode(GeometryNode):
    """Cylinder ``(stretch_u*(u-cu))^2+(stretch_v*(v-cv))^2=r^2`` (``stretch_*`` default ``1``)."""

    axis_u: str = "x"
    axis_v: str = "y"
    center_u: float = 0.0
    center_v: float = 0.0
    radius_sq: float = 1.0
    stretch_u: float = 1.0
    stretch_v: float = 1.0
    extrusion_axis: str = "z"
    z_min: float = 0.0
    z_max: float = 1.0
    theta_segments: int = 40
    z_segments: int = 12


@dataclass(slots=True)
class ParametricUVPatchNode(GeometryNode):
    """Parametric surface ``(x(u,v), y(u,v), z(u,v))`` with ``u,v`` in ``[0,1]`` (Desmos defaults)."""

    x_expr: str = ""
    y_expr: str = ""
    z_expr: str = ""
    u_segments: int = 40
    v_segments: int = 40


@dataclass(slots=True)
class ParametricTCurveNode(GeometryNode):
    """Parametric curve ``(x(t), y(t), z(t))`` with ``t`` in ``[0,1]``."""

    x_expr: str = ""
    y_expr: str = ""
    z_expr: str = ""
    segments: int = 96


@dataclass(slots=True)
class DiskExtrusionSolidNode(GeometryNode):
    """Voxel-meshed solid (u-cu)^2+(v-cv)^2<=radius_sq within an axis-aligned bbox."""

    axis_u: str = "x"
    axis_v: str = "y"
    center_u: float = 0.0
    center_v: float = 0.0
    radius_sq: float = 1.0
    x_min: float = -1.0
    x_max: float = 1.0
    y_min: float = -1.0
    y_max: float = 1.0
    z_min: float = -1.0
    z_max: float = 1.0
    voxel_resolution: int = 28


@dataclass(slots=True)
class SphereSolidNode(GeometryNode):
    """Voxel-meshed solid ball ``(x-cx)^2+(y-cy)^2+(z-cz)^2<=radius_sq`` clipped to a bbox."""

    center_x: float = 0.0
    center_y: float = 0.0
    center_z: float = 0.0
    radius_sq: float = 1.0
    x_min: float = -1.0
    x_max: float = 1.0
    y_min: float = -1.0
    y_max: float = 1.0
    z_min: float = -1.0
    z_max: float = 1.0
    voxel_resolution: int = 28


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
    """Closed face as point IDs and/or inline ``(x,y,z)`` expressions (future ``polygon`` support)."""

    point_refs: list[str] = field(default_factory=list)
    inline_vertices: list[tuple[str, str, str]] = field(default_factory=list)


@dataclass(slots=True)
class UnsupportedExpressionNode(GeometryNode):
    """Explicit IR when an expression is classified but intentionally not converted to geometry."""

    unsupported_reason: str = ""
    fingerprint: str = ""


GeometryConcreteNode = (
    PlanePatchNode
    | VerticalCylinderSurfaceNode
    | ParametricUVPatchNode
    | ParametricTCurveNode
    | DiskExtrusionSolidNode
    | SphereSolidNode
    | BoxVolumeNode
    | ZSlabNode
    | XSlabNode
    | YSlabNode
    | SampledSurfaceNode
    | PointNode
    | PolygonFaceNode
    | UnsupportedExpressionNode
)


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
    geometry_ineligible: int = 0


@dataclass(slots=True)
class FileAuditReport:
    source_file: str
    total_expressions: int
    supported_count: int
    recognized_unsupported_count: int
    unrecognized_count: int
    geometry_ineligible_count: int
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
                "geometry_ineligible": sum(r["geometry_ineligible_count"] for r in reports),
            },
            "blocked_files": [r["source_file"] for r in reports if r["status"] == AuditStatus.FAIL],
            "files": reports,
        }
