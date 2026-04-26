from __future__ import annotations

from desmos3d_pipeline.ir.models import (
    ClassificationStatus,
    ExpressionFamily,
    PolygonFaceNode,
    SourceRef,
    UnsupportedExpressionNode,
)
from desmos3d_pipeline.mesh.meshers import mesh_geometry_nodes, mesh_polygon_face


def _src() -> SourceRef:
    return SourceRef(source_file="x.json", expression_id="99", folder_id=None, folder_name=None, index=0)


def test_mesh_polygon_face_inline_triangle() -> None:
    node = PolygonFaceNode(
        node_type="polygon_face",
        source_ref=_src(),
        family=ExpressionFamily.POLYGON_CALL,
        status=ClassificationStatus.SUPPORTED,
        original_latex="",
        normalized_latex="",
        color=None,
        hidden=False,
        metadata={"resolved_symbols": {}, "python_symbol_map": {}},
        inline_vertices=[("0", "0", "0"), ("1", "0", "0"), ("0", "1", "0")],
    )
    mesh = mesh_polygon_face(node)
    assert len(mesh.vertices) == 3
    assert len(mesh.faces) == 1


def test_mesh_polygon_face_segment_thin_quad() -> None:
    node = PolygonFaceNode(
        node_type="polygon_face",
        source_ref=_src(),
        family=ExpressionFamily.SEGMENT_CALL,
        status=ClassificationStatus.SUPPORTED,
        original_latex="",
        normalized_latex="",
        color=None,
        hidden=False,
        metadata={"resolved_symbols": {}, "python_symbol_map": {}},
        inline_vertices=[("0", "0", "0"), ("2", "0", "0")],
    )
    mesh = mesh_polygon_face(node)
    assert len(mesh.vertices) == 4
    assert len(mesh.faces) == 2


def test_mesh_geometry_nodes_reports_unsupported_expression_node() -> None:
    node = UnsupportedExpressionNode(
        node_type="unsupported_expression",
        source_ref=_src(),
        family=ExpressionFamily.TRIANGLE_CALL,
        status=ClassificationStatus.RECOGNIZED_UNSUPPORTED,
        original_latex="",
        normalized_latex="",
        color=None,
        hidden=False,
        metadata={},
        unsupported_reason="triangle() not meshed",
        fingerprint="abc",
    )
    meshes, failures = mesh_geometry_nodes([node])
    assert meshes == []
    assert len(failures) == 1
    assert "UnsupportedExpressionNode" in failures[0]["error"]
