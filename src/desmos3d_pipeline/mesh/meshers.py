from __future__ import annotations

import math
import random
import re
from typing import Iterable

from desmos3d_pipeline.ir.models import (
    BoxVolumeNode,
    DiskExtrusionSolidNode,
    SphereSolidNode,
    GeometryNode,
    Mesh,
    ParametricTCurveNode,
    ParametricUVPatchNode,
    PlanePatchNode,
    PointNode,
    PolygonFaceNode,
    RangeConstraint,
    SampledSurfaceNode,
    UnsupportedExpressionNode,
    VerticalCylinderSurfaceNode,
    XSlabNode,
    YSlabNode,
    ZSlabNode,
)
from desmos3d_pipeline.parse.math_eval import safe_eval, to_python_expr


def _unit_perpendicular(d: tuple[float, float, float]) -> tuple[float, float, float]:
    dx, dy, dz = d
    n = math.sqrt(dx * dx + dy * dy + dz * dz)
    if n < 1e-12:
        return (1.0, 0.0, 0.0)
    cx, cy, cz = dy, -dx, 0.0
    cn = math.sqrt(cx * cx + cy * cy + cz * cz)
    if cn < 1e-9:
        cx, cy, cz = 0.0, dz, -dy
        cn = math.sqrt(cx * cx + cy * cy + cz * cz)
    if cn < 1e-9:
        return (1.0, 0.0, 0.0)
    return cx / cn, cy / cn, cz / cn


def _polar_hint_points_radial_sqrt(
    python_expr: str,
    rx0: float,
    rx1: float,
    ry0: float,
    ry1: float,
    env: dict[str, float],
    raw_restrictions: list[str],
    metadata,
) -> list[tuple[float, float]]:
    """Extra (x,y) samples near ``z=c-k*sqrt((x-cx)^2+(y-cy)^2)`` rings when uniform grids miss thin bands."""
    m_minus = re.search(r"\(x-([-+]?\d+(?:\.\d+)?)\)\*\*2", python_expr)
    m_plus = re.search(r"\(x\+([-+]?\d+(?:\.\d+)?)\)\*\*2", python_expr)
    my_minus = re.search(r"\(y-([-+]?\d+(?:\.\d+)?)\)\*\*2", python_expr)
    my_plus = re.search(r"\(y\+([-+]?\d+(?:\.\d+)?)\)\*\*2", python_expr)
    if m_minus:
        cx = float(m_minus.group(1))
    elif m_plus:
        cx = -float(m_plus.group(1))
    else:
        return []
    if my_minus:
        cy = float(my_minus.group(1))
    elif my_plus:
        cy = -float(my_plus.group(1))
    else:
        cy = 0.0
    out: list[tuple[float, float]] = []
    for r in (0.03, 0.06, 0.1, 0.15, 0.22, 0.3, 0.4, 0.55, 0.75, 1.0, 1.35, 1.8):
        for k in range(72):
            ang = 2.0 * math.pi * k / 72
            x = cx + r * math.cos(ang)
            y = cy + r * math.sin(ang)
            if not (rx0 <= x <= rx1 and ry0 <= y <= ry1):
                continue
            try:
                z = safe_eval(python_expr, {**env, "x": x, "y": y, "z": 0.0})
            except Exception:
                continue
            if _evaluate_restrictions(raw_restrictions, x, y, z, metadata):
                out.append((x, y))
    return out


def _adaptive_xy_bbox_for_restricted_surface(
    node: SampledSurfaceNode,
    python_expr: str,
    py_map: dict,
    env: dict[str, float],
    raw_restrictions: list[str],
) -> tuple[float, float, float, float] | None:
    """When the initial axis-aligned grid misses a thin feasible region, scan the viewport for valid (x,y)."""
    vp = node.metadata.get("viewport") or {}
    rx0 = float(vp.get("xmin", -50.0))
    rx1 = float(vp.get("xmax", 50.0))
    ry0 = float(vp.get("ymin", -50.0))
    ry1 = float(vp.get("ymax", 50.0))
    if rx0 >= rx1 or ry0 >= ry1:
        return None
    hits: list[tuple[float, float]] = []
    nx, ny = 96, 96
    for j in range(ny + 1):
        y = ry0 + (ry1 - ry0) * j / ny
        for i in range(nx + 1):
            x = rx0 + (rx1 - rx0) * i / nx
            try:
                z = safe_eval(python_expr, {**env, "x": x, "y": y, "z": 0.0})
            except Exception:
                continue
            if _evaluate_restrictions(raw_restrictions, x, y, z, node.metadata):
                hits.append((x, y))
    if len(hits) < 6:
        seed = hash((node.source_ref.expression_id or "", node.source_ref.source_file)) % (2**32)
        rng = random.Random(seed)
        for _ in range(50000):
            x = rx0 + (rx1 - rx0) * rng.random()
            y = ry0 + (ry1 - ry0) * rng.random()
            try:
                z = safe_eval(python_expr, {**env, "x": x, "y": y, "z": 0.0})
            except Exception:
                continue
            if _evaluate_restrictions(raw_restrictions, x, y, z, node.metadata):
                hits.append((x, y))
    if len(hits) < 30 and "sqrt" in python_expr:
        hits.extend(
            _polar_hint_points_radial_sqrt(python_expr, rx0, rx1, ry0, ry1, env, raw_restrictions, node.metadata)
        )
    if len(hits) < 4:
        return None
    mx0 = min(h[0] for h in hits)
    mx1 = max(h[0] for h in hits)
    my0 = min(h[1] for h in hits)
    my1 = max(h[1] for h in hits)
    dx = mx1 - mx0 or 1.0
    dy = my1 - my0 or 1.0
    padx = max(0.1 * dx, 0.02 * max(rx1 - rx0, ry1 - ry0))
    pady = max(0.1 * dy, 0.02 * max(rx1 - rx0, ry1 - ry0))
    return mx0 - padx, mx1 + padx, my0 - pady, my1 + pady


def _sampled_surface_vertices_and_faces(
    node: SampledSurfaceNode,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    xs: int,
    ys: int,
    python_expr: str,
    env: dict[str, float],
    py_map: dict,
    raw_restrictions: list[str],
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    verts: list[tuple[float, float, float]] = []
    for yi in range(ys + 1):
        y = y0 + (y1 - y0) * yi / ys
        for xi in range(xs + 1):
            x = x0 + (x1 - x0) * xi / xs
            z = safe_eval(python_expr, {**env, "x": x, "y": y, "z": 0.0})
            verts.append((x, y, z))
    faces: list[tuple[int, int, int]] = []
    stride = xs + 1
    valid = [[True for _ in range(xs + 1)] for _ in range(ys + 1)]
    if raw_restrictions:
        for yi in range(ys + 1):
            for xi in range(xs + 1):
                idx = yi * stride + xi
                x, y, z = verts[idx]
                valid[yi][xi] = _evaluate_restrictions(raw_restrictions, x, y, z, node.metadata)
    for yi in range(ys):
        for xi in range(xs):
            if not (valid[yi][xi] and valid[yi][xi + 1] and valid[yi + 1][xi] and valid[yi + 1][xi + 1]):
                continue
            a = yi * stride + xi + 1
            b = a + 1
            c = a + stride
            d = c + 1
            faces.append((a, b, d))
            faces.append((a, d, c))
    return verts, faces


def mesh_geometry_nodes(nodes: Iterable[GeometryNode]) -> tuple[list[Mesh], list[dict[str, str]]]:
    meshes: list[Mesh] = []
    failures: list[dict[str, str]] = []
    for node in nodes:
        try:
            if isinstance(node, PlanePatchNode):
                meshes.append(mesh_plane_patch(node))
            elif isinstance(node, BoxVolumeNode):
                meshes.append(mesh_box_volume(node))
            elif isinstance(node, DiskExtrusionSolidNode):
                meshes.append(mesh_disk_extrusion_solid(node))
            elif isinstance(node, SphereSolidNode):
                meshes.append(mesh_sphere_solid(node))
            elif isinstance(node, ZSlabNode):
                meshes.append(mesh_z_slab(node))
            elif isinstance(node, XSlabNode):
                meshes.append(mesh_x_slab(node))
            elif isinstance(node, YSlabNode):
                meshes.append(mesh_y_slab(node))
            elif isinstance(node, SampledSurfaceNode):
                meshes.append(mesh_sampled_surface(node))
            elif isinstance(node, VerticalCylinderSurfaceNode):
                meshes.append(mesh_vertical_cylinder_surface(node))
            elif isinstance(node, ParametricUVPatchNode):
                meshes.append(mesh_parametric_uv_patch(node))
            elif isinstance(node, ParametricTCurveNode):
                meshes.append(mesh_parametric_t_curve(node))
            elif isinstance(node, PointNode):
                meshes.append(mesh_point(node))
            elif isinstance(node, PolygonFaceNode):
                meshes.append(mesh_polygon_face(node))
            elif isinstance(node, UnsupportedExpressionNode):
                failures.append(
                    {
                        "source_file": node.source_ref.source_file,
                        "expression_id": str(node.source_ref.expression_id),
                        "error": f"UnsupportedExpressionNode: {node.unsupported_reason or 'not meshed'}",
                    }
                )
            else:
                failures.append(
                    {
                        "source_file": node.source_ref.source_file,
                        "expression_id": str(node.source_ref.expression_id),
                        "error": f"No mesher registered for {type(node).__name__}",
                    }
                )
        except Exception as exc:
            failures.append({"source_file": node.source_ref.source_file, "expression_id": str(node.source_ref.expression_id), "error": str(exc)})
    return meshes, failures


def mesh_polygon_face(node: PolygonFaceNode) -> Mesh:
    """Triangle-fan mesh from numeric inline vertices (expressions must eval as constants in graph env)."""
    if node.point_refs:
        raise ValueError("PolygonFaceNode point_refs are not yet resolved to coordinates")
    if len(node.inline_vertices) < 2:
        raise ValueError("PolygonFaceNode requires at least two inline vertices")
    py_map = node.metadata.get("python_symbol_map", {})
    env = dict(node.metadata.get("resolved_symbols", {}))
    env.setdefault("x", 0.0)
    env.setdefault("y", 0.0)
    env.setdefault("z", 0.0)
    verts: list[tuple[float, float, float]] = []
    for xs, ys, zs in node.inline_vertices:
        verts.append(
            (
                safe_eval(to_python_expr(xs, py_map), env),
                safe_eval(to_python_expr(ys, py_map), env),
                safe_eval(to_python_expr(zs, py_map), env),
            )
        )
    faces: list[tuple[int, int, int]]
    if len(verts) == 2:
        ax, ay, az = verts[0]
        bx, by, bz = verts[1]
        dx, dy, dz = bx - ax, by - ay, bz - az
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        h = max(1e-4, 0.012 * max(length, 1e-9))
        px, py, pz = _unit_perpendicular((dx, dy, dz))
        ox, oy, oz = px * h, py * h, pz * h
        verts = [
            (ax + ox, ay + oy, az + oz),
            (bx + ox, by + oy, bz + oz),
            (bx - ox, by - oy, bz - oz),
            (ax - ox, ay - oy, az - oz),
        ]
        faces = [(1, 2, 3), (1, 3, 4)]
    else:
        faces = []
        for i in range(1, len(verts) - 1):
            faces.append((1, i + 1, i + 2))
    return Mesh(
        name=_mesh_name(node),
        color=node.color,
        vertices=verts,
        faces=faces,
        source_file=node.source_ref.source_file,
        expression_id=node.source_ref.expression_id,
        family=node.family.value,
    )


def mesh_plane_patch(node: PlanePatchNode) -> Mesh:
    plane_value = _eval_expr(node.value, node.metadata)
    varying = [axis for axis in ("x", "y", "z") if axis != node.axis]
    try:
        resolved = _resolve_axis_bounds(node.bounds, node.metadata)
        a0, a1 = _require_bounds(resolved, varying[0])
        b0, b1 = _require_bounds(resolved, varying[1])
        verts = []
        for av, bv in ((a0, b0), (a1, b0), (a1, b1), (a0, b1)):
            coords = {node.axis: plane_value, varying[0]: av, varying[1]: bv}
            verts.append((coords["x"], coords["y"], coords["z"]))
        return Mesh(name=_mesh_name(node), color=node.color, vertices=verts, faces=[(1, 2, 3), (1, 3, 4)], source_file=node.source_ref.source_file, expression_id=node.source_ref.expression_id, family=node.family.value)
    except Exception:
        return _mesh_plane_patch_fallback(node, plane_value, varying)


def mesh_box_volume(node: BoxVolumeNode) -> Mesh:
    try:
        resolved = _resolve_axis_bounds(node.ranges, node.metadata)
        x0, x1 = _require_bounds(resolved, "x")
        y0, y1 = _require_bounds(resolved, "y")
        z0, z1 = _require_bounds(resolved, "z")
        verts = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        faces = [(1, 2, 3), (1, 3, 4), (5, 6, 7), (5, 7, 8), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 4, 8), (3, 8, 7), (4, 1, 5), (4, 5, 8)]
        return Mesh(name=_mesh_name(node), color=node.color, vertices=verts, faces=faces, source_file=node.source_ref.source_file, expression_id=node.source_ref.expression_id, family=node.family.value)
    except Exception:
        return _mesh_box_volume_voxel_fallback(node)


def mesh_disk_extrusion_solid(node: DiskExtrusionSolidNode) -> Mesh:
    """Voxel surface of (u-cu)^2+(v-cv)^2<=radius_sq clipped to the node bbox."""

    def _inside(x: float, y: float, z: float) -> bool:
        u = {"x": x, "y": y, "z": z}[node.axis_u]
        v = {"x": x, "y": y, "z": z}[node.axis_v]
        du = u - node.center_u
        dv = v - node.center_v
        return du * du + dv * dv <= node.radius_sq + 1e-9

    x0, x1 = node.x_min, node.x_max
    y0, y1 = node.y_min, node.y_max
    z0, z1 = node.z_min, node.z_max
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    vol_scale = max((max(dx, 1e-9) * max(dy, 1e-9) * max(dz, 1e-9)) ** (1.0 / 3.0), 1e-9)
    r = math.sqrt(max(node.radius_sq, 1e-15))
    # Finer grid when the disk radius is small relative to the bbox (small solid in a large box).
    nx = ny = nz = int(max(10, min(48, round(10 + 34 * r / vol_scale))))
    filled = [[[False for _ in range(nz)] for _ in range(ny)] for _ in range(nx)]

    for i in range(nx):
        x = x0 + (x1 - x0) * (i + 0.5) / nx
        for j in range(ny):
            y = y0 + (y1 - y0) * (j + 0.5) / ny
            for k in range(nz):
                z = z0 + (z1 - z0) * (k + 0.5) / nz
                if _inside(x, y, z):
                    filled[i][j][k] = True

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    vertex_map: dict[tuple[float, float, float], int] = {}

    def vid(v: tuple[float, float, float]) -> int:
        key = (round(v[0], 6), round(v[1], 6), round(v[2], 6))
        idx = vertex_map.get(key)
        if idx is None:
            vertices.append(v)
            idx = len(vertices)
            vertex_map[key] = idx
        return idx

    def add_quad(a, b, c, d):
        ia, ib, ic, id_ = vid(a), vid(b), vid(c), vid(d)
        faces.append((ia, ib, ic))
        faces.append((ia, ic, id_))

    for i in range(nx):
        xa = x0 + (x1 - x0) * i / nx
        xb = x0 + (x1 - x0) * (i + 1) / nx
        for j in range(ny):
            ya = y0 + (y1 - y0) * j / ny
            yb = y0 + (y1 - y0) * (j + 1) / ny
            for k in range(nz):
                if not filled[i][j][k]:
                    continue
                za = z0 + (z1 - z0) * k / nz
                zb = z0 + (z1 - z0) * (k + 1) / nz
                neighbors = [
                    (i - 1, j, k, ((xa, ya, za), (xa, yb, za), (xa, yb, zb), (xa, ya, zb))),
                    (i + 1, j, k, ((xb, ya, za), (xb, ya, zb), (xb, yb, zb), (xb, yb, za))),
                    (i, j - 1, k, ((xa, ya, za), (xb, ya, za), (xb, ya, zb), (xa, ya, zb))),
                    (i, j + 1, k, ((xa, yb, za), (xa, yb, zb), (xb, yb, zb), (xb, yb, za))),
                    (i, j, k - 1, ((xa, ya, za), (xa, yb, za), (xb, yb, za), (xb, ya, za))),
                    (i, j, k + 1, ((xa, ya, zb), (xb, ya, zb), (xb, yb, zb), (xa, yb, zb))),
                ]
                for ni, nj, nk, quad in neighbors:
                    if ni < 0 or nj < 0 or nk < 0 or ni >= nx or nj >= ny or nk >= nz or not filled[ni][nj][nk]:
                        add_quad(*quad)

    if not faces:
        raise ValueError("disk extrusion voxel mesher produced no faces")
    return Mesh(
        name=_mesh_name(node),
        color=node.color,
        vertices=vertices,
        faces=faces,
        source_file=node.source_ref.source_file,
        expression_id=node.source_ref.expression_id,
        family=node.family.value,
    )


def mesh_sphere_solid(node: SphereSolidNode) -> Mesh:
    """Voxel surface of ``(x-cx)^2+(y-cy)^2+(z-cz)^2<=radius_sq`` clipped to the node bbox."""

    def _inside(x: float, y: float, z: float) -> bool:
        dx = x - node.center_x
        dy = y - node.center_y
        dz = z - node.center_z
        return dx * dx + dy * dy + dz * dz <= node.radius_sq + 1e-9

    x0, x1 = node.x_min, node.x_max
    y0, y1 = node.y_min, node.y_max
    z0, z1 = node.z_min, node.z_max
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    vol_scale = max((max(dx, 1e-9) * max(dy, 1e-9) * max(dz, 1e-9)) ** (1.0 / 3.0), 1e-9)
    r = math.sqrt(max(node.radius_sq, 1e-15))
    nx = ny = nz = int(max(10, min(48, round(10 + 34 * r / vol_scale))))
    filled = [[[False for _ in range(nz)] for _ in range(ny)] for _ in range(nx)]

    for i in range(nx):
        x = x0 + (x1 - x0) * (i + 0.5) / nx
        for j in range(ny):
            y = y0 + (y1 - y0) * (j + 0.5) / ny
            for k in range(nz):
                z = z0 + (z1 - z0) * (k + 0.5) / nz
                if _inside(x, y, z):
                    filled[i][j][k] = True

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    vertex_map: dict[tuple[float, float, float], int] = {}

    def vid(v: tuple[float, float, float]) -> int:
        key = (round(v[0], 6), round(v[1], 6), round(v[2], 6))
        idx = vertex_map.get(key)
        if idx is None:
            vertices.append(v)
            idx = len(vertices)
            vertex_map[key] = idx
        return idx

    def add_quad(a, b, c, d):
        ia, ib, ic, id_ = vid(a), vid(b), vid(c), vid(d)
        faces.append((ia, ib, ic))
        faces.append((ia, ic, id_))

    for i in range(nx):
        xa = x0 + (x1 - x0) * i / nx
        xb = x0 + (x1 - x0) * (i + 1) / nx
        for j in range(ny):
            ya = y0 + (y1 - y0) * j / ny
            yb = y0 + (y1 - y0) * (j + 1) / ny
            for k in range(nz):
                if not filled[i][j][k]:
                    continue
                za = z0 + (z1 - z0) * k / nz
                zb = z0 + (z1 - z0) * (k + 1) / nz
                neighbors = [
                    (i - 1, j, k, ((xa, ya, za), (xa, yb, za), (xa, yb, zb), (xa, ya, zb))),
                    (i + 1, j, k, ((xb, ya, za), (xb, ya, zb), (xb, yb, zb), (xb, yb, za))),
                    (i, j - 1, k, ((xa, ya, za), (xb, ya, za), (xb, ya, zb), (xa, ya, zb))),
                    (i, j + 1, k, ((xa, yb, za), (xa, yb, zb), (xb, yb, zb), (xb, yb, za))),
                    (i, j, k - 1, ((xa, ya, za), (xa, yb, za), (xb, yb, za), (xb, ya, za))),
                    (i, j, k + 1, ((xa, ya, zb), (xb, ya, zb), (xb, yb, zb), (xa, yb, zb))),
                ]
                for ni, nj, nk, quad in neighbors:
                    if ni < 0 or nj < 0 or nk < 0 or ni >= nx or nj >= ny or nk >= nz or not filled[ni][nj][nk]:
                        add_quad(*quad)

    if not faces:
        raise ValueError("sphere solid voxel mesher produced no faces")
    return Mesh(
        name=_mesh_name(node),
        color=node.color,
        vertices=vertices,
        faces=faces,
        source_file=node.source_ref.source_file,
        expression_id=node.source_ref.expression_id,
        family=node.family.value,
    )


def mesh_sampled_surface(node: SampledSurfaceNode) -> Mesh:
    resolved = _resolve_axis_bounds(
        node.bounds,
        node.metadata,
        include_axes={"x", "y"},
        tolerate_unresolved=True,
    )
    if node.dependent_axis != "z":
        raise ValueError("Only z=f(x,y) sampled surfaces are supported in this phase")
    x0, x1 = _require_bounds_or_infer(resolved, "x", node)
    y0, y1 = _require_bounds_or_infer(resolved, "y", node)
    xs, ys = node.sampling_hint
    py_map = node.metadata.get("python_symbol_map", {})
    python_expr = to_python_expr(node.function_expr, py_map)
    env = dict(node.metadata.get("resolved_symbols", {}))
    raw_restrictions: list[str] = node.metadata.get("raw_restrictions", [])
    if raw_restrictions:
        xs, ys = max(xs, 96), max(ys, 96)
    verts, faces = _sampled_surface_vertices_and_faces(
        node, x0, x1, y0, y1, xs, ys, python_expr, env, py_map, raw_restrictions
    )
    if not faces and raw_restrictions:
        bbox = _adaptive_xy_bbox_for_restricted_surface(node, python_expr, py_map, env, raw_restrictions)
        if bbox is not None:
            ax0, ax1, ay0, ay1 = bbox
            for dense in (192, 320, 480):
                verts, faces = _sampled_surface_vertices_and_faces(
                    node, ax0, ax1, ay0, ay1, max(xs, dense), max(ys, dense), python_expr, env, py_map, raw_restrictions
                )
                if faces:
                    break
    if not faces:
        raise ValueError("No valid sampled cells after applying restrictions")
    return Mesh(name=_mesh_name(node), color=node.color, vertices=verts, faces=faces, source_file=node.source_ref.source_file, expression_id=node.source_ref.expression_id, family=node.family.value)


def mesh_z_slab(node: ZSlabNode) -> Mesh:
    resolved = _resolve_axis_bounds(
        node.bounds,
        node.metadata,
        include_axes={"x", "y"},
        tolerate_unresolved=True,
    )
    x0, x1 = _require_bounds_or_viewport(resolved, "x", node.metadata)
    y0, y1 = _require_bounds_or_viewport(resolved, "y", node.metadata)
    xs, ys = node.sampling_hint
    py_map = node.metadata.get("python_symbol_map", {})
    env = dict(node.metadata.get("resolved_symbols", {}))
    lower_py = to_python_expr(node.lower_expr, py_map)
    upper_py = to_python_expr(node.upper_expr, py_map)

    lower_verts: list[tuple[float, float, float]] = []
    upper_verts: list[tuple[float, float, float]] = []
    for yi in range(ys + 1):
        y = y0 + (y1 - y0) * yi / ys
        for xi in range(xs + 1):
            x = x0 + (x1 - x0) * xi / xs
            ev = {**env, "x": x, "y": y, "z": 0.0}
            zlo = safe_eval(lower_py, ev, clamp_sqrt=True)
            zhi = safe_eval(upper_py, ev, clamp_sqrt=True)
            if zlo > zhi:
                zlo, zhi = zhi, zlo
            lower_verts.append((x, y, zlo))
            upper_verts.append((x, y, zhi))

    verts = lower_verts + upper_verts
    faces: list[tuple[int, int, int]] = []
    stride = xs + 1
    base_upper = len(lower_verts)

    # Bottom and top surfaces
    for yi in range(ys):
        for xi in range(xs):
            a = yi * stride + xi
            b = a + 1
            c = a + stride
            d = c + 1
            # bottom (wind consistent)
            faces.append((a + 1, c + 1, d + 1))
            faces.append((a + 1, d + 1, b + 1))
            # top
            au = base_upper + a
            bu = base_upper + b
            cu = base_upper + c
            du = base_upper + d
            faces.append((au + 1, bu + 1, du + 1))
            faces.append((au + 1, du + 1, cu + 1))

    # Side walls around perimeter
    def quad(i0, i1, j0, j1):
        faces.append((i0 + 1, j0 + 1, j1 + 1))
        faces.append((i0 + 1, j1 + 1, i1 + 1))

    # y = y0 edge
    for xi in range(xs):
        a = 0 * stride + xi
        b = a + 1
        quad(a, b, base_upper + a, base_upper + b)
    # y = y1 edge
    for xi in range(xs):
        a = ys * stride + xi
        b = a + 1
        quad(b, a, base_upper + b, base_upper + a)
    # x = x0 edge
    for yi in range(ys):
        a = yi * stride + 0
        b = (yi + 1) * stride + 0
        quad(b, a, base_upper + b, base_upper + a)
    # x = x1 edge
    for yi in range(ys):
        a = yi * stride + xs
        b = (yi + 1) * stride + xs
        quad(a, b, base_upper + a, base_upper + b)

    return Mesh(name=_mesh_name(node), color=node.color, vertices=verts, faces=faces, source_file=node.source_ref.source_file, expression_id=node.source_ref.expression_id, family=node.family.value)


def mesh_x_slab(node: XSlabNode) -> Mesh:
    # Sample x-lower/x-upper across (y,z) grid within finite y/z bounds.
    resolved = _resolve_axis_bounds(
        node.bounds,
        node.metadata,
        include_axes={"y", "z"},
        tolerate_unresolved=False,
    )
    y0, y1 = _require_bounds(resolved, "y")
    z0, z1 = _require_bounds(resolved, "z")
    ys, zs = node.sampling_hint

    py_map = node.metadata.get("python_symbol_map", {})
    env = dict(node.metadata.get("resolved_symbols", {}))
    lower_py = to_python_expr(node.lower_expr, py_map)
    upper_py = to_python_expr(node.upper_expr, py_map)

    lower_verts: list[tuple[float, float, float]] = []
    upper_verts: list[tuple[float, float, float]] = []
    for zi in range(zs + 1):
        z = z0 + (z1 - z0) * zi / zs
        for yi in range(ys + 1):
            y = y0 + (y1 - y0) * yi / ys
            xlo = safe_eval(lower_py, {**env, "x": 0.0, "y": y, "z": z})
            xhi = safe_eval(upper_py, {**env, "x": 0.0, "y": y, "z": z})
            if xlo > xhi:
                xlo, xhi = xhi, xlo
            lower_verts.append((xlo, y, z))
            upper_verts.append((xhi, y, z))

    verts = lower_verts + upper_verts
    faces: list[tuple[int, int, int]] = []
    stride = ys + 1
    base_upper = len(lower_verts)

    # Left and right surfaces (x = lower/upper)
    for zi in range(zs):
        for yi in range(ys):
            a = zi * stride + yi
            b = a + 1
            c = a + stride
            d = c + 1
            # lower surface
            faces.append((a + 1, c + 1, d + 1))
            faces.append((a + 1, d + 1, b + 1))
            # upper surface
            au = base_upper + a
            bu = base_upper + b
            cu = base_upper + c
            du = base_upper + d
            faces.append((au + 1, bu + 1, du + 1))
            faces.append((au + 1, du + 1, cu + 1))

    # Side walls around perimeter in (y,z)
    def quad(i0, i1, j0, j1):
        faces.append((i0 + 1, j0 + 1, j1 + 1))
        faces.append((i0 + 1, j1 + 1, i1 + 1))

    # z = z0 edge
    for yi in range(ys):
        a = 0 * stride + yi
        b = a + 1
        quad(a, b, base_upper + a, base_upper + b)
    # z = z1 edge
    for yi in range(ys):
        a = zs * stride + yi
        b = a + 1
        quad(b, a, base_upper + b, base_upper + a)
    # y = y0 edge
    for zi in range(zs):
        a = zi * stride + 0
        b = (zi + 1) * stride + 0
        quad(b, a, base_upper + b, base_upper + a)
    # y = y1 edge
    for zi in range(zs):
        a = zi * stride + ys
        b = (zi + 1) * stride + ys
        quad(a, b, base_upper + a, base_upper + b)

    return Mesh(name=_mesh_name(node), color=node.color, vertices=verts, faces=faces, source_file=node.source_ref.source_file, expression_id=node.source_ref.expression_id, family=node.family.value)


def mesh_y_slab(node: YSlabNode) -> Mesh:
    # Sample y-lower/y-upper across (x,z) grid within finite x bounds and a z range.
    resolved = _resolve_axis_bounds(
        node.bounds,
        node.metadata,
        include_axes={"x", "y", "z"},
        tolerate_unresolved=True,
    )
    x0, x1 = _require_bounds(resolved, "x")
    y_clip = None
    y_bounds = resolved.get("y")
    if y_bounds and y_bounds["lower"] is not None and y_bounds["upper"] is not None:
        y_clip = (y_bounds["lower"], y_bounds["upper"])
    z_bounds = resolved.get("z")
    if z_bounds and z_bounds["lower"] is not None and z_bounds["upper"] is not None:
        z0, z1 = z_bounds["lower"], z_bounds["upper"]
    else:
        viewport = node.metadata.get("viewport", {})
        if "zmin" not in viewport or "zmax" not in viewport:
            raise ValueError("Missing finite bounds for axis z")
        z0, z1 = float(viewport["zmin"]), float(viewport["zmax"])

    xs, zs = node.sampling_hint

    py_map = node.metadata.get("python_symbol_map", {})
    env = dict(node.metadata.get("resolved_symbols", {}))
    lower_py = to_python_expr(node.lower_expr, py_map)
    upper_py = to_python_expr(node.upper_expr, py_map)

    lower_verts: list[tuple[float, float, float]] = []
    upper_verts: list[tuple[float, float, float]] = []
    valid: list[bool] = []
    for zi in range(zs + 1):
        z = z0 + (z1 - z0) * zi / zs
        for xi in range(xs + 1):
            x = x0 + (x1 - x0) * xi / xs
            ylo = safe_eval(lower_py, {**env, "x": x, "y": 0.0, "z": z})
            yhi = safe_eval(upper_py, {**env, "x": x, "y": 0.0, "z": z})
            if ylo > yhi:
                ylo, yhi = yhi, ylo
            if y_clip is not None:
                c0, c1 = y_clip
                ylo = max(ylo, c0)
                yhi = min(yhi, c1)
            is_valid = ylo < yhi
            valid.append(is_valid)
            lower_verts.append((x, ylo, z))
            upper_verts.append((x, yhi, z))

    verts = lower_verts + upper_verts
    faces: list[tuple[int, int, int]] = []
    stride = xs + 1
    base_upper = len(lower_verts)

    # Lower and upper surfaces
    for zi in range(zs):
        for xi in range(xs):
            a = zi * stride + xi
            b = a + 1
            c = a + stride
            d = c + 1
            if not (valid[a] and valid[b] and valid[c] and valid[d]):
                continue
            # lower surface
            faces.append((a + 1, c + 1, d + 1))
            faces.append((a + 1, d + 1, b + 1))
            # upper surface
            au = base_upper + a
            bu = base_upper + b
            cu = base_upper + c
            du = base_upper + d
            faces.append((au + 1, bu + 1, du + 1))
            faces.append((au + 1, du + 1, cu + 1))

    # Side walls around perimeter in (x,z)
    def quad(i0, i1, j0, j1):
        faces.append((i0 + 1, j0 + 1, j1 + 1))
        faces.append((i0 + 1, j1 + 1, i1 + 1))

    # z = z0 edge
    for xi in range(xs):
        a = 0 * stride + xi
        b = a + 1
        if not (valid[a] and valid[b]):
            continue
        quad(a, b, base_upper + a, base_upper + b)
    # z = z1 edge
    for xi in range(xs):
        a = zs * stride + xi
        b = a + 1
        if not (valid[a] and valid[b]):
            continue
        quad(b, a, base_upper + b, base_upper + a)
    # x = x0 edge
    for zi in range(zs):
        a = zi * stride + 0
        b = (zi + 1) * stride + 0
        if not (valid[a] and valid[b]):
            continue
        quad(b, a, base_upper + b, base_upper + a)
    # x = x1 edge
    for zi in range(zs):
        a = zi * stride + xs
        b = (zi + 1) * stride + xs
        if not (valid[a] and valid[b]):
            continue
        quad(a, b, base_upper + a, base_upper + b)

    if not faces:
        raise ValueError("No valid sampled cells after applying y-slab clipping")

    return Mesh(name=_mesh_name(node), color=node.color, vertices=verts, faces=faces, source_file=node.source_ref.source_file, expression_id=node.source_ref.expression_id, family=node.family.value)


def mesh_point(node: PointNode) -> Mesh:
    x = _eval_expr(node.x, node.metadata)
    y = _eval_expr(node.y, node.metadata)
    z = _eval_expr(node.z, node.metadata)
    viewport = node.metadata.get("viewport", {})
    span = max(
        abs(viewport.get("xmax", 100.0) - viewport.get("xmin", -100.0)),
        abs(viewport.get("ymax", 100.0) - viewport.get("ymin", -100.0)),
        abs(viewport.get("zmax", 100.0) - viewport.get("zmin", -100.0)),
    )
    radius = max(span * 0.004, 0.08)
    height = radius * 3.0
    segments = 14
    cone_color = "#6042a6" if (node.color or "").lower() == "#6042a6" else "#ffffff"

    verts: list[tuple[float, float, float]] = [(x, y, z + height)]
    for i in range(segments):
        a = 2.0 * 3.141592653589793 * i / segments
        verts.append((x + radius * __import__("math").cos(a), y + radius * __import__("math").sin(a), z))
    verts.append((x, y, z))

    tip_idx = 1
    base_center_idx = len(verts)
    faces: list[tuple[int, int, int]] = []
    for i in range(segments):
        a = 2 + i
        b = 2 + ((i + 1) % segments)
        faces.append((tip_idx, a, b))
    for i in range(segments):
        a = 2 + i
        b = 2 + ((i + 1) % segments)
        faces.append((base_center_idx, b, a))

    return Mesh(
        name=_mesh_name(node),
        color=cone_color,
        vertices=verts,
        faces=faces,
        source_file=node.source_ref.source_file,
        expression_id=node.source_ref.expression_id,
        family=node.family.value,
    )


def _resolve_axis_bounds(
    ranges: list[RangeConstraint],
    metadata,
    include_axes: set[str] | None = None,
    tolerate_unresolved: bool = False,
):
    resolved_symbols = dict(metadata.get("resolved_symbols", {}))
    symbol_map = metadata.get("python_symbol_map", {})
    for axis, expr in metadata.get("fixed_axes", {}).items():
        resolved_symbols[axis] = safe_eval(to_python_expr(expr, symbol_map), resolved_symbols)
    by_axis = {}
    for constraint in ranges:
        if include_axes is not None and constraint.axis not in include_axes:
            continue
        try:
            lower = _eval_optional(constraint.lower, resolved_symbols, symbol_map)
            upper = _eval_optional(constraint.upper, resolved_symbols, symbol_map)
        except Exception:
            if tolerate_unresolved:
                continue
            raise
        current = by_axis.setdefault(constraint.axis, {"lower": None, "upper": None})
        if lower is not None:
            current["lower"] = lower if current["lower"] is None else max(current["lower"], lower)
        if upper is not None:
            current["upper"] = upper if current["upper"] is None else min(current["upper"], upper)
    return by_axis


def _require_bounds(bounds, axis):
    axis_bounds = bounds.get(axis)
    if axis_bounds is None or axis_bounds["lower"] is None or axis_bounds["upper"] is None:
        raise ValueError(f"Missing finite bounds for axis {axis}")
    return axis_bounds["lower"], axis_bounds["upper"]


def _require_bounds_or_viewport(bounds, axis: str, metadata) -> tuple[float, float]:
    """Finite interval on ``axis`` from resolved constraints, using graph viewport for open ends."""
    axis_bounds = bounds.get(axis) or {"lower": None, "upper": None}
    lo, hi = axis_bounds.get("lower"), axis_bounds.get("upper")
    vp = metadata.get("viewport") or {}
    kmin, kmax = f"{axis}min", f"{axis}max"
    if lo is None:
        if kmin not in vp:
            raise ValueError(f"Missing finite lower bound for axis {axis} and no viewport")
        lo = float(vp[kmin])
    if hi is None:
        if kmax not in vp:
            raise ValueError(f"Missing finite upper bound for axis {axis} and no viewport")
        hi = float(vp[kmax])
    if lo >= hi:
        mid = (lo + hi) / 2.0
        pad = max(1e-3, abs(mid) * 1e-6 + 1e-3)
        lo, hi = mid - pad, mid + pad
    return lo, hi


def _require_bounds_or_infer(bounds, axis: str, node: SampledSurfaceNode):
    axis_bounds = bounds.get(axis) or {"lower": None, "upper": None}
    lo, hi = axis_bounds.get("lower"), axis_bounds.get("upper")
    if lo is not None and hi is not None:
        return float(lo), float(hi)

    inferred = _infer_axis_bounds_from_z_constraints(node, axis, bounds)
    if inferred is not None:
        return inferred
    merged = {**bounds, axis: {"lower": lo, "upper": hi}}
    return _require_bounds_or_viewport(merged, axis, node.metadata)


def _infer_axis_bounds_from_z_constraints(node: SampledSurfaceNode, missing_axis: str, known_bounds):
    viewport = node.metadata.get("viewport", {})
    py_symbol_map = node.metadata.get("python_symbol_map", {})
    env_base = dict(node.metadata.get("resolved_symbols", {}))
    z_expr = to_python_expr(node.function_expr, py_symbol_map)
    z_constraints = [c for c in node.bounds if c.axis == "z"]
    if not z_constraints:
        return None

    vmin_key, vmax_key = f"{missing_axis}min", f"{missing_axis}max"
    if vmin_key not in viewport or vmax_key not in viewport:
        return None
    scan_min, scan_max = viewport[vmin_key], viewport[vmax_key]
    if scan_min >= scan_max:
        return None

    other_axis = "y" if missing_axis == "x" else "x"
    other = known_bounds.get(other_axis)
    if other and other["lower"] is not None and other["upper"] is not None:
        other_value = (other["lower"] + other["upper"]) / 2.0
    else:
        ovmin_key, ovmax_key = f"{other_axis}min", f"{other_axis}max"
        if ovmin_key not in viewport or ovmax_key not in viewport:
            return None
        other_value = (viewport[ovmin_key] + viewport[ovmax_key]) / 2.0

    valid: list[float] = []
    steps = 240
    for idx in range(steps + 1):
        candidate = scan_min + (scan_max - scan_min) * idx / steps
        env = dict(env_base)
        env[missing_axis] = candidate
        env[other_axis] = other_value
        env["z"] = 0.0
        try:
            z_val = safe_eval(z_expr, env)
        except Exception:
            continue
        ok = True
        for c in z_constraints:
            try:
                lower = _eval_optional(c.lower, env, py_symbol_map)
                upper = _eval_optional(c.upper, env, py_symbol_map)
            except Exception:
                ok = False
                break
            if lower is not None and z_val < lower:
                ok = False
                break
            if upper is not None and z_val > upper:
                ok = False
                break
        if ok:
            valid.append(candidate)
    if not valid:
        return None
    return min(valid), max(valid)


def _eval_optional(expr: str | None, resolved_symbols, symbol_map):
    if expr is None:
        return None
    return safe_eval(to_python_expr(expr, symbol_map), resolved_symbols)


def _eval_expr(expr: str, metadata) -> float:
    resolved_symbols = metadata.get("resolved_symbols", {})
    symbol_map = metadata.get("python_symbol_map", {})
    return safe_eval(to_python_expr(expr, symbol_map), resolved_symbols)


def mesh_vertical_cylinder_surface(node: VerticalCylinderSurfaceNode) -> Mesh:
    u, v = node.axis_u, node.axis_v
    w = node.extrusion_axis
    if {u, v, w} != {"x", "y", "z"}:
        raise ValueError("Cylinder disk axes and extrusion axis must be a permutation of x,y,z")
    r = math.sqrt(max(node.radius_sq, 1e-15))
    cu, cv = node.center_u, node.center_v
    su = max(abs(node.stretch_u), 1e-15)
    sv = max(abs(node.stretch_v), 1e-15)
    w0, w1 = node.z_min, node.z_max
    nt, nz = node.theta_segments, node.z_segments
    verts: list[tuple[float, float, float]] = []
    for j in range(nz + 1):
        wv = w0 + (w1 - w0) * j / nz
        for i in range(nt):
            th = 2 * math.pi * i / nt
            du = (r / su) * math.cos(th)
            dv = (r / sv) * math.sin(th)
            pos = {"x": 0.0, "y": 0.0, "z": 0.0}
            pos[u] = cu + du
            pos[v] = cv + dv
            pos[w] = wv
            verts.append((pos["x"], pos["y"], pos["z"]))
    faces: list[tuple[int, int, int]] = []
    for j in range(nz):
        for i in range(nt):
            a = j * nt + i + 1
            b = j * nt + ((i + 1) % nt) + 1
            d = (j + 1) * nt + i + 1
            c = (j + 1) * nt + ((i + 1) % nt) + 1
            faces.append((a, b, d))
            faces.append((a, d, c))
    return Mesh(
        name=_mesh_name(node),
        color=node.color,
        vertices=verts,
        faces=faces,
        source_file=node.source_ref.source_file,
        expression_id=node.source_ref.expression_id,
        family=node.family.value,
    )


def mesh_parametric_uv_patch(node: ParametricUVPatchNode) -> Mesh:
    py_map = node.metadata.get("python_symbol_map", {})
    env0 = dict(node.metadata.get("resolved_symbols", {}))
    x_py = to_python_expr(node.x_expr, py_map)
    y_py = to_python_expr(node.y_expr, py_map)
    z_py = to_python_expr(node.z_expr, py_map)
    nu, nv = node.u_segments, node.v_segments
    verts: list[tuple[float, float, float]] = []
    for j in range(nv + 1):
        vv = j / nv if nv else 0.0
        for i in range(nu + 1):
            uu = i / nu if nu else 0.0
            env = {**env0, "u": uu, "v": vv, "t": 0.0, "x": 0.0, "y": 0.0, "z": 0.0}
            xv = safe_eval(x_py, env)
            yv = safe_eval(y_py, env)
            zv = safe_eval(z_py, env)
            verts.append((xv, yv, zv))
    faces: list[tuple[int, int, int]] = []
    stride = nu + 1
    for j in range(nv):
        for i in range(nu):
            a = j * stride + i + 1
            b = a + 1
            c = a + stride
            d = c + 1
            faces.append((a, b, d))
            faces.append((a, d, c))
    if not faces:
        raise ValueError("Empty parametric UV mesh")
    return Mesh(
        name=_mesh_name(node),
        color=node.color,
        vertices=verts,
        faces=faces,
        source_file=node.source_ref.source_file,
        expression_id=node.source_ref.expression_id,
        family=node.family.value,
    )


def _vec_sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_len(v: tuple[float, float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _vec_scale(v: tuple[float, float, float], s: float) -> tuple[float, float, float]:
    return (v[0] * s, v[1] * s, v[2] * s)


def _vec_norm(v: tuple[float, float, float]) -> tuple[float, float, float]:
    ln = _vec_len(v)
    if ln < 1e-12:
        return (0.0, 0.0, 1.0)
    return (v[0] / ln, v[1] / ln, v[2] / ln)


def _vec_cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _orthonormal_frame(d: tuple[float, float, float]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    d = _vec_norm(d)
    aux = (1.0, 0.0, 0.0) if abs(d[0]) < 0.9 else (0.0, 1.0, 0.0)
    e1 = _vec_norm(_vec_cross(d, aux))
    e2 = _vec_norm(_vec_cross(d, e1))
    return e1, e2


def mesh_parametric_t_curve(node: ParametricTCurveNode) -> Mesh:
    py_map = node.metadata.get("python_symbol_map", {})
    env0 = dict(node.metadata.get("resolved_symbols", {}))
    x_py = to_python_expr(node.x_expr, py_map)
    y_py = to_python_expr(node.y_expr, py_map)
    z_py = to_python_expr(node.z_expr, py_map)
    viewport = node.metadata.get("viewport", {})
    span = max(
        abs(float(viewport.get("xmax", 100.0)) - float(viewport.get("xmin", -100.0))),
        abs(float(viewport.get("ymax", 100.0)) - float(viewport.get("ymin", -100.0))),
        abs(float(viewport.get("zmax", 100.0)) - float(viewport.get("zmin", -100.0))),
    )
    radius = max(span * 0.003, 0.05)
    n = max(2, node.segments)
    pts: list[tuple[float, float, float]] = []
    for i in range(n):
        tt = i / (n - 1) if n > 1 else 0.0
        env = {**env0, "t": tt, "u": tt, "v": tt, "x": 0.0, "y": 0.0, "z": 0.0}
        pts.append(
            (
                float(safe_eval(x_py, env)),
                float(safe_eval(y_py, env)),
                float(safe_eval(z_py, env)),
            )
        )
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    def add_quad(i0: int, i1: int, i2: int, i3: int) -> None:
        faces.append((i0, i1, i2))
        faces.append((i0, i2, i3))

    for k in range(len(pts) - 1):
        a, b = pts[k], pts[k + 1]
        d = _vec_sub(b, a)
        ln = _vec_len(d)
        if ln < 1e-9:
            continue
        d = _vec_norm(d)
        e1, e2 = _orthonormal_frame(d)
        r1 = _vec_scale(e1, radius)
        r2 = _vec_scale(e2, radius)
        corners_a = (
            (a[0] + r1[0] + r2[0], a[1] + r1[1] + r2[1], a[2] + r1[2] + r2[2]),
            (a[0] + r1[0] - r2[0], a[1] + r1[1] - r2[1], a[2] + r1[2] - r2[2]),
            (a[0] - r1[0] - r2[0], a[1] - r1[1] - r2[1], a[2] - r1[2] - r2[2]),
            (a[0] - r1[0] + r2[0], a[1] - r1[1] + r2[1], a[2] - r1[2] + r2[2]),
        )
        corners_b = (
            (b[0] + r1[0] + r2[0], b[1] + r1[1] + r2[1], b[2] + r1[2] + r2[2]),
            (b[0] + r1[0] - r2[0], b[1] + r1[1] - r2[1], b[2] + r1[2] - r2[2]),
            (b[0] - r1[0] - r2[0], b[1] - r1[1] - r2[1], b[2] - r1[2] - r2[2]),
            (b[0] - r1[0] + r2[0], b[1] - r1[1] + r2[1], b[2] - r1[2] + r2[2]),
        )
        base = len(verts)
        verts.extend(corners_a)
        verts.extend(corners_b)
        # corners_a: 0..3, corners_b: 4..7 -> 1-based indices base+1 .. base+8
        o = base + 1
        add_quad(o + 0, o + 1, o + 2, o + 3)
        add_quad(o + 4, o + 7, o + 6, o + 5)
        add_quad(o + 0, o + 4, o + 5, o + 1)
        add_quad(o + 1, o + 5, o + 6, o + 2)
        add_quad(o + 2, o + 6, o + 7, o + 3)
        add_quad(o + 3, o + 7, o + 4, o + 0)
    if not faces:
        raise ValueError("Empty parametric curve mesh")
    return Mesh(
        name=_mesh_name(node),
        color=node.color,
        vertices=verts,
        faces=faces,
        source_file=node.source_ref.source_file,
        expression_id=node.source_ref.expression_id,
        family=node.family.value,
    )


def _mesh_name(node: GeometryNode) -> str:
    expr_id = node.source_ref.expression_id or str(node.source_ref.index)
    return f"{node.source_ref.source_file.rsplit('.', 1)[0]}_{expr_id}"


def _mesh_box_volume_voxel_fallback(node: BoxVolumeNode) -> Mesh:
    bbox = _estimate_bbox(node)
    if bbox is None:
        raise ValueError("Unable to estimate bounding box for voxel fallback")
    sampled_bbox = _sample_occupied_bbox(node, bbox, samples=10)
    if sampled_bbox is not None:
        bbox = sampled_bbox

    for resolution in (16, 24, 32):
        mesh = _voxelize_box_node(node, bbox, resolution)
        if mesh is not None:
            return mesh
    raise ValueError("Voxel fallback produced no faces")


def _estimate_bbox(node: BoxVolumeNode) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None:
    viewport = node.metadata.get("viewport", {})
    bounds = {
        "x": [viewport.get("xmin", -100.0), viewport.get("xmax", 100.0)],
        "y": [viewport.get("ymin", -100.0), viewport.get("ymax", 100.0)],
        "z": [viewport.get("zmin", -100.0), viewport.get("zmax", 100.0)],
    }
    resolved = node.metadata.get("resolved_symbols", {})
    symbol_map = node.metadata.get("python_symbol_map", {})
    for c in node.ranges:
        if c.lower is not None:
            try:
                val = safe_eval(to_python_expr(c.lower, symbol_map), resolved)
                bounds[c.axis][0] = max(bounds[c.axis][0], val)
            except Exception:
                pass
        if c.upper is not None:
            try:
                val = safe_eval(to_python_expr(c.upper, symbol_map), resolved)
                bounds[c.axis][1] = min(bounds[c.axis][1], val)
            except Exception:
                pass
    _refine_bounds_from_dependencies(node, bounds)
    if any(bounds[a][0] >= bounds[a][1] for a in ("x", "y", "z")):
        return None
    return (tuple(bounds["x"]), tuple(bounds["y"]), tuple(bounds["z"]))


def _refine_bounds_from_dependencies(node: BoxVolumeNode, bounds: dict[str, list[float]]) -> None:
    symbol_map = node.metadata.get("python_symbol_map", {})
    resolved = node.metadata.get("resolved_symbols", {})
    for _ in range(3):
        changed = False
        for c in node.ranges:
            if c.lower is not None:
                rng = _estimate_expr_range(c.lower, bounds, symbol_map, resolved)
                if rng is not None:
                    new_lower = max(bounds[c.axis][0], rng[0])
                    if new_lower > bounds[c.axis][0]:
                        bounds[c.axis][0] = new_lower
                        changed = True
            if c.upper is not None:
                rng = _estimate_expr_range(c.upper, bounds, symbol_map, resolved)
                if rng is not None:
                    new_upper = min(bounds[c.axis][1], rng[1])
                    if new_upper < bounds[c.axis][1]:
                        bounds[c.axis][1] = new_upper
                        changed = True
        if not changed:
            break


def _estimate_expr_range(
    expr: str,
    bounds: dict[str, list[float]],
    symbol_map: dict[str, str],
    resolved: dict[str, float],
) -> tuple[float, float] | None:
    py_expr = to_python_expr(expr, symbol_map)
    vars_used = sorted(set(re.findall(r"[xyz]", py_expr)))
    if not vars_used:
        try:
            v = safe_eval(py_expr, resolved)
            return v, v
        except Exception:
            return None
    samples_per_axis = 4
    values: list[float] = []

    def rec(axis_idx: int, env: dict[str, float]) -> None:
        if axis_idx >= len(vars_used):
            try:
                values.append(safe_eval(py_expr, {**resolved, **env}))
            except Exception:
                pass
            return
        axis = vars_used[axis_idx]
        lo, hi = bounds[axis]
        if lo >= hi:
            return
        for s in range(samples_per_axis):
            t = s / (samples_per_axis - 1)
            env[axis] = lo + (hi - lo) * t
            rec(axis_idx + 1, env)

    rec(0, {})
    if not values:
        return None
    return min(values), max(values)


def _sample_occupied_bbox(
    node: BoxVolumeNode,
    bbox: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    samples: int,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None:
    (x0, x1), (y0, y1), (z0, z1) = bbox
    occupied: list[tuple[float, float, float]] = []
    for i in range(samples):
        x = x0 + (x1 - x0) * (i + 0.5) / samples
        for j in range(samples):
            y = y0 + (y1 - y0) * (j + 0.5) / samples
            for k in range(samples):
                z = z0 + (z1 - z0) * (k + 0.5) / samples
                if _node_predicate(node, x, y, z):
                    occupied.append((x, y, z))
    if not occupied:
        return None
    xs = [p[0] for p in occupied]
    ys = [p[1] for p in occupied]
    zs = [p[2] for p in occupied]
    pad_x = max((max(xs) - min(xs)) * 0.1, 0.25)
    pad_y = max((max(ys) - min(ys)) * 0.1, 0.25)
    pad_z = max((max(zs) - min(zs)) * 0.1, 0.25)
    return (
        (min(xs) - pad_x, max(xs) + pad_x),
        (min(ys) - pad_y, max(ys) + pad_y),
        (min(zs) - pad_z, max(zs) + pad_z),
    )


def _voxelize_box_node(
    node: BoxVolumeNode,
    bbox: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    resolution: int,
) -> Mesh | None:
    (x0, x1), (y0, y1), (z0, z1) = bbox
    nx = ny = nz = resolution
    filled = [[[False for _ in range(nz)] for _ in range(ny)] for _ in range(nx)]

    for i in range(nx):
        x = x0 + (x1 - x0) * (i + 0.5) / nx
        for j in range(ny):
            y = y0 + (y1 - y0) * (j + 0.5) / ny
            for k in range(nz):
                z = z0 + (z1 - z0) * (k + 0.5) / nz
                if _node_predicate(node, x, y, z):
                    filled[i][j][k] = True

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    vertex_map: dict[tuple[float, float, float], int] = {}

    def vid(v: tuple[float, float, float]) -> int:
        key = (round(v[0], 6), round(v[1], 6), round(v[2], 6))
        idx = vertex_map.get(key)
        if idx is None:
            vertices.append(v)
            idx = len(vertices)
            vertex_map[key] = idx
        return idx

    def add_quad(a, b, c, d):
        ia, ib, ic, id_ = vid(a), vid(b), vid(c), vid(d)
        faces.append((ia, ib, ic))
        faces.append((ia, ic, id_))

    for i in range(nx):
        xa = x0 + (x1 - x0) * i / nx
        xb = x0 + (x1 - x0) * (i + 1) / nx
        for j in range(ny):
            ya = y0 + (y1 - y0) * j / ny
            yb = y0 + (y1 - y0) * (j + 1) / ny
            for k in range(nz):
                if not filled[i][j][k]:
                    continue
                za = z0 + (z1 - z0) * k / nz
                zb = z0 + (z1 - z0) * (k + 1) / nz
                neighbors = [
                    (i - 1, j, k, ((xa, ya, za), (xa, yb, za), (xa, yb, zb), (xa, ya, zb))),
                    (i + 1, j, k, ((xb, ya, za), (xb, ya, zb), (xb, yb, zb), (xb, yb, za))),
                    (i, j - 1, k, ((xa, ya, za), (xb, ya, za), (xb, ya, zb), (xa, ya, zb))),
                    (i, j + 1, k, ((xa, yb, za), (xa, yb, zb), (xb, yb, zb), (xb, yb, za))),
                    (i, j, k - 1, ((xa, ya, za), (xa, yb, za), (xb, yb, za), (xb, ya, za))),
                    (i, j, k + 1, ((xa, ya, zb), (xb, ya, zb), (xb, yb, zb), (xa, yb, zb))),
                ]
                for ni, nj, nk, quad in neighbors:
                    if ni < 0 or nj < 0 or nk < 0 or ni >= nx or nj >= ny or nk >= nz or not filled[ni][nj][nk]:
                        add_quad(*quad)

    if not faces:
        return None

    return Mesh(
        name=_mesh_name(node),
        color=node.color,
        vertices=vertices,
        faces=faces,
        source_file=node.source_ref.source_file,
        expression_id=node.source_ref.expression_id,
        family=node.family.value,
    )


def _node_predicate(node: BoxVolumeNode, x: float, y: float, z: float) -> bool:
    raw_restrictions: list[str] = node.metadata.get("raw_restrictions", [])
    core_expr: str = node.metadata.get("core_expr", "")
    checks = [core_expr] if core_expr else []
    checks.extend(raw_restrictions)
    env = dict(node.metadata.get("resolved_symbols", {}))
    env.update({"x": x, "y": y, "z": z})
    symbol_map = node.metadata.get("python_symbol_map", {})
    for expr in checks:
        if expr and not _evaluate_relation_expression(expr, env, symbol_map):
            return False
    return True


def _tighten_plane_fallback_viewport(
    node: PlanePatchNode, v0: str, v1: str, v0_lo: float, v0_hi: float, v1_lo: float, v1_hi: float
) -> tuple[float, float, float, float]:
    """Intersect fallback sampling range with axis bounds that do not reference coordinates."""
    sym = node.metadata.get("python_symbol_map", {})
    resolved = dict(node.metadata.get("resolved_symbols", {}))

    def span(axis: str, lo: float, hi: float) -> tuple[float, float]:
        out_lo, out_hi = lo, hi
        for c in node.bounds:
            if c.axis != axis:
                continue
            if c.lower and re.search(r"[xyz]", c.lower) is None:
                v = float(safe_eval(to_python_expr(c.lower, sym), resolved))
                if not c.lower_inclusive:
                    v += 1e-4 * max(1.0, abs(v))
                out_lo = max(out_lo, v)
            if c.upper and re.search(r"[xyz]", c.upper) is None:
                v = float(safe_eval(to_python_expr(c.upper, sym), resolved))
                if not c.upper_inclusive:
                    v -= 1e-4 * max(1.0, abs(v))
                out_hi = min(out_hi, v)
        if out_lo + 1e-9 < out_hi:
            return out_lo, out_hi
        return lo, hi

    a0, a1 = span(v0, v0_lo, v0_hi)
    b0, b1 = span(v1, v1_lo, v1_hi)
    return a0, a1, b0, b1


def _mesh_plane_patch_fallback(node: PlanePatchNode, plane_value: float, varying: list[str]) -> Mesh:
    viewport = node.metadata.get("viewport", {})
    v0_min = viewport.get(f"{varying[0]}min", -100.0)
    v0_max = viewport.get(f"{varying[0]}max", 100.0)
    v1_min = viewport.get(f"{varying[1]}min", -100.0)
    v1_max = viewport.get(f"{varying[1]}max", 100.0)
    v0_min, v0_max, v1_min, v1_max = _tighten_plane_fallback_viewport(
        node, varying[0], varying[1], v0_min, v0_max, v1_min, v1_max
    )
    # Thin feasible strips (e.g. ``0.5`` on ``y`` over a wide viewport) can miss all four corners of every
    # coarse cell — use cell-centre validity so narrow domains still produce faces.
    nx = ny = 80
    raw_restrictions: list[str] = node.metadata.get("raw_restrictions", [])
    if raw_restrictions:
        span0 = max(v0_max - v0_min, 1e-9)
        span1 = max(v1_max - v1_min, 1e-9)
        nx = ny = int(max(80, min(480, round(80 * max(span0, span1) / 0.35))))

    verts: list[tuple[float, float, float]] = []
    for j in range(ny + 1):
        b = v1_min + (v1_max - v1_min) * j / ny
        for i in range(nx + 1):
            a = v0_min + (v0_max - v0_min) * i / nx
            coords = {node.axis: plane_value, varying[0]: a, varying[1]: b}
            x, y, z = coords["x"], coords["y"], coords["z"]
            verts.append((x, y, z))

    cell_ok = [[False for _ in range(nx)] for _ in range(ny)]
    for j in range(ny):
        bc = v1_min + (v1_max - v1_min) * (j + 0.5) / ny
        for i in range(nx):
            ac = v0_min + (v0_max - v0_min) * (i + 0.5) / nx
            coords = {node.axis: plane_value, varying[0]: ac, varying[1]: bc}
            x, y, z = coords["x"], coords["y"], coords["z"]
            cell_ok[j][i] = _evaluate_restrictions(raw_restrictions, x, y, z, node.metadata)

    faces: list[tuple[int, int, int]] = []
    stride = nx + 1
    for j in range(ny):
        for i in range(nx):
            if not cell_ok[j][i]:
                continue
            a = j * stride + i + 1
            b = a + 1
            c = a + stride
            d = c + 1
            faces.append((a, b, d))
            faces.append((a, d, c))
    if not faces:
        raise ValueError("Fallback plane mesher produced no faces")
    return Mesh(name=_mesh_name(node), color=node.color, vertices=verts, faces=faces, source_file=node.source_ref.source_file, expression_id=node.source_ref.expression_id, family=node.family.value)


def _evaluate_restrictions(restrictions: list[str], x: float, y: float, z: float, metadata) -> bool:
    env = dict(metadata.get("resolved_symbols", {}))
    env.update({"x": x, "y": y, "z": z})
    symbol_map = metadata.get("python_symbol_map", {})
    for restriction in restrictions:
        if not _evaluate_relation_expression(restriction, env, symbol_map):
            return False
    return True


def _evaluate_relation_expression(expr: str, env: dict[str, float], symbol_map: dict[str, str]) -> bool:
    tokens = [t for t in re.split(r"(<=|>=|=|<|>)", expr) if t != ""]
    if len(tokens) < 3:
        return True
    values = []
    ops = []
    for idx, token in enumerate(tokens):
        if idx % 2 == 0:
            values.append(safe_eval(to_python_expr(token, symbol_map), env))
        else:
            ops.append(token)
    for i, op in enumerate(ops):
        left = values[i]
        right = values[i + 1]
        if op == "<" and not (left < right):
            return False
        if op == "<=" and not (left <= right):
            return False
        if op == ">" and not (left > right):
            return False
        if op == ">=" and not (left >= right):
            return False
        if op == "=" and not (abs(left - right) <= 1e-6):
            return False
    return True
