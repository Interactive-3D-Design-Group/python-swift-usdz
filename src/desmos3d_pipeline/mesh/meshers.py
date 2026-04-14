from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable

from desmos3d_pipeline.ir.models import BoxVolumeNode, GeometryNode, PlanePatchNode, PointNode, RangeConstraint, SampledSurfaceNode
from desmos3d_pipeline.parse.math_eval import safe_eval, to_python_expr


@dataclass(slots=True)
class Mesh:
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


def mesh_geometry_nodes(nodes: Iterable[GeometryNode]) -> tuple[list[Mesh], list[dict[str, str]]]:
    meshes: list[Mesh] = []
    failures: list[dict[str, str]] = []
    for node in nodes:
        try:
            if isinstance(node, PlanePatchNode):
                meshes.append(mesh_plane_patch(node))
            elif isinstance(node, BoxVolumeNode):
                meshes.append(mesh_box_volume(node))
            elif isinstance(node, SampledSurfaceNode):
                meshes.append(mesh_sampled_surface(node))
            elif isinstance(node, PointNode):
                meshes.append(mesh_point(node))
        except Exception as exc:
            failures.append({"source_file": node.source_ref.source_file, "expression_id": str(node.source_ref.expression_id), "error": str(exc)})
    return meshes, failures


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
    python_expr = to_python_expr(node.function_expr, node.metadata.get("python_symbol_map", {}))
    env = dict(node.metadata.get("resolved_symbols", {}))
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
    raw_restrictions: list[str] = node.metadata.get("raw_restrictions", [])
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
    if not faces:
        raise ValueError("No valid sampled cells after applying restrictions")
    return Mesh(name=_mesh_name(node), color=node.color, vertices=verts, faces=faces, source_file=node.source_ref.source_file, expression_id=node.source_ref.expression_id, family=node.family.value)


def mesh_point(node: PointNode) -> Mesh:
    x = _eval_expr(node.x, node.metadata)
    y = _eval_expr(node.y, node.metadata)
    z = _eval_expr(node.z, node.metadata)
    return Mesh(name=_mesh_name(node), color=node.color, vertices=[(x, y, z)], faces=[], source_file=node.source_ref.source_file, expression_id=node.source_ref.expression_id, family=node.family.value)


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


def _require_bounds_or_infer(bounds, axis: str, node: SampledSurfaceNode):
    axis_bounds = bounds.get(axis)
    if axis_bounds and axis_bounds["lower"] is not None and axis_bounds["upper"] is not None:
        return axis_bounds["lower"], axis_bounds["upper"]

    inferred = _infer_axis_bounds_from_z_constraints(node, axis, bounds)
    if inferred is not None:
        return inferred
    raise ValueError(f"Missing finite bounds for axis {axis}")


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


def _mesh_plane_patch_fallback(node: PlanePatchNode, plane_value: float, varying: list[str]) -> Mesh:
    viewport = node.metadata.get("viewport", {})
    v0_min = viewport.get(f"{varying[0]}min", -100.0)
    v0_max = viewport.get(f"{varying[0]}max", 100.0)
    v1_min = viewport.get(f"{varying[1]}min", -100.0)
    v1_max = viewport.get(f"{varying[1]}max", 100.0)
    nx = ny = 80
    verts: list[tuple[float, float, float]] = []
    valid = [[False for _ in range(nx + 1)] for _ in range(ny + 1)]
    raw_restrictions: list[str] = node.metadata.get("raw_restrictions", [])

    for j in range(ny + 1):
        b = v1_min + (v1_max - v1_min) * j / ny
        for i in range(nx + 1):
            a = v0_min + (v0_max - v0_min) * i / nx
            coords = {node.axis: plane_value, varying[0]: a, varying[1]: b}
            x, y, z = coords["x"], coords["y"], coords["z"]
            verts.append((x, y, z))
            valid[j][i] = _evaluate_restrictions(raw_restrictions, x, y, z, node.metadata)

    faces: list[tuple[int, int, int]] = []
    stride = nx + 1
    for j in range(ny):
        for i in range(nx):
            if not (valid[j][i] and valid[j][i + 1] and valid[j + 1][i] and valid[j + 1][i + 1]):
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
