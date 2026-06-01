from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .config import PhantomConfig, ScanConfig
from .phantom import LumpSpec

LUMP_COLORS: tuple[str, ...] = (
    "#d94f3d",
    "#1687d9",
    "#eba21a",
    "#3fa662",
    "#9367c7",
    "#d56a9f",
)


def require_pyvista():
    try:
        import pyvista as pv
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Native visualization requires pyvista/vtk. Recreate the environment from environment.yml "
            "or install pyvista, vtk, pyvistaqt, pyside6, and pyqtgraph."
        ) from exc
    return pv


def lump_implicit_value(points: np.ndarray, lump: LumpSpec) -> np.ndarray:
    """Return an implicit value whose zero contour is the continuous lump surface."""
    pts = np.asarray(points, dtype=np.float32)
    center = np.asarray(lump.center, dtype=np.float32)
    radii = np.maximum(np.asarray(lump.radii, dtype=np.float32), 1e-9)
    rel = _rotate_xy(pts - center, -float(lump.yaw))

    if lump.shape in {"sphere", "ellipsoid"}:
        return np.sqrt(np.sum((rel / radii) ** 2, axis=-1)) - 1.0
    if lump.shape == "box":
        q = np.abs(rel) - radii
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        inside = np.minimum(np.max(q, axis=-1), 0.0)
        return (outside + inside) / float(np.min(radii))
    if lump.shape == "cylinder":
        radial = np.sqrt((rel[..., 0] / radii[0]) ** 2 + (rel[..., 1] / radii[1]) ** 2) - 1.0
        axial = np.abs(rel[..., 2]) / radii[2] - 1.0
        return np.maximum(radial, axial)
    if lump.shape == "capsule":
        radius = float(radii[0])
        half_axis = float(radii[2])
        closest_z = np.clip(rel[..., 2], -half_axis, half_axis)
        nearest = np.stack([np.zeros_like(closest_z), np.zeros_like(closest_z), closest_z], axis=-1)
        return (np.linalg.norm(rel - nearest, axis=-1) - radius) / max(radius, 1e-9)
    raise ValueError(f"Unsupported lump shape: {lump.shape}")


def analytic_lump_polydata(lump: LumpSpec, *, resolution: int = 56, padding: float = 0.25):
    """Build a PyVista PolyData surface from the analytic lump implicit field."""
    pv = require_pyvista()
    try:
        from skimage.measure import marching_cubes
    except ModuleNotFoundError as exc:
        raise RuntimeError("Analytic lump surface extraction requires scikit-image.") from exc

    bounds = _lump_bounds(lump, padding=padding)
    nx, ny, nz = _grid_resolution(bounds, resolution)
    xs = np.linspace(bounds[0], bounds[1], nx, dtype=np.float32)
    ys = np.linspace(bounds[2], bounds[3], ny, dtype=np.float32)
    zs = np.linspace(bounds[4], bounds[5], nz, dtype=np.float32)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.stack([xx, yy, zz], axis=-1)
    values = lump_implicit_value(points, lump).astype(np.float32)
    if float(values.min()) > 0.0 or float(values.max()) < 0.0:
        return pv.PolyData()

    spacing = (
        float(xs[1] - xs[0]) if nx > 1 else 1.0,
        float(ys[1] - ys[0]) if ny > 1 else 1.0,
        float(zs[1] - zs[0]) if nz > 1 else 1.0,
    )
    verts, faces, _normals, _values = marching_cubes(values, level=0.0, spacing=spacing)
    verts += np.asarray([bounds[0], bounds[2], bounds[4]], dtype=np.float32)
    vtk_faces = np.empty((faces.shape[0], 4), dtype=np.int64)
    vtk_faces[:, 0] = 3
    vtk_faces[:, 1:] = faces.astype(np.int64)
    mesh = pv.PolyData(verts.astype(np.float32), vtk_faces.reshape(-1))
    mesh["stiffness_multiplier"] = np.full(mesh.n_points, float(lump.stiffness_multiplier), dtype=np.float32)
    return mesh


def phantom_box_polydata(phantom: PhantomConfig):
    pv = require_pyvista()
    bounds = (
        -0.5 * phantom.size_x,
        0.5 * phantom.size_x,
        -0.5 * phantom.size_y,
        0.5 * phantom.size_y,
        0.0,
        phantom.height,
    )
    return pv.Box(bounds=bounds)


def top_surface_polydata(phantom: PhantomConfig, *, resolution: int = 96):
    pv = require_pyvista()
    n = max(int(resolution), 2)
    xs = np.linspace(-0.5 * phantom.size_x, 0.5 * phantom.size_x, n, dtype=np.float32)
    ys = np.linspace(-0.5 * phantom.size_y, 0.5 * phantom.size_y, n, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    zz = np.full_like(xx, float(phantom.height), dtype=np.float32)
    points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    faces = []
    for row in range(n - 1):
        for col in range(n - 1):
            a = row * n + col
            faces.extend([4, a, a + 1, a + n + 1, a + n])
    mesh = pv.PolyData(points, np.asarray(faces, dtype=np.int64))
    mesh["visual_z_displacement_m"] = np.zeros(mesh.n_points, dtype=np.float32)
    return mesh


def deformed_surface_points(
    base_points: np.ndarray,
    phantom: PhantomConfig,
    scan: ScanConfig,
    *,
    x: float,
    y: float,
    depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(base_points, dtype=np.float32).copy()
    sigma = max(float(scan.probe_radius) * 1.35, 1e-6)
    r2 = (points[:, 0] - float(x)) ** 2 + (points[:, 1] - float(y)) ** 2
    displacement = float(depth) * np.exp(-0.5 * r2 / (sigma * sigma)).astype(np.float32)
    points[:, 2] = float(phantom.height) - displacement
    return points, displacement


def deformed_body_points(
    base_points: np.ndarray,
    phantom: PhantomConfig,
    scan: ScanConfig,
    *,
    x: float,
    y: float,
    depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a lightweight press-shaped visual deformation to arbitrary mesh points."""
    points = np.asarray(base_points, dtype=np.float32).copy()
    sigma = max(float(scan.probe_radius) * 1.35, 1e-6)
    r2 = (points[:, 0] - float(x)) ** 2 + (points[:, 1] - float(y)) ** 2
    below_surface = np.maximum(float(phantom.height) - points[:, 2], 0.0)
    depth_sigma = max(float(phantom.height) * 0.42, 1e-6)
    displacement = (
        float(depth)
        * np.exp(-0.5 * r2 / (sigma * sigma))
        * np.exp(-below_surface / depth_sigma)
    ).astype(np.float32)
    points[:, 2] = np.maximum(0.0, points[:, 2] - displacement)
    return points, displacement


def scan_points_polydata(xy: np.ndarray, values: np.ndarray | None = None, *, z: float = 0.0):
    pv = require_pyvista()
    xy = np.asarray(xy, dtype=np.float32)
    points = np.column_stack(
        [
            xy[..., 0].reshape(-1),
            xy[..., 1].reshape(-1),
            np.full(xy.shape[0] * xy.shape[1], float(z), dtype=np.float32),
        ]
    )
    cloud = pv.PolyData(points)
    if values is not None:
        cloud["value"] = np.asarray(values, dtype=np.float32).reshape(-1)
    rows, cols = xy.shape[:2]
    cloud["row"] = np.repeat(np.arange(rows, dtype=np.int32), cols)
    cloud["col"] = np.tile(np.arange(cols, dtype=np.int32), rows)
    return cloud


def save_lump_surfaces(
    out_dir: Path,
    lumps: Sequence[LumpSpec],
    *,
    resolution: int = 56,
    prefix: str = "lump",
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for idx, lump in enumerate(lumps):
        mesh = analytic_lump_polydata(lump, resolution=resolution)
        mesh["lump_id"] = np.full(mesh.n_points, idx, dtype=np.int32)
        path = out_dir / f"{prefix}_{idx:02d}_{lump.shape}_analytic.vtp"
        mesh.save(path)
        paths.append(path)
    return paths


def _lump_bounds(lump: LumpSpec, *, padding: float) -> tuple[float, float, float, float, float, float]:
    cx, cy, cz = (float(v) for v in lump.center)
    rx, ry, rz = (float(v) for v in lump.radii)
    if lump.shape == "capsule":
        rz = rz + rx
    xy_extent = max(rx, ry)
    z_extent = rz
    extent = max(xy_extent, z_extent, 1e-4)
    pad = max(float(padding) * extent, 1e-5)
    xy_extent += pad
    z_extent += pad
    return (cx - xy_extent, cx + xy_extent, cy - xy_extent, cy + xy_extent, cz - z_extent, cz + z_extent)


def _grid_resolution(bounds: tuple[float, float, float, float, float, float], target: int) -> tuple[int, int, int]:
    spans = np.asarray([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]], dtype=np.float64)
    max_span = max(float(spans.max()), 1e-9)
    base = max(int(target), 12)
    counts = np.maximum(np.ceil(base * spans / max_span).astype(int), 12)
    return int(counts[0]), int(counts[1]), int(counts[2])


def _rotate_xy(rel: np.ndarray, yaw: float) -> np.ndarray:
    c = np.cos(yaw)
    s = np.sin(yaw)
    out = np.array(rel, copy=True)
    x = rel[..., 0]
    y = rel[..., 1]
    out[..., 0] = c * x - s * y
    out[..., 1] = s * x + c * y
    return out
