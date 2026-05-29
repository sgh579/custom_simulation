from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence

import numpy as np

from .config import MaterialConfig, PhantomConfig, ScanConfig

LumpShape = Literal["sphere", "ellipsoid", "box", "cylinder", "capsule"]
SURFACE_CLEARANCE = 0.005


@dataclass
class StructuredTetMesh:
    vertices: np.ndarray
    tets: np.ndarray
    bottom_vertex_mask: np.ndarray
    tet_centroids: np.ndarray

    @property
    def flat_indices(self) -> list[int]:
        return self.tets.astype(np.int32, copy=False).reshape(-1).tolist()


@dataclass
class LumpSpec:
    """One embedded inclusion/lump inside the phantom."""

    shape: LumpShape
    center: tuple[float, float, float]
    radii: tuple[float, float, float]
    stiffness_multiplier: float
    yaw: float = 0.0

    def to_dict(self, phantom: PhantomConfig | None = None) -> dict[str, object]:
        data = asdict(self)
        if phantom is not None:
            center_z = float(self.center[2])
            extent_z = _z_extent(self.radii, self.shape)
            top_z = center_z + extent_z
            bottom_z = center_z - extent_z
            data["center_depth_from_top"] = float(phantom.height - center_z)
            data["top_depth_from_top"] = float(max(phantom.height - top_z, 0.0))
            data["bottom_depth_from_top"] = float(max(phantom.height - bottom_z, 0.0))
            data["z_extent"] = float(extent_z)
        return data


def create_structured_tet_mesh(config: PhantomConfig) -> StructuredTetMesh:
    """Create a rectangular tet mesh using the same 5-tet cube split as Newton examples."""
    nx, ny, nz = config.cells_x, config.cells_y, config.cells_z
    xs = np.linspace(-0.5 * config.size_x, 0.5 * config.size_x, nx + 1, dtype=np.float32)
    ys = np.linspace(-0.5 * config.size_y, 0.5 * config.size_y, ny + 1, dtype=np.float32)
    zs = np.linspace(0.0, config.height, nz + 1, dtype=np.float32)

    vertices = np.zeros(((nx + 1) * (ny + 1) * (nz + 1), 3), dtype=np.float32)
    for z_i, z in enumerate(zs):
        for y_i, y in enumerate(ys):
            for x_i, x in enumerate(xs):
                vertices[_grid_index(x_i, y_i, z_i, nx, ny)] = (x, y, z)

    tets: list[tuple[int, int, int, int]] = []
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                v0 = _grid_index(x, y, z, nx, ny)
                v1 = _grid_index(x + 1, y, z, nx, ny)
                v2 = _grid_index(x + 1, y, z + 1, nx, ny)
                v3 = _grid_index(x, y, z + 1, nx, ny)
                v4 = _grid_index(x, y + 1, z, nx, ny)
                v5 = _grid_index(x + 1, y + 1, z, nx, ny)
                v6 = _grid_index(x + 1, y + 1, z + 1, nx, ny)
                v7 = _grid_index(x, y + 1, z + 1, nx, ny)

                if (x & 1) ^ (y & 1) ^ (z & 1):
                    tets.extend(
                        [
                            (v0, v1, v4, v3),
                            (v2, v3, v6, v1),
                            (v5, v4, v1, v6),
                            (v7, v6, v3, v4),
                            (v4, v1, v6, v3),
                        ]
                    )
                else:
                    tets.extend(
                        [
                            (v1, v2, v5, v0),
                            (v3, v0, v7, v2),
                            (v4, v7, v0, v5),
                            (v6, v5, v2, v7),
                            (v5, v2, v7, v0),
                        ]
                    )

    tet_array = np.asarray(tets, dtype=np.int32)
    bottom_mask = np.isclose(vertices[:, 2], 0.0)
    centroids = vertices[tet_array].mean(axis=1).astype(np.float32)
    return StructuredTetMesh(vertices=vertices, tets=tet_array, bottom_vertex_mask=bottom_mask, tet_centroids=centroids)


def sample_lump(
    rng: np.random.Generator,
    phantom: PhantomConfig,
    material: MaterialConfig,
    shapes: tuple[LumpShape, ...] = ("sphere", "ellipsoid", "box"),
    size_scale: float = 1.0,
    center_depth_range: tuple[float, float] | None = None,
    max_radius_fraction: float = 0.2,
    z_extent_limit: float | None = None,
) -> LumpSpec:
    """Randomly sample a lump that remains inside the phantom volume."""
    shape = str(rng.choice(shapes))
    scale = max(float(size_scale), 0.1)
    max_rx = max(0.002, min(0.035 * scale, max_radius_fraction * phantom.size_x))
    max_ry = max(0.002, min(0.035 * scale, max_radius_fraction * phantom.size_y))
    max_xy_radius = min(max_rx, max_ry)
    max_z_radius = min(0.022 * scale, 0.35 * phantom.height)
    if z_extent_limit is not None:
        max_z_radius = min(max_z_radius, max(float(z_extent_limit), 0.001))
    max_z_radius = max(max_z_radius, 0.001)
    min_xy_radius = min(0.010 * scale, 0.65 * max_xy_radius)
    min_z_radius = min(0.006 * scale, 0.65 * max_z_radius)
    base_radius_hi = min(max_xy_radius, max_z_radius)
    base_radius_lo = min(min_xy_radius, 0.65 * base_radius_hi)
    base_radius = _sample_inside(rng, base_radius_lo, base_radius_hi)
    if shape == "sphere":
        radii = (base_radius, base_radius, base_radius)
    elif shape == "ellipsoid":
        radii = (
            float(rng.uniform(min_xy_radius, max_rx)),
            float(rng.uniform(0.75 * min_xy_radius, 0.85 * max_ry)),
            float(rng.uniform(min_z_radius, max_z_radius)),
        )
    elif shape == "box":
        radii = (
            float(rng.uniform(0.85 * min_xy_radius, 0.9 * max_rx)),
            float(rng.uniform(0.85 * min_xy_radius, 0.9 * max_ry)),
            float(rng.uniform(0.8 * min_z_radius, max_z_radius)),
        )
    elif shape == "cylinder":
        radii = (
            float(rng.uniform(min_xy_radius, 0.85 * max_rx)),
            float(rng.uniform(min_xy_radius, 0.85 * max_ry)),
            float(rng.uniform(min_z_radius, max_z_radius)),
        )
    elif shape == "capsule":
        radius_hi = min(0.75 * max_xy_radius, 0.65 * max_z_radius)
        radius = _sample_inside(rng, min(0.75 * min_xy_radius, radius_hi), radius_hi)
        half_axis_hi = max(max_z_radius - radius, 0.0)
        half_axis = _sample_inside(rng, 0.0, half_axis_hi)
        radii = (radius, radius, half_axis)
    else:
        raise ValueError(f"Unsupported lump shape: {shape}")

    x_margin = max(radii[0], radii[1]) + 0.006
    y_margin = max(radii[0], radii[1]) + 0.006
    z_extent = _z_extent(radii, shape)  # type: ignore[arg-type]
    x_lo = -0.5 * phantom.size_x + x_margin
    x_hi = 0.5 * phantom.size_x - x_margin
    y_lo = -0.5 * phantom.size_y + y_margin
    y_hi = 0.5 * phantom.size_y - y_margin
    z_lo, z_hi = _z_center_bounds(phantom, z_extent, center_depth_range)
    _require_valid_interval(x_lo, x_hi, "x")
    _require_valid_interval(y_lo, y_hi, "y")
    _require_valid_interval(z_lo, z_hi, "z")
    center = (
        _sample_inside(rng, x_lo, x_hi),
        _sample_inside(rng, y_lo, y_hi),
        _sample_inside(rng, z_lo, z_hi),
    )
    multiplier = float(np.exp(rng.uniform(np.log(material.lump_stiffness_min), np.log(material.lump_stiffness_max))))
    yaw = float(rng.uniform(-np.pi, np.pi))
    return LumpSpec(shape=shape, center=center, radii=radii, stiffness_multiplier=multiplier, yaw=yaw)  # type: ignore[arg-type]


def sample_lumps(
    rng: np.random.Generator,
    phantom: PhantomConfig,
    material: MaterialConfig,
    count_min: int = 1,
    count_max: int = 1,
    shapes: tuple[LumpShape, ...] = ("sphere", "ellipsoid", "box", "cylinder", "capsule"),
    size_scale: float = 1.0,
    center_depth_range: tuple[float, float] | None = None,
    max_radius_fraction: float = 0.2,
    allow_overlap: bool = False,
    separate_z: bool = True,
    z_gap: float = 0.0,
) -> list[LumpSpec]:
    """Sample one or more complete lumps.

    With ``separate_z=True`` only occupied z-interval separation is enforced;
    x/y projections may overlap. With ``separate_z=False`` the older bounding
    overlap check is used unless ``allow_overlap`` is true.
    """
    lo = max(0, int(count_min))
    hi = max(lo, int(count_max))
    count = int(rng.integers(lo, hi + 1)) if hi > 0 else 0
    lumps: list[LumpSpec] = []
    if separate_z and count > 0:
        return _sample_z_separated_lumps(
            rng,
            phantom,
            material,
            count=count,
            shapes=shapes,
            size_scale=size_scale,
            center_depth_range=center_depth_range,
            max_radius_fraction=max_radius_fraction,
            allow_overlap=allow_overlap,
            z_gap=z_gap,
        )

    attempts = 0
    max_attempts = max(1000, 400 * max(count, 1))
    while len(lumps) < count and attempts < max_attempts:
        attempts += 1
        candidate = sample_lump(
            rng,
            phantom,
            material,
            shapes=shapes,
            size_scale=size_scale,
            center_depth_range=center_depth_range,
            max_radius_fraction=max_radius_fraction,
        )
        xy_ok = allow_overlap or all(not _strongly_overlaps(candidate, lump) for lump in lumps)
        if xy_ok:
            lumps.append(candidate)

    while len(lumps) < count:
        lumps.append(
            sample_lump(
                rng,
                phantom,
                material,
                shapes=shapes,
                size_scale=size_scale,
                center_depth_range=center_depth_range,
                max_radius_fraction=max_radius_fraction,
            )
        )
    return lumps


def _sample_z_separated_lumps(
    rng: np.random.Generator,
    phantom: PhantomConfig,
    material: MaterialConfig,
    count: int,
    shapes: tuple[LumpShape, ...],
    size_scale: float,
    center_depth_range: tuple[float, float] | None,
    max_radius_fraction: float,
    allow_overlap: bool,
    z_gap: float,
) -> list[LumpSpec]:
    """Sample complete lumps whose occupied z intervals do not overlap.

    The z constraint is applied only after each full lump geometry has been
    sampled. We intentionally do not shrink a lump to fit a pre-cut z slab.
    """
    del allow_overlap
    z_extent_limit = _z_extent_limit_for_count(phantom, count, center_depth_range, z_gap)
    max_layout_attempts = 300
    max_candidate_attempts = max(300, 100 * count)
    for _ in range(max_layout_attempts):
        lumps: list[LumpSpec] = []
        for _lump_idx in range(count):
            placed = False
            for _ in range(max_candidate_attempts):
                try:
                    candidate = sample_lump(
                        rng,
                        phantom,
                        material,
                        shapes=shapes,
                        size_scale=size_scale,
                        center_depth_range=center_depth_range,
                        max_radius_fraction=max_radius_fraction,
                        z_extent_limit=z_extent_limit,
                    )
                except RuntimeError:
                    continue

                z_center = _sample_nonoverlapping_z_center(
                    rng,
                    phantom,
                    candidate,
                    lumps,
                    center_depth_range=center_depth_range,
                    gap=z_gap,
                )
                if z_center is None:
                    continue
                candidate = LumpSpec(
                    shape=candidate.shape,
                    center=(candidate.center[0], candidate.center[1], z_center),
                    radii=candidate.radii,
                    stiffness_multiplier=candidate.stiffness_multiplier,
                    yaw=candidate.yaw,
                )
                lumps.append(candidate)
                placed = True
                break
            if not placed:
                break
        if len(lumps) == count:
            return lumps

    raise RuntimeError(
        f"Could not sample {count} complete z-separated lumps in phantom height {phantom.height:.4f} m. "
        "Increase phantom height, reduce lump size/count/z-gap, or pass --allow-z-overlap."
    )


def lump_membership(points: np.ndarray, lump: LumpSpec, project_xy: bool = False) -> np.ndarray:
    """Return a boolean mask for points inside the lump, optionally using xy projection only."""
    pts = np.asarray(points, dtype=np.float32)
    center = np.asarray(lump.center, dtype=np.float32)
    radii = np.asarray(lump.radii, dtype=np.float32)
    rel = pts - center
    rel = _rotate_xy(rel, -lump.yaw)

    if project_xy:
        rel_eval = rel[..., :2]
        radii_eval = radii[:2]
    else:
        rel_eval = rel
        radii_eval = radii

    if lump.shape in {"sphere", "ellipsoid"}:
        value = np.sum((rel_eval / radii_eval) ** 2, axis=-1)
        return value <= 1.0
    if lump.shape == "box":
        return np.all(np.abs(rel_eval) <= radii_eval, axis=-1)
    if lump.shape == "cylinder":
        if project_xy:
            return np.sum((rel_eval / radii_eval) ** 2, axis=-1) <= 1.0
        radial = (rel[..., 0] / radii[0]) ** 2 + (rel[..., 1] / radii[1]) ** 2
        return (radial <= 1.0) & (np.abs(rel[..., 2]) <= radii[2])
    if lump.shape == "capsule":
        if project_xy:
            return np.sum((rel_eval / radii_eval) ** 2, axis=-1) <= 1.0
        radius = float(radii[0])
        half_axis = float(radii[2])
        closest_z = np.clip(rel[..., 2], -half_axis, half_axis)
        nearest = np.stack([np.zeros_like(closest_z), np.zeros_like(closest_z), closest_z], axis=-1)
        return np.linalg.norm(rel - nearest, axis=-1) <= radius
    raise ValueError(f"Unsupported lump shape: {lump.shape}")


def lumps_membership(points: np.ndarray, lumps: LumpSpec | Sequence[LumpSpec], project_xy: bool = False) -> np.ndarray:
    lump_list = normalize_lumps(lumps)
    if not lump_list:
        return np.zeros(np.asarray(points).shape[:-1], dtype=bool)
    mask = np.zeros(np.asarray(points).shape[:-1], dtype=bool)
    for lump in lump_list:
        mask |= lump_membership(points, lump, project_xy=project_xy)
    return mask


def material_arrays_for_lump(
    mesh: StructuredTetMesh,
    material: MaterialConfig,
    lump: LumpSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k_mu, k_lambda, k_damp, inside, _lump_id = material_arrays_for_lumps(mesh, material, [lump])
    return k_mu, k_lambda, k_damp, inside


def material_arrays_for_lumps(
    mesh: StructuredTetMesh,
    material: MaterialConfig,
    lumps: LumpSpec | Sequence[LumpSpec],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lump_list = normalize_lumps(lumps)
    stiffness = np.ones(mesh.tets.shape[0], dtype=np.float32)
    lump_id = np.full(mesh.tets.shape[0], -1, dtype=np.int32)
    for idx, lump in enumerate(lump_list):
        inside = lump_membership(mesh.tet_centroids, lump, project_xy=False)
        update = inside & (np.float32(lump.stiffness_multiplier) > stiffness)
        stiffness[update] = np.float32(lump.stiffness_multiplier)
        lump_id[update] = idx
    inside_any = lump_id >= 0
    k_mu = np.full(mesh.tets.shape[0], material.k_mu, dtype=np.float32)
    k_lambda = np.full(mesh.tets.shape[0], material.k_lambda, dtype=np.float32)
    k_damp = np.full(mesh.tets.shape[0], material.k_damp, dtype=np.float32)
    k_mu *= stiffness
    k_lambda *= stiffness
    k_damp *= np.sqrt(stiffness)
    return k_mu, k_lambda, k_damp, inside_any.astype(np.uint8), lump_id


def mask_for_scan_grid(scan: ScanConfig, phantom: PhantomConfig, lumps: LumpSpec | Sequence[LumpSpec]) -> np.ndarray:
    xs = np.asarray(scan.x_values(phantom), dtype=np.float32)
    ys = np.asarray(scan.y_values(phantom), dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    points = np.stack([xv, yv, np.zeros_like(xv)], axis=-1)
    return lumps_membership(points, lumps, project_xy=True).astype(np.float32)


def normalize_lumps(lumps: LumpSpec | Sequence[LumpSpec]) -> list[LumpSpec]:
    if isinstance(lumps, LumpSpec):
        return [lumps]
    return list(lumps)


def lumps_to_json(lumps: LumpSpec | Sequence[LumpSpec], phantom: PhantomConfig) -> str:
    return json_dumps_lumps(normalize_lumps(lumps), phantom)


def json_dumps_lumps(lumps: Sequence[LumpSpec], phantom: PhantomConfig) -> str:
    import json

    return json.dumps([lump.to_dict(phantom) for lump in lumps])


def _z_center_bounds(
    phantom: PhantomConfig,
    z_extent: float,
    center_depth_range: tuple[float, float] | None,
) -> tuple[float, float]:
    z_lo = float(z_extent) + SURFACE_CLEARANCE
    z_hi = phantom.height - float(z_extent) - SURFACE_CLEARANCE
    if center_depth_range is not None:
        depth_min, depth_max = sorted(center_depth_range)
        z_lo = max(z_lo, phantom.height - depth_max)
        z_hi = min(z_hi, phantom.height - depth_min)
    return z_lo, z_hi


def _z_extent_limit_for_count(
    phantom: PhantomConfig,
    count: int,
    center_depth_range: tuple[float, float] | None,
    z_gap: float,
) -> float | None:
    if count <= 0:
        return None
    usable_height = phantom.height - 2.0 * SURFACE_CLEARANCE - float(z_gap) * max(count - 1, 0)
    if usable_height <= 0.0:
        raise RuntimeError(
            f"Could not reserve {count} complete z-separated lumps in phantom height {phantom.height:.4f} m. "
            "Increase phantom height, reduce z-gap, or pass --allow-z-overlap."
        )

    extent_limit = 0.5 * usable_height / float(count)
    if center_depth_range is not None and count > 1:
        depth_min, depth_max = sorted(center_depth_range)
        center_span = float(depth_max - depth_min)
        center_limit = (center_span - float(z_gap) * float(count - 1)) / (2.0 * float(count - 1))
        extent_limit = min(extent_limit, center_limit)

    if extent_limit <= 0.0:
        raise RuntimeError(
            f"Could not fit {count} complete z-separated lumps in the requested z/depth range. "
            "Increase phantom height/depth range, reduce count/z-gap, or pass --allow-z-overlap."
        )
    return extent_limit


def _sample_nonoverlapping_z_center(
    rng: np.random.Generator,
    phantom: PhantomConfig,
    candidate: LumpSpec,
    placed: Sequence[LumpSpec],
    *,
    center_depth_range: tuple[float, float] | None,
    gap: float,
) -> float | None:
    z_extent = _z_extent(candidate.radii, candidate.shape)
    z_lo, z_hi = _z_center_bounds(phantom, z_extent, center_depth_range)
    if z_hi < z_lo:
        return None

    available = [(z_lo, z_hi)]
    for lump in placed:
        lump_lo, lump_hi = _z_interval(lump)
        forbidden_lo = lump_lo - float(gap) - z_extent
        forbidden_hi = lump_hi + float(gap) + z_extent
        available = _subtract_interval(available, forbidden_lo, forbidden_hi)
        if not available:
            return None

    lengths = np.asarray([max(hi - lo, 0.0) for lo, hi in available], dtype=np.float64)
    total = float(np.sum(lengths))
    if total <= 1e-12:
        lo, hi = available[int(rng.integers(0, len(available)))]
        return float(0.5 * (lo + hi))

    pick = float(rng.uniform(0.0, total))
    cursor = 0.0
    for (lo, hi), length in zip(available, lengths):
        next_cursor = cursor + float(length)
        if pick <= next_cursor:
            return float(lo + rng.uniform(0.0, float(length)))
        cursor = next_cursor
    return float(available[-1][1])


def _subtract_interval(
    intervals: Sequence[tuple[float, float]],
    cut_lo: float,
    cut_hi: float,
) -> list[tuple[float, float]]:
    if cut_hi < cut_lo:
        cut_lo, cut_hi = cut_hi, cut_lo
    out: list[tuple[float, float]] = []
    for lo, hi in intervals:
        if cut_hi <= lo or cut_lo >= hi:
            out.append((lo, hi))
            continue
        if cut_lo > lo:
            out.append((lo, min(cut_lo, hi)))
        if cut_hi < hi:
            out.append((max(cut_hi, lo), hi))
    return [(lo, hi) for lo, hi in out if hi >= lo]


def _require_valid_interval(lo: float, hi: float, axis: str) -> None:
    if hi < lo:
        raise RuntimeError(f"Cannot place a complete lump inside the phantom along {axis}: interval [{lo:.6f}, {hi:.6f}]")


def _grid_index(x: int, y: int, z: int, nx: int, ny: int) -> int:
    return (nx + 1) * (ny + 1) * z + (nx + 1) * y + x


def _sample_inside(rng: np.random.Generator, lo: float, hi: float) -> float:
    if hi <= lo:
        return float(0.5 * (lo + hi))
    return float(rng.uniform(lo, hi))


def _z_extent(radii: tuple[float, float, float], shape: LumpShape) -> float:
    if shape == "capsule":
        return float(radii[0] + radii[2])
    return float(radii[2])


def _strongly_overlaps(a: LumpSpec, b: LumpSpec) -> bool:
    ca = np.asarray(a.center, dtype=np.float32)
    cb = np.asarray(b.center, dtype=np.float32)
    ea = np.asarray([max(a.radii[0], a.radii[1]), max(a.radii[0], a.radii[1]), _z_extent(a.radii, a.shape)])
    eb = np.asarray([max(b.radii[0], b.radii[1]), max(b.radii[0], b.radii[1]), _z_extent(b.radii, b.shape)])
    scaled = np.abs(ca - cb) / np.maximum(ea + eb, 1e-6)
    return bool(np.linalg.norm(scaled) < 0.85)


def _z_overlaps(a: LumpSpec, b: LumpSpec, gap: float = 0.0) -> bool:
    a_lo, a_hi = _z_interval(a)
    b_lo, b_hi = _z_interval(b)
    return (a_lo - gap) < b_hi and (b_lo - gap) < a_hi


def _z_inside(lump: LumpSpec, interval: tuple[float, float]) -> bool:
    lo, hi = sorted(interval)
    lump_lo, lump_hi = _z_interval(lump)
    eps = 1e-7
    return bool(lump_lo >= lo - eps and lump_hi <= hi + eps)


def _z_interval(lump: LumpSpec) -> tuple[float, float]:
    extent = _z_extent(lump.radii, lump.shape)
    center_z = float(lump.center[2])
    return center_z - extent, center_z + extent


def _rotate_xy(rel: np.ndarray, yaw: float) -> np.ndarray:
    c = np.cos(yaw)
    s = np.sin(yaw)
    out = np.array(rel, copy=True)
    x = rel[..., 0]
    y = rel[..., 1]
    out[..., 0] = c * x - s * y
    out[..., 1] = s * x + c * y
    return out
