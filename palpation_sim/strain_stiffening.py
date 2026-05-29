from __future__ import annotations

import json
from typing import Sequence

import numpy as np

from .config import MaterialConfig, PhantomConfig, ScanConfig
from .phantom import LumpSpec, lumps_to_json, mask_for_scan_grid, normalize_lumps


def run_strain_stiffening_sample(
    phantom: PhantomConfig,
    material: MaterialConfig,
    scan: ScanConfig,
    lumps: LumpSpec | Sequence[LumpSpec],
    rng: np.random.Generator,
    *,
    hardening_b: float = 1.8,
    noise_std: float = 0.0,
    enforce_convex: bool = True,
) -> dict[str, np.ndarray | str]:
    """Small explicit strain-stiffening palpation model.

    This is a fast calibration/smoke-test model, not a Newton FEM solver.  Its
    local force law is a Fung-like 1D reduction:

        F(z) = K z exp(b (z / z_max)^2)

    so the tangent stiffness dF/dz increases with indentation when b > 0.
    """
    xs = np.asarray(scan.x_values(phantom), dtype=np.float32)
    ys = np.asarray(scan.y_values(phantom), dtype=np.float32)
    depths = np.asarray(scan.indentation_values(), dtype=np.float32)
    h, w, t = scan.grid_h, scan.grid_w, scan.press_steps

    presses = np.zeros((h, w, t, 2), dtype=np.float32)
    probe_pose = np.zeros((h, w, t, 7), dtype=np.float32)
    indentation = np.broadcast_to(depths, (h, w, t)).copy().astype(np.float32)
    fz = np.zeros((h, w, t), dtype=np.float32)
    contact_features = np.zeros((h, w, t, 5), dtype=np.float32)
    nonlinearity_ratio = np.zeros((h, w), dtype=np.float32)

    lump_list = normalize_lumps(lumps)
    base_k = 1400.0 * (material.k_mu / 2.0e5) ** 0.5
    z_scale = max(float(scan.max_indentation), float(depths[-1]) if depths.size else 0.0, 1e-6)
    strain = np.clip(depths / z_scale, 0.0, 1.5).astype(np.float32)
    hardening = np.exp(np.clip(float(hardening_b) * strain * strain, 0.0, 12.0)).astype(np.float32)

    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            xy = np.asarray([x, y], dtype=np.float32)
            depth_gain = np.zeros_like(depths, dtype=np.float32)
            spatial_peak = 0.0

            for lump in lump_list:
                center_xy = np.asarray(lump.center[:2], dtype=np.float32)
                radii_xy = np.asarray(lump.radii[:2], dtype=np.float32)
                shape_gain, transition_scale, footprint_scale = _shape_response_parameters(lump.shape)
                lateral_sigma = footprint_scale * 0.75 * float(np.max(radii_xy)) + scan.probe_radius
                dist = float(np.linalg.norm((xy - center_xy) / max(lateral_sigma, 1e-6)))
                spatial_gain = float(np.exp(-0.5 * dist * dist))
                spatial_peak = max(spatial_peak, spatial_gain)

                top_depth = _lump_top_depth_from_surface(phantom, lump)
                transition = transition_scale * max(0.08 * scan.probe_radius, 0.04 * z_scale, 1e-6)
                activation = _sigmoid((depths - np.float32(top_depth)) / np.float32(transition))
                depth_gain += (
                    np.float32(shape_gain)
                    * np.float32(lump.stiffness_multiplier - 1.0)
                    * np.float32(spatial_gain)
                    * (0.10 + 0.90 * activation)
                )

            local_multiplier = 1.0 + depth_gain
            force = (base_k * depths * hardening * local_multiplier).astype(np.float32)

            if noise_std > 0.0:
                scale = noise_std * max(float(np.max(force)), 1e-4)
                force = force + rng.normal(0.0, scale, size=force.shape).astype(np.float32)
                force = np.maximum(force, 0.0)
            if enforce_convex:
                force = _enforce_monotone_convex(depths, force)
            else:
                force = np.maximum.accumulate(force).astype(np.float32)

            tangent = _finite_difference_tangent(depths, force)
            nonlinearity_ratio[row, col] = _late_early_slope_ratio(depths, force)

            z = phantom.height + scan.probe_radius + scan.preload_gap - depths
            probe_pose[row, col, :, 0] = x
            probe_pose[row, col, :, 1] = y
            probe_pose[row, col, :, 2] = z
            probe_pose[row, col, :, 6] = 1.0

            presses[row, col, :, 0] = depths
            presses[row, col, :, 1] = force
            fz[row, col] = force
            contact_features[row, col, :, 0] = strain
            contact_features[row, col, :, 1] = hardening
            contact_features[row, col, :, 2] = local_multiplier
            contact_features[row, col, :, 3] = tangent
            contact_features[row, col, :, 4] = spatial_peak

    mask = mask_for_scan_grid(scan, phantom, lump_list)
    xy_grid = np.stack(np.meshgrid(xs, ys), axis=-1).astype(np.float32)
    first_lump_json = json.dumps(lump_list[0].to_dict(phantom)) if lump_list else "{}"
    return {
        "presses": presses,
        "mask": mask,
        "xy": xy_grid,
        "probe_pose": probe_pose,
        "indentation_depth": indentation,
        "fz": fz,
        "contact_features": contact_features,
        "nonlinearity_ratio": nonlinearity_ratio,
        "lump_json": first_lump_json,
        "lumps_json": lumps_to_json(lump_list, phantom),
        "num_lumps": np.asarray(len(lump_list), dtype=np.int32),
        "backend": np.asarray("strain_stiffening"),
    }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def _enforce_monotone_convex(depths: np.ndarray, force: np.ndarray) -> np.ndarray:
    if depths.size < 2:
        return np.maximum(force, 0.0).astype(np.float32)
    dz = np.diff(depths).astype(np.float32)
    dz = np.maximum(dz, np.float32(1e-12))
    f0 = max(float(force[0]), 0.0)
    slopes = np.diff(np.maximum(force, 0.0).astype(np.float32)) / dz
    slopes = np.maximum.accumulate(np.maximum(slopes, 0.0)).astype(np.float32)
    out = np.empty_like(force, dtype=np.float32)
    out[0] = np.float32(f0)
    out[1:] = np.float32(f0) + np.cumsum(slopes * dz, dtype=np.float32)
    return out


def _finite_difference_tangent(depths: np.ndarray, force: np.ndarray) -> np.ndarray:
    if depths.size < 2:
        return np.zeros_like(force, dtype=np.float32)
    return np.gradient(force.astype(np.float32), depths.astype(np.float32), edge_order=1).astype(np.float32)


def _late_early_slope_ratio(depths: np.ndarray, force: np.ndarray) -> float:
    early = _segment_slope(depths, force, 0.10, 0.35)
    late = _segment_slope(depths, force, 0.65, 0.90)
    if early <= 1e-12:
        return 0.0
    return float(late / early)


def _segment_slope(depths: np.ndarray, force: np.ndarray, lo: float, hi: float) -> float:
    z_min = float(np.min(depths))
    z_max = float(np.max(depths))
    span = z_max - z_min
    if span <= 1e-12:
        return 0.0
    keep = (depths >= z_min + lo * span) & (depths <= z_min + hi * span)
    if int(np.count_nonzero(keep)) < 2:
        return 0.0
    z = depths[keep].astype(np.float64)
    f = force[keep].astype(np.float64)
    zc = z - float(np.mean(z))
    denom = float(np.sum(zc * zc))
    if denom <= 1e-18:
        return 0.0
    return float(np.sum(zc * (f - float(np.mean(f)))) / denom)


def _lump_top_depth_from_surface(phantom: PhantomConfig, lump: LumpSpec) -> float:
    top_z = float(lump.center[2]) + _lump_z_extent(lump)
    return max(float(phantom.height) - top_z, 0.0)


def _lump_z_extent(lump: LumpSpec) -> float:
    rx, ry, rz = (float(v) for v in lump.radii)
    if lump.shape == "capsule":
        return rx + rz
    return rz


def _shape_response_parameters(shape: str) -> tuple[float, float, float]:
    if shape == "box":
        return 1.18, 0.72, 1.06
    if shape == "cylinder":
        return 1.10, 0.82, 1.02
    if shape == "capsule":
        return 1.04, 1.05, 0.96
    if shape == "ellipsoid":
        return 0.94, 1.18, 0.92
    return 1.0, 1.0, 1.0
