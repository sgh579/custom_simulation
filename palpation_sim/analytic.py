from __future__ import annotations

import json
from typing import Sequence

import numpy as np

from .config import MaterialConfig, PhantomConfig, ScanConfig
from .phantom import LumpSpec, lumps_to_json, mask_for_scan_grid, normalize_lumps


def run_analytic_sample(
    phantom: PhantomConfig,
    material: MaterialConfig,
    scan: ScanConfig,
    lumps: LumpSpec | Sequence[LumpSpec],
    rng: np.random.Generator,
    noise_std: float = 0.03,
) -> dict[str, np.ndarray | str]:
    """Fast deterministic-ish surrogate for smoke tests and ML pipeline checks.

    This is not a replacement for Newton/VBD. It produces Hertz-like monotonic
    force curves with a localized stiffness bump from the lump so the NN stack
    can be validated without launching the simulator.
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

    lump_list = normalize_lumps(lumps)
    base_k = 950.0 * (material.k_mu / 2.0e5) ** 0.45

    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            xy = np.asarray([x, y], dtype=np.float32)
            lump_gain = 0.0
            spatial_peak = 0.0
            for lump in lump_list:
                center_xy = np.asarray(lump.center[:2], dtype=np.float32)
                radii_xy = np.asarray(lump.radii[:2], dtype=np.float32)
                lateral_sigma = 0.8 * float(np.max(radii_xy)) + scan.probe_radius
                dist = float(np.linalg.norm((xy - center_xy) / max(lateral_sigma, 1e-6)))
                spatial_gain = float(np.exp(-0.5 * dist * dist))
                center_depth = max(float(phantom.height - lump.center[2]), 0.0)
                depth_gain = float(np.exp(-center_depth / max(0.45 * phantom.height, 1e-6)))
                lump_gain += (lump.stiffness_multiplier - 1.0) * spatial_gain * (0.25 + 0.75 * depth_gain)
                spatial_peak = max(spatial_peak, spatial_gain)
            local_k = base_k * (1.0 + lump_gain)

            force = local_k * np.power(np.maximum(depths, 0.0), 1.35)
            if noise_std > 0.0:
                scale = noise_std * max(float(np.max(force)), 1e-4)
                force = force + rng.normal(0.0, scale, size=force.shape).astype(np.float32)
            force = np.maximum.accumulate(np.maximum(force, 0.0)).astype(np.float32)

            z = phantom.height + scan.probe_radius + scan.preload_gap - depths
            probe_pose[row, col, :, 0] = x
            probe_pose[row, col, :, 1] = y
            probe_pose[row, col, :, 2] = z
            probe_pose[row, col, :, 6] = 1.0

            presses[row, col, :, 0] = depths
            presses[row, col, :, 1] = force
            fz[row, col] = force
            contact_features[row, col, :, 0] = np.clip(depths / max(scan.max_indentation, 1e-6), 0.0, 1.0) * 12.0
            contact_features[row, col, :, 1] = np.maximum(depths - scan.preload_gap, 0.0)
            contact_features[row, col, :, 2] = z
            contact_features[row, col, :, 3] = -1.0
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
        "lump_json": first_lump_json,
        "lumps_json": lumps_to_json(lump_list, phantom),
        "num_lumps": np.asarray(len(lump_list), dtype=np.int32),
        "backend": np.asarray("analytic"),
    }
