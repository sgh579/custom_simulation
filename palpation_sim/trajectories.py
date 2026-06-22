from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import PhantomConfig, ScanConfig


@dataclass(frozen=True)
class PointedEllipseTrajectoryConfig:
    amplitude_min: float = 0.0006
    amplitude_max: float = 0.0030
    aspect_min: float = 0.30
    aspect_max: float = 0.85
    cycles_min: float = 0.75
    cycles_max: float = 1.35
    sharpness_min: float = 1.15
    sharpness_max: float = 2.40
    skew_min: float = -0.45
    skew_max: float = 0.45
    pointiness_min: float = 0.10
    pointiness_max: float = 0.45

    def to_dict(self) -> dict[str, float | str]:
        return {
            "mode": "pointed_ellipse",
            "amplitude_min": self.amplitude_min,
            "amplitude_max": self.amplitude_max,
            "aspect_min": self.aspect_min,
            "aspect_max": self.aspect_max,
            "cycles_min": self.cycles_min,
            "cycles_max": self.cycles_max,
            "sharpness_min": self.sharpness_min,
            "sharpness_max": self.sharpness_max,
            "skew_min": self.skew_min,
            "skew_max": self.skew_max,
            "pointiness_min": self.pointiness_min,
            "pointiness_max": self.pointiness_max,
        }


def sample_pointed_ellipse_offsets(
    rng: np.random.Generator,
    scan: ScanConfig,
    phantom: PhantomConfig,
    config: PointedEllipseTrajectoryConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample close-to-vertical, start/end-centered lateral probe trajectories."""

    cfg = config or PointedEllipseTrajectoryConfig()
    _validate_config(cfg, scan, phantom)
    s = np.linspace(0.0, 1.0, int(scan.press_steps), dtype=np.float32)
    envelope_base = np.sin(np.pi * s).astype(np.float32)

    offsets = np.zeros((scan.grid_h, scan.grid_w, scan.press_steps, 2), dtype=np.float32)
    max_offsets = np.zeros((scan.grid_h, scan.grid_w), dtype=np.float32)
    path_lengths = np.zeros((scan.grid_h, scan.grid_w), dtype=np.float32)

    for row in range(scan.grid_h):
        for col in range(scan.grid_w):
            amplitude = float(rng.uniform(cfg.amplitude_min, cfg.amplitude_max))
            aspect = float(rng.uniform(cfg.aspect_min, cfg.aspect_max))
            cycles = float(rng.uniform(cfg.cycles_min, cfg.cycles_max))
            sharpness = float(rng.uniform(cfg.sharpness_min, cfg.sharpness_max))
            skew = float(rng.uniform(cfg.skew_min, cfg.skew_max))
            pointiness = float(rng.uniform(cfg.pointiness_min, cfg.pointiness_max))
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            rotation = float(rng.uniform(0.0, 2.0 * np.pi))

            envelope = np.power(np.maximum(envelope_base, 0.0), sharpness)
            theta = (2.0 * np.pi * cycles * s + phase).astype(np.float32)
            taper = np.clip(1.0 + skew * (2.0 * s - 1.0), 0.20, 1.80).astype(np.float32)
            point = (1.0 + pointiness * np.sign(np.cos(theta)) * np.power(np.abs(np.cos(theta)), 2.0)).astype(
                np.float32
            )

            local_x = amplitude * envelope * point * np.cos(theta)
            local_y = amplitude * aspect * envelope * taper * np.sin(theta)
            c = float(np.cos(rotation))
            r = float(np.sin(rotation))
            xy = np.stack((c * local_x - r * local_y, r * local_x + c * local_y), axis=-1).astype(np.float32)
            max_norm = float(np.linalg.norm(xy, axis=-1).max())
            if max_norm > amplitude:
                xy *= np.float32(amplitude / max_norm)
            xy[0] = 0.0
            xy[-1] = 0.0
            offsets[row, col] = xy
            max_offsets[row, col] = float(np.linalg.norm(xy, axis=-1).max())
            path_lengths[row, col] = float(np.linalg.norm(np.diff(xy, axis=0), axis=-1).sum())

    metadata = {
        **cfg.to_dict(),
        "max_offset_m": float(max_offsets.max()),
        "mean_max_offset_m": float(max_offsets.mean()),
        "mean_lateral_path_length_m": float(path_lengths.mean()),
        "press_steps": int(scan.press_steps),
        "grid_h": int(scan.grid_h),
        "grid_w": int(scan.grid_w),
    }
    return offsets, metadata


def _validate_config(cfg: PointedEllipseTrajectoryConfig, scan: ScanConfig, phantom: PhantomConfig) -> None:
    if int(scan.press_steps) < 2:
        raise ValueError("pointed ellipse trajectories require at least two press steps")
    if cfg.amplitude_min < 0.0 or cfg.amplitude_max <= 0.0 or cfg.amplitude_min > cfg.amplitude_max:
        raise ValueError("invalid trajectory amplitude range")
    if cfg.aspect_min <= 0.0 or cfg.aspect_min > cfg.aspect_max:
        raise ValueError("invalid trajectory aspect range")
    if cfg.cycles_min <= 0.0 or cfg.cycles_min > cfg.cycles_max:
        raise ValueError("invalid trajectory cycle range")
    if cfg.sharpness_min <= 0.0 or cfg.sharpness_min > cfg.sharpness_max:
        raise ValueError("invalid trajectory sharpness range")
    if cfg.pointiness_min < 0.0 or cfg.pointiness_min > cfg.pointiness_max:
        raise ValueError("invalid trajectory pointiness range")
    spacing_x = phantom.size_x / max(int(scan.grid_w) - 1, 1)
    spacing_y = phantom.size_y / max(int(scan.grid_h) - 1, 1)
    conservative_limit = 0.35 * min(spacing_x, spacing_y, float(scan.probe_radius))
    if cfg.amplitude_max > conservative_limit:
        raise ValueError(
            f"trajectory amplitude_max {cfg.amplitude_max:.6g} is large for the scan spacing/probe radius; "
            f"use <= {conservative_limit:.6g}"
        )
