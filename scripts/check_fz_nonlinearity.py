from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether generated F-z curves are nonlinear/convex.")
    parser.add_argument("npz", type=Path)
    parser.add_argument("--min-median-ratio", type=float, default=1.5)
    parser.add_argument("--min-convex-fraction", type=float, default=0.95)
    parser.add_argument("--tol", type=float, default=1e-5)
    args = parser.parse_args()

    with np.load(args.npz, allow_pickle=True) as sample:
        if "fz" in sample:
            fz = np.asarray(sample["fz"], dtype=np.float32)
        else:
            fz = np.asarray(sample["presses"][..., 1], dtype=np.float32)
        if "indentation_depth" in sample:
            depth = np.asarray(sample["indentation_depth"], dtype=np.float32)
        else:
            depth = np.asarray(sample["presses"][..., 0], dtype=np.float32)
        backend = _sample_scalar(sample, "backend")

    if fz.ndim != 3 or depth.ndim != 3:
        raise SystemExit(f"Expected fz/depth shape [H,W,T], got {fz.shape} and {depth.shape}")

    ratios = np.zeros(fz.shape[:2], dtype=np.float32)
    convex = np.zeros(fz.shape[:2], dtype=np.float32)
    slope_start = np.zeros(fz.shape[:2], dtype=np.float32)
    slope_end = np.zeros(fz.shape[:2], dtype=np.float32)

    for row in range(fz.shape[0]):
        for col in range(fz.shape[1]):
            z = depth[row, col].astype(np.float64)
            f = (fz[row, col] - fz[row, col, 0]).astype(np.float64)
            slopes = _slopes(z, f)
            if slopes.size == 0:
                continue
            slope_start[row, col] = float(slopes[0])
            slope_end[row, col] = float(slopes[-1])
            convex[row, col] = float(np.mean(np.diff(slopes) >= -abs(args.tol))) if slopes.size > 1 else 1.0
            early = _segment_slope(z, f, 0.10, 0.35)
            late = _segment_slope(z, f, 0.65, 0.90)
            ratios[row, col] = float(late / early) if early > 1e-12 else 0.0

    summary = {
        "npz": str(args.npz),
        "backend": backend,
        "curve_shape": list(fz.shape),
        "median_late_early_slope_ratio": float(np.median(ratios)),
        "min_late_early_slope_ratio": float(np.min(ratios)),
        "median_convex_fraction": float(np.median(convex)),
        "min_convex_fraction": float(np.min(convex)),
        "median_first_step_slope": float(np.median(slope_start)),
        "median_last_step_slope": float(np.median(slope_end)),
        "pass": bool(np.median(ratios) >= args.min_median_ratio and np.min(convex) >= args.min_convex_fraction),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["pass"]:
        sys.exit(1)


def _sample_scalar(sample: np.lib.npyio.NpzFile, key: str) -> str | None:
    if key not in sample:
        return None
    value = sample[key]
    try:
        return str(value.item())
    except ValueError:
        return str(value)


def _slopes(z: np.ndarray, f: np.ndarray) -> np.ndarray:
    if z.size < 2:
        return np.zeros(0, dtype=np.float64)
    dz = np.diff(z)
    df = np.diff(f)
    out = np.zeros_like(df, dtype=np.float64)
    np.divide(df, dz, out=out, where=np.abs(dz) > 1e-12)
    return out


def _segment_slope(z: np.ndarray, f: np.ndarray, lo: float, hi: float) -> float:
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    span = z_max - z_min
    if span <= 1e-12:
        return 0.0
    keep = (z >= z_min + lo * span) & (z <= z_min + hi * span)
    if int(np.count_nonzero(keep)) < 2:
        return 0.0
    zz = z[keep]
    ff = f[keep]
    zc = zz - float(np.mean(zz))
    denom = float(np.sum(zc * zc))
    if denom <= 1e-18:
        return 0.0
    return float(np.sum(zc * (ff - float(np.mean(ff)))) / denom)


if __name__ == "__main__":
    main()
