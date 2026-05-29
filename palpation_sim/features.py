from __future__ import annotations

from typing import Iterable

import numpy as np


FEATURE_NAMES: list[str] = [
    "f_max",
    "z_max",
    "global_stiffness",
    "early_stiffness",
    "late_stiffness",
    "loading_work",
    "hysteresis",
    "force_at_z25",
    "force_at_z50",
    "force_at_z75",
]


def extract_press_features(curve: np.ndarray) -> np.ndarray:
    if curve.ndim != 2 or curve.shape[-1] < 2:
        raise ValueError(f"Expected curve shape [T, 2], got {curve.shape}")

    z_raw = curve[:, 0]
    f_raw = curve[:, 1]
    valid = np.isfinite(z_raw) & np.isfinite(f_raw)
    if int(valid.sum()) < 2:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    z, f = _make_compression_positive(z_raw[valid], f_raw[valid])
    peak_idx = int(np.argmax(z))
    loading_z = z[: peak_idx + 1]
    loading_f = f[: peak_idx + 1]
    if loading_z.size < 2:
        loading_z = z
        loading_f = f

    f_max = float(np.max(loading_f))
    z_max = float(np.max(loading_z))
    global_stiffness = _slope(loading_z, loading_f)
    early_stiffness = _segment_slope(loading_z, loading_f, 0.10, 0.40)
    late_stiffness = _segment_slope(loading_z, loading_f, 0.60, 0.90)
    loading_work = _trapezoid(loading_f, loading_z) if loading_z.size >= 2 else 0.0

    if peak_idx + 2 < z.size:
        unloading_z = z[peak_idx:]
        unloading_f = f[peak_idx:]
        load_area = abs(_trapezoid(loading_f, loading_z))
        unload_area = abs(_trapezoid(unloading_f[::-1], unloading_z[::-1]))
        hysteresis = load_area - unload_area
    else:
        hysteresis = 0.0

    features = np.array(
        [
            f_max,
            z_max,
            global_stiffness,
            early_stiffness,
            late_stiffness,
            loading_work,
            hysteresis,
            _interp_force(loading_z, loading_f, 0.25),
            _interp_force(loading_z, loading_f, 0.50),
            _interp_force(loading_z, loading_f, 0.75),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def extract_feature_map(presses: np.ndarray) -> np.ndarray:
    if presses.ndim != 4 or presses.shape[-1] < 2:
        raise ValueError(f"Expected presses shape [H, W, T, 2], got {presses.shape}")

    fast = _extract_feature_map_fast_loading(presses)
    if fast is not None:
        return fast

    h, w, _, _ = presses.shape
    feature_map = np.zeros((len(FEATURE_NAMES), h, w), dtype=np.float32)
    for row in range(h):
        for col in range(w):
            feature_map[:, row, col] = extract_press_features(presses[row, col])
    return feature_map


def normalize_feature_map(feature_map: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    feature_map = np.asarray(feature_map, dtype=np.float32)
    mean = np.nanmean(feature_map, axis=(1, 2), keepdims=True)
    std = np.nanstd(feature_map, axis=(1, 2), keepdims=True)
    normalized = (feature_map - mean) / (std + eps)
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def ensure_chw(features: np.ndarray) -> np.ndarray:
    if features.ndim != 3:
        raise ValueError(f"Expected a 3D feature array, got {features.shape}")
    if features.shape[0] <= 64 and features.shape[1] > 1 and features.shape[2] > 1:
        return features.astype(np.float32)
    return np.moveaxis(features, -1, 0).astype(np.float32)


def feature_names() -> Iterable[str]:
    return tuple(FEATURE_NAMES)


def _extract_feature_map_fast_loading(presses: np.ndarray) -> np.ndarray | None:
    z_raw = np.asarray(presses[..., 0], dtype=np.float32)
    f_raw = np.asarray(presses[..., 1], dtype=np.float32)
    if z_raw.shape[-1] < 2 or not (np.isfinite(z_raw).all() and np.isfinite(f_raw).all()):
        return None

    z = z_raw - z_raw[..., :1]
    f = f_raw - f_raw[..., :1]
    z_min = np.nanmin(z, axis=-1, keepdims=True)
    z_max = np.nanmax(z, axis=-1, keepdims=True)
    f_min = np.nanmin(f, axis=-1, keepdims=True)
    f_max = np.nanmax(f, axis=-1, keepdims=True)
    z = np.where(np.abs(z_min) > np.abs(z_max), -z, z)
    f = np.where(np.abs(f_min) > np.abs(f_max), -f, f)

    # The generated palpation samples are loading-only curves: indentation rises
    # monotonically until the final point. Keep the scalar fallback for any future
    # hysteresis/unloading sample that does not match this shape.
    if not np.all(np.argmax(z, axis=-1) == z.shape[-1] - 1):
        return None

    h, w, t = z.shape
    z2 = z.reshape(-1, t)
    f2 = f.reshape(-1, t)
    n = z2.shape[0]
    features = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)

    features[:, 0] = np.max(f2, axis=1)
    features[:, 1] = np.max(z2, axis=1)
    features[:, 2] = _slope_many(z2, f2)
    features[:, 3] = _segment_slope_many(z2, f2, 0.10, 0.40)
    features[:, 4] = _segment_slope_many(z2, f2, 0.60, 0.90)
    features[:, 5] = _trapz_many(f2, z2)
    features[:, 6] = 0.0
    features[:, 7] = _interp_force_many(z2, f2, 0.25)
    features[:, 8] = _interp_force_many(z2, f2, 0.50)
    features[:, 9] = _interp_force_many(z2, f2, 0.75)

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return np.moveaxis(features.reshape(h, w, len(FEATURE_NAMES)), -1, 0).astype(np.float32)


def _slope_many(z: np.ndarray, f: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is None:
        z_mean = np.mean(z, axis=1, keepdims=True)
        f_mean = np.mean(f, axis=1, keepdims=True)
        z_centered = z - z_mean
        denom = np.sum(z_centered * z_centered, axis=1)
        numer = np.sum(z_centered * (f - f_mean), axis=1)
    else:
        weights = mask.astype(np.float32)
        counts = np.sum(weights, axis=1, keepdims=True)
        valid = counts >= 2.0
        counts = np.maximum(counts, 1.0)
        z_mean = np.sum(z * weights, axis=1, keepdims=True) / counts
        f_mean = np.sum(f * weights, axis=1, keepdims=True) / counts
        z_centered = (z - z_mean) * weights
        denom = np.sum(z_centered * z_centered, axis=1)
        numer = np.sum(z_centered * (f - f_mean) * weights, axis=1)
        denom = np.where(valid[:, 0], denom, 0.0)
    out = np.zeros_like(numer, dtype=np.float32)
    np.divide(numer, denom, out=out, where=denom >= 1e-12)
    return out.astype(np.float32)


def _segment_slope_many(z: np.ndarray, f: np.ndarray, low: float, high: float) -> np.ndarray:
    z_min = np.min(z, axis=1, keepdims=True)
    z_max = np.max(z, axis=1, keepdims=True)
    span = z_max - z_min
    keep = (span >= 1e-12) & (z >= z_min + low * span) & (z <= z_min + high * span)
    return _slope_many(z, f, keep)


def _trapz_many(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    dx = x[:, 1:] - x[:, :-1]
    area = 0.5 * (y[:, 1:] + y[:, :-1]) * dx
    return np.sum(area, axis=1).astype(np.float32)


def _interp_force_many(z: np.ndarray, f: np.ndarray, fraction: float) -> np.ndarray:
    z_min = np.min(z, axis=1)
    z_max = np.max(z, axis=1)
    span = z_max - z_min
    target = z_min + fraction * span
    monotonic = np.all(np.diff(z, axis=1) >= -1e-9, axis=1)
    if not bool(np.all(monotonic)):
        out = np.zeros(z.shape[0], dtype=np.float32)
        for i in range(z.shape[0]):
            out[i] = _interp_force(z[i], f[i], fraction)
        return out

    idx_hi = np.sum(z < target[:, None], axis=1)
    idx_hi = np.clip(idx_hi, 1, z.shape[1] - 1)
    idx_lo = idx_hi - 1
    rows = np.arange(z.shape[0])
    z0 = z[rows, idx_lo]
    z1 = z[rows, idx_hi]
    f0 = f[rows, idx_lo]
    f1 = f[rows, idx_hi]
    denom = z1 - z0
    alpha = np.where(np.abs(denom) > 1e-12, (target - z0) / denom, 0.0)
    out = f0 + alpha * (f1 - f0)
    return np.where(span >= 1e-12, out, np.max(f, axis=1)).astype(np.float32)


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    return float(integrate(y, x))


def _make_compression_positive(z: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z_rel = z.astype(np.float32) - np.float32(z[0])
    f_rel = f.astype(np.float32) - np.float32(f[0])
    if abs(float(np.nanmin(z_rel))) > abs(float(np.nanmax(z_rel))):
        z_rel = -z_rel
    if abs(float(np.nanmin(f_rel))) > abs(float(np.nanmax(f_rel))):
        f_rel = -f_rel
    return z_rel, f_rel


def _slope(z: np.ndarray, f: np.ndarray) -> float:
    if z.size < 2:
        return 0.0
    z_centered = z - float(np.mean(z))
    denom = float(np.sum(z_centered * z_centered))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(z_centered * (f - float(np.mean(f)))) / denom)


def _segment_slope(z: np.ndarray, f: np.ndarray, low: float, high: float) -> float:
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    span = z_max - z_min
    if span < 1e-12:
        return 0.0
    keep = (z >= z_min + low * span) & (z <= z_min + high * span)
    return _slope(z[keep], f[keep])


def _interp_force(z: np.ndarray, f: np.ndarray, fraction: float) -> float:
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    span = z_max - z_min
    if span < 1e-12:
        return float(np.max(f)) if f.size else 0.0
    order = np.argsort(z)
    z_sorted = z[order]
    f_sorted = f[order]
    z_unique, unique_idx = np.unique(z_sorted, return_index=True)
    f_unique = f_sorted[unique_idx]
    return float(np.interp(z_min + fraction * span, z_unique, f_unique))
