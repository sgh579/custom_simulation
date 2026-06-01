from __future__ import annotations

import numpy as np

XIAO_DEPTH_CLASSES_MM: tuple[int, ...] = (0, 5, 8, 10)


def xiao_sequence_from_press(
    press: np.ndarray,
    *,
    sequence_length: int = 50,
    ratio_eps: float = 1e-6,
) -> np.ndarray:
    """Convert a raw [z, Fz] palpation curve to [Fz, z, Fz/z] sequence features."""
    press = np.asarray(press, dtype=np.float32)
    if press.ndim != 2 or press.shape[-1] < 2:
        raise ValueError(f"Expected press shape [T, 2], got {press.shape}")
    valid = np.isfinite(press[:, 0]) & np.isfinite(press[:, 1])
    if int(valid.sum()) < 2:
        return np.zeros((sequence_length, 3), dtype=np.float32)

    z = press[valid, 0].astype(np.float32, copy=False)
    fz = press[valid, 1].astype(np.float32, copy=False)
    z = z - np.float32(z[0])
    fz = fz - np.float32(fz[0])
    if abs(float(np.nanmin(z))) > abs(float(np.nanmax(z))):
        z = -z
    if abs(float(np.nanmin(fz))) > abs(float(np.nanmax(fz))):
        fz = -fz
    ratio = np.divide(fz, z, out=np.zeros_like(fz), where=np.abs(z) > np.float32(ratio_eps))
    sequence = np.stack([fz, z, ratio], axis=-1).astype(np.float32)
    return resample_xiao_sequence(sequence, sequence_length)


def resample_xiao_sequence(sequence: np.ndarray, sequence_length: int = 50) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2:
        raise ValueError(f"Expected a 2D sequence, got {sequence.shape}")
    if sequence.shape[0] == sequence_length:
        return np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if sequence.shape[0] < 2:
        return np.zeros((sequence_length, sequence.shape[1]), dtype=np.float32)

    src_t = np.linspace(0.0, 1.0, sequence.shape[0], dtype=np.float32)
    dst_t = np.linspace(0.0, 1.0, sequence_length, dtype=np.float32)
    out = np.empty((sequence_length, sequence.shape[1]), dtype=np.float32)
    for channel in range(sequence.shape[1]):
        out[:, channel] = np.interp(dst_t, src_t, sequence[:, channel]).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def normalize_sequence_samplewise(sequence: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    mean = np.nanmean(sequence, axis=0, keepdims=True)
    std = np.nanstd(sequence, axis=0, keepdims=True)
    normalized = (sequence - mean) / (std + np.float32(eps))
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
