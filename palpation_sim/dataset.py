from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import (
    ensure_chw,
    extract_feature_map,
    fz_to_channel_map,
    normalize_feature_map,
    presses_to_channel_map,
)
from .xiao2020 import (
    XIAO_DEPTH_CLASSES_MM,
    normalize_sequence_samplewise,
    resample_xiao_sequence,
    xiao_sequence_from_press,
)

SequenceNormalizeMode = Literal["none", "sample", "dataset"]


def resolve_npz_files(path_or_files: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(path_or_files, (str, Path)):
        path = Path(path_or_files)
        files = sorted(path.glob("*.npz")) if path.is_dir() else [path]
    else:
        files = [Path(file) for file in path_or_files]
    if not files:
        raise FileNotFoundError(f"No .npz files found in {path_or_files}")
    return files


class PalpationProcessDataset(Dataset):
    """Load palpation process data and convert it to U-Net input maps.

    Expected sample format:
    - ``fz``: [H, W, T], raw probe reaction force trajectory
    - ``presses``: [H, W, T, 2], with channels indentation and Fz
    - ``mask``: [H, W], binary inclusion projection label

    By default, the raw Fz trajectory is used as channels: [H, W, T]
    becomes [T, H, W]. The full raw press record can be used by setting
    ``input_mode="presses"``. Precomputed engineered ``features`` [C, H, W]
    or [H, W, C] can still be used by setting ``input_mode="features"``.
    """

    def __init__(
        self,
        path_or_files: str | Path | Sequence[str | Path],
        normalize: bool = True,
        input_mode: str = "fz",
    ) -> None:
        if input_mode not in {"fz", "presses", "features", "auto"}:
            raise ValueError("input_mode must be one of: 'fz', 'presses', 'features', 'auto'")
        self.files = resolve_npz_files(path_or_files)
        self.normalize = normalize
        self.input_mode = input_mode

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.files[idx]
        with np.load(path) as sample:
            if self.input_mode == "fz":
                features = _load_fz_channels(sample, path)
            elif self.input_mode == "presses":
                if "presses" not in sample:
                    raise KeyError(f"{path} must contain 'presses' when input_mode='presses'")
                features = presses_to_channel_map(sample["presses"])
            elif self.input_mode == "features" and "features" in sample:
                features = ensure_chw(sample["features"])
            elif self.input_mode == "features" and "presses" in sample:
                features = extract_feature_map(sample["presses"])
            elif self.input_mode == "auto" and ("fz" in sample or "presses" in sample):
                features = _load_fz_channels(sample, path)
            elif self.input_mode == "auto" and "features" in sample:
                features = ensure_chw(sample["features"])
            else:
                raise KeyError(f"{path} must contain data compatible with input_mode='{self.input_mode}'")

            if "mask" not in sample:
                raise KeyError(f"{path} must contain 'mask'")
            mask = sample["mask"].astype(np.float32)

        if self.normalize:
            features = normalize_feature_map(features)
        if mask.ndim == 2:
            mask = mask[None, ...]
        elif mask.ndim == 3 and mask.shape[-1] == 1:
            mask = np.moveaxis(mask, -1, 0)

        return torch.from_numpy(features.astype(np.float32)), torch.from_numpy(mask.astype(np.float32))


def _load_fz_channels(sample: np.lib.npyio.NpzFile, path: Path) -> np.ndarray:
    if "fz" in sample:
        return fz_to_channel_map(sample["fz"])
    if "presses" in sample:
        return fz_to_channel_map(sample["presses"])
    raise KeyError(f"{path} must contain 'fz' or 'presses' when input_mode='fz'")


class XiaoDepthSequenceDataset(Dataset):
    """Load one-palpation sequences for Xiao et al. 2020 LSTM classification.

    Expected sample format:
    - ``sequence``: [T, 3], ordered as Fz, z, Fz/z
    - ``label``: scalar class index for depths ``(0, 5, 8, 10)`` mm by default

    For convenience, raw ``press`` [T, 2] or 1x1-grid ``presses`` [1, 1, T, 2]
    arrays are also accepted and converted to the paper's vector sequence.
    """

    def __init__(
        self,
        path_or_files: str | Path | Sequence[str | Path],
        *,
        sequence_length: int = 50,
        normalize: SequenceNormalizeMode = "dataset",
        class_depths_mm: Sequence[int | float] = XIAO_DEPTH_CLASSES_MM,
        sequence_mean: Sequence[float] | np.ndarray | None = None,
        sequence_std: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        self.files = resolve_npz_files(path_or_files)
        self.sequence_length = int(sequence_length)
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if normalize not in ("none", "sample", "dataset"):
            raise ValueError("normalize must be 'none', 'sample', or 'dataset'")
        self.normalize = normalize
        self.class_depths_mm = tuple(float(value) for value in class_depths_mm)
        self.sequence_mean: np.ndarray | None = None
        self.sequence_std: np.ndarray | None = None
        if self.normalize == "dataset":
            if sequence_mean is None or sequence_std is None:
                self.sequence_mean, self.sequence_std = compute_xiao_sequence_stats(
                    self.files,
                    sequence_length=self.sequence_length,
                )
            else:
                self.sequence_mean = np.asarray(sequence_mean, dtype=np.float32).reshape(1, -1)
                self.sequence_std = np.asarray(sequence_std, dtype=np.float32).reshape(1, -1)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.files[idx]
        with np.load(path) as sample:
            sequence = load_xiao_sequence_from_npz(sample, sequence_length=self.sequence_length)
            label = label_from_npz(sample, self.class_depths_mm)

        if self.normalize == "sample":
            sequence = normalize_sequence_samplewise(sequence)
        elif self.normalize == "dataset":
            assert self.sequence_mean is not None and self.sequence_std is not None
            sequence = (sequence - self.sequence_mean) / (self.sequence_std + np.float32(1e-6))
            sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        features = torch.from_numpy(sequence.astype(np.float32, copy=False))
        target = torch.tensor(label, dtype=torch.long)
        return features, target


def load_xiao_sequence_from_npz(sample: np.lib.npyio.NpzFile, *, sequence_length: int = 50) -> np.ndarray:
    if "sequence" in sample:
        sequence = np.asarray(sample["sequence"], dtype=np.float32)
        if sequence.ndim != 2 or sequence.shape[-1] != 3:
            raise ValueError(f"Expected sequence shape [T, 3], got {sequence.shape}")
        return resample_xiao_sequence(sequence, sequence_length)

    if "press" in sample:
        press = np.asarray(sample["press"], dtype=np.float32)
    elif "presses" in sample:
        presses = np.asarray(sample["presses"], dtype=np.float32)
        if presses.ndim != 4 or presses.shape[-1] < 2:
            raise ValueError(f"Expected presses shape [H, W, T, 2], got {presses.shape}")
        if presses.shape[0] != 1 or presses.shape[1] != 1:
            raise ValueError("XiaoDepthSequenceDataset can only infer sequences from 1x1-grid presses")
        press = presses[0, 0, :, :2]
    else:
        raise KeyError("sample must contain 'sequence', 'press', or 1x1-grid 'presses'")

    return xiao_sequence_from_press(press, sequence_length=sequence_length)


def label_from_npz(sample: np.lib.npyio.NpzFile, class_depths_mm: Sequence[float]) -> int:
    if "label" in sample:
        return int(np.asarray(sample["label"]).reshape(()))
    if "depth_mm" not in sample:
        raise KeyError("sample must contain either 'label' or 'depth_mm'")
    depth = float(np.asarray(sample["depth_mm"]).reshape(()))
    distances = [abs(depth - class_depth) for class_depth in class_depths_mm]
    return int(np.argmin(distances))


def compute_xiao_sequence_stats(
    path_or_files: str | Path | Sequence[str | Path],
    *,
    sequence_length: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    files = resolve_npz_files(path_or_files)
    count = 0
    sum_features = np.zeros((1, 3), dtype=np.float64)
    sum_squares = np.zeros((1, 3), dtype=np.float64)
    for path in files:
        with np.load(path) as sample:
            sequence = load_xiao_sequence_from_npz(sample, sequence_length=sequence_length).astype(np.float64)
        sum_features += np.sum(sequence, axis=0, keepdims=True)
        sum_squares += np.sum(sequence * sequence, axis=0, keepdims=True)
        count += int(sequence.shape[0])
    if count == 0:
        return np.zeros((1, 3), dtype=np.float32), np.ones((1, 3), dtype=np.float32)
    mean = sum_features / float(count)
    variance = np.maximum(sum_squares / float(count) - mean * mean, 1e-12)
    std = np.sqrt(variance)
    return mean.astype(np.float32), std.astype(np.float32)
