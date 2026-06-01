from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import ensure_chw, extract_feature_map, fz_to_channel_map, normalize_feature_map, presses_to_channel_map


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
