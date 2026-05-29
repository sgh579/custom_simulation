from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import ensure_chw, extract_feature_map, normalize_feature_map


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
    """Load palpation process data and convert it to feature maps for U-Net.

    Expected sample format:
    - ``presses``: [H, W, T, 2], with channels indentation and Fz
    - ``mask``: [H, W], binary inclusion projection label

    Precomputed ``features`` [C, H, W] or [H, W, C] are also accepted.
    """

    def __init__(self, path_or_files: str | Path | Sequence[str | Path], normalize: bool = True) -> None:
        self.files = resolve_npz_files(path_or_files)
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.files[idx]
        with np.load(path) as sample:
            if "features" in sample:
                features = ensure_chw(sample["features"])
            elif "presses" in sample:
                features = extract_feature_map(sample["presses"])
            else:
                raise KeyError(f"{path} must contain either 'features' or 'presses'")

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
