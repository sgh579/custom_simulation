from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_highres_grid_unet_transfer.py"
SPEC = importlib.util.spec_from_file_location("evaluate_highres_grid_unet_transfer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_checkpoint_dataset_normalization_uses_saved_channel_stats() -> None:
    value = np.asarray(
        [[[[1.0]], [[5.0]]], [[[3.0]], [[9.0]]]], dtype=np.float32
    )
    contract = {
        "schema_version": 1,
        "preprocessing": "raw_fz",
        "mode": "dataset",
        "mean": [2.0, 7.0],
        "std": [1.0, 2.0],
        "epsilon": 0.0,
    }
    actual = MODULE.apply_checkpoint_normalization(value, contract)
    expected = np.asarray(
        [[[[-1.0]], [[-1.0]]], [[[1.0]], [[1.0]]]], dtype=np.float32
    )
    np.testing.assert_array_equal(actual, expected)


def test_channel_resampling_preserves_endpoints() -> None:
    value = np.arange(3, dtype=np.float32).reshape(1, 3, 1, 1)
    actual = MODULE.resample_channels(value, 5)
    np.testing.assert_allclose(actual[:, (0, -1)], value[:, (0, -1)])


def test_per_sample_response_normalization_removes_preload_and_force_scale() -> None:
    base = np.asarray([0.0, 1.0, 2.0, 4.0], dtype=np.float32).reshape(1, 4, 1, 1)
    first = base + 10.0
    second = base * 7.0 + 300.0
    contract = {
        "schema_version": 1,
        "mode": "per_sample",
        "preprocessing": "response=max(fz-fz_at_first_depth,0)",
    }
    actual_first = MODULE.apply_checkpoint_normalization(first, contract)
    actual_second = MODULE.apply_checkpoint_normalization(second, contract)
    np.testing.assert_allclose(actual_first, actual_second, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(actual_first[:, :1], np.zeros_like(actual_first[:, :1]))
