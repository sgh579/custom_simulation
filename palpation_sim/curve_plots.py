from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


os.environ.setdefault("MPLBACKEND", "Agg")

MM = 1.0e-3
CENTER_SAMPLE_NAME = "center_sphere_newton_sample.npz"
RUN_LABEL_PREFIX_RE = re.compile(r"^\d{8}-\d{6}-")


def draw_sample_curves(
    sample: Mapping[str, Any],
    path: str | Path,
    *,
    title: str | None = None,
    max_columns: int = 17,
) -> dict[str, object]:
    """Write representative F-z curves for one assembled scan sample."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    plot_info = _plot_sample_on_axis(ax, sample, title=title or path.parent.name, max_columns=max_columns)
    ax.set_xlabel("indentation [mm]")
    ax.legend(fontsize=7, ncol=min(5, len(plot_info["columns"])), loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {
        "plot": str(path),
        **plot_info,
    }


def draw_run_curves(
    run_dir: str | Path,
    *,
    output_name: str = "fz_curves.png",
    max_columns: int = 17,
) -> dict[str, object]:
    """Write representative F-z curves for an assembled run directory."""
    run_dir = Path(run_dir)
    sample_path = run_dir / CENTER_SAMPLE_NAME
    if not sample_path.exists():
        raise FileNotFoundError(f"assembled sample not found: {sample_path}")
    with np.load(sample_path, allow_pickle=False) as sample:
        result = draw_sample_curves(
            sample,
            run_dir / output_name,
            title=_display_run_label(run_dir),
            max_columns=max_columns,
        )
    result["sample"] = str(sample_path)
    return result


def draw_run_group_curves(
    run_root: str | Path,
    *,
    output_name: str = "all_run_fz_curves.png",
    max_columns: int = 17,
    draw_child_plots: bool = True,
) -> dict[str, object]:
    """Write one comparison plot for all assembled child runs under a root."""
    import matplotlib.pyplot as plt

    run_root = Path(run_root)
    run_dirs = _assembled_child_runs(run_root)
    skipped = [
        str(path)
        for path in sorted(run_root.iterdir())
        if path.is_dir() and not (path / CENTER_SAMPLE_NAME).exists()
    ]
    if not run_dirs:
        raise FileNotFoundError(f"no assembled child runs found under {run_root}")

    if draw_child_plots:
        child_plots = [draw_run_curves(run_dir, max_columns=max_columns) for run_dir in run_dirs]
    else:
        child_plots = []

    output_path = run_root / output_name
    fig_height = max(3.0, 2.35 * len(run_dirs))
    fig, axes = plt.subplots(len(run_dirs), 1, figsize=(8.6, fig_height), sharex=True)
    axes_list = np.atleast_1d(axes).tolist()
    panel_info: list[dict[str, object]] = []
    for index, (ax, run_dir) in enumerate(zip(axes_list, run_dirs)):
        with np.load(run_dir / CENTER_SAMPLE_NAME, allow_pickle=False) as sample:
            info = _plot_sample_on_axis(
                ax,
                sample,
                title=_display_run_label(run_dir),
                max_columns=max_columns,
            )
        panel_info.append({"run_dir": str(run_dir), "sample": str(run_dir / CENTER_SAMPLE_NAME), **info})
        if index == 0:
            ax.legend(fontsize=7, ncol=min(8, len(info["columns"])), loc="upper left", bbox_to_anchor=(0.0, 1.42))
    axes_list[-1].set_xlabel("indentation [mm]")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {
        "plot": str(output_path),
        "run_root": str(run_root),
        "child_plots": child_plots,
        "panels": panel_info,
        "skipped_child_dirs": skipped,
    }


def _plot_sample_on_axis(
    ax: Any,
    sample: Mapping[str, Any],
    *,
    title: str,
    max_columns: int,
) -> dict[str, object]:
    depth_all = np.asarray(sample["indentation_depth"], dtype=np.float32)
    fz_all = np.asarray(sample["fz"], dtype=np.float32)
    xy_all = np.asarray(sample["xy"], dtype=np.float32)
    row_idx = _center_row_index(xy_all)
    col_indices = _representative_columns(fz_all.shape[1], max_columns=max_columns)
    depth = depth_all[row_idx]
    fz = fz_all[row_idx]
    xy = xy_all[row_idx]

    for col in col_indices:
        x_mm = float(xy[col, 0] / MM)
        width = 2.5 if abs(x_mm) < 1.0e-4 else 1.35
        ax.plot(depth[col] / MM, fz[col], linewidth=width, label=f"x={x_mm:.0f}mm")

    y_mm = float(xy[0, 1] / MM)
    ax.set_title(f"{title} center row y={y_mm:.0f}mm", fontsize=10)
    ax.set_ylabel("Fz [N]")
    ax.grid(True, alpha=0.22)
    return {
        "row": int(row_idx),
        "columns": [int(col) for col in col_indices],
        "x_mm": [float(xy[col, 0] / MM) for col in col_indices],
        "y_mm": y_mm,
    }


def _assembled_child_runs(run_root: Path) -> list[Path]:
    return sorted(path for path in run_root.iterdir() if path.is_dir() and (path / CENTER_SAMPLE_NAME).exists())


def _center_row_index(xy: np.ndarray) -> int:
    y = np.asarray(xy[:, 0, 1], dtype=np.float32)
    return int(np.argmin(np.abs(y)))


def _representative_columns(count: int, *, max_columns: int) -> list[int]:
    max_columns = max(int(max_columns), 1)
    if count <= max_columns:
        return list(range(count))
    return sorted({int(v) for v in np.linspace(0, count - 1, max_columns)})


def _display_run_label(run_dir: Path) -> str:
    return RUN_LABEL_PREFIX_RE.sub("", run_dir.name)
