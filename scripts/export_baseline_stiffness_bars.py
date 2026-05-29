from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export equivalent-stiffness maps as 3D bar plots.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing .npz samples.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory. Defaults to --data-dir.")
    parser.add_argument("--pattern", type=str, default="sample_*.npz")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--html", action="store_true", help="Write interactive Plotly HTML for each sample.")
    parser.add_argument("--png", action="store_true", help="Write static PNG views for each sample.")
    parser.add_argument("--top-png", action="store_true", help="Also write a top-view static PNG.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    _load_dependencies()

    files = sorted(args.data_dir.glob(args.pattern))
    if args.max_samples is not None:
        files = files[: args.max_samples]
    if not files:
        raise SystemExit(f"No samples matched {args.data_dir / args.pattern}")

    out_dir = args.out_dir or args.data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_png = args.png or not args.html
    write_html = args.html

    for idx, path in enumerate(files, start=1):
        outputs = export_sample(
            path,
            out_dir,
            write_html=write_html,
            write_png=write_png,
            write_top_png=args.top_png,
            overwrite=args.overwrite,
        )
        print(f"[{idx}/{len(files)}] wrote {path.stem}: {', '.join(str(item.name) for item in outputs)}")


def export_sample(
    sample_path,
    out_dir,
    *,
    write_html: bool,
    write_png: bool,
    write_top_png: bool,
    overwrite: bool,
):
    sample_path = Path(sample_path)
    out_dir = Path(out_dir)
    stem = sample_path.stem
    stiffness_npy = out_dir / f"{stem}_baseline_stiffness.npy"
    png_path = out_dir / f"{stem}_baseline_stiffness_bar3d.png"
    top_png_path = out_dir / f"{stem}_baseline_stiffness_bar3d_top.png"
    html_path = out_dir / f"{stem}_baseline_stiffness_bar3d.html"
    stats_path = out_dir / f"{stem}_baseline_stiffness_bar3d_stats.json"

    expected = [stiffness_npy, stats_path]
    if write_png:
        expected.append(png_path)
    if write_top_png:
        expected.append(top_png_path)
    if write_html:
        expected.append(html_path)
    if not overwrite and all(path.exists() for path in expected):
        return expected

    with np.load(sample_path) as sample:
        if "presses" not in sample:
            raise ValueError(f"{sample_path} does not contain presses")
        stiffness = equivalent_stiffness_map(sample["presses"])
        xy = sample["xy"].astype(np.float64) if "xy" in sample else _default_xy(stiffness.shape)
        mask = sample["mask"].astype(np.uint8) if "mask" in sample else np.zeros(stiffness.shape, dtype=np.uint8)

    np.save(stiffness_npy, stiffness.astype(np.float32))
    x = xy[..., 0].astype(np.float64) * 1000.0
    y = xy[..., 1].astype(np.float64) * 1000.0
    z = stiffness.astype(np.float64) / 1000.0
    face_colors, norm = _bar_colors(z)

    if write_png:
        _write_static_bar_png(png_path, x, y, z, mask, face_colors, norm, elev=34, azim=-58, title=f"{stem} baseline k")
    if write_top_png:
        _write_static_bar_png(
            top_png_path,
            x,
            y,
            z,
            mask,
            face_colors,
            norm,
            elev=72,
            azim=-62,
            title=f"{stem} baseline k - top view",
        )
    if write_html:
        _write_interactive_bar_html(html_path, x, y, z, mask, norm, title=f"{stem} baseline equivalent stiffness")

    finite = stiffness[np.isfinite(stiffness)]
    stats = {
        "sample": str(sample_path),
        "shape": list(stiffness.shape),
        "unit": "N/m in npy, kN/m in plots",
        "min_n_per_m": float(finite.min()) if finite.size else 0.0,
        "max_n_per_m": float(finite.max()) if finite.size else 0.0,
        "mean_n_per_m": float(finite.mean()) if finite.size else 0.0,
        "median_n_per_m": float(np.median(finite)) if finite.size else 0.0,
        "percentiles_n_per_m": {str(p): float(np.percentile(finite, p)) for p in [1, 2, 5, 25, 50, 75, 95, 98, 99]}
        if finite.size
        else {},
        "gt_positive_cells": int(mask.sum()),
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return expected


def equivalent_stiffness_map(presses):
    presses = np.asarray(presses, dtype=np.float32)
    if presses.ndim != 4 or presses.shape[-1] < 2:
        raise ValueError(f"Expected presses shape [H, W, T, 2], got {presses.shape}")
    out = np.zeros(presses.shape[:2], dtype=np.float32)
    for row in range(presses.shape[0]):
        for col in range(presses.shape[1]):
            out[row, col] = equivalent_stiffness(presses[row, col, :, 0], presses[row, col, :, 1])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def equivalent_stiffness(displacement, force) -> float:
    z_raw = np.asarray(displacement, dtype=np.float32)
    f_raw = np.asarray(force, dtype=np.float32)
    valid = np.isfinite(z_raw) & np.isfinite(f_raw)
    if int(valid.sum()) < 2:
        return 0.0
    z = z_raw[valid] - np.float32(z_raw[valid][0])
    f = f_raw[valid] - np.float32(f_raw[valid][0])
    if abs(float(np.nanmin(z))) > abs(float(np.nanmax(z))):
        z = -z
    if abs(float(np.nanmin(f))) > abs(float(np.nanmax(f))):
        f = -f
    peak_idx = int(np.nanargmax(z))
    loading_z = z[: peak_idx + 1]
    loading_f = f[: peak_idx + 1]
    if loading_z.size < 2:
        loading_z = z
        loading_f = f
    peak_idx = int(np.nanargmax(loading_z))
    dz = float(loading_z[peak_idx] - loading_z[0])
    if abs(dz) < 1e-9:
        return 0.0
    df = float(loading_f[peak_idx] - loading_f[0])
    return max(df / dz, 0.0)


def _bar_geometry(x, y):
    x_vals = x[0]
    y_vals = y[:, 0]
    dx = float(np.median(np.diff(x_vals))) * 0.72 if x_vals.size > 1 else 1.0
    dy = float(np.median(np.diff(y_vals))) * 0.72 if y_vals.size > 1 else 1.0
    return dx, dy


def _bar_colors(z):
    finite = z[np.isfinite(z)]
    if finite.size:
        z_p2, z_p98 = np.percentile(finite, [2.0, 98.0])
        if z_p98 <= z_p2:
            z_p98 = z_p2 + max(abs(float(z_p2)) * 0.1, 1.0)
    else:
        z_p2, z_p98 = 0.0, 1.0
    norm = colors.Normalize(vmin=float(z_p2), vmax=float(z_p98))
    face_colors = cm.viridis(norm(np.clip(z, z_p2, z_p98)))
    return face_colors, norm


def _write_static_bar_png(path, x, y, z, mask, face_colors, norm, *, elev, azim, title):
    h, w = z.shape
    dx, dy = _bar_geometry(x, y)
    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(
        (x - dx / 2.0).ravel(),
        (y - dy / 2.0).ravel(),
        np.zeros(h * w),
        dx,
        dy,
        z.ravel(),
        color=face_colors.reshape(-1, 4),
        shade=True,
        linewidth=0.08,
        edgecolor=(0.08, 0.08, 0.08, 0.28),
    )
    positive = mask.astype(bool)
    if int(positive.sum()):
        ax.scatter(x[positive], y[positive], np.zeros(int(positive.sum())), c="#d63f3f", s=12, depthshade=False)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("k (kN/m)")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    ax.set_zlim(0.0, max(float(np.nanmax(z)) * 1.08, 1.0))
    mappable = cm.ScalarMappable(norm=norm, cmap="viridis")
    mappable.set_array(z)
    fig.colorbar(mappable, ax=ax, shrink=0.62, pad=0.08, label="k (kN/m)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_interactive_bar_html(path, x, y, z, mask, norm, *, title):
    h, w = z.shape
    dx, dy = _bar_geometry(x, y)
    verts = []
    intensity = []
    i_faces = []
    j_faces = []
    k_faces = []
    triangles = [
        (0, 1, 2),
        (0, 2, 3),
        (4, 6, 5),
        (4, 7, 6),
        (0, 4, 5),
        (0, 5, 1),
        (1, 5, 6),
        (1, 6, 2),
        (2, 6, 7),
        (2, 7, 3),
        (3, 7, 4),
        (3, 4, 0),
    ]
    for row in range(h):
        for col in range(w):
            xc = float(x[row, col])
            yc = float(y[row, col])
            top = float(z[row, col])
            x0, x1 = xc - dx / 2.0, xc + dx / 2.0
            y0, y1 = yc - dy / 2.0, yc + dy / 2.0
            base = len(verts)
            verts.extend(
                [
                    (x0, y0, 0.0),
                    (x1, y0, 0.0),
                    (x1, y1, 0.0),
                    (x0, y1, 0.0),
                    (x0, y0, top),
                    (x1, y0, top),
                    (x1, y1, top),
                    (x0, y1, top),
                ]
            )
            intensity.extend([top] * 8)
            for a, b, c in triangles:
                i_faces.append(base + a)
                j_faces.append(base + b)
                k_faces.append(base + c)
    verts = np.asarray(verts, dtype=np.float64)
    mesh = go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=i_faces,
        j=j_faces,
        k=k_faces,
        intensity=np.asarray(intensity),
        colorscale="Viridis",
        cmin=float(norm.vmin),
        cmax=float(norm.vmax),
        colorbar=dict(title="k (kN/m)"),
        flatshading=True,
        opacity=0.96,
        name="baseline stiffness bars",
        hovertemplate="x=%{x:.1f} mm<br>y=%{y:.1f} mm<br>k=%{z:.2f} kN/m<extra></extra>",
    )
    data = [mesh]
    positive = mask.astype(bool)
    if int(positive.sum()):
        data.append(
            go.Scatter3d(
                x=x[positive],
                y=y[positive],
                z=np.zeros(int(positive.sum())),
                mode="markers",
                marker=dict(size=3, color="red", opacity=0.9),
                name="GT scan positive on floor",
            )
        )
    fig = go.Figure(data=data)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x (mm)",
            yaxis_title="y (mm)",
            zaxis_title="k (kN/m)",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.55),
            camera=dict(eye=dict(x=1.45, y=-1.55, z=1.05)),
        ),
        margin=dict(l=0, r=0, t=52, b=0),
    )
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)


def _default_xy(shape):
    h, w = shape
    y, x = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    return np.stack([x, y], axis=-1)


def _load_dependencies() -> None:
    global cm, colors, go, np, plt
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt_mod
        from matplotlib import cm as cm_mod
        from matplotlib import colors as colors_mod
        import numpy as np_mod
        import plotly.graph_objects as go_mod
    except ModuleNotFoundError as exc:
        raise SystemExit("Need numpy, matplotlib, and plotly in the active environment.") from exc

    np = np_mod
    plt = plt_mod
    cm = cm_mod
    colors = colors_mod
    go = go_mod


if __name__ == "__main__":
    main()
