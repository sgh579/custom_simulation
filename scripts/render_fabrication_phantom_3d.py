#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


SHAPE_COLORS = {
    "sphere": "#2ca25f",
    "ellipsoid": "#3182bd",
    "cylinder": "#addd8e",
    "capsule": "#fdae6b",
}

SHAPE_LABELS = {
    "sphere": "SPH",
    "ellipsoid": "ELL",
    "cylinder": "CYL",
    "capsule": "CAP",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Render 3D schematic images for fabrication phantom configs.")
    parser.add_argument("batch_dir", type=Path, help="Directory containing random_fabrication_manifest.json.")
    parser.add_argument("--out-dir-name", type=str, default="visualizations")
    parser.add_argument("--release", action="store_true", help="Copy rendered images into release/visualizations and refresh release zip.")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    batch_dir = args.batch_dir.expanduser().resolve()
    manifest_path = batch_dir / "random_fabrication_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    out_dir = batch_dir / args.out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered_paths = []
    for item in manifest["configurations"]:
        config = json.loads(Path(item["config_json"]).read_text(encoding="utf-8"))
        fig = plt.figure(figsize=(8.0, 7.0))
        ax = fig.add_subplot(111, projection="3d")
        render_config(ax, config, show_title=True, label_lumps=True)
        out_path = out_dir / f"{config['config_id']}_3d.png"
        fig.savefig(out_path, dpi=int(args.dpi), bbox_inches="tight")
        plt.close(fig)
        rendered_paths.append(out_path)
        print(f"wrote {out_path}", flush=True)

    overview = out_dir / "random_fabrication_3d_overview.png"
    render_overview(manifest, overview, dpi=int(args.dpi))
    rendered_paths.append(overview)
    (out_dir / "README.md").write_text(_readme(manifest), encoding="utf-8")

    if args.release:
        release_vis = batch_dir / "release" / "visualizations"
        if release_vis.exists():
            shutil.rmtree(release_vis)
        shutil.copytree(out_dir, release_vis)
        zip_path = batch_dir / "release_printable.zip"
        _zip_dir(batch_dir / "release", zip_path)
        print(f"refreshed release zip -> {zip_path}", flush=True)

    print(f"Wrote {len(rendered_paths)} visualization images to {out_dir}")


def render_overview(manifest: dict, out_path: Path, *, dpi: int) -> None:
    fig = plt.figure(figsize=(18.0, 13.0))
    for idx, item in enumerate(manifest["configurations"]):
        config = json.loads(Path(item["config_json"]).read_text(encoding="utf-8"))
        ax = fig.add_subplot(3, 4, idx + 1, projection="3d")
        render_config(ax, config, show_title=True, label_lumps=False, compact=True)
    fig.suptitle("3D fabrication phantom overview: bottom cast volume and embedded lumps", fontsize=16)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"wrote {out_path}", flush=True)


def render_config(ax, config: dict, *, show_title: bool, label_lumps: bool, compact: bool = False) -> None:
    params = config["fabrication_params"]
    wall = float(params["wall_thickness_mm"])
    outer = float(params["outer_size_mm"])
    cavity_min = wall
    cavity_max = outer - wall
    z_bottom = float(params["ecoflex_bottom_z_mm"])
    z_top = float(params["ecoflex_top_z_mm"])

    draw_box(ax, cavity_min, cavity_max, cavity_min, cavity_max, z_bottom, z_top, color="#31a354", alpha=0.08, lw=0.8)
    draw_wall_outline(ax, outer=outer, z_min=0.0, z_max=90.0)

    for idx, lump in enumerate(config["lumps"]):
        color = SHAPE_COLORS.get(str(lump["shape"]), "#9e9e9e")
        draw_lump(ax, lump, color=color, resolution=26 if compact else 36)
        x = float(lump["center_x_mm_in_mold_frame"])
        y = float(lump["center_y_mm_in_mold_frame"])
        z = float(lump["center_z_mm_from_mold_bottom"])
        if label_lumps:
            label = f"L{idx}-{SHAPE_LABELS.get(str(lump['shape']), str(lump['shape'])[:3].upper())}"
            ax.text(x, y, z + 1.2, label, fontsize=8, ha="center", va="center", color="#111111")

    ax.set_xlim(0.0, 90.0)
    ax.set_ylim(0.0, 90.0)
    ax.set_zlim(0.0, 95.0)
    ax.set_box_aspect((90.0, 90.0, 95.0))
    ax.view_init(elev=23, azim=-52)
    ax.set_axis_off()
    if show_title:
        ax.set_title(f"{config['config_id']}  n={config['lump_count']}", fontsize=10 if compact else 12)


def draw_lump(ax, lump: dict, *, color: str, resolution: int) -> None:
    shape = str(lump["shape"])
    center = np.asarray(
        [
            float(lump["center_x_mm_in_mold_frame"]),
            float(lump["center_y_mm_in_mold_frame"]),
            float(lump["center_z_mm_from_mold_bottom"]),
        ],
        dtype=np.float64,
    )
    rx, ry, rz = (float(value) for value in lump["radii_mm"])
    yaw = math.radians(float(lump["yaw_deg"]))

    if shape in {"sphere", "ellipsoid"}:
        x, y, z = ellipsoid_surface(center, (rx, ry, rz), yaw, resolution=resolution)
        ax.plot_surface(x, y, z, color=color, alpha=0.78, linewidth=0.0, shade=True)
        return
    if shape == "cylinder":
        surfaces = cylinder_surfaces(center, (rx, ry, rz), yaw, resolution=resolution)
        for x, y, z in surfaces:
            ax.plot_surface(x, y, z, color=color, alpha=0.78, linewidth=0.0, shade=True)
        return
    if shape == "capsule":
        surfaces = capsule_surfaces(center, radius=rx, half_axis=rz, yaw=yaw, resolution=resolution)
        for x, y, z in surfaces:
            ax.plot_surface(x, y, z, color=color, alpha=0.78, linewidth=0.0, shade=True)
        return
    if shape == "box":
        hx, hy, hz = rx, ry, rz
        draw_oriented_box(ax, center, (hx, hy, hz), yaw, color=color)
        return
    raise ValueError(f"Unsupported shape: {shape}")


def ellipsoid_surface(center: np.ndarray, radii: tuple[float, float, float], yaw: float, *, resolution: int):
    u = np.linspace(0.0, 2.0 * np.pi, resolution)
    v = np.linspace(0.0, np.pi, max(resolution // 2, 12))
    uu, vv = np.meshgrid(u, v)
    x = radii[0] * np.cos(uu) * np.sin(vv)
    y = radii[1] * np.sin(uu) * np.sin(vv)
    z = radii[2] * np.cos(vv)
    return transform_surface(x, y, z, center, yaw)


def cylinder_surfaces(center: np.ndarray, radii: tuple[float, float, float], yaw: float, *, resolution: int):
    theta = np.linspace(0.0, 2.0 * np.pi, resolution)
    z_line = np.linspace(-radii[2], radii[2], 2)
    tt, zz = np.meshgrid(theta, z_line)
    side_x = radii[0] * np.cos(tt)
    side_y = radii[1] * np.sin(tt)
    side = transform_surface(side_x, side_y, zz, center, yaw)

    caps = []
    rr = np.linspace(0.0, 1.0, max(resolution // 3, 8))
    tt2, rr2 = np.meshgrid(theta, rr)
    for z_cap in (-radii[2], radii[2]):
        cap_x = radii[0] * rr2 * np.cos(tt2)
        cap_y = radii[1] * rr2 * np.sin(tt2)
        cap_z = np.full_like(cap_x, z_cap)
        caps.append(transform_surface(cap_x, cap_y, cap_z, center, yaw))
    return [side, *caps]


def capsule_surfaces(center: np.ndarray, *, radius: float, half_axis: float, yaw: float, resolution: int):
    theta = np.linspace(0.0, 2.0 * np.pi, resolution)
    z_line = np.linspace(-half_axis, half_axis, 2)
    tt, zz = np.meshgrid(theta, z_line)
    side_x = radius * np.cos(tt)
    side_y = radius * np.sin(tt)
    side = transform_surface(side_x, side_y, zz, center, yaw)

    phi_top = np.linspace(0.0, 0.5 * np.pi, max(resolution // 3, 8))
    tt_top, pp_top = np.meshgrid(theta, phi_top)
    top_x = radius * np.cos(tt_top) * np.sin(pp_top)
    top_y = radius * np.sin(tt_top) * np.sin(pp_top)
    top_z = half_axis + radius * np.cos(pp_top)
    top = transform_surface(top_x, top_y, top_z, center, yaw)

    phi_bottom = np.linspace(0.5 * np.pi, np.pi, max(resolution // 3, 8))
    tt_bottom, pp_bottom = np.meshgrid(theta, phi_bottom)
    bottom_x = radius * np.cos(tt_bottom) * np.sin(pp_bottom)
    bottom_y = radius * np.sin(tt_bottom) * np.sin(pp_bottom)
    bottom_z = -half_axis + radius * np.cos(pp_bottom)
    bottom = transform_surface(bottom_x, bottom_y, bottom_z, center, yaw)
    return [side, top, bottom]


def transform_surface(x, y, z, center: np.ndarray, yaw: float):
    c = math.cos(yaw)
    s = math.sin(yaw)
    xw = c * x - s * y + center[0]
    yw = s * x + c * y + center[1]
    zw = z + center[2]
    return xw, yw, zw


def draw_box(ax, xmin, xmax, ymin, ymax, zmin, zmax, *, color: str, alpha: float, lw: float) -> None:
    vertices = np.asarray(
        [
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax],
        ],
        dtype=np.float64,
    )
    faces = [
        [vertices[i] for i in [0, 1, 2, 3]],
        [vertices[i] for i in [4, 5, 6, 7]],
        [vertices[i] for i in [0, 1, 5, 4]],
        [vertices[i] for i in [1, 2, 6, 5]],
        [vertices[i] for i in [2, 3, 7, 6]],
        [vertices[i] for i in [3, 0, 4, 7]],
    ]
    collection = Poly3DCollection(faces, facecolors=color, edgecolors=color, linewidths=lw, alpha=alpha)
    ax.add_collection3d(collection)


def draw_wall_outline(ax, *, outer: float, z_min: float, z_max: float) -> None:
    color = "#9e9e9e"
    xs = [0.0, outer, outer, 0.0, 0.0]
    ys = [0.0, 0.0, outer, outer, 0.0]
    for z in (z_min, z_max):
        ax.plot(xs, ys, [z] * len(xs), color=color, lw=0.55, alpha=0.36)
    for x, y in [(0.0, 0.0), (outer, 0.0), (outer, outer), (0.0, outer)]:
        ax.plot([x, x], [y, y], [z_min, z_max], color=color, lw=0.5, alpha=0.28)


def draw_top_hanger(ax, *, outer: float, center: float, z: float) -> None:
    ax.plot([0.0, outer], [center, center], [z, z], color="#d7301f", lw=1.6)
    ax.plot([center, center], [0.0, outer], [z, z], color="#d7301f", lw=1.6)


def draw_oriented_box(ax, center: np.ndarray, half_size: tuple[float, float, float], yaw: float, *, color: str) -> None:
    hx, hy, hz = half_size
    corners = np.asarray(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float64,
    )
    c = math.cos(yaw)
    s = math.sin(yaw)
    x = corners[:, 0].copy()
    y = corners[:, 1].copy()
    corners[:, 0] = c * x - s * y
    corners[:, 1] = s * x + c * y
    corners += center
    face_ids = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    faces = [[corners[i] for i in ids] for ids in face_ids]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, edgecolors="#333333", linewidths=0.5, alpha=0.78))


def _readme(manifest: dict) -> str:
    return "\n".join(
        [
            "# 3D Fabrication Phantom Visualizations",
            "",
            "These PNG files are schematic 3D renderings generated from the fabrication config JSON files.",
            "They intentionally omit support rods, top hanging-frame rods, axes, and tick marks.",
            "",
            "Color/label legend:",
            "",
            "- SPH: sphere",
            "- ELL: ellipsoid",
            "- CYL: cylinder",
            "- CAP: capsule",
            "",
            f"Cast phantom height: {manifest['fabrication_params']['phantom_height_mm']} mm.",
            f"Ecoflex interval: z={manifest['fabrication_params']['ecoflex_bottom_z_mm']}..{manifest['fabrication_params']['ecoflex_top_z_mm']} mm.",
            f"Hanging top rod center: z={manifest['hanging_layout']['top_rod_center_z_mm']} mm.",
            "",
        ]
    )


def _zip_dir(src_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    import zipfile

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(src_dir.parent))


if __name__ == "__main__":
    main()
