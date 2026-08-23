#!/usr/bin/env python3
"""Render a point-synchronized palpation replay from saved simulation states.

Newton/VBD geometry is rendered off-line from saved ``particle_q`` states.  The
website receives a compressed video and never reconstructs or deforms the
tetrahedral mesh in browser JavaScript.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont


BACKGROUND = "#f5f4f0"
INK = "#202529"
MUTED = "#687077"
ACCENT = "#176d8f"
LUMP_COLORS = ("#ff6b35", "#ffc145", "#ef476f", "#9b5de5")

VIDEO_SIZE = (1140, 500)
GEOMETRY_SIZE = (430, 270)
PRESS_PANEL_SIZE = (430, 390)
BASELINE_PANEL_SIZE = (270, 390)
CURVE_PANEL_SIZE = (390, 390)
CONTENT_Y = 58
KEYFRAME_INDICES = (0, 3, 6, 9, 12, 15, 17, 19)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-npz", required=True, type=Path)
    parser.add_argument("--sample-json", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--poster", required=True, type=Path)
    parser.add_argument("--outputs", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--point-limit",
        type=int,
        default=None,
        help="Optional raster-order point limit for a quick rendering test.",
    )
    parser.add_argument(
        "--tet-stride",
        type=int,
        default=128,
        help="Render one tetrahedron in every N cells as a moving wireframe.",
    )
    return parser.parse_args()


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        ("/System/Library/Fonts/SFNS.ttf", 1 if bold else 0),
        ("/System/Library/Fonts/SFNSRounded.ttf", 1 if bold else 0),
        ("/System/Library/Fonts/Helvetica.ttc", 1 if bold else 0),
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            0,
        ),
    ]
    for candidate, index in candidates:
        try:
            return ImageFont.truetype(candidate, size=size, index=index)
        except OSError:
            continue
    return ImageFont.load_default()


def rgba(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4)) + (alpha,)


def make_tet_grid(vertices: np.ndarray, tets: np.ndarray) -> pv.UnstructuredGrid:
    cell_sizes = np.full((tets.shape[0], 1), 4, dtype=np.int64)
    cells = np.hstack([cell_sizes, tets.astype(np.int64)]).reshape(-1)
    cell_types = np.full(tets.shape[0], int(pv.CellType.TETRA), dtype=np.uint8)
    return pv.UnstructuredGrid(cells, cell_types, vertices)


def sampled_tet_grid(
    vertices: np.ndarray,
    tets: np.ndarray,
    stride: int,
) -> tuple[pv.UnstructuredGrid, np.ndarray]:
    selected = np.asarray(tets[:: max(stride, 1)], dtype=np.int64)
    vertex_ids, inverse = np.unique(selected.reshape(-1), return_inverse=True)
    compact = inverse.reshape(-1, 4)
    return make_tet_grid(vertices[vertex_ids].copy(), compact), vertex_ids


def original_point_ids(mesh: pv.DataSet) -> np.ndarray:
    key = "vtkOriginalPointIds"
    if key not in mesh.point_data:
        raise RuntimeError(f"PyVista did not preserve {key}; replay cannot be mapped safely.")
    return np.asarray(mesh.point_data[key], dtype=np.int64)


def render_geometry_frames(
    replay: np.lib.npyio.NpzFile,
    tet_stride: int,
) -> tuple[list[Image.Image], np.ndarray]:
    vertices = np.asarray(replay["mesh_vertices"], dtype=np.float32)
    positions = np.asarray(replay["positions"], dtype=np.float32)
    tets = np.asarray(replay["mesh_tets"], dtype=np.int64)
    tet_lump_id = np.asarray(replay["tet_lump_id"], dtype=np.int32)
    probe_positions = np.asarray(replay["probe_positions"], dtype=np.float32)
    depths_mm = np.asarray(replay["indentation_depth"], dtype=np.float32) * 1000.0

    full_grid = make_tet_grid(vertices.copy(), tets)
    tissue_surface = full_grid.extract_surface(algorithm="dataset_surface")
    tissue_ids = original_point_ids(tissue_surface)

    lump_surfaces: list[tuple[pv.PolyData, np.ndarray]] = []
    for lump_id in sorted(value for value in np.unique(tet_lump_id) if value >= 0):
        cell_ids = np.flatnonzero(tet_lump_id == lump_id)
        lump_grid = full_grid.extract_cells(cell_ids)
        lump_point_ids = original_point_ids(lump_grid)
        surface = lump_grid.extract_surface(algorithm="dataset_surface")
        surface_local_ids = original_point_ids(surface)
        lump_surfaces.append((surface, lump_point_ids[surface_local_ids]))

    wire_grid, wire_ids = sampled_tet_grid(vertices, tets, tet_stride)
    probe = pv.Sphere(radius=0.0125, theta_resolution=36, phi_resolution=24)
    probe_origin = probe.points.copy()

    plotter = pv.Plotter(off_screen=True, window_size=GEOMETRY_SIZE)
    plotter.set_background(BACKGROUND)
    plotter.add_mesh(
        tissue_surface,
        color="#8fbaca",
        opacity=0.42,
        smooth_shading=True,
        show_edges=False,
    )
    for index, (surface, _ids) in enumerate(lump_surfaces):
        plotter.add_mesh(
            surface,
            color=LUMP_COLORS[index % len(LUMP_COLORS)],
            opacity=0.95,
            show_edges=True,
            edge_color="#4b382f",
            line_width=0.7,
            lighting=False,
        )
    plotter.add_mesh(
        wire_grid,
        style="wireframe",
        color="#35484f",
        opacity=0.38,
        line_width=0.75,
    )
    plotter.add_mesh(
        probe,
        color="#d4d7d8",
        smooth_shading=True,
        specular=0.30,
    )
    plotter.camera_position = [
        (0.245, -0.285, 0.235),
        (0.0, 0.0, 0.038),
        (0.0, 0.0, 1.0),
    ]
    plotter.enable_parallel_projection()
    plotter.camera.parallel_scale = 0.12

    frames: list[Image.Image] = []
    for frame_index, state in enumerate(positions):
        tissue_surface.points = state[tissue_ids]
        for surface, point_ids in lump_surfaces:
            surface.points = state[point_ids]
        wire_grid.points = state[wire_ids]
        probe.points = probe_origin + probe_positions[frame_index]
        plotter.render()
        frames.append(Image.fromarray(plotter.screenshot(return_img=True)).convert("RGB"))

    plotter.close()
    return frames, depths_mm


def render_force_chart(
    depth_mm: np.ndarray,
    force_n: np.ndarray,
    visible_step: int,
    point_row: int,
    point_col: int,
) -> Image.Image:
    width, height = (430, 120)
    panel = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(panel)

    left, top, right, bottom = 18, 10, 414, 108
    draw.line((left, bottom, right, bottom), fill=rgba("#9ba1a3"), width=1)
    draw.line((left, top, left, bottom), fill=rgba("#9ba1a3"), width=1)

    x_max = max(float(np.max(depth_mm)), 1e-6)
    y_max = max(float(np.max(force_n)) * 1.08, 1.0)

    def xy(step: int) -> tuple[float, float]:
        x = left + float(depth_mm[step]) / x_max * (right - left)
        y = bottom - float(force_n[step]) / y_max * (bottom - top)
        return x, y

    visible_step = max(0, min(int(visible_step), len(force_n) - 1))
    if visible_step >= 1:
        points = [xy(step) for step in range(visible_step + 1)]
        draw.line(points, fill=rgba(ACCENT), width=3, joint="curve")
    marker_x, marker_y = xy(visible_step)
    draw.ellipse(
        (marker_x - 4, marker_y - 4, marker_x + 4, marker_y + 4),
        fill=rgba("#d05a3a"),
    )
    return panel


def render_press_panel(
    geometry_frame: Image.Image,
    depth_mm: np.ndarray,
    force_n: np.ndarray,
    visible_step: int,
    point_row: int,
    point_col: int,
) -> Image.Image:
    panel = Image.new("RGB", PRESS_PANEL_SIZE, BACKGROUND)
    panel.paste(geometry_frame, (0, 0))
    chart = render_force_chart(depth_mm, force_n, visible_step, point_row, point_col)
    panel.paste(chart, (0, 270))
    return panel


def colorized_array(
    values: np.ndarray,
    mask: np.ndarray,
    cmap_name: str,
    value_range: tuple[float, float],
) -> Image.Image:
    low, high = value_range
    normalized = np.clip((values - low) / max(high - low, 1e-8), 0.0, 1.0)
    pixels = (colormaps[cmap_name](normalized)[:, :, :3] * 255).astype(np.uint8)
    pixels[~mask] = np.asarray([230, 230, 226], dtype=np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def render_baseline_panel(
    stiffness: np.ndarray,
    completed_points: int,
    value_range: tuple[float, float],
) -> Image.Image:
    panel = Image.new("RGB", BASELINE_PANEL_SIZE, BACKGROUND)
    draw = ImageDraw.Draw(panel)

    mask = np.zeros_like(stiffness, dtype=bool)
    mask.reshape(-1)[:completed_points] = True
    heatmap = colorized_array(
        stiffness / 1000.0,
        mask,
        "viridis",
        (value_range[0] / 1000.0, value_range[1] / 1000.0),
    ).resize((218, 218), Image.Resampling.NEAREST)
    panel.paste(heatmap, (20, 82))
    draw.rectangle((20, 82, 238, 300), outline=rgba("#a8aca9"), width=1)

    bar_pixels = np.asarray(
        colormaps["viridis"](np.linspace(1.0, 0.0, 218))[:, :3] * 255,
        dtype=np.uint8,
    )
    bar = Image.fromarray(bar_pixels[:, None, :], mode="RGB").resize((12, 218))
    panel.paste(bar, (244, 82))
    return panel


def render_curve_panel(
    fz: np.ndarray,
    completed_points: int,
    value_range: tuple[float, float],
) -> Image.Image:
    rows, cols, steps = fz.shape
    figure = plt.figure(figsize=(3.9, 3.9), dpi=100, facecolor=BACKGROUND)
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0), projection="3d")
    axis.set_facecolor(BACKGROUND)

    if completed_points:
        flat_points = np.arange(completed_points)
        point_rows = flat_points // cols
        point_cols = flat_points % cols
        step_grid = np.tile(np.arange(steps), completed_points)
        col_grid = np.repeat(point_cols, steps)
        row_grid = np.repeat(point_rows, steps)
        values = fz.reshape(-1, steps)[:completed_points].reshape(-1)
        normalized = np.clip(
            (values - value_range[0]) / max(value_range[1] - value_range[0], 1e-8),
            0.0,
            1.0,
        )
        colors = colormaps["magma"](normalized)
        colors[:, 3] = 0.30 + 0.62 * colors[:, 0]
        axis.scatter(
            col_grid,
            row_grid,
            step_grid,
            c=colors,
            s=5.5,
            marker="s",
            linewidths=0,
            depthshade=False,
        )

    axis.set_xlim(0, cols - 1)
    axis.set_ylim(0, rows - 1)
    axis.set_zlim(0, steps - 1)
    axis.set_xticks(())
    axis.set_yticks(())
    axis.set_zticks(())
    axis.view_init(elev=24, azim=-58)
    axis.set_box_aspect((1.0, 1.0, 1.18))
    for axis_dimension in (axis.xaxis, axis.yaxis, axis.zaxis):
        axis_dimension.pane.set_facecolor((0.97, 0.97, 0.95, 0.9))
        axis_dimension.pane.set_edgecolor((0.72, 0.74, 0.73, 0.5))
        axis_dimension._axinfo["grid"]["color"] = (0.72, 0.74, 0.73, 0.25)

    figure.canvas.draw()
    image = Image.fromarray(np.asarray(figure.canvas.buffer_rgba())).convert("RGB")
    plt.close(figure)
    return image.resize(CURVE_PANEL_SIZE, Image.Resampling.LANCZOS)


def compose_frame(
    press_panel: Image.Image,
    baseline_panel: Image.Image,
    curve_panel: Image.Image,
    batch_index: int,
    total_batches: int,
    point_row: int,
    point_col: int,
) -> Image.Image:
    frame = Image.new("RGB", VIDEO_SIZE, BACKGROUND)
    frame.paste(press_panel, (8, CONTENT_Y))
    frame.paste(baseline_panel, (446, CONTENT_Y))
    frame.paste(curve_panel, (730, CONTENT_Y))
    draw = ImageDraw.Draw(frame, "RGBA")
    draw.rectangle((0, 0, VIDEO_SIZE[0], 52), fill=(255, 255, 255, 242))
    status = f"({batch_index + 1}/{total_batches})"
    draw.text(
        (18, 11),
        status,
        fill=rgba(ACCENT),
        font=font(23, bold=True),
    )
    draw.line((440, 66, 440, 443), fill=rgba("#d3d4d1"), width=1)
    draw.line((722, 66, 722, 443), fill=rgba("#d3d4d1"), width=1)
    return frame


def start_video_encoder(output_path: Path, fps: int) -> subprocess.Popen:
    ffmpeg = os.environ.get("PALPATION_FFMPEG") or shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode the replay video.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{VIDEO_SIZE[0]}x{VIDEO_SIZE[1]}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def build_video(
    geometry_frames: list[Image.Image],
    geometry_depth_mm: np.ndarray,
    scan_depth_mm: np.ndarray,
    stiffness: np.ndarray,
    fz: np.ndarray,
    output_path: Path,
    poster_path: Path,
    fps: int,
    point_limit: int | None,
) -> None:
    rows, cols = stiffness.shape
    total_available = rows * cols
    total_batches = total_available if point_limit is None else min(point_limit, total_available)
    stiffness_range = tuple(float(value) for value in np.percentile(stiffness, (2.0, 98.0)))
    force_range = tuple(float(value) for value in np.percentile(fz, (2.0, 98.0)))

    encoder = start_video_encoder(output_path, fps)
    if encoder.stdin is None:
        raise RuntimeError("ffmpeg did not expose an input pipe.")

    baseline_before = render_baseline_panel(stiffness, 0, stiffness_range)
    curve_before = render_curve_panel(fz, 0, force_range)
    last_frame: Image.Image | None = None
    try:
        for point_index in range(total_batches):
            point_row, point_col = divmod(point_index, cols)
            point_force = fz[point_row, point_col]
            baseline_after = render_baseline_panel(stiffness, point_index + 1, stiffness_range)
            curve_after = render_curve_panel(fz, point_index + 1, force_range)

            for local_index, source_step in enumerate(KEYFRAME_INDICES):
                reveal_point = local_index == len(KEYFRAME_INDICES) - 1
                press_panel = render_press_panel(
                    geometry_frames[source_step],
                    scan_depth_mm,
                    point_force,
                    source_step,
                    point_row,
                    point_col,
                )
                last_frame = compose_frame(
                    press_panel,
                    baseline_after if reveal_point else baseline_before,
                    curve_after if reveal_point else curve_before,
                    point_index,
                    total_batches,
                    point_row,
                    point_col,
                )
                encoder.stdin.write(np.asarray(last_frame, dtype=np.uint8).tobytes())

            baseline_before = baseline_after
            curve_before = curve_after
            if (point_index + 1) % 20 == 0 or point_index + 1 == total_batches:
                print(f"Rendered {point_index + 1} / {total_batches} point batches", flush=True)

        if last_frame is None:
            raise RuntimeError("No replay frames were generated.")
        for _ in range(fps * 2):
            encoder.stdin.write(np.asarray(last_frame, dtype=np.uint8).tobytes())
        poster_path.parent.mkdir(parents=True, exist_ok=True)
        last_frame.save(poster_path)
    finally:
        encoder.stdin.close()
        return_code = encoder.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}.")


def build_output_figure(sample: dict, output_path: Path) -> None:
    output = sample["output"]
    ground_truth = np.asarray(output["groundTruth"], dtype=np.float32)
    panels = (
        ("Proposed · full curve", np.asarray(output["curveProbability"])),
        (
            "Baseline · scalar stiffness",
            np.asarray(output["stiffnessProbability"]),
        ),
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 3.7), dpi=160, facecolor=BACKGROUND)
    for axis, (title, probability) in zip(axes, panels, strict=True):
        axis.imshow(probability, cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")
        axis.contour(ground_truth, levels=[0.5], colors=["#f4d35e"], linewidths=1.5)
        axis.set_title(title, fontsize=12, weight="bold", color=INK, pad=8)
        axis.set_xticks(())
        axis.set_yticks(())
        for spine in axis.spines.values():
            spine.set_edgecolor("#b9bbb8")
    figure.suptitle(
        "Saved held-out predictions · sample 0299",
        fontsize=14,
        weight="bold",
        color=INK,
        y=0.98,
    )
    figure.subplots_adjust(left=0.06, right=0.94, bottom=0.12, top=0.84, wspace=0.12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, facecolor=BACKGROUND, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    with args.sample_json.open("r", encoding="utf-8") as handle:
        sample = json.load(handle)
    replay = np.load(args.replay_npz, allow_pickle=False)

    geometry_frames, geometry_depth_mm = render_geometry_frames(replay, args.tet_stride)
    stiffness = np.asarray(sample["scan"]["stiffnessNPerM"], dtype=np.float32)
    fz = np.asarray(sample["scan"]["fzN"], dtype=np.float32)
    scan_depth_mm = np.asarray(sample["scan"]["depthMm"], dtype=np.float32)
    if not np.allclose(geometry_depth_mm, scan_depth_mm, atol=1e-3):
        raise RuntimeError("Geometry and recorded scan depth grids do not match.")

    build_video(
        geometry_frames,
        geometry_depth_mm,
        scan_depth_mm,
        stiffness,
        fz,
        args.video,
        args.poster,
        args.fps,
        args.point_limit,
    )
    build_output_figure(sample, args.outputs)
    print(f"Wrote {args.video}")
    print(f"Wrote {args.poster}")
    print(f"Wrote {args.outputs}")


if __name__ == "__main__":
    main()
