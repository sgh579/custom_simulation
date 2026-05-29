from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from palpation_sim.config import MaterialConfig, PhantomConfig, ScanConfig
from palpation_sim.exports import write_phantom_gltf
from palpation_sim.features import extract_feature_map
from palpation_sim.phantom import LumpSpec, create_structured_tet_mesh, material_arrays_for_lumps
from palpation_sim.strain_stiffening import run_strain_stiffening_sample
from write_real_mesh_webgl_player import write_real_mesh_webgl_player


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one high-mesh phantom strain-stiffening experiment.")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/one_phantom_high_mesh_strain_stiffening"))
    parser.add_argument("--cells-x", type=int, default=80)
    parser.add_argument("--cells-y", type=int, default=80)
    parser.add_argument("--cells-z", type=int, default=30)
    parser.add_argument("--grid", type=int, default=41)
    parser.add_argument("--press-steps", type=int, default=96)
    parser.add_argument("--max-indentation", type=float, default=0.020)
    parser.add_argument("--hardening-b", type=float, default=1.8)
    parser.add_argument("--seed", type=int, default=28)
    args = parser.parse_args()

    start = time.time()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    phantom = PhantomConfig(cells_x=args.cells_x, cells_y=args.cells_y, cells_z=args.cells_z)
    material = MaterialConfig()
    scan = ScanConfig(
        grid_h=args.grid,
        grid_w=args.grid,
        press_steps=args.press_steps,
        max_indentation=args.max_indentation,
    )
    lumps = _one_phantom_lumps(phantom)

    print("building high-resolution tet mesh...", flush=True)
    mesh = create_structured_tet_mesh(phantom)
    _, _, _, tet_lump_mask, tet_lump_id = material_arrays_for_lumps(mesh, material, lumps)

    print("generating strain-stiffening scan response...", flush=True)
    sample = run_strain_stiffening_sample(
        phantom,
        material,
        scan,
        lumps,
        rng,
        hardening_b=args.hardening_b,
        noise_std=0.0,
        enforce_convex=True,
    )
    sample["features"] = extract_feature_map(np.asarray(sample["presses"], dtype=np.float32))
    sample["tet_lump_mask"] = tet_lump_mask
    sample["tet_lump_id"] = tet_lump_id
    sample["mesh_vertices"] = mesh.vertices.astype(np.float32)
    sample["mesh_tets"] = mesh.tets.astype(np.int32)
    sample["phantom_json"] = np.asarray(json.dumps(phantom.to_dict()))
    sample["material_json"] = np.asarray(json.dumps(material.to_dict()))
    sample["scan_json"] = np.asarray(json.dumps(scan.to_dict()))
    sample["mesh_json"] = np.asarray(
        json.dumps(
            {
                "cells": [phantom.cells_x, phantom.cells_y, phantom.cells_z],
                "vertex_count": int(mesh.vertices.shape[0]),
                "tet_count": int(mesh.tets.shape[0]),
                "tet_lump_count": int(np.count_nonzero(tet_lump_mask)),
            }
        )
    )

    npz_path = out_dir / "one_phantom_high_mesh_sample.npz"
    print(f"writing {npz_path}...", flush=True)
    np.savez_compressed(npz_path, **sample)

    selected = _selected_press_records(scan, phantom, lumps, sample)
    _write_summary(out_dir / "selected_press_summary.csv", selected)
    _write_metadata(out_dir / "experiment_metadata.json", args, phantom, scan, material, lumps, mesh, selected, time.time() - start)

    print("writing plots...", flush=True)
    _write_plots(out_dir, sample, selected)

    print("writing lightweight analytic player...", flush=True)
    _write_player(out_dir / "press_player_lightweight.html", phantom, scan, lumps, sample, selected)

    print("writing real high-mesh WebGL player...", flush=True)
    write_real_mesh_webgl_player(out_dir, filename="press_player.html")
    write_real_mesh_webgl_player(out_dir, filename="real_webgl_press_player.html")

    print("writing high-mesh glTF preview...", flush=True)
    write_phantom_gltf(out_dir / "phantom_high_mesh_material_preview.gltf", phantom, lumps, material, normal_alpha=0.12)

    elapsed = time.time() - start
    print(f"done in {elapsed:.1f}s")
    print(f"outputs: {out_dir}")


def _one_phantom_lumps(phantom: PhantomConfig) -> list[LumpSpec]:
    specs = [
        ("sphere", (-0.050, -0.045), 0.026, (0.010, 0.010, 0.010), 2.0, 0.0),
        ("ellipsoid", (0.000, -0.048), 0.034, (0.016, 0.009, 0.008), 4.0, math.radians(25.0)),
        ("box", (0.050, -0.043), 0.041, (0.012, 0.010, 0.007), 8.0, math.radians(18.0)),
        ("cylinder", (-0.035, 0.040), 0.049, (0.012, 0.012, 0.008), 6.0, math.radians(45.0)),
        ("capsule", (0.035, 0.040), 0.057, (0.008, 0.008, 0.008), 12.0, math.radians(35.0)),
    ]
    lumps = []
    for shape, xy, depth, radii, stiffness, yaw in specs:
        lumps.append(
            LumpSpec(
                shape=shape,
                center=(float(xy[0]), float(xy[1]), float(phantom.height - depth)),
                radii=tuple(float(v) for v in radii),
                stiffness_multiplier=float(stiffness),
                yaw=float(yaw),
            )
        )
    return lumps


def _selected_press_records(
    scan: ScanConfig,
    phantom: PhantomConfig,
    lumps: list[LumpSpec],
    sample: dict[str, object],
) -> list[dict[str, object]]:
    xy_grid = np.asarray(sample["xy"], dtype=np.float32)
    fz = np.asarray(sample["fz"], dtype=np.float32)
    depth = np.asarray(sample["indentation_depth"], dtype=np.float32)
    records: list[dict[str, object]] = []
    for idx, lump in enumerate(lumps):
        target = np.asarray(lump.center[:2], dtype=np.float32)
        dist2 = np.sum((xy_grid - target) ** 2, axis=-1)
        row, col = (int(v) for v in np.unravel_index(np.argmin(dist2), dist2.shape))
        curve_depth = depth[row, col]
        curve_force = fz[row, col]
        summary = _curve_summary(curve_depth, curve_force)
        records.append(
            {
                "id": idx,
                "shape": lump.shape,
                "row": row,
                "col": col,
                "x_m": float(xy_grid[row, col, 0]),
                "y_m": float(xy_grid[row, col, 1]),
                "target_x_m": float(lump.center[0]),
                "target_y_m": float(lump.center[1]),
                "center_depth_m": float(phantom.height - lump.center[2]),
                "top_depth_m": float(_top_depth(phantom, lump)),
                "stiffness_multiplier": float(lump.stiffness_multiplier),
                **summary,
            }
        )
    return records


def _top_depth(phantom: PhantomConfig, lump: LumpSpec) -> float:
    return max(float(phantom.height - lump.center[2] - _z_extent(lump)), 0.0)


def _z_extent(lump: LumpSpec) -> float:
    if lump.shape == "capsule":
        return float(lump.radii[0] + lump.radii[2])
    return float(lump.radii[2])


def _curve_summary(depth: np.ndarray, force: np.ndarray) -> dict[str, float]:
    slopes = np.diff(force) / np.maximum(np.diff(depth), 1e-12)
    early = _segment_slope(depth, force, 0.10, 0.35)
    late = _segment_slope(depth, force, 0.65, 0.90)
    return {
        "peak_force_n": float(np.max(force)),
        "early_slope_n_per_m": early,
        "late_slope_n_per_m": late,
        "late_early_slope_ratio": float(late / early) if early > 1e-12 else 0.0,
        "convex_fraction": float(np.mean(np.diff(slopes) >= -1e-5)) if slopes.size > 1 else 1.0,
        "loading_work_j": _trapz(force, depth),
    }


def _segment_slope(depth: np.ndarray, force: np.ndarray, lo: float, hi: float) -> float:
    z_min = float(np.min(depth))
    z_max = float(np.max(depth))
    span = z_max - z_min
    keep = (depth >= z_min + lo * span) & (depth <= z_min + hi * span)
    if int(np.count_nonzero(keep)) < 2:
        return 0.0
    z = depth[keep].astype(np.float64)
    f = force[keep].astype(np.float64)
    zc = z - float(np.mean(z))
    denom = float(np.sum(zc * zc))
    if denom <= 1e-18:
        return 0.0
    return float(np.sum(zc * (f - float(np.mean(f)))) / denom)


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(y, x))


def _write_summary(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _write_metadata(
    path: Path,
    args: argparse.Namespace,
    phantom: PhantomConfig,
    scan: ScanConfig,
    material: MaterialConfig,
    lumps: list[LumpSpec],
    mesh: object,
    selected: list[dict[str, object]],
    elapsed: float,
) -> None:
    metadata = {
        "schema_version": 1,
        "backend": "strain_stiffening",
        "note": "One high-resolution tet-mesh phantom. Curves are generated by the explicit strain-stiffening surrogate.",
        "elapsed_seconds": elapsed,
        "requested_parameters": _json_safe(vars(args)),
        "phantom": phantom.to_dict(),
        "scan": scan.to_dict(),
        "material": material.to_dict(),
        "mesh": {
            "cells": [phantom.cells_x, phantom.cells_y, phantom.cells_z],
            "vertex_count": int(mesh.vertices.shape[0]),
            "tet_count": int(mesh.tets.shape[0]),
        },
        "lumps": [lump.to_dict(phantom) for lump in lumps],
        "selected_presses": selected,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_plots(out_dir: Path, sample: dict[str, object], selected: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to write plots") from exc

    fz = np.asarray(sample["fz"], dtype=np.float32)
    depth = np.asarray(sample["indentation_depth"], dtype=np.float32)
    ratio = np.asarray(sample["nonlinearity_ratio"], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
    for record in selected:
        row = int(record["row"])
        col = int(record["col"])
        label = (
            f"{record['shape']} | depth {1000.0 * float(record['center_depth_m']):.0f} mm | "
            f"{float(record['stiffness_multiplier']):.0f}x"
        )
        ax.plot(depth[row, col] * 1000.0, fz[row, col], linewidth=2.2, label=label)
    ax.set_title("Selected F-z curves from one high-mesh phantom")
    ax.set_xlabel("indentation [mm]")
    ax.set_ylabel("Fz [N]")
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "selected_fz_curve_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    im = ax.imshow(fz[..., -1], origin="lower", cmap="magma")
    ax.set_title("Peak Fz map at maximum indentation")
    ax.set_xlabel("scan col")
    ax.set_ylabel("scan row")
    for record in selected:
        ax.scatter(int(record["col"]), int(record["row"]), s=45, facecolors="none", edgecolors="#7fffd4", linewidths=1.8)
        ax.text(int(record["col"]) + 0.4, int(record["row"]) + 0.4, str(record["id"]), color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="Fz [N]")
    fig.savefig(out_dir / "peak_fz_map.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    im = ax.imshow(ratio, origin="lower", cmap="viridis")
    ax.set_title("Late / early slope ratio map")
    ax.set_xlabel("scan col")
    ax.set_ylabel("scan row")
    for record in selected:
        ax.scatter(int(record["col"]), int(record["row"]), s=45, facecolors="none", edgecolors="#ffffff", linewidths=1.8)
        ax.text(int(record["col"]) + 0.4, int(record["row"]) + 0.4, str(record["id"]), color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="late / early slope")
    fig.savefig(out_dir / "slope_ratio_map.png", dpi=180)
    plt.close(fig)


def _write_player(
    path: Path,
    phantom: PhantomConfig,
    scan: ScanConfig,
    lumps: list[LumpSpec],
    sample: dict[str, object],
    selected: list[dict[str, object]],
) -> None:
    fz = np.asarray(sample["fz"], dtype=np.float32)
    depth = np.asarray(sample["indentation_depth"], dtype=np.float32)
    curves = []
    for record in selected:
        row = int(record["row"])
        col = int(record["col"])
        curves.append(
            {
                "id": int(record["id"]),
                "row": row,
                "col": col,
                "label": f"{record['id']} {record['shape']} {1000.0 * float(record['center_depth_m']):.0f}mm {float(record['stiffness_multiplier']):.0f}x",
                "depths": depth[row, col].tolist(),
                "forces": fz[row, col].tolist(),
                "summary": record,
            }
        )
    data = {
        "phantom": phantom.to_dict(),
        "scan": scan.to_dict(),
        "mesh": {"cells": [phantom.cells_x, phantom.cells_y, phantom.cells_z]},
        "lumps": [lump.to_dict(phantom) for lump in lumps],
        "curves": curves,
    }
    path.write_text(_player_html(json.dumps(data, separators=(",", ":")).replace("</", "<\\/")), encoding="utf-8")


def _player_html(data_json: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>One Phantom Press Player</title>
  <style>
    html, body {{
      width: 100%;
      height: 100%;
      margin: 0;
      overflow: hidden;
      background: #111417;
      color: #f2eee6;
      font: 14px/1.4 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    .layout {{ height: 100%; display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(360px, 0.9fr); }}
    #scene {{ width: 100%; height: 100%; display: block; }}
    .side {{ display: grid; grid-template-rows: auto minmax(260px, 1fr) auto auto; background: #171b1f; border-left: 1px solid rgba(255,255,255,0.15); }}
    .header, .controls, .stats {{ padding: 12px 14px; }}
    .header {{ border-bottom: 1px solid rgba(255,255,255,0.12); }}
    .header h1 {{ margin: 0 0 6px; font-size: 18px; }}
    .header div {{ color: #cbc4b8; }}
    #chart {{ width: 100%; height: 100%; display: block; background: #f6f1e8; }}
    .controls {{ display: grid; grid-template-columns: auto auto minmax(160px, 1fr); gap: 10px; align-items: center; border-top: 1px solid rgba(255,255,255,0.12); }}
    .controls select {{ grid-column: 1 / -1; }}
    button, select {{
      border: 1px solid rgba(255,255,255,0.24);
      border-radius: 6px;
      background: #eee8dc;
      color: #151719;
      padding: 7px 10px;
      font: inherit;
    }}
    input[type="range"] {{ width: 100%; }}
    .stats {{ color: #d8d0c4; border-top: 1px solid rgba(255,255,255,0.12); white-space: pre-wrap; }}
    @media (max-width: 900px) {{ .layout {{ grid-template-columns: 1fr; grid-template-rows: 58vh 42vh; }} .side {{ border-left: 0; border-top: 1px solid rgba(255,255,255,0.15); }} }}
  </style>
</head>
<body>
  <div class="layout">
    <canvas id="scene"></canvas>
    <aside class="side">
      <div class="header">
        <h1>One phantom: press playback + F-z curve</h1>
        <div id="subtitle"></div>
      </div>
      <canvas id="chart"></canvas>
      <div class="controls">
        <select id="curveSelect"></select>
        <button id="play" type="button">Pause</button>
        <button id="reset" type="button">Reset</button>
        <input id="scrub" type="range" min="0" max="0" value="0">
      </div>
      <div id="stats" class="stats"></div>
    </aside>
  </div>
  <script type="importmap">
    {{
      "imports": {{
        "three": "https://unpkg.com/three@0.165.0/build/three.module.js",
        "three/addons/": "https://unpkg.com/three@0.165.0/examples/jsm/"
      }}
    }}
  </script>
  <script type="module">
    import * as THREE from "three";
    import {{ OrbitControls }} from "three/addons/controls/OrbitControls.js";

    const data = {data_json};
    const phantom = data.phantom;
    const scan = data.scan;
    const canvas = document.getElementById("scene");
    const chart = document.getElementById("chart");
    const ctx = chart.getContext("2d");
    const subtitle = document.getElementById("subtitle");
    const stats = document.getElementById("stats");
    const select = document.getElementById("curveSelect");
    const playBtn = document.getElementById("play");
    const resetBtn = document.getElementById("reset");
    const scrub = document.getElementById("scrub");
    let curveIndex = 0;
    let step = 0;
    let playing = true;

    subtitle.textContent = `mesh ${{data.mesh.cells.join(" x ")}} cells, ${{data.lumps.length}} embedded lumps`;
    data.curves.forEach((curve, idx) => {{
      const option = document.createElement("option");
      option.value = String(idx);
      option.textContent = curve.label;
      select.appendChild(option);
    }});

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111417);
    const camera = new THREE.PerspectiveCamera(42, 1, 0.001, 20);
    const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true }});
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 0, phantom.height * 0.46);

    scene.add(new THREE.HemisphereLight(0xffffff, 0x48535c, 2.1));
    const light = new THREE.DirectionalLight(0xffffff, 2.6);
    light.position.set(0.25, -0.35, 0.7);
    scene.add(light);

    const box = new THREE.Mesh(
      new THREE.BoxGeometry(phantom.size_x, phantom.size_y, phantom.height),
      new THREE.MeshStandardMaterial({{ color: 0x77b6d8, transparent: true, opacity: 0.13, roughness: 0.76 }})
    );
    box.position.z = phantom.height * 0.5;
    scene.add(box);
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(box.geometry), new THREE.LineBasicMaterial({{ color: 0xc0e7f2, transparent: true, opacity: 0.42 }}));
    edges.position.copy(box.position);
    scene.add(edges);

    const surface = makeSurface();
    scene.add(surface);
    const lumpMeshes = data.lumps.map((lump, i) => makeLumpMesh(lump, i));
    lumpMeshes.forEach(mesh => scene.add(mesh));
    const probe = new THREE.Mesh(new THREE.SphereGeometry(scan.probe_radius, 42, 22), new THREE.MeshStandardMaterial({{ color: 0xf4eee2, metalness: 0.08, roughness: 0.3 }}));
    scene.add(probe);
    const ring = new THREE.Mesh(new THREE.RingGeometry(scan.probe_radius * 0.75, scan.probe_radius * 1.16, 56), new THREE.MeshBasicMaterial({{ color: 0xffc857, transparent: true, opacity: 0.9, side: THREE.DoubleSide }}));
    scene.add(ring);

    camera.position.set(phantom.size_x * 0.72, -phantom.size_y * 0.88, phantom.height * 1.35);
    controls.update();

    select.addEventListener("change", () => {{ curveIndex = Number(select.value); step = 0; update(); }});
    playBtn.addEventListener("click", () => {{ playing = !playing; playBtn.textContent = playing ? "Pause" : "Play"; }});
    resetBtn.addEventListener("click", () => {{ step = 0; update(); }});
    scrub.addEventListener("input", () => {{ step = Number(scrub.value); playing = false; playBtn.textContent = "Play"; update(); }});

    function makeSurface() {{
      const geo = new THREE.PlaneGeometry(phantom.size_x, phantom.size_y, 72, 72);
      geo.translate(0, 0, phantom.height);
      return new THREE.Mesh(geo, new THREE.MeshStandardMaterial({{ color: 0x8ac7e8, transparent: true, opacity: 0.58, roughness: 0.72, side: THREE.DoubleSide }}));
    }}

    function makeLumpMesh(lump, idx) {{
      const colors = [0xe0523f, 0x1692e6, 0xf2a20b, 0x45b36b, 0xa56bd6];
      const r = lump.radii;
      const mat = new THREE.MeshStandardMaterial({{ color: colors[idx % colors.length], transparent: true, opacity: 0.78, roughness: 0.45 }});
      let mesh;
      if (lump.shape === "box") mesh = new THREE.Mesh(new THREE.BoxGeometry(2*r[0], 2*r[1], 2*r[2]), mat);
      else if (lump.shape === "cylinder") {{ mesh = new THREE.Mesh(new THREE.CylinderGeometry(r[0], r[0], 2*r[2], 48), mat); mesh.rotation.x = Math.PI / 2; }}
      else if (lump.shape === "capsule") {{ mesh = new THREE.Mesh(new THREE.CapsuleGeometry(r[0], 2*r[2], 12, 36), mat); mesh.rotation.x = Math.PI / 2; }}
      else {{ mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 52, 26), mat); mesh.scale.set(r[0], r[1], r[2]); }}
      mesh.position.set(lump.center[0], lump.center[1], lump.center[2]);
      mesh.rotation.z += lump.yaw || 0;
      return mesh;
    }}

    function activeCurve() {{ return data.curves[curveIndex]; }}
    function activeTarget() {{
      const summary = activeCurve().summary;
      return [summary.x_m, summary.y_m];
    }}

    function deformSurface(px, py, depth) {{
      const pos = surface.geometry.attributes.position;
      const sigma = Math.max(scan.probe_radius * 1.35, 0.001);
      const sigma2 = sigma * sigma;
      for (let i = 0; i < pos.count; i++) {{
        const x = pos.getX(i), y = pos.getY(i);
        const falloff = Math.exp(-0.5 * ((x-px)*(x-px) + (y-py)*(y-py)) / sigma2);
        pos.setZ(i, phantom.height - depth * falloff);
      }}
      pos.needsUpdate = true;
      surface.geometry.computeVertexNormals();
    }}

    function update() {{
      const curve = activeCurve();
      scrub.max = String(curve.depths.length - 1);
      const depth = curve.depths[step];
      const force = curve.forces[step];
      const [px, py] = activeTarget();
      probe.position.set(px, py, phantom.height + scan.probe_radius + (scan.preload_gap || 0) - depth);
      ring.position.set(px, py, phantom.height + 0.0005);
      deformSurface(px, py, depth);
      scrub.value = String(step);
      const s = curve.summary;
      stats.textContent =
        `${{curve.label}}\\n` +
        `step ${{step + 1}} / ${{curve.depths.length}}\\n` +
        `indentation: ${{(depth * 1000).toFixed(2)}} mm\\n` +
        `Fz: ${{force.toFixed(2)}} N\\n` +
        `peak Fz: ${{Number(s.peak_force_n).toFixed(2)}} N\\n` +
        `late/early slope: ${{Number(s.late_early_slope_ratio).toFixed(2)}}`;
      drawChart();
    }}

    function resize() {{
      const rect = canvas.getBoundingClientRect();
      renderer.setSize(rect.width, rect.height, false);
      camera.aspect = rect.width / Math.max(rect.height, 1);
      camera.updateProjectionMatrix();
      const cRect = chart.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      chart.width = Math.max(1, Math.floor(cRect.width * dpr));
      chart.height = Math.max(1, Math.floor(cRect.height * dpr));
      drawChart();
    }}
    window.addEventListener("resize", resize);

    function drawChart() {{
      const curve = activeCurve();
      const w = chart.width, h = chart.height;
      if (!w || !h) return;
      const depths = curve.depths, forces = curve.forces;
      const maxZ = Math.max(...depths), maxF = Math.max(...forces);
      const padL = 58, padR = 18, padT = 24, padB = 46;
      const x0 = padL, y0 = h - padB, x1 = w - padR, y1 = padT;
      ctx.fillStyle = "#f6f1e8"; ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = "#c7c0b2"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x0, y1); ctx.lineTo(x0, y0); ctx.lineTo(x1, y0); ctx.stroke();
      ctx.font = `${{Math.max(11, Math.round(w / 58))}}px system-ui`;
      ctx.fillStyle = "#303438"; ctx.fillText("Fz [N]", x0, y1 - 8); ctx.fillText("indentation [mm]", Math.max(x0, x1 - 145), h - 14);
      function px(z) {{ return x0 + (z / maxZ) * (x1 - x0); }}
      function py(f) {{ return y0 - (f / maxF) * (y0 - y1); }}
      for (let i = 0; i <= 4; i++) {{
        const yy = y0 - (i / 4) * (y0 - y1);
        ctx.strokeStyle = "#ddd6c8"; ctx.beginPath(); ctx.moveTo(x0, yy); ctx.lineTo(x1, yy); ctx.stroke();
        ctx.fillStyle = "#55595d"; ctx.fillText(String(Math.round((i / 4) * maxF)), 8, yy + 4);
      }}
      ctx.strokeStyle = "#a1a8ad"; ctx.lineWidth = 2; ctx.beginPath();
      depths.forEach((z, i) => {{ const x = px(z), y = py(forces[i]); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }});
      ctx.stroke();
      ctx.strokeStyle = "#176d8f"; ctx.lineWidth = 4; ctx.beginPath();
      for (let i = 0; i <= step; i++) {{ const x = px(depths[i]), y = py(forces[i]); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }}
      ctx.stroke();
      ctx.fillStyle = "#d34a24"; ctx.beginPath(); ctx.arc(px(depths[step]), py(forces[step]), 6, 0, Math.PI * 2); ctx.fill();
    }}

    let last = performance.now();
    function tick(now) {{
      const dt = now - last; last = now;
      if (playing && dt < 90) {{
        const curve = activeCurve();
        step = (step + 1) % curve.depths.length;
        update();
      }}
      controls.update();
      renderer.render(scene, camera);
      requestAnimationFrame(tick);
    }}
    resize();
    update();
    requestAnimationFrame(tick);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
