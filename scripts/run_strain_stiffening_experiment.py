from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from palpation_sim.config import MaterialConfig, PhantomConfig, ScanConfig
from palpation_sim.phantom import LumpSpec
from palpation_sim.strain_stiffening import run_strain_stiffening_sample


SHAPES = ("sphere", "ellipsoid", "box", "cylinder", "capsule")
DEPTHS = (0.022, 0.028, 0.034, 0.040, 0.046, 0.052, 0.058)
STIFFNESSES = (1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
SIZE_SCALES = (0.75, 1.0, 1.25)
YAWS = (0.0, math.pi / 6.0, math.pi / 3.0, math.pi / 2.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a controlled strain-stiffening response sweep.")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/strain_stiffening_depth_shape_hardness"))
    parser.add_argument("--grid-size", type=int, default=9)
    parser.add_argument("--press-steps", type=int, default=72)
    parser.add_argument("--max-indentation", type=float, default=0.018)
    parser.add_argument("--hardening-b", type=float, default=1.8)
    parser.add_argument("--seed", type=int, default=28)
    args = parser.parse_args()

    start = time.time()
    rng = np.random.default_rng(args.seed)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    phantom = PhantomConfig(cells_x=32, cells_y=32, cells_z=12)
    material = MaterialConfig()
    scan = ScanConfig(
        grid_h=args.grid_size,
        grid_w=args.grid_size,
        press_steps=args.press_steps,
        max_indentation=args.max_indentation,
    )

    xy = np.stack(np.meshgrid(scan.x_values(phantom), scan.y_values(phantom)), axis=-1).astype(np.float32)
    center_row = args.grid_size // 2
    center_col = args.grid_size // 2
    depths = np.asarray(scan.indentation_values(), dtype=np.float32)

    cases: list[dict[str, object]] = []
    fz_maps: list[np.ndarray] = []
    center_curves: list[np.ndarray] = []
    ratio_maps: list[np.ndarray] = []

    total = len(SHAPES) * len(DEPTHS) * len(STIFFNESSES) * len(SIZE_SCALES) * len(YAWS)
    case_id = 0
    for shape in SHAPES:
        for center_depth in DEPTHS:
            for stiffness in STIFFNESSES:
                for size_scale in SIZE_SCALES:
                    for yaw in YAWS:
                        radii = _radii_for_shape(shape, size_scale)
                        if not _valid_depth(phantom, shape, radii, center_depth):
                            continue
                        lump = LumpSpec(
                            shape=shape,
                            center=(0.0, 0.0, phantom.height - center_depth),
                            radii=radii,
                            stiffness_multiplier=stiffness,
                            yaw=yaw,
                        )
                        sample = run_strain_stiffening_sample(
                            phantom,
                            material,
                            scan,
                            [lump],
                            rng,
                            hardening_b=args.hardening_b,
                            noise_std=0.0,
                            enforce_convex=True,
                        )
                        fz = np.asarray(sample["fz"], dtype=np.float32)
                        curve = fz[center_row, center_col].copy()
                        ratio_map = np.asarray(sample["nonlinearity_ratio"], dtype=np.float32)
                        summary = _curve_summary(depths, curve)
                        cases.append(
                            {
                                "case_id": case_id,
                                "shape": shape,
                                "center_depth_m": center_depth,
                                "top_depth_m": _top_depth(phantom, shape, radii, center_depth),
                                "stiffness_multiplier": stiffness,
                                "size_scale": size_scale,
                                "yaw_rad": yaw,
                                "peak_force_center_n": summary["peak_force_n"],
                                "early_slope_n_per_m": summary["early_slope_n_per_m"],
                                "late_slope_n_per_m": summary["late_slope_n_per_m"],
                                "late_early_slope_ratio": summary["late_early_slope_ratio"],
                                "loading_work_center_j": summary["loading_work_j"],
                                "convex_fraction_center": summary["convex_fraction"],
                            }
                        )
                        fz_maps.append(fz)
                        center_curves.append(curve)
                        ratio_maps.append(ratio_map)
                        case_id += 1
                        if case_id % 100 == 0:
                            elapsed = time.time() - start
                            print(f"generated {case_id}/{total} cases in {elapsed:.1f}s", flush=True)

    fz_array = np.stack(fz_maps, axis=0).astype(np.float32)
    center_array = np.stack(center_curves, axis=0).astype(np.float32)
    ratio_array = np.stack(ratio_maps, axis=0).astype(np.float32)
    case_table = _case_table_arrays(cases)

    np.savez_compressed(
        out_dir / "strain_stiffening_sweep.npz",
        fz=fz_array,
        center_curves=center_array,
        nonlinearity_ratio=ratio_array,
        indentation_depth=depths,
        xy=xy,
        center_row=np.asarray(center_row, dtype=np.int32),
        center_col=np.asarray(center_col, dtype=np.int32),
        phantom_json=np.asarray(json.dumps(phantom.to_dict())),
        material_json=np.asarray(json.dumps(material.to_dict())),
        scan_json=np.asarray(json.dumps(scan.to_dict())),
        **case_table,
    )
    _write_summary_csv(out_dir / "summary.csv", cases)
    _write_metadata(out_dir / "experiment_metadata.json", args, phantom, scan, material, len(cases), time.time() - start)
    _write_plots(out_dir, cases, depths, center_array)
    _write_player(out_dir / "press_player.html", cases, depths, center_array, phantom, scan)

    elapsed = time.time() - start
    print(f"done: {len(cases)} cases, fz shape {fz_array.shape}, elapsed {elapsed:.1f}s")
    print(f"outputs: {out_dir}")


def _radii_for_shape(shape: str, scale: float) -> tuple[float, float, float]:
    if shape == "sphere":
        base = (0.010, 0.010, 0.010)
    elif shape == "ellipsoid":
        base = (0.014, 0.009, 0.008)
    elif shape == "box":
        base = (0.011, 0.011, 0.007)
    elif shape == "cylinder":
        base = (0.012, 0.012, 0.008)
    elif shape == "capsule":
        base = (0.0075, 0.0075, 0.006)
    else:
        raise ValueError(f"unsupported shape {shape}")
    return tuple(float(v * scale) for v in base)


def _valid_depth(phantom: PhantomConfig, shape: str, radii: tuple[float, float, float], center_depth: float) -> bool:
    extent = _z_extent(shape, radii)
    return center_depth - extent >= 0.005 and center_depth + extent <= phantom.height - 0.005


def _top_depth(phantom: PhantomConfig, shape: str, radii: tuple[float, float, float], center_depth: float) -> float:
    del phantom
    return max(center_depth - _z_extent(shape, radii), 0.0)


def _z_extent(shape: str, radii: tuple[float, float, float]) -> float:
    if shape == "capsule":
        return float(radii[0] + radii[2])
    return float(radii[2])


def _curve_summary(z: np.ndarray, f: np.ndarray) -> dict[str, float]:
    slopes = np.diff(f) / np.maximum(np.diff(z), 1e-12)
    early = _segment_slope(z, f, 0.10, 0.35)
    late = _segment_slope(z, f, 0.65, 0.90)
    return {
        "peak_force_n": float(np.max(f)),
        "early_slope_n_per_m": early,
        "late_slope_n_per_m": late,
        "late_early_slope_ratio": float(late / early) if early > 1e-12 else 0.0,
        "loading_work_j": _trapz(f, z),
        "convex_fraction": float(np.mean(np.diff(slopes) >= -1e-5)) if slopes.size > 1 else 1.0,
    }


def _segment_slope(z: np.ndarray, f: np.ndarray, lo: float, hi: float) -> float:
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    span = z_max - z_min
    keep = (z >= z_min + lo * span) & (z <= z_min + hi * span)
    if int(np.count_nonzero(keep)) < 2:
        return 0.0
    zz = z[keep].astype(np.float64)
    ff = f[keep].astype(np.float64)
    zc = zz - float(np.mean(zz))
    denom = float(np.sum(zc * zc))
    if denom <= 1e-18:
        return 0.0
    return float(np.sum(zc * (ff - float(np.mean(ff)))) / denom)


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    integrate = getattr(np, "trapezoid", None) or np.trapz
    return float(integrate(y, x))


def _case_table_arrays(cases: list[dict[str, object]]) -> dict[str, np.ndarray]:
    return {
        "case_id": np.asarray([c["case_id"] for c in cases], dtype=np.int32),
        "shape": np.asarray([c["shape"] for c in cases]),
        "center_depth_m": np.asarray([c["center_depth_m"] for c in cases], dtype=np.float32),
        "top_depth_m": np.asarray([c["top_depth_m"] for c in cases], dtype=np.float32),
        "stiffness_multiplier": np.asarray([c["stiffness_multiplier"] for c in cases], dtype=np.float32),
        "size_scale": np.asarray([c["size_scale"] for c in cases], dtype=np.float32),
        "yaw_rad": np.asarray([c["yaw_rad"] for c in cases], dtype=np.float32),
        "peak_force_center_n": np.asarray([c["peak_force_center_n"] for c in cases], dtype=np.float32),
        "late_early_slope_ratio": np.asarray([c["late_early_slope_ratio"] for c in cases], dtype=np.float32),
    }


def _write_summary_csv(path: Path, cases: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(cases[0].keys()))
        writer.writeheader()
        writer.writerows(cases)


def _write_metadata(
    path: Path,
    args: argparse.Namespace,
    phantom: PhantomConfig,
    scan: ScanConfig,
    material: MaterialConfig,
    case_count: int,
    elapsed: float,
) -> None:
    metadata = {
        "schema_version": 1,
        "backend": "strain_stiffening",
        "description": "Controlled sweep of lump center depth, shape, and stiffness multiplier.",
        "case_count": case_count,
        "elapsed_seconds": elapsed,
        "requested_parameters": vars(args),
        "phantom": phantom.to_dict(),
        "scan": scan.to_dict(),
        "material": material.to_dict(),
        "shapes": list(SHAPES),
        "center_depths_m": list(DEPTHS),
        "stiffness_multipliers": list(STIFFNESSES),
        "size_scales": list(SIZE_SCALES),
        "yaws_rad": list(YAWS),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def _write_plots(out_dir: Path, cases: list[dict[str, object]], depths: np.ndarray, curves: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for experiment plots") from exc

    _plot_family(
        out_dir / "comparison_depth_sweep.png",
        cases,
        depths,
        curves,
        select={"shape": "sphere", "stiffness_multiplier": 6.0, "size_scale": 1.0, "yaw_rad": 0.0},
        group_key="center_depth_m",
        title="Depth sweep: sphere, stiffness 6x",
    )
    _plot_family(
        out_dir / "comparison_hardness_sweep.png",
        cases,
        depths,
        curves,
        select={"shape": "sphere", "center_depth_m": 0.040, "size_scale": 1.0, "yaw_rad": 0.0},
        group_key="stiffness_multiplier",
        title="Hardness sweep: sphere, center depth 40 mm",
    )
    _plot_family(
        out_dir / "comparison_shape_sweep.png",
        cases,
        depths,
        curves,
        select={"center_depth_m": 0.040, "stiffness_multiplier": 6.0, "size_scale": 1.0, "yaw_rad": 0.0},
        group_key="shape",
        title="Shape sweep: depth 40 mm, stiffness 6x",
    )
    _plot_heatmaps(out_dir / "slope_ratio_heatmaps.png", cases)
    _plot_peak_force(out_dir / "peak_force_depth_shape.png", cases)
    plt.close("all")


def _plot_family(
    path: Path,
    cases: list[dict[str, object]],
    depths: np.ndarray,
    curves: np.ndarray,
    *,
    select: dict[str, object],
    group_key: str,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    selected = []
    for i, case in enumerate(cases):
        if all(_matches(case[k], v) for k, v in select.items()):
            selected.append((case[group_key], i))
    selected.sort(key=lambda item: str(item[0]) if isinstance(item[0], str) else float(item[0]))
    for value, idx in selected:
        label = _label(group_key, value)
        ax.plot(depths * 1000.0, curves[idx], linewidth=2.0, label=label)
    ax.set_title(title)
    ax.set_xlabel("indentation [mm]")
    ax.set_ylabel("Fz [N]")
    ax.grid(True, alpha=0.28)
    ax.legend(ncols=2, fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_heatmaps(path: Path, cases: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    depths = np.asarray(DEPTHS)
    stiffnesses = np.asarray(STIFFNESSES)
    fig, axes = plt.subplots(1, len(SHAPES), figsize=(18, 3.9), constrained_layout=True, sharey=True)
    for ax, shape in zip(axes, SHAPES):
        heat = np.zeros((len(depths), len(stiffnesses)), dtype=np.float32)
        for i, depth in enumerate(depths):
            for j, stiffness in enumerate(stiffnesses):
                values = [
                    float(c["late_early_slope_ratio"])
                    for c in cases
                    if c["shape"] == shape
                    and _matches(c["center_depth_m"], depth)
                    and _matches(c["stiffness_multiplier"], stiffness)
                ]
                heat[i, j] = float(np.median(values)) if values else np.nan
        im = ax.imshow(heat, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(shape)
        ax.set_xlabel("stiffness x")
        ax.set_xticks(range(len(stiffnesses)), [f"{v:g}" for v in stiffnesses], rotation=45)
        ax.set_yticks(range(len(depths)), [f"{v*1000:.0f}" for v in depths])
    axes[0].set_ylabel("center depth [mm]")
    fig.colorbar(im, ax=axes, shrink=0.86, label="late/early slope ratio")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_peak_force(path: Path, cases: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for shape in SHAPES:
        x = []
        y = []
        for depth in DEPTHS:
            values = [
                float(c["peak_force_center_n"])
                for c in cases
                if c["shape"] == shape
                and _matches(c["center_depth_m"], depth)
                and _matches(c["stiffness_multiplier"], 6.0)
                and _matches(c["size_scale"], 1.0)
            ]
            if values:
                x.append(depth * 1000.0)
                y.append(float(np.median(values)))
        ax.plot(x, y, marker="o", linewidth=2.0, label=shape)
    ax.set_title("Peak Fz vs lump center depth, stiffness 6x")
    ax.set_xlabel("lump center depth from top [mm]")
    ax.set_ylabel("peak Fz at 18 mm indentation [N]")
    ax.grid(True, alpha=0.28)
    ax.legend(ncols=3, fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _matches(actual: object, expected: object) -> bool:
    if isinstance(expected, float) or isinstance(actual, float):
        return abs(float(actual) - float(expected)) < 1e-6
    return actual == expected


def _label(key: str, value: object) -> str:
    if key.endswith("_m"):
        return f"{float(value) * 1000:.0f} mm"
    if key == "stiffness_multiplier":
        return f"{float(value):g}x"
    return str(value)


def _write_player(
    path: Path,
    cases: list[dict[str, object]],
    depths: np.ndarray,
    curves: np.ndarray,
    phantom: PhantomConfig,
    scan: ScanConfig,
) -> None:
    idx = _find_case(
        cases,
        shape="capsule",
        center_depth_m=0.040,
        stiffness_multiplier=8.0,
        size_scale=1.25,
        yaw_rad=math.pi / 6.0,
    )
    if idx is None:
        idx = 0
    case = cases[idx]
    radii = _radii_for_shape(str(case["shape"]), float(case["size_scale"]))
    lump = {
        "shape": case["shape"],
        "center": [0.0, 0.0, phantom.height - float(case["center_depth_m"])],
        "radii": list(radii),
        "stiffness_multiplier": case["stiffness_multiplier"],
        "yaw": case["yaw_rad"],
    }
    data = {
        "case": case,
        "phantom": phantom.to_dict(),
        "scan": scan.to_dict(),
        "lump": lump,
        "depths": depths.tolist(),
        "forces": curves[idx].tolist(),
    }
    path.write_text(_player_html(json.dumps(data, separators=(",", ":")).replace("</", "<\\/")), encoding="utf-8")


def _find_case(cases: list[dict[str, object]], **select: object) -> int | None:
    for i, case in enumerate(cases):
        if all(_matches(case[k], v) for k, v in select.items()):
            return i
    return None


def _player_html(data_json: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Strain-Stiffening Press Player</title>
  <style>
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #111417;
      color: #f4f1ea;
      font: 14px/1.4 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(340px, 0.85fr);
      height: 100%;
    }}
    #scene {{ width: 100%; height: 100%; display: block; }}
    .side {{
      border-left: 1px solid rgba(255,255,255,0.16);
      background: #171b1f;
      display: grid;
      grid-template-rows: auto minmax(260px, 1fr) auto;
      min-width: 0;
    }}
    .header, .controls, .stats {{ padding: 12px 14px; }}
    .header {{ border-bottom: 1px solid rgba(255,255,255,0.12); }}
    .header h1 {{ margin: 0 0 5px; font-size: 18px; font-weight: 700; }}
    .header div {{ color: #c9c4b8; }}
    #chart {{ width: 100%; height: 100%; display: block; background: #f6f2e9; }}
    .controls {{
      display: grid;
      grid-template-columns: auto auto 1fr;
      gap: 10px;
      align-items: center;
      border-top: 1px solid rgba(255,255,255,0.12);
    }}
    button {{
      border: 1px solid rgba(255,255,255,0.22);
      border-radius: 6px;
      background: #ece7dd;
      color: #151719;
      padding: 7px 10px;
      font: inherit;
      cursor: pointer;
    }}
    input[type="range"] {{ width: 100%; }}
    .stats {{
      color: #d8d2c7;
      border-top: 1px solid rgba(255,255,255,0.12);
      white-space: pre-wrap;
    }}
    @media (max-width: 860px) {{
      .layout {{ grid-template-columns: 1fr; grid-template-rows: 56vh 44vh; }}
      .side {{ border-left: 0; border-top: 1px solid rgba(255,255,255,0.16); }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <canvas id="scene"></canvas>
    <aside class="side">
      <div class="header">
        <h1>3D press and F-z response</h1>
        <div id="subtitle"></div>
      </div>
      <canvas id="chart"></canvas>
      <div class="controls">
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
    const lump = data.lump;
    const depths = data.depths;
    const forces = data.forces;
    const sceneCanvas = document.getElementById("scene");
    const chart = document.getElementById("chart");
    const ctx = chart.getContext("2d");
    const subtitle = document.getElementById("subtitle");
    const stats = document.getElementById("stats");
    const playBtn = document.getElementById("play");
    const resetBtn = document.getElementById("reset");
    const scrub = document.getElementById("scrub");
    scrub.max = String(depths.length - 1);
    subtitle.textContent = `${{lump.shape}}, center depth ${{(data.case.center_depth_m * 1000).toFixed(0)}} mm, stiffness ${{data.case.stiffness_multiplier}}x`;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111417);
    const camera = new THREE.PerspectiveCamera(42, 1, 0.001, 10);
    const renderer = new THREE.WebGLRenderer({{ canvas: sceneCanvas, antialias: true }});
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 0, phantom.height * 0.45);

    scene.add(new THREE.HemisphereLight(0xffffff, 0x46515a, 2.0));
    const light = new THREE.DirectionalLight(0xffffff, 2.6);
    light.position.set(0.2, -0.35, 0.7);
    scene.add(light);

    const box = new THREE.Mesh(
      new THREE.BoxGeometry(phantom.size_x, phantom.size_y, phantom.height),
      new THREE.MeshStandardMaterial({{ color: 0x77b6d8, transparent: true, opacity: 0.16, roughness: 0.75 }})
    );
    box.position.z = phantom.height * 0.5;
    scene.add(box);
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(box.geometry),
      new THREE.LineBasicMaterial({{ color: 0xb8dfed, transparent: true, opacity: 0.45 }})
    );
    edges.position.copy(box.position);
    scene.add(edges);

    const surface = makeSurface();
    scene.add(surface);
    const lumpMesh = makeLumpMesh();
    scene.add(lumpMesh);
    const probe = new THREE.Mesh(
      new THREE.SphereGeometry(scan.probe_radius, 40, 20),
      new THREE.MeshStandardMaterial({{ color: 0xf4eee2, metalness: 0.08, roughness: 0.3 }})
    );
    scene.add(probe);
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(scan.probe_radius * 0.75, scan.probe_radius * 1.18, 56),
      new THREE.MeshBasicMaterial({{ color: 0xffc857, transparent: true, opacity: 0.9, side: THREE.DoubleSide }})
    );
    scene.add(ring);

    camera.position.set(phantom.size_x * 0.75, -phantom.size_y * 0.9, phantom.height * 1.35);
    controls.update();

    let step = 0;
    let playing = true;
    let last = performance.now();
    playBtn.addEventListener("click", () => {{
      playing = !playing;
      playBtn.textContent = playing ? "Pause" : "Play";
    }});
    resetBtn.addEventListener("click", () => {{
      step = 0;
      update();
    }});
    scrub.addEventListener("input", () => {{
      step = Number(scrub.value);
      playing = false;
      playBtn.textContent = "Play";
      update();
    }});

    function makeSurface() {{
      const geo = new THREE.PlaneGeometry(phantom.size_x, phantom.size_y, 52, 52);
      geo.translate(0, 0, phantom.height);
      const mat = new THREE.MeshStandardMaterial({{ color: 0x8ac7e8, transparent: true, opacity: 0.62, roughness: 0.7, side: THREE.DoubleSide }});
      return new THREE.Mesh(geo, mat);
    }}

    function makeLumpMesh() {{
      const r = lump.radii;
      let mesh;
      const mat = new THREE.MeshStandardMaterial({{ color: 0xe0523f, transparent: true, opacity: 0.78, roughness: 0.45 }});
      if (lump.shape === "box") {{
        mesh = new THREE.Mesh(new THREE.BoxGeometry(2 * r[0], 2 * r[1], 2 * r[2]), mat);
      }} else if (lump.shape === "cylinder") {{
        mesh = new THREE.Mesh(new THREE.CylinderGeometry(r[0], r[0], 2 * r[2], 48), mat);
        mesh.rotation.x = Math.PI / 2;
      }} else if (lump.shape === "capsule") {{
        mesh = new THREE.Mesh(new THREE.CapsuleGeometry(r[0], 2 * r[2], 12, 32), mat);
        mesh.rotation.x = Math.PI / 2;
      }} else {{
        mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 24), mat);
        mesh.scale.set(r[0], r[1], r[2]);
      }}
      mesh.position.set(lump.center[0], lump.center[1], lump.center[2]);
      mesh.rotation.z = lump.yaw || 0;
      return mesh;
    }}

    function deformSurface(depth) {{
      const pos = surface.geometry.attributes.position;
      const sigma = Math.max(scan.probe_radius * 1.35, 0.001);
      const sigma2 = sigma * sigma;
      for (let i = 0; i < pos.count; i++) {{
        const x = pos.getX(i);
        const y = pos.getY(i);
        const falloff = Math.exp(-0.5 * (x * x + y * y) / sigma2);
        pos.setZ(i, phantom.height - depth * falloff);
      }}
      pos.needsUpdate = true;
      surface.geometry.computeVertexNormals();
    }}

    function resize() {{
      const rect = sceneCanvas.getBoundingClientRect();
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

    function update() {{
      const depth = depths[step];
      const force = forces[step];
      probe.position.set(0, 0, phantom.height + scan.probe_radius + (scan.preload_gap || 0) - depth);
      ring.position.set(0, 0, phantom.height + 0.0005);
      deformSurface(depth);
      scrub.value = String(step);
      stats.textContent =
        `step ${{step + 1}} / ${{depths.length}}\\n` +
        `indentation: ${{(depth * 1000).toFixed(2)}} mm\\n` +
        `Fz: ${{force.toFixed(2)}} N\\n` +
        `late/early slope ratio: ${{Number(data.case.late_early_slope_ratio).toFixed(2)}}`;
      drawChart();
    }}

    function drawChart() {{
      const w = chart.width;
      const h = chart.height;
      if (!w || !h) return;
      ctx.clearRect(0, 0, w, h);
      const padL = 58, padR = 18, padT = 24, padB = 46;
      const x0 = padL, y0 = h - padB, x1 = w - padR, y1 = padT;
      const maxZ = Math.max(...depths);
      const maxF = Math.max(...forces);
      ctx.fillStyle = "#f6f2e9";
      ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = "#c7c0b2";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x0, y1);
      ctx.lineTo(x0, y0);
      ctx.lineTo(x1, y0);
      ctx.stroke();
      ctx.fillStyle = "#2c3034";
      ctx.font = `${{Math.max(11, Math.round(w / 58))}}px system-ui`;
      ctx.fillText("Fz [N]", x0, y1 - 8);
      ctx.fillText("indentation [mm]", Math.max(x0, x1 - 145), h - 14);
      for (let i = 0; i <= 4; i++) {{
        const yy = y0 - (i / 4) * (y0 - y1);
        ctx.strokeStyle = "#ddd6c8";
        ctx.beginPath();
        ctx.moveTo(x0, yy);
        ctx.lineTo(x1, yy);
        ctx.stroke();
        ctx.fillStyle = "#4d5155";
        ctx.fillText(String(Math.round((i / 4) * maxF)), 8, yy + 4);
      }}
      function px(z) {{ return x0 + (z / maxZ) * (x1 - x0); }}
      function py(f) {{ return y0 - (f / maxF) * (y0 - y1); }}
      ctx.strokeStyle = "#9aa3a8";
      ctx.lineWidth = 2;
      ctx.beginPath();
      depths.forEach((z, i) => {{
        const x = px(z), y = py(forces[i]);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }});
      ctx.stroke();
      ctx.strokeStyle = "#176d8f";
      ctx.lineWidth = 4;
      ctx.beginPath();
      for (let i = 0; i <= step; i++) {{
        const x = px(depths[i]), y = py(forces[i]);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }}
      ctx.stroke();
      const mx = px(depths[step]), my = py(forces[step]);
      ctx.fillStyle = "#d34a24";
      ctx.beginPath();
      ctx.arc(mx, my, 6, 0, Math.PI * 2);
      ctx.fill();
    }}

    function tick(now) {{
      const dt = now - last;
      last = now;
      if (playing && dt < 80) {{
        step = (step + 1) % depths.length;
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
