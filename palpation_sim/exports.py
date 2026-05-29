from __future__ import annotations

import base64
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import MaterialConfig, PhantomConfig, ScanConfig
from .phantom import LumpSpec, create_structured_tet_mesh, material_arrays_for_lumps, normalize_lumps

TET_FACES: tuple[tuple[int, int, int], ...] = ((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2))
TET_EDGES: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def build_ground_truth_metadata(
    *,
    sample_id: str,
    split: str,
    phantom: PhantomConfig,
    material: MaterialConfig,
    scan: ScanConfig,
    lumps: LumpSpec | Sequence[LumpSpec],
    sample: dict[str, object] | None = None,
    npz_path: Path | None = None,
    gltf_path: Path | None = None,
    press_records_dir: Path | None = None,
    scan_animation_path: Path | None = None,
) -> dict[str, object]:
    """Create a JSON-friendly metadata record for one generated phantom."""
    lump_list = normalize_lumps(lumps)
    metadata: dict[str, object] = {
        "schema_version": 1,
        "sample_id": sample_id,
        "split": split,
        "files": {
            "npz": npz_path.name if npz_path is not None else None,
            "phantom_3d": gltf_path.name if gltf_path is not None else None,
            "press_records": press_records_dir.name if press_records_dir is not None else None,
            "scan_animation": scan_animation_path.name if scan_animation_path is not None else None,
        },
        "backend": _sample_scalar(sample, "backend"),
        "phantom": phantom.to_dict(),
        "material": material.to_dict(),
        "scan": {
            **scan.to_dict(),
            "x_values": scan.x_values(phantom),
            "y_values": scan.y_values(phantom),
            "indentation_values": scan.indentation_values(),
        },
        "num_lumps": len(lump_list),
        "lumps": [_lump_metadata(idx, lump, phantom, material) for idx, lump in enumerate(lump_list)],
    }

    if sample is not None:
        arrays = {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in sample.items()
            if isinstance(value, np.ndarray)
        }
        metadata["arrays"] = arrays
        if "mask" in sample:
            mask = np.asarray(sample["mask"])
            positive = int(np.count_nonzero(mask > 0.5))
            total = int(mask.size)
            metadata["mask"] = {
                "shape": list(mask.shape),
                "positive_cells": positive,
                "total_cells": total,
                "coverage_fraction": float(positive / max(total, 1)),
            }

    return metadata


def write_press_records(
    out_dir: Path,
    sample: dict[str, object],
    *,
    sample_id: str,
    split: str,
) -> None:
    """Write one CSV and one F-z plot for every press location in a sample."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to write press record plots. "
            "Use /home/goodmansun/miniconda3/envs/torchnightly/bin/python."
        ) from exc

    presses = _require_sample_array(sample, "presses")
    if presses.ndim != 4 or presses.shape[-1] < 2:
        raise ValueError(f"Expected presses shape [H, W, T, 2], got {presses.shape}")

    h, w, t, _ = presses.shape
    xy = _optional_sample_array(sample, "xy")
    probe_pose = _optional_sample_array(sample, "probe_pose")
    indentation = _optional_sample_array(sample, "indentation_depth")
    fz = _optional_sample_array(sample, "fz")
    contact_features = _optional_sample_array(sample, "contact_features")

    out_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, object]] = []
    f_min = float(np.nanmin(presses[..., 1]))
    f_max = float(np.nanmax(presses[..., 1]))
    z_min = float(np.nanmin(presses[..., 0]))
    z_max = float(np.nanmax(presses[..., 0]))

    for row in range(h):
        for col in range(w):
            stem = f"press_r{row:03d}_c{col:03d}"
            csv_name = f"{stem}.csv"
            png_name = f"{stem}_fz.png"
            x_value, y_value = _xy_at(xy, row, col)
            z_disp = _curve_or_fallback(indentation, presses, row, col, channel=0)
            force = _curve_or_fallback(fz, presses, row, col, channel=1)
            probe_z = _probe_z_at(probe_pose, row, col, t)
            contacts = _contact_features_at(contact_features, row, col, t)

            _write_press_csv(
                out_dir / csv_name,
                z_disp=z_disp,
                force=force,
                probe_z=probe_z,
                x_value=x_value,
                y_value=y_value,
                contacts=contacts,
            )
            _write_press_plot(
                plt,
                out_dir / png_name,
                z_disp=z_disp,
                force=force,
                x_value=x_value,
                y_value=y_value,
                title=f"{sample_id} r{row:03d} c{col:03d}",
                z_limits=(z_min, z_max),
                f_limits=(f_min, f_max),
            )
            index_rows.append(
                {
                    "row": row,
                    "col": col,
                    "x_m": x_value,
                    "y_m": y_value,
                    "csv": csv_name,
                    "plot": png_name,
                    "force_min_n": float(np.nanmin(force)),
                    "force_max_n": float(np.nanmax(force)),
                    "z_displacement_min_m": float(np.nanmin(z_disp)),
                    "z_displacement_max_m": float(np.nanmax(z_disp)),
                }
            )

    _write_press_index(out_dir / "index.csv", index_rows)
    manifest = {
        "schema_version": 1,
        "sample_id": sample_id,
        "split": split,
        "grid_shape": [h, w],
        "press_steps": t,
        "num_presses": h * w,
        "columns": _press_csv_columns(),
        "files": {
            "index": "index.csv",
            "per_press_csv_pattern": "press_r{row:03d}_c{col:03d}.csv",
            "per_press_plot_pattern": "press_r{row:03d}_c{col:03d}_fz.png",
        },
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def write_ground_truth_metadata(path: Path, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def _require_sample_array(sample: dict[str, object], key: str) -> np.ndarray:
    if key not in sample:
        raise KeyError(f"sample must contain '{key}'")
    return np.asarray(sample[key], dtype=np.float32)


def _optional_sample_array(sample: dict[str, object], key: str) -> np.ndarray | None:
    if key not in sample:
        return None
    return np.asarray(sample[key], dtype=np.float32)


def _xy_at(xy: np.ndarray | None, row: int, col: int) -> tuple[float | None, float | None]:
    if xy is None or xy.ndim != 3 or xy.shape[-1] < 2:
        return None, None
    return float(xy[row, col, 0]), float(xy[row, col, 1])


def _curve_or_fallback(
    array: np.ndarray | None,
    presses: np.ndarray,
    row: int,
    col: int,
    *,
    channel: int,
) -> np.ndarray:
    if array is not None and array.ndim == 3:
        return np.asarray(array[row, col], dtype=np.float32)
    return np.asarray(presses[row, col, :, channel], dtype=np.float32)


def _probe_z_at(probe_pose: np.ndarray | None, row: int, col: int, steps: int) -> np.ndarray:
    if probe_pose is not None and probe_pose.ndim == 4 and probe_pose.shape[-1] >= 3:
        return np.asarray(probe_pose[row, col, :, 2], dtype=np.float32)
    return np.full(steps, np.nan, dtype=np.float32)


def _contact_features_at(contact_features: np.ndarray | None, row: int, col: int, steps: int) -> np.ndarray:
    if contact_features is not None and contact_features.ndim == 4:
        return np.asarray(contact_features[row, col], dtype=np.float32)
    return np.full((steps, 0), np.nan, dtype=np.float32)


def _press_csv_columns() -> list[str]:
    return [
        "step",
        "x_m",
        "y_m",
        "z_displacement_m",
        "probe_z_m",
        "force_z_n",
        "contact_feature_0",
        "contact_feature_1",
        "contact_feature_2",
        "contact_feature_3",
        "contact_feature_4",
    ]


def _write_press_csv(
    path: Path,
    *,
    z_disp: np.ndarray,
    force: np.ndarray,
    probe_z: np.ndarray,
    x_value: float | None,
    y_value: float | None,
    contacts: np.ndarray,
) -> None:
    columns = _press_csv_columns()
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for step in range(z_disp.shape[0]):
            contact = contacts[step] if contacts.size else np.asarray([], dtype=np.float32)
            row = {
                "step": step,
                "x_m": _csv_float(x_value),
                "y_m": _csv_float(y_value),
                "z_displacement_m": _csv_float(float(z_disp[step])),
                "probe_z_m": _csv_float(float(probe_z[step])),
                "force_z_n": _csv_float(float(force[step])),
                "contact_feature_0": _csv_float(float(contact[0])) if contact.shape[0] > 0 else "",
                "contact_feature_1": _csv_float(float(contact[1])) if contact.shape[0] > 1 else "",
                "contact_feature_2": _csv_float(float(contact[2])) if contact.shape[0] > 2 else "",
                "contact_feature_3": _csv_float(float(contact[3])) if contact.shape[0] > 3 else "",
                "contact_feature_4": _csv_float(float(contact[4])) if contact.shape[0] > 4 else "",
            }
            writer.writerow(row)


def _write_press_plot(
    plt: object,
    path: Path,
    *,
    z_disp: np.ndarray,
    force: np.ndarray,
    x_value: float | None,
    y_value: float | None,
    title: str,
    z_limits: tuple[float, float],
    f_limits: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.2), constrained_layout=True)
    ax.plot(z_disp, force, color="#176d8f", linewidth=1.8)
    ax.scatter(z_disp, force, s=12, color="#d34a24", zorder=3)
    ax.set_xlabel("z displacement / indentation [m]")
    ax.set_ylabel("Fz [N]")
    ax.set_title(title)
    if x_value is not None and y_value is not None:
        ax.text(
            0.02,
            0.96,
            f"x={x_value:.4f} m\ny={y_value:.4f} m",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 3},
        )
    z_lo, z_hi = _expanded_limits(z_limits)
    f_lo, f_hi = _expanded_limits(f_limits)
    ax.set_xlim(z_lo, z_hi)
    ax.set_ylim(f_lo, f_hi)
    ax.grid(True, linewidth=0.6, alpha=0.32)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _expanded_limits(limits: tuple[float, float]) -> tuple[float, float]:
    lo, hi = limits
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 1.0
    if hi <= lo:
        pad = max(abs(lo) * 0.08, 1e-6)
        return lo - pad, hi + pad
    pad = 0.06 * (hi - lo)
    return lo - pad, hi + pad


def _write_press_index(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "row",
        "col",
        "x_m",
        "y_m",
        "csv",
        "plot",
        "force_min_n",
        "force_max_n",
        "z_displacement_min_m",
        "z_displacement_max_m",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _csv_float(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{value:.10g}"


def write_scan_animation_html(
    path: Path,
    phantom: PhantomConfig,
    scan: ScanConfig,
    sample: dict[str, object],
    lumps: LumpSpec | Sequence[LumpSpec],
) -> None:
    """Write an interactive 3D HTML animation of the scan/probe/deformation process."""
    presses = _require_sample_array(sample, "presses")
    xy = _require_sample_array(sample, "xy")
    if presses.ndim != 4 or presses.shape[-1] < 2:
        raise ValueError(f"Expected presses shape [H, W, T, 2], got {presses.shape}")
    if xy.ndim != 3 or xy.shape[-1] < 2:
        raise ValueError(f"Expected xy shape [H, W, 2], got {xy.shape}")

    lump_list = normalize_lumps(lumps)
    mesh = create_structured_tet_mesh(phantom)
    _, _, _, _, tet_lump_id = material_arrays_for_lumps(mesh, MaterialConfig(), lump_list)
    normal_faces = _selected_tet_boundary_faces(mesh.tets, tet_lump_id < 0)
    lump_surfaces = []
    for idx, lump in enumerate(lump_list):
        faces = _selected_tet_boundary_faces(mesh.tets, tet_lump_id == idx)
        if faces.size == 0:
            continue
        lump_surfaces.append(
            {
                "id": idx,
                "shape": lump.shape,
                "triangles": np.asarray(faces, dtype=np.uint32).reshape(-1).tolist(),
            }
        )

    data = {
        "sample_id": path.stem.replace("_scan_animation", ""),
        "phantom": phantom.to_dict(),
        "scan": scan.to_dict(),
        "xy": np.asarray(xy, dtype=np.float32).tolist(),
        "indentation": np.asarray(presses[..., 0], dtype=np.float32).tolist(),
        "force_z": np.asarray(presses[..., 1], dtype=np.float32).tolist(),
        "lumps": [lump.to_dict(phantom) for lump in lump_list],
        "mesh": {
            "vertices": np.asarray(mesh.vertices, dtype=np.float32).reshape(-1).tolist(),
            "normal_triangles": np.asarray(normal_faces, dtype=np.uint32).reshape(-1).tolist(),
            "tet_edges": _all_tet_edges(mesh.tets).reshape(-1).tolist(),
            "lump_surfaces": lump_surfaces,
        },
    }
    data_json = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(_scan_animation_html(data_json))


def _scan_animation_html(data_json: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Phantom Scan Animation</title>
  <style>
    html, body {{
      width: 100%;
      height: 100%;
      margin: 0;
      overflow: hidden;
      background: #101214;
      color: #f3f1ea;
      font: 14px/1.4 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    #viewport {{ position: fixed; inset: 0; }}
    .panel {{
      position: fixed;
      top: 12px;
      left: 12px;
      right: 12px;
      z-index: 2;
      display: grid;
      grid-template-columns: minmax(180px, 1fr) auto;
      gap: 10px;
      align-items: center;
      padding: 8px 10px;
      border: 1px solid rgba(255,255,255,0.16);
      border-radius: 8px;
      background: rgba(16,18,20,0.74);
      backdrop-filter: blur(8px);
    }}
    .title {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-weight: 650;
    }}
    .controls {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    label {{ display: inline-flex; align-items: center; gap: 5px; user-select: none; }}
    button {{
      border: 1px solid rgba(255,255,255,0.28);
      border-radius: 6px;
      background: #f4f0e8;
      color: #17191b;
      padding: 5px 9px;
      font: inherit;
      cursor: pointer;
    }}
    input[type="range"] {{ width: 120px; }}
    .status {{
      position: fixed;
      left: 12px;
      bottom: 12px;
      z-index: 2;
      max-width: min(760px, calc(100vw - 24px));
      padding: 7px 9px;
      border: 1px solid rgba(255,255,255,0.16);
      border-radius: 8px;
      background: rgba(16,18,20,0.74);
      color: #dedbd1;
      backdrop-filter: blur(8px);
      white-space: pre-wrap;
    }}
    @media (max-width: 720px) {{
      .panel {{ grid-template-columns: 1fr; }}
      .controls {{ justify-content: flex-start; }}
    }}
  </style>
</head>
<body>
  <canvas id="viewport"></canvas>
  <div class="panel">
    <div id="title" class="title">scan animation</div>
    <div class="controls">
      <button id="playButton" type="button">Pause</button>
      <button id="resetButton" type="button">Reset</button>
      <label>speed <input id="speedRange" type="range" min="1" max="120" value="36"></label>
      <label><input id="autoViewToggle" type="checkbox" checked> auto view</label>
      <label><input id="surfaceToggle" type="checkbox" checked> tissue</label>
      <label><input id="lumpToggle" type="checkbox" checked> lumps</label>
      <label><input id="wireToggle" type="checkbox" checked> tet wire</label>
      <label><input id="vertexToggle" type="checkbox" checked> vertices</label>
    </div>
  </div>
  <div id="status" class="status">Loading...</div>
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

    const scanData = {data_json};
    const canvas = document.getElementById("viewport");
    const title = document.getElementById("title");
    const status = document.getElementById("status");
    const playButton = document.getElementById("playButton");
    const resetButton = document.getElementById("resetButton");
    const speedRange = document.getElementById("speedRange");
    const autoViewToggle = document.getElementById("autoViewToggle");
    const surfaceToggle = document.getElementById("surfaceToggle");
    const lumpToggle = document.getElementById("lumpToggle");
    const wireToggle = document.getElementById("wireToggle");
    const vertexToggle = document.getElementById("vertexToggle");

    const phantom = scanData.phantom;
    const scan = scanData.scan;
    const rows = scanData.xy.length;
    const cols = scanData.xy[0].length;
    const steps = scanData.indentation[0][0].length;
    const totalFrames = rows * cols * steps;
    const sampleId = scanData.sample_id || "phantom";
    title.textContent = `${{sampleId}} scan animation`;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101214);
    const camera = new THREE.PerspectiveCamera(46, window.innerWidth / window.innerHeight, 0.001, 20);
    const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.6;

    scene.add(new THREE.HemisphereLight(0xffffff, 0x4b555d, 2.2));
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.4);
    keyLight.position.set(0.25, -0.45, 0.8);
    scene.add(keyLight);

    const tissueGroup = new THREE.Group();
    const lumpGroup = new THREE.Group();
    const wireGroup = new THREE.Group();
    const vertexGroup = new THREE.Group();
    scene.add(tissueGroup, lumpGroup, wireGroup, vertexGroup);

    const baseVertices = new Float32Array(scanData.mesh.vertices);
    const deformTargets = [];
    const normalSurface = createIndexedMesh(
      baseVertices,
      scanData.mesh.normal_triangles,
      new THREE.MeshStandardMaterial({{
        color: 0x8dc4e6,
        transparent: true,
        opacity: 0.26,
        roughness: 0.68,
        side: THREE.DoubleSide
      }})
    );
    tissueGroup.add(normalSurface.mesh);
    deformTargets.push(normalSurface);

    const tetWire = createIndexedLines(
      baseVertices,
      scanData.mesh.tet_edges,
      new THREE.LineBasicMaterial({{ color: 0x22333d, transparent: true, opacity: 0.20 }})
    );
    wireGroup.add(tetWire.lines);
    deformTargets.push(tetWire);

    const vertices = createVertexPoints(baseVertices);
    vertexGroup.add(vertices.points);
    deformTargets.push(vertices);

    const pathLine = createScanPath();
    scene.add(pathLine);

    const probe = new THREE.Mesh(
      new THREE.SphereGeometry(scan.probe_radius, 32, 16),
      new THREE.MeshStandardMaterial({{ color: 0xf2f0e7, metalness: 0.05, roughness: 0.35 }})
    );
    scene.add(probe);

    const contactRing = new THREE.Mesh(
      new THREE.RingGeometry(scan.probe_radius * 0.75, scan.probe_radius * 1.15, 48),
      new THREE.MeshBasicMaterial({{ color: 0xffd15c, transparent: true, opacity: 0.9, side: THREE.DoubleSide }})
    );
    contactRing.rotation.x = 0;
    scene.add(contactRing);

    addLumpTetSurfaces(baseVertices, deformTargets);
    resetCamera();

    let playing = true;
    let frame = 0;
    let lastTime = performance.now();

    playButton.addEventListener("click", () => {{
      playing = !playing;
      playButton.textContent = playing ? "Pause" : "Play";
    }});
    resetButton.addEventListener("click", () => {{
      frame = 0;
      updateFrame();
      resetCamera();
    }});
    autoViewToggle.addEventListener("change", () => {{
      controls.autoRotate = autoViewToggle.checked;
    }});
    surfaceToggle.addEventListener("change", () => {{
      tissueGroup.visible = surfaceToggle.checked;
    }});
    lumpToggle.addEventListener("change", () => {{
      lumpGroup.visible = lumpToggle.checked;
    }});
    wireToggle.addEventListener("change", () => {{
      wireGroup.visible = wireToggle.checked;
    }});
    vertexToggle.addEventListener("change", () => {{
      vertexGroup.visible = vertexToggle.checked;
    }});

    window.addEventListener("resize", () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }});

    function createIndexedMesh(base, indices, material) {{
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(base), 3));
      geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(indices), 1));
      geometry.computeVertexNormals();
      return {{ mesh: new THREE.Mesh(geometry, material), geometry }};
    }}

    function createIndexedLines(base, indices, material) {{
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(base), 3));
      geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(indices), 1));
      return {{ lines: new THREE.LineSegments(geometry, material), geometry }};
    }}

    function createVertexPoints(base) {{
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(base), 3));
      const material = new THREE.PointsMaterial({{
        color: 0xf6f1df,
        size: Math.max(phantom.size_x, phantom.size_y, phantom.height) * 0.007,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.78
      }});
      return {{ points: new THREE.Points(geometry, material), geometry }};
    }}

    function createScanPath() {{
      const points = [];
      for (let r = 0; r < rows; r++) {{
        const colRange = r % 2 === 0 ? [...Array(cols).keys()] : [...Array(cols).keys()].reverse();
        for (const c of colRange) {{
          const xy = scanData.xy[r][c];
          points.push(new THREE.Vector3(xy[0], xy[1], phantom.height + 0.0015));
        }}
      }}
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      return new THREE.Line(
        geometry,
        new THREE.LineBasicMaterial({{ color: 0xffd15c, transparent: true, opacity: 0.75 }})
      );
    }}

    function addLumpTetSurfaces(base, targets) {{
      const colors = [0xe53d34, 0x1692e6, 0xf2a20b, 0x40a85a, 0x9c63d6, 0x00a29a];
      for (let i = 0; i < scanData.mesh.lump_surfaces.length; i++) {{
        const lumpSurface = scanData.mesh.lump_surfaces[i];
        const color = colors[i % colors.length];
        const surface = createIndexedMesh(
          base,
          lumpSurface.triangles,
          new THREE.MeshStandardMaterial({{ color, transparent: true, opacity: 0.72, roughness: 0.5, side: THREE.DoubleSide }})
        );
        lumpGroup.add(surface.mesh);
        targets.push(surface);
        const wire = createIndexedLines(
          base,
          triangleEdges(lumpSurface.triangles),
          new THREE.LineBasicMaterial({{ color: 0x0b0b0b, transparent: true, opacity: 0.72 }})
        );
        wireGroup.add(wire.lines);
        targets.push(wire);
      }}
    }}

    function triangleEdges(triangles) {{
      const edges = [];
      const seen = new Set();
      for (let i = 0; i < triangles.length; i += 3) {{
        const tri = [triangles[i], triangles[i + 1], triangles[i + 2]];
        for (const pair of [[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]]) {{
          const a = Math.min(pair[0], pair[1]);
          const b = Math.max(pair[0], pair[1]);
          const key = `${{a}}:${{b}}`;
          if (!seen.has(key)) {{
            seen.add(key);
            edges.push(a, b);
          }}
        }}
      }}
      return edges;
    }}

    function updateTetVertices(px, py, depth) {{
      const sigma = Math.max(scan.probe_radius * 1.45, 0.001);
      const sigma2 = sigma * sigma;
      const deform = Math.min(depth, phantom.height * 0.65);
      const depthSigma = Math.max(phantom.height * 0.42, 0.001);
      for (const target of deformTargets) {{
        const pos = target.geometry.attributes.position;
        for (let i = 0; i < pos.count; i++) {{
          const baseIndex = i * 3;
          const x = baseVertices[baseIndex];
          const y = baseVertices[baseIndex + 1];
          const z = baseVertices[baseIndex + 2];
          const dx = x - px;
          const dy = y - py;
          const lateral = Math.exp(-0.5 * (dx * dx + dy * dy) / sigma2);
          const belowSurface = Math.max(phantom.height - z, 0.0);
          const depthFalloff = Math.exp(-belowSurface / depthSigma);
          const dz = deform * lateral * depthFalloff;
          pos.setXYZ(i, x, y, Math.max(0.0, z - dz));
        }}
        pos.needsUpdate = true;
        if (target.mesh) {{
          target.geometry.computeVertexNormals();
        }}
      }}
    }}

    function updateFrame() {{
      const step = frame % steps;
      const pressIndex = Math.floor(frame / steps);
      const row = Math.floor(pressIndex / cols);
      const col = pressIndex % cols;
      const xy = scanData.xy[row][col];
      const depth = scanData.indentation[row][col][step];
      const force = scanData.force_z[row][col][step];
      const pz = phantom.height + scan.probe_radius + (scan.preload_gap || 0.0) - depth;
      probe.position.set(xy[0], xy[1], pz);
      contactRing.position.set(xy[0], xy[1], phantom.height + 0.0004);
      updateTetVertices(xy[0], xy[1], depth);
      status.textContent =
        `${{sampleId}}\\nframe ${{frame + 1}} / ${{totalFrames}}   press r${{row}} c${{col}}   step ${{step + 1}} / ${{steps}}\\n` +
        `x=${{xy[0].toFixed(4)}} m  y=${{xy[1].toFixed(4)}} m  indentation=${{depth.toFixed(5)}} m  Fz=${{force.toFixed(4)}} N`;
    }}

    function resetCamera() {{
      const radius = Math.max(phantom.size_x, phantom.size_y, phantom.height);
      camera.position.set(radius * 0.9, -radius * 1.35, radius * 0.9);
      controls.target.set(0, 0, phantom.height * 0.45);
      camera.near = Math.max(radius / 1000, 0.0001);
      camera.far = radius * 100;
      camera.updateProjectionMatrix();
      controls.update();
    }}

    function animate(now) {{
      const dt = Math.max((now - lastTime) / 1000, 0.0);
      lastTime = now;
      if (playing) {{
        const framesToAdvance = Math.max(1, Math.floor(Number(speedRange.value) * dt));
        frame = (frame + framesToAdvance) % totalFrames;
      }}
      updateFrame();
      controls.update();
      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    }}

    updateFrame();
    requestAnimationFrame(animate);
  </script>
</body>
</html>
"""


def write_phantom_gltf(
    path: Path,
    phantom: PhantomConfig,
    lumps: LumpSpec | Sequence[LumpSpec],
    material: MaterialConfig | None = None,
    *,
    normal_alpha: float = 0.16,
) -> None:
    """Write a self-contained glTF preview of the actual tetrahedral material assignment."""
    lump_list = normalize_lumps(lumps)
    material = material or MaterialConfig()
    mesh = create_structured_tet_mesh(phantom)
    _, _, _, _, tet_lump_id = material_arrays_for_lumps(mesh, material, lump_list)
    gltf: dict[str, object] = {
        "asset": {"version": "2.0", "generator": "custom_simulation.palpation_sim.exports"},
        "scene": 0,
        "scenes": [{"name": "phantom_scene", "nodes": []}],
        "materials": [],
        "meshes": [],
        "nodes": [],
        "buffers": [],
        "bufferViews": [],
        "accessors": [],
        "extras": {
            "units": "meters",
            "geometry_source": "structured_tetrahedral_mesh_material_assignment",
            "phantom": phantom.to_dict(),
            "tet_count": int(mesh.tets.shape[0]),
            "vertex_count": int(mesh.vertices.shape[0]),
            "num_lumps": len(lump_list),
            "lumps": [lump.to_dict(phantom) for lump in lump_list],
        },
    }

    normal_material = _add_material(
        gltf,
        name="normal_tissue_tet_surface",
        color=(0.62, 0.78, 1.0, float(normal_alpha)),
    )
    normal_wire_material = _add_material(
        gltf,
        name="normal_tissue_tet_wire",
        color=(0.18, 0.26, 0.30, 0.30),
    )
    normal_faces = _selected_tet_boundary_faces(mesh.tets, tet_lump_id < 0)
    normal_vertices, normal_indices = _compact_indexed_geometry(mesh.vertices, normal_faces)
    _add_mesh_node(gltf, "normal_tissue_tet_surface", normal_vertices, normal_indices, normal_material)
    normal_wire_vertices, normal_wire_indices = _surface_wire_geometry(normal_vertices, normal_indices)
    _add_mesh_node(
        gltf,
        "normal_tissue_tet_wire",
        normal_wire_vertices,
        normal_wire_indices,
        normal_wire_material,
        mode=1,
    )

    for idx, lump in enumerate(lump_list):
        material_idx = _add_material(
            gltf,
            name=f"lump_{idx:02d}_{lump.shape}_tet_surface",
            color=_lump_color(idx),
        )
        wire_material_idx = _add_material(
            gltf,
            name=f"lump_{idx:02d}_{lump.shape}_tet_wire",
            color=(0.05, 0.05, 0.05, 0.86),
        )
        faces = _selected_tet_boundary_faces(mesh.tets, tet_lump_id == idx)
        if faces.size == 0:
            continue
        vertices, indices = _compact_indexed_geometry(mesh.vertices, faces)
        label = f"lump_{idx:02d}_{lump.shape}_tet_stiffness_{lump.stiffness_multiplier:.3f}"
        _add_mesh_node(gltf, label, vertices, indices, material_idx)

        wire_vertices, wire_indices = _selected_tet_wire_geometry(mesh.vertices, mesh.tets, tet_lump_id == idx)
        if wire_indices.size > 0:
            _add_mesh_node(gltf, f"{label}_wire", wire_vertices, wire_indices, wire_material_idx, mode=1)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(gltf, f, indent=2)


def _lump_metadata(idx: int, lump: LumpSpec, phantom: PhantomConfig, material: MaterialConfig) -> dict[str, object]:
    center = tuple(float(v) for v in lump.center)
    radii = tuple(float(v) for v in lump.radii)
    extent_z = _z_extent(lump)
    data = lump.to_dict(phantom)
    data.update(
        {
            "id": idx,
            "z_interval": [center[2] - extent_z, center[2] + extent_z],
            "xy_bbox_approx": [
                center[0] - max(radii[0], radii[1]),
                center[1] - max(radii[0], radii[1]),
                center[0] + max(radii[0], radii[1]),
                center[1] + max(radii[0], radii[1]),
            ],
            "effective_material": {
                "k_mu": float(material.k_mu * lump.stiffness_multiplier),
                "k_lambda": float(material.k_lambda * lump.stiffness_multiplier),
                "k_damp": float(material.k_damp * np.sqrt(lump.stiffness_multiplier)),
            },
        }
    )
    return data


def _sample_scalar(sample: dict[str, object] | None, key: str) -> object:
    if sample is None or key not in sample:
        return None
    value = sample[key]
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        return value.tolist()
    return value


def _z_extent(lump: LumpSpec) -> float:
    if lump.shape == "capsule":
        return float(lump.radii[0] + lump.radii[2])
    return float(lump.radii[2])


def _add_material(gltf: dict[str, object], *, name: str, color: tuple[float, float, float, float]) -> int:
    materials = gltf["materials"]
    assert isinstance(materials, list)
    alpha = float(color[3])
    material: dict[str, object] = {
        "name": name,
        "pbrMetallicRoughness": {
            "baseColorFactor": [float(c) for c in color],
            "metallicFactor": 0.0,
            "roughnessFactor": 0.58,
        },
        "doubleSided": True,
    }
    if alpha < 1.0:
        material["alphaMode"] = "BLEND"
    materials.append(material)
    return len(materials) - 1


def _add_mesh_node(
    gltf: dict[str, object],
    name: str,
    vertices: np.ndarray,
    indices: np.ndarray,
    material_idx: int,
    *,
    mode: int = 4,
) -> None:
    vertices = np.asarray(vertices, dtype="<f4")
    indices = np.asarray(indices, dtype="<u4")
    pos_accessor = _add_buffer_accessor(gltf, vertices, component_type=5126, accessor_type="VEC3")
    idx_accessor = _add_buffer_accessor(gltf, indices, component_type=5125, accessor_type="SCALAR")

    meshes = gltf["meshes"]
    nodes = gltf["nodes"]
    scenes = gltf["scenes"]
    assert isinstance(meshes, list)
    assert isinstance(nodes, list)
    assert isinstance(scenes, list)

    mesh_idx = len(meshes)
    meshes.append(
        {
            "name": name,
            "primitives": [
                {
                    "attributes": {"POSITION": pos_accessor},
                    "indices": idx_accessor,
                    "material": material_idx,
                    "mode": mode,
                }
            ],
        }
    )
    node_idx = len(nodes)
    nodes.append({"name": name, "mesh": mesh_idx})
    scene = scenes[0]
    assert isinstance(scene, dict)
    scene_nodes = scene["nodes"]
    assert isinstance(scene_nodes, list)
    scene_nodes.append(node_idx)


def _add_buffer_accessor(
    gltf: dict[str, object],
    values: np.ndarray,
    *,
    component_type: int,
    accessor_type: str,
) -> int:
    buffers = gltf["buffers"]
    buffer_views = gltf["bufferViews"]
    accessors = gltf["accessors"]
    assert isinstance(buffers, list)
    assert isinstance(buffer_views, list)
    assert isinstance(accessors, list)

    values = np.asarray(values)
    if accessor_type == "SCALAR":
        values = values.reshape(-1)
    raw = np.ascontiguousarray(values).tobytes()
    buffer_idx = len(buffers)
    buffers.append(
        {
            "uri": "data:application/octet-stream;base64," + base64.b64encode(raw).decode("ascii"),
            "byteLength": len(raw),
        }
    )
    view_idx = len(buffer_views)
    buffer_views.append({"buffer": buffer_idx, "byteOffset": 0, "byteLength": len(raw)})
    accessor: dict[str, object] = {
        "bufferView": view_idx,
        "byteOffset": 0,
        "componentType": component_type,
        "count": int(values.shape[0] if accessor_type != "SCALAR" else values.size),
        "type": accessor_type,
    }
    if accessor_type == "VEC3":
        accessor["min"] = [float(v) for v in np.min(values, axis=0)]
        accessor["max"] = [float(v) for v in np.max(values, axis=0)]
    accessors.append(accessor)
    return len(accessors) - 1


def _selected_tet_boundary_faces(tets: np.ndarray, selected_mask: np.ndarray) -> np.ndarray:
    selected = np.asarray(selected_mask, dtype=bool)
    if int(np.count_nonzero(selected)) == 0:
        return np.zeros((0, 3), dtype=np.uint32)

    face_counts: dict[tuple[int, int, int], int] = {}
    face_orientation: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    for tet in np.asarray(tets, dtype=np.int64)[selected]:
        for local in TET_FACES:
            face = (int(tet[local[0]]), int(tet[local[1]]), int(tet[local[2]]))
            key = tuple(sorted(face))
            face_counts[key] = face_counts.get(key, 0) + 1
            face_orientation.setdefault(key, face)

    boundary = [face_orientation[key] for key, count in face_counts.items() if count == 1]
    if not boundary:
        return np.zeros((0, 3), dtype=np.uint32)
    return np.asarray(boundary, dtype=np.uint32)


def _all_tet_edges(tets: np.ndarray) -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    for tet in np.asarray(tets, dtype=np.int64):
        for local in TET_EDGES:
            edge = (int(tet[local[0]]), int(tet[local[1]]))
            edges.add(tuple(sorted(edge)))
    if not edges:
        return np.zeros((0, 2), dtype=np.uint32)
    return np.asarray(sorted(edges), dtype=np.uint32)


def _compact_indexed_geometry(vertices: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=np.uint32)
    if indices.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(indices.shape, dtype=np.uint32)
    unique, inverse = np.unique(indices.reshape(-1), return_inverse=True)
    compact_vertices = np.asarray(vertices, dtype=np.float32)[unique]
    compact_indices = inverse.reshape(indices.shape).astype(np.uint32)
    return compact_vertices, compact_indices


def _surface_wire_geometry(vertices: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    triangles = np.asarray(triangles, dtype=np.uint32)
    if triangles.size == 0:
        return np.asarray(vertices, dtype=np.float32), np.zeros((0, 2), dtype=np.uint32)
    edges: set[tuple[int, int]] = set()
    for tri in triangles.reshape(-1, 3):
        a, b, c = (int(v) for v in tri)
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))
    return np.asarray(vertices, dtype=np.float32), np.asarray(sorted(edges), dtype=np.uint32)


def _selected_tet_wire_geometry(
    vertices: np.ndarray,
    tets: np.ndarray,
    selected_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(selected_mask, dtype=bool)
    if int(np.count_nonzero(selected)) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 2), dtype=np.uint32)
    edges: set[tuple[int, int]] = set()
    for tet in np.asarray(tets, dtype=np.int64)[selected]:
        for local in TET_EDGES:
            edge = (int(tet[local[0]]), int(tet[local[1]]))
            edges.add(tuple(sorted(edge)))
    edge_array = np.asarray(sorted(edges), dtype=np.uint32)
    return _compact_indexed_geometry(vertices, edge_array)


def _mesh_for_lump(lump: LumpSpec) -> tuple[np.ndarray, np.ndarray]:
    if lump.shape in {"sphere", "ellipsoid"}:
        return _ellipsoid_mesh(lump.center, lump.radii, lump.yaw)
    if lump.shape == "box":
        return _box_mesh(lump.center, lump.radii, lump.yaw)
    if lump.shape == "cylinder":
        return _cylinder_mesh(lump.center, lump.radii, lump.yaw)
    if lump.shape == "capsule":
        return _capsule_mesh(lump.center, lump.radii, lump.yaw)
    raise ValueError(f"Unsupported lump shape: {lump.shape}")


def _box_mesh(
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
    yaw: float,
) -> tuple[np.ndarray, np.ndarray]:
    rx, ry, rz = (float(v) for v in radii)
    vertices = np.asarray(
        [
            (-rx, -ry, -rz),
            (rx, -ry, -rz),
            (rx, ry, -rz),
            (-rx, ry, -rz),
            (-rx, -ry, rz),
            (rx, -ry, rz),
            (rx, ry, rz),
            (-rx, ry, rz),
        ],
        dtype=np.float32,
    )
    indices = np.asarray(
        [
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 7),
            (2, 7, 6),
            (3, 0, 4),
            (3, 4, 7),
        ],
        dtype=np.uint32,
    )
    return _transform_vertices(vertices, center, yaw), indices


def _ellipsoid_mesh(
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
    yaw: float,
    *,
    slices: int = 32,
    stacks: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    rx, ry, rz = (float(v) for v in radii)
    vertices = []
    for i in range(stacks + 1):
        theta = np.pi * i / stacks
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        for j in range(slices):
            phi = 2.0 * np.pi * j / slices
            vertices.append((rx * sin_t * np.cos(phi), ry * sin_t * np.sin(phi), rz * cos_t))
    indices = _ring_indices(stacks + 1, slices)
    return _transform_vertices(np.asarray(vertices, dtype=np.float32), center, yaw), indices


def _cylinder_mesh(
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
    yaw: float,
    *,
    slices: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    rx, ry, rz = (float(v) for v in radii)
    vertices = []
    for z in (-rz, rz):
        for j in range(slices):
            phi = 2.0 * np.pi * j / slices
            vertices.append((rx * np.cos(phi), ry * np.sin(phi), z))
    bottom_center = len(vertices)
    vertices.append((0.0, 0.0, -rz))
    top_center = len(vertices)
    vertices.append((0.0, 0.0, rz))

    indices = []
    for j in range(slices):
        next_j = (j + 1) % slices
        b0, b1 = j, next_j
        t0, t1 = slices + j, slices + next_j
        indices.extend([(b0, t0, b1), (b1, t0, t1)])
        indices.append((bottom_center, b1, b0))
        indices.append((top_center, t0, t1))
    return _transform_vertices(np.asarray(vertices, dtype=np.float32), center, yaw), np.asarray(indices, dtype=np.uint32)


def _capsule_mesh(
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
    yaw: float,
    *,
    slices: int = 32,
    hemi_segments: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    radius = float(radii[0])
    half_axis = float(radii[2])
    if half_axis <= 1e-7:
        return _ellipsoid_mesh(center, (radius, radius, radius), yaw, slices=slices, stacks=hemi_segments * 2)

    rings: list[tuple[float, float]] = []
    for i in range(hemi_segments + 1):
        theta = -0.5 * np.pi + 0.5 * np.pi * i / hemi_segments
        rings.append((radius * float(np.cos(theta)), -half_axis + radius * float(np.sin(theta))))
    rings.append((radius, half_axis))
    for i in range(1, hemi_segments + 1):
        theta = 0.5 * np.pi * i / hemi_segments
        rings.append((radius * float(np.cos(theta)), half_axis + radius * float(np.sin(theta))))

    vertices = []
    for ring_radius, z in rings:
        for j in range(slices):
            phi = 2.0 * np.pi * j / slices
            vertices.append((ring_radius * np.cos(phi), ring_radius * np.sin(phi), z))
    indices = _ring_indices(len(rings), slices)
    return _transform_vertices(np.asarray(vertices, dtype=np.float32), center, yaw), indices


def _ring_indices(num_rings: int, slices: int) -> np.ndarray:
    indices = []
    for i in range(num_rings - 1):
        row = i * slices
        next_row = (i + 1) * slices
        for j in range(slices):
            next_j = (j + 1) % slices
            a = row + j
            b = row + next_j
            c = next_row + j
            d = next_row + next_j
            indices.extend([(a, c, b), (b, c, d)])
    return np.asarray(indices, dtype=np.uint32)


def _transform_vertices(vertices: np.ndarray, center: tuple[float, float, float], yaw: float) -> np.ndarray:
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    out = np.asarray(vertices, dtype=np.float32).copy()
    x = vertices[:, 0]
    y = vertices[:, 1]
    out[:, 0] = c * x - s * y + float(center[0])
    out[:, 1] = s * x + c * y + float(center[1])
    out[:, 2] = vertices[:, 2] + float(center[2])
    return out


def _lump_color(idx: int) -> tuple[float, float, float, float]:
    palette = (
        (0.90, 0.22, 0.18, 0.94),
        (0.08, 0.52, 0.92, 0.94),
        (0.98, 0.65, 0.08, 0.94),
        (0.25, 0.66, 0.28, 0.94),
        (0.62, 0.36, 0.86, 0.94),
        (0.00, 0.64, 0.62, 0.94),
    )
    return palette[idx % len(palette)]
