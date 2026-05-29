from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from palpation_sim.exports import _all_tet_edges, _selected_tet_boundary_faces


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/one_phantom_high_mesh_strain_stiffening")
    filename = sys.argv[2] if len(sys.argv) > 2 else "real_webgl_press_player.html"
    print(write_real_mesh_webgl_player(out_dir, filename=filename))


def write_real_mesh_webgl_player(
    out_dir: Path | str = Path("runs/one_phantom_high_mesh_strain_stiffening"),
    *,
    filename: str = "real_webgl_press_player.html",
) -> Path:
    out_dir = Path(out_dir)
    sample_path = out_dir / "one_phantom_high_mesh_sample.npz"
    meta_path = out_dir / "experiment_metadata.json"

    with np.load(sample_path, allow_pickle=True) as sample:
        vertices = np.asarray(sample["mesh_vertices"], dtype=np.float32)
        tets = np.asarray(sample["mesh_tets"], dtype=np.int32)
        tet_lump_id = np.asarray(sample["tet_lump_id"], dtype=np.int32)
        fz = np.asarray(sample["fz"], dtype=np.float32)
        depth = np.asarray(sample["indentation_depth"], dtype=np.float32)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    normal_faces = _selected_tet_boundary_faces(tets, tet_lump_id < 0)
    tet_edges = _all_tet_edges(tets)
    lump_surfaces = []
    for record in meta["selected_presses"]:
        idx = int(record["id"])
        faces = _selected_tet_boundary_faces(tets, tet_lump_id == idx)
        if faces.size == 0:
            continue
        lump_surfaces.append(
            {
                "id": idx,
                "shape": str(record["shape"]),
                "triangles": np.asarray(faces, dtype=np.uint32).reshape(-1).tolist(),
                "triangle_count": int(faces.shape[0]),
            }
        )

    curves = []
    for record in meta["selected_presses"]:
        row = int(record["row"])
        col = int(record["col"])
        curves.append(
            {
                "id": int(record["id"]),
                "label": (
                    f"{record['id']} {record['shape']} "
                    f"{1000.0 * float(record['center_depth_m']):.0f}mm "
                    f"{float(record['stiffness_multiplier']):.0f}x"
                ),
                "summary": record,
                "depths": np.round(depth[row, col], 8).tolist(),
                "forces": np.round(fz[row, col], 6).tolist(),
            }
        )

    data = {
        "phantom": meta["phantom"],
        "scan": meta["scan"],
        "mesh": {
            **meta["mesh"],
            "vertices": np.round(vertices.reshape(-1), 7).tolist(),
            "normal_triangles": np.asarray(normal_faces, dtype=np.uint32).reshape(-1).tolist(),
            "tet_edges": np.asarray(tet_edges, dtype=np.uint32).reshape(-1).tolist(),
            "normal_triangle_count": int(normal_faces.shape[0]),
            "tet_edge_count": int(tet_edges.shape[0]),
            "lump_surfaces": lump_surfaces,
        },
        "lumps": meta["lumps"],
        "curves": curves,
    }

    path = out_dir / filename
    path.write_text(_html(json.dumps(data, separators=(",", ":")).replace("</", "<\\/")), encoding="utf-8")
    return path


def _html(data_json: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Real Tet Mesh WebGL Press Player</title>
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
    .layout {{
      height: 100%;
      display: grid;
      grid-template-columns: minmax(0, 1.38fr) minmax(380px, 0.9fr);
    }}
    #viewport, #chart {{ width: 100%; height: 100%; display: block; }}
    .side {{
      display: grid;
      grid-template-rows: auto minmax(260px, 1fr) auto auto;
      background: #171b1f;
      border-left: 1px solid rgba(255,255,255,0.15);
    }}
    .header, .controls, .stats {{ padding: 12px 14px; }}
    .header {{ border-bottom: 1px solid rgba(255,255,255,0.12); }}
    .header h1 {{ margin: 0 0 6px; font-size: 18px; }}
    .header div {{ color: #cbc4b8; }}
    #chart {{ background: #f6f1e8; }}
    .controls {{
      display: grid;
      grid-template-columns: auto auto minmax(150px, 1fr);
      gap: 10px;
      align-items: center;
      border-top: 1px solid rgba(255,255,255,0.12);
    }}
    .controls select, .controls .wide {{ grid-column: 1 / -1; }}
    button, select {{
      border: 1px solid rgba(255,255,255,0.24);
      border-radius: 6px;
      background: #eee8dc;
      color: #151719;
      padding: 7px 10px;
      font: inherit;
    }}
    label {{ color: #d8d0c4; display: inline-flex; align-items: center; gap: 8px; }}
    input[type="range"] {{ width: 100%; }}
    .stats {{
      color: #d8d0c4;
      border-top: 1px solid rgba(255,255,255,0.12);
      white-space: pre-wrap;
    }}
    .status {{
      position: fixed;
      left: 12px;
      bottom: 12px;
      z-index: 2;
      max-width: min(820px, calc(100vw - 24px));
      padding: 7px 9px;
      border: 1px solid rgba(255,255,255,0.16);
      border-radius: 8px;
      background: rgba(16,18,20,0.74);
      color: #dedbd1;
      backdrop-filter: blur(8px);
      white-space: pre-wrap;
    }}
    @media (max-width: 940px) {{
      .layout {{ grid-template-columns: 1fr; grid-template-rows: 58vh 42vh; }}
      .side {{ border-left: 0; border-top: 1px solid rgba(255,255,255,0.15); }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <canvas id="viewport"></canvas>
    <aside class="side">
      <div class="header">
        <h1>Real tet mesh press playback</h1>
        <div id="subtitle"></div>
      </div>
      <canvas id="chart"></canvas>
      <div class="controls">
        <select id="curveSelect"></select>
        <button id="playButton" type="button">Pause</button>
        <button id="resetButton" type="button">Reset</button>
        <input id="scrub" type="range" min="0" max="0" value="0">
        <label class="wide">speed <input id="speedRange" type="range" min="1" max="90" value="26"></label>
        <label><input id="surfaceToggle" type="checkbox" checked> tissue surface</label>
        <label><input id="lumpToggle" type="checkbox" checked> lump tet surfaces</label>
        <label><input id="wireToggle" type="checkbox" checked> full tet wire</label>
        <label><input id="vertexToggle" type="checkbox"> vertices</label>
      </div>
      <div id="stats" class="stats"></div>
    </aside>
  </div>
  <div id="status" class="status">Loading real mesh...</div>
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
    const canvas = document.getElementById("viewport");
    const chart = document.getElementById("chart");
    const ctx = chart.getContext("2d");
    const subtitle = document.getElementById("subtitle");
    const status = document.getElementById("status");
    const select = document.getElementById("curveSelect");
    const playButton = document.getElementById("playButton");
    const resetButton = document.getElementById("resetButton");
    const scrub = document.getElementById("scrub");
    const speedRange = document.getElementById("speedRange");
    const surfaceToggle = document.getElementById("surfaceToggle");
    const lumpToggle = document.getElementById("lumpToggle");
    const wireToggle = document.getElementById("wireToggle");
    const vertexToggle = document.getElementById("vertexToggle");
    const stats = document.getElementById("stats");

    subtitle.textContent =
      `mesh ${{data.mesh.cells.join(" x ")}} cells, ` +
      `${{data.mesh.vertex_count.toLocaleString()}} vertices, ` +
      `${{data.mesh.tet_count.toLocaleString()}} tets, ` +
      `${{data.mesh.tet_edge_count.toLocaleString()}} tet edges`;
    data.curves.forEach((curve, idx) => {{
      const option = document.createElement("option");
      option.value = String(idx);
      option.textContent = curve.label;
      select.appendChild(option);
    }});

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101214);
    const camera = new THREE.PerspectiveCamera(46, 1, 0.001, 20);
    let renderer;
    try {{
      renderer = new THREE.WebGLRenderer({{ canvas, antialias: true, powerPreference: "high-performance" }});
    }} catch (error) {{
      console.error(error);
      status.textContent = "WebGL context failed here. Open this URL in VSCode Simple Browser or a GPU-enabled browser.";
      throw error;
    }}
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.45;

    scene.add(new THREE.HemisphereLight(0xffffff, 0x4b555d, 2.2));
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.4);
    keyLight.position.set(0.25, -0.45, 0.8);
    scene.add(keyLight);

    const tissueGroup = new THREE.Group();
    const lumpGroup = new THREE.Group();
    const wireGroup = new THREE.Group();
    const vertexGroup = new THREE.Group();
    scene.add(tissueGroup, lumpGroup, wireGroup, vertexGroup);

    const baseVertices = new Float32Array(data.mesh.vertices);
    const dynamicVertices = new Float32Array(baseVertices.length);
    dynamicVertices.set(baseVertices);
    const sharedPositions = new THREE.BufferAttribute(dynamicVertices, 3);
    sharedPositions.setUsage(THREE.DynamicDrawUsage);
    const normalMeshes = [];

    const normalSurface = createIndexedMesh(
      data.mesh.normal_triangles,
      new THREE.MeshStandardMaterial({{
        color: 0x8dc4e6,
        transparent: true,
        opacity: 0.25,
        roughness: 0.68,
        side: THREE.DoubleSide
      }})
    );
    tissueGroup.add(normalSurface.mesh);
    normalMeshes.push(normalSurface.mesh);

    const tetWire = createIndexedLines(
      data.mesh.tet_edges,
      new THREE.LineBasicMaterial({{ color: 0x24404d, transparent: true, opacity: 0.18 }})
    );
    wireGroup.add(tetWire.lines);

    const vertices = createVertexPoints();
    vertexGroup.add(vertices.points);
    vertexGroup.visible = false;

    addLumpTetSurfaces();
    const pathLine = createSelectedPath();
    scene.add(pathLine);

    const probe = new THREE.Mesh(
      new THREE.SphereGeometry(scan.probe_radius, 36, 18),
      new THREE.MeshStandardMaterial({{ color: 0xf2f0e7, metalness: 0.05, roughness: 0.35 }})
    );
    scene.add(probe);

    const contactRing = new THREE.Mesh(
      new THREE.RingGeometry(scan.probe_radius * 0.75, scan.probe_radius * 1.15, 48),
      new THREE.MeshBasicMaterial({{ color: 0xffd15c, transparent: true, opacity: 0.9, side: THREE.DoubleSide }})
    );
    scene.add(contactRing);

    let curveIndex = 0;
    let step = 0;
    let playing = true;
    let lastTime = performance.now();

    select.addEventListener("change", () => {{
      curveIndex = Number(select.value);
      step = 0;
      updateFrame();
    }});
    playButton.addEventListener("click", () => {{
      playing = !playing;
      playButton.textContent = playing ? "Pause" : "Play";
    }});
    resetButton.addEventListener("click", () => {{
      step = 0;
      resetCamera();
      updateFrame();
    }});
    scrub.addEventListener("input", () => {{
      step = Number(scrub.value);
      playing = false;
      playButton.textContent = "Play";
      updateFrame();
    }});
    surfaceToggle.addEventListener("change", () => {{ tissueGroup.visible = surfaceToggle.checked; }});
    lumpToggle.addEventListener("change", () => {{ lumpGroup.visible = lumpToggle.checked; }});
    wireToggle.addEventListener("change", () => {{ wireGroup.visible = wireToggle.checked; }});
    vertexToggle.addEventListener("change", () => {{ vertexGroup.visible = vertexToggle.checked; }});
    window.addEventListener("resize", resize);

    function createIndexedMesh(indices, material) {{
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", sharedPositions);
      geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(indices), 1));
      geometry.computeVertexNormals();
      return {{ mesh: new THREE.Mesh(geometry, material), geometry }};
    }}

    function createIndexedLines(indices, material) {{
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", sharedPositions);
      geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(indices), 1));
      return {{ lines: new THREE.LineSegments(geometry, material), geometry }};
    }}

    function createVertexPoints() {{
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", sharedPositions);
      const material = new THREE.PointsMaterial({{
        color: 0xf6f1df,
        size: Math.max(phantom.size_x, phantom.size_y, phantom.height) * 0.0055,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.58
      }});
      return {{ points: new THREE.Points(geometry, material), geometry }};
    }}

    function addLumpTetSurfaces() {{
      const colors = [0xe53d34, 0x1692e6, 0xf2a20b, 0x40a85a, 0x9c63d6, 0x00a29a];
      for (let i = 0; i < data.mesh.lump_surfaces.length; i++) {{
        const lumpSurface = data.mesh.lump_surfaces[i];
        const color = colors[i % colors.length];
        const surface = createIndexedMesh(
          lumpSurface.triangles,
          new THREE.MeshStandardMaterial({{ color, transparent: true, opacity: 0.72, roughness: 0.5, side: THREE.DoubleSide }})
        );
        lumpGroup.add(surface.mesh);
        normalMeshes.push(surface.mesh);
        const wire = createIndexedLines(
          triangleEdges(lumpSurface.triangles),
          new THREE.LineBasicMaterial({{ color: 0x0b0b0b, transparent: true, opacity: 0.72 }})
        );
        wireGroup.add(wire.lines);
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

    function createSelectedPath() {{
      const points = data.curves.map((curve) => {{
        const s = curve.summary;
        return new THREE.Vector3(Number(s.x_m), Number(s.y_m), phantom.height + 0.0015);
      }});
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      return new THREE.Line(
        geometry,
        new THREE.LineBasicMaterial({{ color: 0xffd15c, transparent: true, opacity: 0.8 }})
      );
    }}

    function activeCurve() {{ return data.curves[curveIndex]; }}

    function activeXY() {{
      const s = activeCurve().summary;
      return [Number(s.x_m), Number(s.y_m)];
    }}

    function updateTetVertices(px, py, depth) {{
      const sigma = Math.max(scan.probe_radius * 1.45, 0.001);
      const sigma2 = sigma * sigma;
      const deform = Math.min(depth, phantom.height * 0.65);
      const depthSigma = Math.max(phantom.height * 0.42, 0.001);
      for (let i = 0; i < baseVertices.length; i += 3) {{
        const x = baseVertices[i];
        const y = baseVertices[i + 1];
        const z = baseVertices[i + 2];
        const dx = x - px;
        const dy = y - py;
        const lateral = Math.exp(-0.5 * (dx * dx + dy * dy) / sigma2);
        const belowSurface = Math.max(phantom.height - z, 0.0);
        const depthFalloff = Math.exp(-belowSurface / depthSigma);
        const dz = deform * lateral * depthFalloff;
        dynamicVertices[i] = x;
        dynamicVertices[i + 1] = y;
        dynamicVertices[i + 2] = Math.max(0.0, z - dz);
      }}
      sharedPositions.needsUpdate = true;
      for (const mesh of normalMeshes) {{
        mesh.geometry.computeVertexNormals();
      }}
    }}

    function updateFrame() {{
      const curve = activeCurve();
      scrub.max = String(curve.depths.length - 1);
      step = Math.min(step, curve.depths.length - 1);
      scrub.value = String(step);
      const depth = curve.depths[step];
      const force = curve.forces[step];
      const [px, py] = activeXY();
      const pz = phantom.height + scan.probe_radius + (scan.preload_gap || 0.0) - depth;
      probe.position.set(px, py, pz);
      contactRing.position.set(px, py, phantom.height + 0.0004);
      updateTetVertices(px, py, depth);
      drawChart();
      const s = curve.summary;
      stats.textContent =
        `${{curve.label}}\\n` +
        `step ${{step + 1}} / ${{curve.depths.length}}\\n` +
        `indentation: ${{(depth * 1000).toFixed(2)}} mm\\n` +
        `Fz: ${{force.toFixed(2)}} N\\n` +
        `peak Fz: ${{Number(s.peak_force_n).toFixed(2)}} N\\n` +
        `late/early slope: ${{Number(s.late_early_slope_ratio).toFixed(2)}}\\n` +
        `mesh source: actual NPZ vertices/tets/tet_lump_id`;
      status.textContent =
        `Real WebGL tet mesh: ${{data.mesh.vertex_count.toLocaleString()}} vertices, ` +
        `${{data.mesh.tet_count.toLocaleString()}} tets, ` +
        `${{data.mesh.tet_edge_count.toLocaleString()}} tet edges`;
    }}

    function drawChart() {{
      const curve = activeCurve();
      const depths = curve.depths;
      const forces = curve.forces;
      const w = chart.width;
      const h = chart.height;
      if (!w || !h) return;
      const maxZ = Math.max(...depths);
      const maxF = Math.max(...forces);
      const padL = 58, padR = 18, padT = 24, padB = 46;
      const x0 = padL, y0 = h - padB, x1 = w - padR, y1 = padT;
      const px = (z) => x0 + (z / maxZ) * (x1 - x0);
      const py = (f) => y0 - (f / maxF) * (y0 - y1);
      ctx.fillStyle = "#f6f1e8";
      ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = "#c7c0b2";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x0, y1);
      ctx.lineTo(x0, y0);
      ctx.lineTo(x1, y0);
      ctx.stroke();
      ctx.font = `${{Math.max(11, Math.round(w / 58))}}px system-ui`;
      for (let i = 0; i <= 4; i++) {{
        const yy = y0 - (i / 4) * (y0 - y1);
        ctx.strokeStyle = "#ddd6c8";
        ctx.beginPath();
        ctx.moveTo(x0, yy);
        ctx.lineTo(x1, yy);
        ctx.stroke();
        ctx.fillStyle = "#55595d";
        ctx.fillText(String(Math.round((i / 4) * maxF)), 8, yy + 4);
      }}
      ctx.fillStyle = "#303438";
      ctx.fillText("Fz [N]", x0, y1 - 8);
      ctx.fillText("indentation [mm]", Math.max(x0, x1 - 145), h - 14);
      ctx.strokeStyle = "#9aa3a8";
      ctx.lineWidth = 2;
      ctx.beginPath();
      depths.forEach((z, i) => {{
        const x = px(z);
        const y = py(forces[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }});
      ctx.stroke();
      ctx.strokeStyle = "#176d8f";
      ctx.lineWidth = 4;
      ctx.beginPath();
      for (let i = 0; i <= step; i++) {{
        const x = px(depths[i]);
        const y = py(forces[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }}
      ctx.stroke();
      ctx.fillStyle = "#d34a24";
      ctx.beginPath();
      ctx.arc(px(depths[step]), py(forces[step]), 6, 0, Math.PI * 2);
      ctx.fill();
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

    function resize() {{
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      renderer.setSize(Math.max(1, rect.width), Math.max(1, rect.height), false);
      camera.aspect = rect.width / Math.max(rect.height, 1);
      camera.updateProjectionMatrix();
      const chartRect = chart.getBoundingClientRect();
      chart.width = Math.max(1, Math.floor(chartRect.width * dpr));
      chart.height = Math.max(1, Math.floor(chartRect.height * dpr));
      drawChart();
    }}

    function animate(now) {{
      const dt = Math.max((now - lastTime) / 1000, 0.0);
      lastTime = now;
      if (playing) {{
        const framesToAdvance = Math.max(1, Math.floor(Number(speedRange.value) * dt));
        step = (step + framesToAdvance) % activeCurve().depths.length;
      }}
      updateFrame();
      controls.update();
      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    }}

    resize();
    resetCamera();
    updateFrame();
    requestAnimationFrame(animate);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
