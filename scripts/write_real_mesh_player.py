from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from palpation_sim.exports import _compact_indexed_geometry, _selected_tet_boundary_faces, _surface_wire_geometry


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/one_phantom_high_mesh_strain_stiffening")
    sample_path = out_dir / "one_phantom_high_mesh_sample.npz"
    meta_path = out_dir / "experiment_metadata.json"
    sample = np.load(sample_path, allow_pickle=True)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    vertices = np.asarray(sample["mesh_vertices"], dtype=np.float32)
    tets = np.asarray(sample["mesh_tets"], dtype=np.int32)
    tet_lump_id = np.asarray(sample["tet_lump_id"], dtype=np.int32)
    fz = np.asarray(sample["fz"], dtype=np.float32)
    depth = np.asarray(sample["indentation_depth"], dtype=np.float32)

    normal_faces = _selected_tet_boundary_faces(tets, tet_lump_id < 0)
    normal_vertices, normal_triangles = _compact_indexed_geometry(vertices, normal_faces)
    _, normal_edges = _surface_wire_geometry(normal_vertices, normal_triangles)

    lump_meshes = []
    for record in meta["selected_presses"]:
        idx = int(record["id"])
        faces = _selected_tet_boundary_faces(tets, tet_lump_id == idx)
        lump_vertices, lump_triangles = _compact_indexed_geometry(vertices, faces)
        _, lump_edges = _surface_wire_geometry(lump_vertices, lump_triangles)
        lump_meshes.append(
            {
                "id": idx,
                "shape": record["shape"],
                "vertices": _round_vertices(lump_vertices),
                "edges": np.asarray(lump_edges, dtype=np.uint32).reshape(-1).tolist(),
                "edge_count": int(lump_edges.shape[0]),
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
                "depths": depth[row, col].tolist(),
                "forces": fz[row, col].tolist(),
            }
        )

    data = {
        "phantom": meta["phantom"],
        "mesh": {
            **meta["mesh"],
            "normal_surface_edge_count": int(normal_edges.shape[0]),
            "normal_surface_vertex_count": int(normal_vertices.shape[0]),
        },
        "normal_mesh": {
            "vertices": _round_vertices(normal_vertices),
            "edges": np.asarray(normal_edges, dtype=np.uint32).reshape(-1).tolist(),
        },
        "lumps": meta["lumps"],
        "lump_meshes": lump_meshes,
        "curves": curves,
    }

    path = out_dir / "real_mesh_player.html"
    path.write_text(_html(json.dumps(data, separators=(",", ":")).replace("</", "<\\/")), encoding="utf-8")
    print(path)


def _round_vertices(vertices: np.ndarray) -> list[float]:
    return np.round(np.asarray(vertices, dtype=np.float32).reshape(-1), 7).tolist()


def _html(data_json: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Real Tet Mesh Press Player</title>
  <style>
    html, body {{
      width: 100%; height: 100%; margin: 0; overflow: hidden;
      background: #101316; color: #f2eee6;
      font: 14px/1.4 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    .layout {{ height: 100%; display: grid; grid-template-columns: minmax(0, 1.38fr) minmax(380px, .9fr); }}
    canvas {{ display: block; width: 100%; height: 100%; }}
    .side {{ display: grid; grid-template-rows: auto minmax(260px, 1fr) auto auto; background: #171b1f; border-left: 1px solid rgba(255,255,255,.15); }}
    .header, .controls, .stats {{ padding: 12px 14px; }}
    .header {{ border-bottom: 1px solid rgba(255,255,255,.12); }}
    .header h1 {{ margin: 0 0 6px; font-size: 18px; }}
    .header div {{ color: #cbc4b8; }}
    #chart {{ background: #f6f1e8; }}
    .controls {{ display: grid; grid-template-columns: auto auto minmax(160px,1fr); gap: 10px; align-items: center; border-top: 1px solid rgba(255,255,255,.12); }}
    .controls select, .controls .wide {{ grid-column: 1 / -1; }}
    button, select {{
      border: 1px solid rgba(255,255,255,.24); border-radius: 6px;
      background: #eee8dc; color: #151719; padding: 7px 10px; font: inherit;
    }}
    label {{ color: #d8d0c4; display: inline-flex; gap: 8px; align-items: center; }}
    input[type=range] {{ width: 100%; }}
    .stats {{ color: #d8d0c4; border-top: 1px solid rgba(255,255,255,.12); white-space: pre-wrap; }}
    @media (max-width: 940px) {{
      .layout {{ grid-template-columns: 1fr; grid-template-rows: 58vh 42vh; }}
      .side {{ border-left: 0; border-top: 1px solid rgba(255,255,255,.15); }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <canvas id="scene"></canvas>
    <aside class="side">
      <div class="header">
        <h1>Real high-resolution tet mesh playback</h1>
        <div id="subtitle"></div>
      </div>
      <canvas id="chart"></canvas>
      <div class="controls">
        <select id="curveSelect"></select>
        <button id="play" type="button">Pause</button>
        <button id="reset" type="button">Reset</button>
        <input id="scrub" type="range" min="0" max="0" value="0">
        <label class="wide"><input id="autoRotate" type="checkbox" checked> auto rotate true mesh</label>
        <label class="wide">mesh density <input id="density" type="range" min="1" max="8" value="1"></label>
      </div>
      <div id="stats" class="stats"></div>
    </aside>
  </div>
  <script>
    const data = {data_json};
    const phantom = data.phantom;
    const normalVertices = new Float32Array(data.normal_mesh.vertices);
    const normalEdges = new Uint32Array(data.normal_mesh.edges);
    const lumpMeshes = data.lump_meshes.map(m => ({{
      id: m.id,
      shape: m.shape,
      vertices: new Float32Array(m.vertices),
      edges: new Uint32Array(m.edges),
      edgeCount: m.edge_count
    }}));
    const scene = document.getElementById("scene");
    const chart = document.getElementById("chart");
    const sctx = scene.getContext("2d");
    const cctx = chart.getContext("2d");
    const subtitle = document.getElementById("subtitle");
    const select = document.getElementById("curveSelect");
    const play = document.getElementById("play");
    const reset = document.getElementById("reset");
    const scrub = document.getElementById("scrub");
    const stats = document.getElementById("stats");
    const autoRotate = document.getElementById("autoRotate");
    const density = document.getElementById("density");
    const colors = ["#e0523f", "#1692e6", "#f2a20b", "#45b36b", "#a56bd6"];
    let curveIndex = 0;
    let step = 0;
    let playing = true;
    let yaw = -0.72;
    let pitch = 0.72;

    subtitle.textContent =
      `real mesh: ${{data.mesh.cells.join(" x ")}} cells, ` +
      `${{data.mesh.vertex_count.toLocaleString()}} vertices, ${{data.mesh.tet_count.toLocaleString()}} tets, ` +
      `${{data.mesh.normal_surface_edge_count.toLocaleString()}} tissue surface edges`;
    data.curves.forEach((curve, i) => {{
      const opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = curve.label;
      select.appendChild(opt);
    }});

    select.addEventListener("change", () => {{ curveIndex = Number(select.value); step = 0; update(); }});
    play.addEventListener("click", () => {{ playing = !playing; play.textContent = playing ? "Pause" : "Play"; }});
    reset.addEventListener("click", () => {{ step = 0; yaw = -0.72; pitch = 0.72; update(); }});
    scrub.addEventListener("input", () => {{ step = Number(scrub.value); playing = false; play.textContent = "Play"; update(); }});
    density.addEventListener("input", update);
    window.addEventListener("resize", resize);

    let dragging = false;
    let lastPointer = null;
    scene.addEventListener("pointerdown", (e) => {{ dragging = true; lastPointer = [e.clientX, e.clientY]; scene.setPointerCapture(e.pointerId); }});
    scene.addEventListener("pointerup", () => {{ dragging = false; lastPointer = null; }});
    scene.addEventListener("pointermove", (e) => {{
      if (!dragging || !lastPointer) return;
      const dx = e.clientX - lastPointer[0];
      const dy = e.clientY - lastPointer[1];
      yaw += dx * 0.008;
      pitch = Math.max(0.18, Math.min(1.35, pitch + dy * 0.006));
      lastPointer = [e.clientX, e.clientY];
      update();
    }});

    function activeCurve() {{ return data.curves[curveIndex]; }}
    function activeXY() {{
      const s = activeCurve().summary;
      return [Number(s.x_m), Number(s.y_m)];
    }}

    function resize() {{
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      for (const canvas of [scene, chart]) {{
        const r = canvas.getBoundingClientRect();
        canvas.width = Math.max(1, Math.floor(r.width * dpr));
        canvas.height = Math.max(1, Math.floor(r.height * dpr));
      }}
      update();
    }}

    function transformPoint(vertices, index, px, py, depth) {{
      const k = index * 3;
      const x = vertices[k];
      const y = vertices[k + 1];
      const z0 = vertices[k + 2];
      const dz = visualDepression(x, y, z0, px, py, depth);
      const z = Math.max(0, z0 - dz);
      const cy = Math.cos(yaw), sy = Math.sin(yaw);
      const cp = Math.cos(pitch), sp = Math.sin(pitch);
      const xr = cy * x - sy * y;
      const yr = sy * x + cy * y;
      const zr = z;
      const yp = cp * yr - sp * zr;
      const zp = sp * yr + cp * zr;
      const scale = Math.min(scene.width / (phantom.size_x * 2.5), scene.height / (phantom.size_y * 2.2));
      return [scene.width * 0.50 + xr * scale, scene.height * 0.58 + yp * scale - zp * scale * 1.38];
    }}

    function visualDepression(x, y, z, px, py, depth) {{
      const sigma = Math.max(data.scan?.probe_radius || 0.012, 0.001) * 1.45;
      const r2 = (x - px) * (x - px) + (y - py) * (y - py);
      const belowSurface = Math.max(phantom.height - z, 0.0);
      const depthSigma = Math.max(phantom.height * 0.42, 0.001);
      return depth * Math.exp(-0.5 * r2 / (sigma * sigma)) * Math.exp(-belowSurface / depthSigma);
    }}

    function drawMeshEdges(vertices, edges, px, py, depth, color, alpha, stride) {{
      sctx.strokeStyle = color;
      sctx.globalAlpha = alpha;
      sctx.lineWidth = 0.75;
      sctx.beginPath();
      let drawn = 0;
      const edgeCount = edges.length / 2;
      for (let e = 0; e < edgeCount; e += stride) {{
        const a = edges[e * 2];
        const b = edges[e * 2 + 1];
        const p0 = transformPoint(vertices, a, px, py, depth);
        const p1 = transformPoint(vertices, b, px, py, depth);
        sctx.moveTo(p0[0], p0[1]);
        sctx.lineTo(p1[0], p1[1]);
        drawn += 1;
        if (drawn % 6000 === 0) {{
          sctx.stroke();
          sctx.beginPath();
        }}
      }}
      sctx.stroke();
      sctx.globalAlpha = 1;
      return drawn;
    }}

    function drawProbe(px, py, depth, force) {{
      const z = phantom.height + (data.scan?.probe_radius || 0.012) + (data.scan?.preload_gap || 0) - depth;
      const p = transformPoint(new Float32Array([px, py, z]), 0, px, py, 0);
      const r = Math.max(7, Math.min(scene.width, scene.height) * 0.018);
      sctx.fillStyle = "#f4eee2";
      sctx.strokeStyle = "#5c564b";
      sctx.lineWidth = 1.2;
      sctx.beginPath();
      sctx.arc(p[0], p[1], r, 0, Math.PI * 2);
      sctx.fill();
      sctx.stroke();
      sctx.fillStyle = "#f2eee6";
      sctx.font = `${{Math.max(12, scene.width / 86)}}px system-ui`;
      sctx.fillText(`Fz ${{force.toFixed(1)}} N`, p[0] + r + 8, p[1] - r);
    }}

    function drawScene() {{
      const curve = activeCurve();
      const depth = curve.depths[step];
      const force = curve.forces[step];
      const [px, py] = activeXY();
      const w = scene.width, h = scene.height;
      sctx.fillStyle = "#101316";
      sctx.fillRect(0, 0, w, h);
      const stride = Number(density.value);
      const normalDrawn = drawMeshEdges(normalVertices, normalEdges, px, py, depth, "#81c6e8", 0.30, stride);
      let lumpDrawn = 0;
      lumpMeshes.forEach((mesh, i) => {{
        lumpDrawn += drawMeshEdges(mesh.vertices, mesh.edges, px, py, depth, colors[i % colors.length], 0.95, Math.max(1, Math.floor(stride / 2)));
      }});
      drawProbe(px, py, depth, force);
      sctx.fillStyle = "rgba(0,0,0,.42)";
      sctx.fillRect(12, 12, 430, 82);
      sctx.fillStyle = "#f2eee6";
      sctx.font = `${{Math.max(12, scene.width / 98)}}px system-ui`;
      sctx.fillText("Actual high-resolution tet mesh boundary wireframe", 24, 38);
      sctx.fillText(`drawn edges: tissue ${{normalDrawn.toLocaleString()}}, lump boundaries ${{lumpDrawn.toLocaleString()}}`, 24, 61);
      sctx.fillText("drag to rotate | mesh density slider skips edges for speed", 24, 82);
    }}

    function drawChart() {{
      const curve = activeCurve();
      const depths = curve.depths, forces = curve.forces;
      const w = chart.width, h = chart.height;
      cctx.fillStyle = "#f6f1e8"; cctx.fillRect(0, 0, w, h);
      const padL = 58, padR = 18, padT = 24, padB = 46;
      const x0 = padL, y0 = h - padB, x1 = w - padR, y1 = padT;
      const maxZ = Math.max(...depths), maxF = Math.max(...forces);
      function px(z) {{ return x0 + z / maxZ * (x1 - x0); }}
      function py(f) {{ return y0 - f / maxF * (y0 - y1); }}
      cctx.strokeStyle = "#c7c0b2"; cctx.lineWidth = 1;
      cctx.beginPath(); cctx.moveTo(x0, y1); cctx.lineTo(x0, y0); cctx.lineTo(x1, y0); cctx.stroke();
      for (let i = 0; i <= 4; i++) {{
        const yy = y0 - i / 4 * (y0 - y1);
        cctx.strokeStyle = "#ddd6c8"; cctx.beginPath(); cctx.moveTo(x0, yy); cctx.lineTo(x1, yy); cctx.stroke();
        cctx.fillStyle = "#55595d"; cctx.font = `${{Math.max(11, w / 58)}}px system-ui`;
        cctx.fillText(String(Math.round(i / 4 * maxF)), 8, yy + 4);
      }}
      cctx.fillStyle = "#303438"; cctx.fillText("Fz [N]", x0, y1 - 8); cctx.fillText("indentation [mm]", Math.max(x0, x1 - 145), h - 14);
      cctx.strokeStyle = "#a1a8ad"; cctx.lineWidth = 2; cctx.beginPath();
      depths.forEach((z, i) => {{ const x = px(z), y = py(forces[i]); if (i === 0) cctx.moveTo(x, y); else cctx.lineTo(x, y); }});
      cctx.stroke();
      cctx.strokeStyle = "#176d8f"; cctx.lineWidth = 4; cctx.beginPath();
      for (let i = 0; i <= step; i++) {{ const x = px(depths[i]), y = py(forces[i]); if (i === 0) cctx.moveTo(x, y); else cctx.lineTo(x, y); }}
      cctx.stroke();
      cctx.fillStyle = "#d34a24"; cctx.beginPath(); cctx.arc(px(depths[step]), py(forces[step]), 6, 0, Math.PI * 2); cctx.fill();
    }}

    function update() {{
      const curve = activeCurve();
      scrub.max = String(curve.depths.length - 1);
      scrub.value = String(step);
      const s = curve.summary;
      stats.textContent =
        `${{curve.label}}\\n` +
        `step ${{step + 1}} / ${{curve.depths.length}}\\n` +
        `indentation: ${{(curve.depths[step] * 1000).toFixed(2)}} mm\\n` +
        `Fz: ${{curve.forces[step].toFixed(2)}} N\\n` +
        `peak Fz: ${{Number(s.peak_force_n).toFixed(2)}} N\\n` +
        `late/early slope: ${{Number(s.late_early_slope_ratio).toFixed(2)}}\\n` +
        `mesh source: actual vertices/tets/tet_lump_id from one_phantom_high_mesh_sample.npz`;
      drawScene();
      drawChart();
    }}

    let last = performance.now();
    function tick(now) {{
      const dt = now - last;
      last = now;
      if (autoRotate.checked && !dragging) yaw += 0.0025 * Math.min(dt, 40);
      if (playing && dt < 120) {{
        step = (step + 1) % activeCurve().depths.length;
      }}
      update();
      requestAnimationFrame(tick);
    }}

    resize();
    requestAnimationFrame(tick);
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
