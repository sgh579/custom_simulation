from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/one_phantom_high_mesh_strain_stiffening")
    meta = json.loads((out_dir / "experiment_metadata.json").read_text(encoding="utf-8"))
    sample = np.load(out_dir / "one_phantom_high_mesh_sample.npz", allow_pickle=True)
    fz = np.asarray(sample["fz"], dtype=np.float32)
    depth = np.asarray(sample["indentation_depth"], dtype=np.float32)
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
        "mesh": meta["mesh"],
        "scan": meta["scan"],
        "lumps": meta["lumps"],
        "curves": curves,
    }
    html = _html(json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    (out_dir / "press_player_fallback.html").write_text(html, encoding="utf-8")
    print(out_dir / "press_player_fallback.html")


def _html(data_json: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Press Player Fallback</title>
  <style>
    html, body {{
      width: 100%; height: 100%; margin: 0; overflow: hidden;
      background: #111417; color: #f2eee6;
      font: 14px/1.4 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    .layout {{ height: 100%; display: grid; grid-template-columns: minmax(0, 1.3fr) minmax(360px, .9fr); }}
    canvas {{ display: block; width: 100%; height: 100%; }}
    .side {{ display: grid; grid-template-rows: auto minmax(260px, 1fr) auto auto; background: #171b1f; border-left: 1px solid rgba(255,255,255,.15); }}
    .header, .controls, .stats {{ padding: 12px 14px; }}
    .header {{ border-bottom: 1px solid rgba(255,255,255,.12); }}
    .header h1 {{ margin: 0 0 6px; font-size: 18px; }}
    .header div {{ color: #cbc4b8; }}
    #chart {{ background: #f6f1e8; }}
    .controls {{ display: grid; grid-template-columns: auto auto minmax(160px,1fr); gap: 10px; align-items: center; border-top: 1px solid rgba(255,255,255,.12); }}
    .controls select {{ grid-column: 1 / -1; }}
    button, select {{
      border: 1px solid rgba(255,255,255,.24); border-radius: 6px;
      background: #eee8dc; color: #151719; padding: 7px 10px; font: inherit;
    }}
    input[type=range] {{ width: 100%; }}
    .stats {{ color: #d8d0c4; border-top: 1px solid rgba(255,255,255,.12); white-space: pre-wrap; }}
    @media (max-width: 900px) {{
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
        <h1>One phantom: fallback press playback</h1>
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
  <script>
    const data = {data_json};
    const phantom = data.phantom;
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
    let curveIndex = 0;
    let step = 0;
    let playing = true;
    const colors = ["#e0523f", "#1692e6", "#f2a20b", "#45b36b", "#a56bd6"];

    subtitle.textContent = `Canvas fallback: mesh ${{data.mesh.cells.join(" x ")}}, ${{data.lumps.length}} lumps`;
    data.curves.forEach((curve, i) => {{
      const opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = curve.label;
      select.appendChild(opt);
    }});

    select.addEventListener("change", () => {{ curveIndex = Number(select.value); step = 0; update(); }});
    play.addEventListener("click", () => {{ playing = !playing; play.textContent = playing ? "Pause" : "Play"; }});
    reset.addEventListener("click", () => {{ step = 0; update(); }});
    scrub.addEventListener("input", () => {{ step = Number(scrub.value); playing = false; play.textContent = "Play"; update(); }});
    window.addEventListener("resize", resize);

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

    function project(x, y, z) {{
      const w = scene.width, h = scene.height;
      const scale = Math.min(w / (phantom.size_x * 2.8), h / (phantom.size_y * 2.0 + phantom.height * 5.2));
      const cx = w * 0.48, cy = h * 0.68;
      return [
        cx + (x - y) * scale * 1.08,
        cy + (x + y) * scale * 0.42 - z * scale * 2.25
      ];
    }}

    function drawScene() {{
      const curve = activeCurve();
      const depth = curve.depths[step];
      const force = curve.forces[step];
      const [px, py] = activeXY();
      const w = scene.width, h = scene.height;
      sctx.clearRect(0, 0, w, h);
      const grad = sctx.createLinearGradient(0, 0, 0, h);
      grad.addColorStop(0, "#182026");
      grad.addColorStop(1, "#0f1215");
      sctx.fillStyle = grad;
      sctx.fillRect(0, 0, w, h);
      drawPhantomBox();
      drawSurface(px, py, depth);
      data.lumps.forEach((lump, i) => drawLump(lump, i));
      drawProbe(px, py, depth, force);
      drawLegend();
    }}

    function drawPhantomBox() {{
      const sx = phantom.size_x / 2, sy = phantom.size_y / 2, hz = phantom.height;
      const pts = [
        [-sx,-sy,0], [sx,-sy,0], [sx,sy,0], [-sx,sy,0],
        [-sx,-sy,hz], [sx,-sy,hz], [sx,sy,hz], [-sx,sy,hz],
      ].map(p => project(...p));
      const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
      sctx.strokeStyle = "rgba(188,231,242,.45)";
      sctx.lineWidth = 1.2;
      edges.forEach(([a,b]) => line(pts[a], pts[b]));
    }}

    function drawSurface(px, py, depth) {{
      const n = 22, sx = phantom.size_x / 2, sy = phantom.size_y / 2;
      sctx.strokeStyle = "rgba(139,203,232,.38)";
      sctx.lineWidth = 1;
      for (let i = 0; i <= n; i++) {{
        const x = -sx + 2 * sx * i / n;
        drawSurfaceLine(x, -sy, x, sy, px, py, depth);
        const y = -sy + 2 * sy * i / n;
        drawSurfaceLine(-sx, y, sx, y, px, py, depth);
      }}
    }}

    function drawSurfaceLine(x0, y0, x1, y1, px, py, depth) {{
      const steps = 36;
      sctx.beginPath();
      for (let i = 0; i <= steps; i++) {{
        const t = i / steps;
        const x = x0 + (x1 - x0) * t;
        const y = y0 + (y1 - y0) * t;
        const dz = depression(x, y, px, py, depth);
        const p = project(x, y, phantom.height - dz);
        if (i === 0) sctx.moveTo(p[0], p[1]); else sctx.lineTo(p[0], p[1]);
      }}
      sctx.stroke();
    }}

    function depression(x, y, px, py, depth) {{
      const sigma = Math.max(data.scan.probe_radius * 1.35, 0.001);
      const r2 = (x - px) * (x - px) + (y - py) * (y - py);
      return depth * Math.exp(-0.5 * r2 / (sigma * sigma));
    }}

    function drawLump(lump, i) {{
      const p = project(lump.center[0], lump.center[1], lump.center[2]);
      const edge = project(lump.center[0] + lump.radii[0], lump.center[1], lump.center[2]);
      const ryEdge = project(lump.center[0], lump.center[1] + lump.radii[1], lump.center[2]);
      const rx = Math.max(7, Math.abs(edge[0] - p[0]) * 1.8);
      const ry = Math.max(5, Math.abs(ryEdge[1] - p[1]) * 1.8);
      sctx.save();
      sctx.translate(p[0], p[1]);
      sctx.rotate(-0.55 + (lump.yaw || 0));
      sctx.fillStyle = colors[i % colors.length] + "cc";
      sctx.strokeStyle = "#111";
      sctx.lineWidth = 1;
      if (lump.shape === "box") {{
        sctx.fillRect(-rx, -ry, 2*rx, 2*ry);
        sctx.strokeRect(-rx, -ry, 2*rx, 2*ry);
      }} else if (lump.shape === "capsule") {{
        roundRect(-rx, -ry * .75, 2*rx, 1.5*ry, ry * .75);
        sctx.fill(); sctx.stroke();
      }} else {{
        sctx.beginPath();
        sctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
        sctx.fill(); sctx.stroke();
      }}
      sctx.restore();
      sctx.fillStyle = "#fff";
      sctx.font = `${{Math.max(11, scene.width / 90)}}px system-ui`;
      sctx.fillText(String(i), p[0] + rx + 4, p[1]);
    }}

    function drawProbe(px, py, depth, force) {{
      const z = phantom.height + data.scan.probe_radius + (data.scan.preload_gap || 0) - depth;
      const p = project(px, py, z);
      const rP = project(px + data.scan.probe_radius, py, z);
      const r = Math.max(8, Math.abs(rP[0] - p[0]) * 1.25);
      sctx.fillStyle = "#f4eee2";
      sctx.strokeStyle = "#554";
      sctx.lineWidth = 1.2;
      sctx.beginPath();
      sctx.arc(p[0], p[1], r, 0, Math.PI * 2);
      sctx.fill(); sctx.stroke();
      const cp = project(px, py, phantom.height - 0.0006);
      sctx.strokeStyle = "#ffc857";
      sctx.lineWidth = 2;
      sctx.beginPath();
      sctx.ellipse(cp[0], cp[1], r * 0.85, r * 0.34, -0.55, 0, Math.PI * 2);
      sctx.stroke();
      sctx.fillStyle = "#f4eee2";
      sctx.font = `${{Math.max(12, scene.width / 80)}}px system-ui`;
      sctx.fillText(`Fz ${{force.toFixed(1)}} N`, p[0] + r + 8, p[1] - r);
    }}

    function drawLegend() {{
      sctx.fillStyle = "rgba(0,0,0,.35)";
      sctx.fillRect(12, 12, 330, 60);
      sctx.fillStyle = "#f2eee6";
      sctx.font = `${{Math.max(12, scene.width / 95)}}px system-ui`;
      sctx.fillText("No-WebGL fallback: pseudo-3D phantom + animated press", 24, 36);
      sctx.fillText(`mesh ${{data.mesh.cells.join(" x ")}} cells`, 24, 58);
    }}

    function line(a, b) {{
      sctx.beginPath(); sctx.moveTo(a[0], a[1]); sctx.lineTo(b[0], b[1]); sctx.stroke();
    }}
    function roundRect(x, y, w, h, r) {{
      sctx.beginPath();
      sctx.moveTo(x + r, y); sctx.arcTo(x + w, y, x + w, y + h, r);
      sctx.arcTo(x + w, y + h, x, y + h, r); sctx.arcTo(x, y + h, x, y, r);
      sctx.arcTo(x, y, x + w, y, r); sctx.closePath();
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
        `${{curve.label}}\\nstep ${{step + 1}} / ${{curve.depths.length}}\\n` +
        `indentation: ${{(curve.depths[step] * 1000).toFixed(2)}} mm\\n` +
        `Fz: ${{curve.forces[step].toFixed(2)}} N\\n` +
        `peak Fz: ${{Number(s.peak_force_n).toFixed(2)}} N\\n` +
        `late/early slope: ${{Number(s.late_early_slope_ratio).toFixed(2)}}`;
      drawScene();
      drawChart();
    }}

    let last = performance.now();
    function tick(now) {{
      const dt = now - last; last = now;
      if (playing && dt < 90) {{
        step = (step + 1) % activeCurve().depths.length;
        update();
      }}
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
