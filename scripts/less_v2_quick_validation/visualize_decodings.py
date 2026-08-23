#!/usr/bin/env python3
"""Create representative static and interactive 2D/3D decoding views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--html", type=Path, required=True)
    return parser.parse_args()


def metrics(probability: np.ndarray, target: np.ndarray, threshold: float) -> tuple[float, float]:
    prediction = probability >= threshold
    truth = target >= 0.5
    tp = int((prediction & truth).sum())
    fp = int((prediction & ~truth).sum())
    fn = int((~prediction & truth).sum())
    dice = 2 * tp / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    return dice, iou


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir
    manifest = json.loads((result_dir / "run_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
    arrays = np.load(result_dir / "test_predictions.npz")
    ids = [Path(value).stem for value in manifest["selected_samples"]["test"]]
    p2 = arrays["probabilities_2d"].astype(np.float32)
    p3 = arrays["probabilities_3d"].astype(np.float32)
    y2 = arrays["targets_2d"].astype(np.uint8)
    y3 = arrays["targets_3d"].astype(np.uint8)
    threshold_2d = float(summary["thresholds_selected_on_validation"]["2d"])
    threshold_3d = float(summary["thresholds_selected_on_validation"]["3d"])

    scores = [metrics(p3[index], y3[index], threshold_3d)[0] for index in range(len(ids))]
    order = np.argsort(scores)
    selected = [int(order[-1]), int(order[len(order) // 2]), int(order[0])]
    roles = ["strong", "typical", "challenging"]
    experiment_title = str(manifest.get("kind", "Synthetic palpation V2"))
    make_2d_figure(
        result_dir / "decoded_examples_2d.png",
        ids,
        p2,
        y2,
        threshold_2d,
        selected,
        roles,
        experiment_title,
    )
    make_3d_figure(
        result_dir / "decoded_examples_3d.png",
        ids,
        p3,
        y3,
        threshold_3d,
        selected,
        roles,
        experiment_title,
    )
    make_html(
        args.html,
        ids,
        p2,
        p3,
        y2,
        y3,
        threshold_2d,
        threshold_3d,
        selected,
        roles,
    )


def make_2d_figure(
    path: Path,
    ids: list[str],
    probabilities: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    selected: list[int],
    roles: list[str],
    experiment_title: str,
) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(9.4, 9.2), constrained_layout=True)
    for row, (index, role) in enumerate(zip(selected, roles)):
        probability = probabilities[index]
        target = targets[index] >= 0.5
        prediction = probability >= threshold
        dice, iou = metrics(probability, target, threshold)
        overlay = np.zeros((*target.shape, 3), dtype=np.float32)
        overlay[..., 0] = prediction & ~target
        overlay[..., 1] = prediction & target
        overlay[..., 2] = target & ~prediction
        axes[row, 0].imshow(target, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        axes[row, 1].imshow(probability, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        axes[row, 2].imshow(overlay, interpolation="nearest")
        axes[row, 0].set_ylabel(f"{role}\n{ids[index]}\nDice {dice:.3f} · IoU {iou:.3f}")
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    axes[0, 0].set_title("2D ground truth")
    axes[0, 1].set_title("2D probability")
    axes[0, 2].set_title(f"threshold {threshold:.3f}\ngreen TP · red FP · blue FN")
    figure.suptitle(f"{experiment_title}: representative 2D decoded test examples", fontsize=14)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_3d_figure(
    path: Path,
    ids: list[str],
    probabilities: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    selected: list[int],
    roles: list[str],
    experiment_title: str,
) -> None:
    figure = plt.figure(figsize=(12.4, 10.4), constrained_layout=True)
    for row, (index, role) in enumerate(zip(selected, roles)):
        truth = targets[index] >= 0.5
        prediction = probabilities[index] >= threshold
        dice, iou = metrics(probabilities[index], truth, threshold)
        overlap = truth & prediction
        misses = truth & ~prediction
        extras = prediction & ~truth
        for col, (volume, color, title) in enumerate(
            (
                (truth, "tab:blue", "3D ground truth"),
                (prediction, "tab:orange", "3D prediction"),
            )
        ):
            axis = figure.add_subplot(3, 4, row * 4 + col + 1, projection="3d")
            axis.voxels(volume.transpose(2, 1, 0), facecolors=color, edgecolor="none", alpha=0.75)
            style_3d_axis(axis)
            if row == 0:
                axis.set_title(title)
        axis = figure.add_subplot(3, 4, row * 4 + 3, projection="3d")
        axis.voxels(overlap.transpose(2, 1, 0), facecolors="tab:green", edgecolor="none", alpha=0.80)
        axis.voxels(misses.transpose(2, 1, 0), facecolors="tab:blue", edgecolor="none", alpha=0.65)
        axis.voxels(extras.transpose(2, 1, 0), facecolors="tab:red", edgecolor="none", alpha=0.45)
        style_3d_axis(axis)
        if row == 0:
            axis.set_title("overlap: TP/FN/FP")
        depth_counts = truth.sum(axis=(1, 2))
        depth = int(np.argmax(depth_counts))
        classification = np.zeros_like(truth[depth], dtype=np.uint8)
        classification[truth[depth] & prediction[depth]] = 1
        classification[truth[depth] & ~prediction[depth]] = 2
        classification[~truth[depth] & prediction[depth]] = 3
        axis_2d = figure.add_subplot(3, 4, row * 4 + 4)
        axis_2d.imshow(
            classification,
            cmap=ListedColormap(["white", "tab:green", "tab:blue", "tab:red"]),
            vmin=0,
            vmax=3,
            interpolation="nearest",
        )
        axis_2d.set_xticks([])
        axis_2d.set_yticks([])
        axis_2d.set_ylabel(f"{role}\n{ids[index]}\nDice {dice:.3f} · IoU {iou:.3f}")
        if row == 0:
            axis_2d.set_title("most occupied GT depth slice")
    figure.suptitle(
        f"{experiment_title}: representative 3D decoded test examples (threshold {threshold:.3f})",
        fontsize=14,
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def style_3d_axis(axis: plt.Axes) -> None:
    axis.set_xlim(0, 20)
    axis.set_ylim(0, 20)
    axis.set_zlim(0, 16)
    axis.set_box_aspect((20, 20, 16))
    axis.view_init(elev=25, azim=-55)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_zticks([])
    axis.set_xlabel("x", labelpad=-8)
    axis.set_ylabel("y", labelpad=-8)
    axis.set_zlabel("depth", labelpad=-8)


def make_html(
    path: Path,
    ids: list[str],
    p2: np.ndarray,
    p3: np.ndarray,
    y2: np.ndarray,
    y3: np.ndarray,
    threshold_2d: float,
    threshold_3d: float,
    selected: list[int],
    roles: list[str],
) -> None:
    records = []
    role_by_index = {index: role for index, role in zip(selected, roles)}
    for index, sample_id in enumerate(ids):
        dice_2d, iou_2d = metrics(p2[index], y2[index], threshold_2d)
        dice_3d, iou_3d = metrics(p3[index], y3[index], threshold_3d)
        records.append(
            {
                "id": sample_id,
                "role": role_by_index.get(index, "other test case"),
                "p2": np.rint(p2[index] * 255).astype(np.uint8).reshape(-1).tolist(),
                "p3": np.rint(p3[index] * 255).astype(np.uint8).reshape(-1).tolist(),
                "y2": y2[index].reshape(-1).tolist(),
                "y3": y3[index].reshape(-1).tolist(),
                "m2": [round(dice_2d, 4), round(iou_2d, 4)],
                "m3": [round(dice_3d, 4), round(iou_3d, 4)],
            }
        )
    data = json.dumps(
        {"samples": records, "t2": threshold_2d, "t3": threshold_3d},
        separators=(",", ":"),
    )
    template = r'''<div id="less-v2-decoding-viewer" class="lv-root">
  <h2>LESS-style V2 decoded test examples</h2>
  <div class="lv-controls">
    <label>Sample <select id="lv-sample"></select></label>
    <label>3D depth <input id="lv-depth" type="range" min="0" max="15" value="8"><output id="lv-depth-value">8 / 15</output></label>
  </div>
  <div id="lv-metrics" class="lv-metrics" aria-live="polite"></div>
  <section>
    <h3>2D projection decoding</h3>
    <div class="lv-grid lv-grid-3">
      <figure><figcaption>Ground truth</figcaption><canvas id="lv-gt2" width="400" height="400"></canvas></figure>
      <figure><figcaption>Probability</figcaption><canvas id="lv-prob2" width="400" height="400"></canvas></figure>
      <figure><figcaption>Thresholded classification</figcaption><canvas id="lv-class2" width="400" height="400"></canvas></figure>
    </div>
  </section>
  <section>
    <h3>3D occupancy decoding</h3>
    <div class="lv-grid lv-grid-2">
      <figure><figcaption>Isometric voxels: overlap / missed GT / extra prediction</figcaption><canvas id="lv-iso3" width="640" height="460"></canvas></figure>
      <figure><figcaption id="lv-slice-title">Selected depth slice</figcaption><canvas id="lv-slice3" width="460" height="460"></canvas></figure>
    </div>
  </section>
  <div class="lv-legend" aria-label="Classification legend"><span><i class="lv-tp"></i> overlap (TP)</span><span><i class="lv-fn"></i> missed GT (FN)</span><span><i class="lv-fp"></i> extra prediction (FP)</span></div>
</div>
<style>
#less-v2-decoding-viewer{color:var(--foreground);font-size:var(--font-size-base);max-width:100%}
#less-v2-decoding-viewer h2,#less-v2-decoding-viewer h3{font-weight:500;margin:0 0 10px}
#less-v2-decoding-viewer h3{margin-top:18px}
.lv-controls{display:flex;gap:20px;align-items:center;flex-wrap:wrap;margin:8px 0}
.lv-controls label{display:flex;gap:8px;align-items:center}
.lv-controls select{min-width:210px}
.lv-controls input[type=range]{width:220px}
.lv-metrics{min-height:24px;color:var(--muted-foreground);margin:7px 0 4px}
.lv-grid{display:grid;gap:12px}.lv-grid-3{grid-template-columns:repeat(3,minmax(0,1fr))}.lv-grid-2{grid-template-columns:1.25fr .9fr}
.lv-grid figure{margin:0;min-width:0}.lv-grid figcaption{margin-bottom:6px;color:var(--muted-foreground)}
.lv-grid canvas{display:block;width:100%;height:auto;border:1px solid var(--border);background:var(--card)}
.lv-legend{display:flex;gap:18px;flex-wrap:wrap;margin-top:10px;color:var(--muted-foreground)}
.lv-legend span{display:inline-flex;align-items:center;gap:6px}.lv-legend i{width:12px;height:12px;display:inline-block}
.lv-tp{background:var(--green)}.lv-fn{background:var(--blue)}.lv-fp{background:var(--red)}
@media(max-width:700px){.lv-grid-3,.lv-grid-2{grid-template-columns:1fr}.lv-controls input[type=range]{width:160px}}
</style>
<script>
(()=>{
const root=document.getElementById('less-v2-decoding-viewer');
const data=__DATA__;
const style=getComputedStyle(root);
const color=name=>style.getPropertyValue(name).trim();
const C={bg:color('--card'),fg:color('--foreground'),muted:color('--muted-foreground'),border:color('--border'),blue:color('--blue'),orange:color('--orange'),green:color('--green'),red:color('--red'),purple:color('--purple'),yellow:color('--yellow')};
const select=root.querySelector('#lv-sample'),slider=root.querySelector('#lv-depth'),depthOut=root.querySelector('#lv-depth-value'),metrics=root.querySelector('#lv-metrics');
data.samples.forEach((s,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${s.id} — ${s.role}`;select.appendChild(o)});
select.value=data.samples.findIndex(s=>s.role==='typical');
function mix(a,b,t){const pa=parse(a),pb=parse(b);return `rgb(${Math.round(pa[0]+(pb[0]-pa[0])*t)},${Math.round(pa[1]+(pb[1]-pa[1])*t)},${Math.round(pa[2]+(pb[2]-pa[2])*t)})`}
function parse(v){const x=document.createElement('canvas').getContext('2d');x.fillStyle=v;x.fillRect(0,0,1,1);const d=x.getImageData(0,0,1,1).data;return [d[0],d[1],d[2]]}
function heat(t){return t<.5?mix(C.purple,C.orange,t*2):mix(C.orange,C.yellow,(t-.5)*2)}
function grid(canvas,values,kind,threshold,target){const ctx=canvas.getContext('2d'),n=20,cell=canvas.width/n;ctx.fillStyle=C.bg;ctx.fillRect(0,0,canvas.width,canvas.height);for(let y=0;y<n;y++)for(let x=0;x<n;x++){const i=y*n+x,v=values[i]/(kind==='prob'?255:1);if(kind==='truth')ctx.fillStyle=v?C.blue:C.bg;else if(kind==='prob')ctx.fillStyle=heat(v);else{const p=values[i]/255>=threshold,g=target[i]>0;ctx.fillStyle=p&&g?C.green:g?C.blue:p?C.red:C.bg}ctx.fillRect(x*cell,y*cell,cell+.3,cell+.3)}ctx.strokeStyle=C.border;ctx.strokeRect(.5,.5,canvas.width-1,canvas.height-1)}
function slice(sample,z){const off=z*400,prob=sample.p3.slice(off,off+400),truth=sample.y3.slice(off,off+400);grid(root.querySelector('#lv-slice3'),prob,'class',data.t3,truth);root.querySelector('#lv-slice-title').textContent=`Depth slice ${z} / 15`}
function iso(sample){const canvas=root.querySelector('#lv-iso3'),ctx=canvas.getContext('2d');ctx.fillStyle=C.bg;ctx.fillRect(0,0,canvas.width,canvas.height);const vox=[];for(let z=0;z<16;z++)for(let y=0;y<20;y++)for(let x=0;x<20;x++){const i=z*400+y*20+x,p=sample.p3[i]/255>=data.t3,g=sample.y3[i]>0;if(!p&&!g)continue;vox.push({x,y,z,c:p&&g?C.green:g?C.blue:C.red,a:p&&g?.85:g?.72:.42})}vox.sort((a,b)=>(a.x+a.y+a.z)-(b.x+b.y+b.z));const sx=13,sy=6.5,sz=10,ox=canvas.width/2,oy=canvas.height-45;for(const v of vox){const px=ox+(v.x-v.y)*sx,py=oy-(v.x+v.y)*sy-v.z*sz;ctx.globalAlpha=v.a;ctx.fillStyle=v.c;ctx.beginPath();ctx.arc(px,py,4.5,0,Math.PI*2);ctx.fill()}ctx.globalAlpha=1;ctx.fillStyle=C.muted;ctx.font='12px system-ui';ctx.fillText('surface',12,canvas.height-16);ctx.fillText('deep',canvas.width-42,canvas.height-16)}
function render(){const s=data.samples[+select.value];grid(root.querySelector('#lv-gt2'),s.y2,'truth');grid(root.querySelector('#lv-prob2'),s.p2,'prob');grid(root.querySelector('#lv-class2'),s.p2,'class',data.t2,s.y2);iso(s);slice(s,+slider.value);metrics.textContent=`${s.id} (${s.role}) · 2D Dice ${s.m2[0].toFixed(3)}, IoU ${s.m2[1].toFixed(3)} · 3D Dice ${s.m3[0].toFixed(3)}, IoU ${s.m3[1].toFixed(3)} · thresholds: 2D ${data.t2.toFixed(3)}, 3D ${data.t3.toFixed(3)}`;depthOut.textContent=`${slider.value} / 15`}
select.addEventListener('change',render);slider.addEventListener('input',render);render();
})();
</script>'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template.replace("__DATA__", data), encoding="utf-8")


if __name__ == "__main__":
    main()
