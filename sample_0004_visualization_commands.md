# sample_0004 可视化打开指令

样本前缀：

```bash
/home/goodmansun/custom_simulation/data/newton_4lump_33x33_32steps_40_8/train/sample_0004
```

## 1. Phantom 3D 可视化

在仓库根目录运行：

```bash
cd /home/goodmansun/custom_simulation

/home/goodmansun/miniconda3/envs/torchnightly/bin/python scripts/view_phantom_3d.py \
  /home/goodmansun/custom_simulation/data/newton_4lump_33x33_32steps_40_8/train/sample_0004_gt.json
```

这个命令会启动本地 `phantom_viewer.html` 服务并自动打开浏览器。只想打印 URL、不自动打开浏览器时，加 `--no-open`：

```bash
cd /home/goodmansun/custom_simulation

/home/goodmansun/miniconda3/envs/torchnightly/bin/python scripts/view_phantom_3d.py \
  /home/goodmansun/custom_simulation/data/newton_4lump_33x33_32steps_40_8/train/sample_0004_gt.json \
  --no-open
```

## 2. 按压过程可视化

当前这个样本目录里没有现成的 `sample_0004_scan_animation.html`，先从 `.npz` 和 `_gt.json` 生成：

```bash
cd /home/goodmansun/custom_simulation

/home/goodmansun/miniconda3/envs/torchnightly/bin/python - <<'PY'
import json
from pathlib import Path

import numpy as np

from palpation_sim.config import PhantomConfig, ScanConfig
from palpation_sim.exports import write_scan_animation_html
from palpation_sim.phantom import LumpSpec

sample_path = Path("/home/goodmansun/custom_simulation/data/newton_4lump_33x33_32steps_40_8/train/sample_0004.npz")
gt_path = sample_path.with_name("sample_0004_gt.json")
out_path = sample_path.with_name("sample_0004_scan_animation.html")

gt = json.loads(gt_path.read_text(encoding="utf-8"))
phantom = PhantomConfig(**gt["phantom"])
scan_fields = ScanConfig.__dataclass_fields__.keys()
scan = ScanConfig(**{key: gt["scan"][key] for key in scan_fields if key in gt["scan"]})
lumps = [
    LumpSpec(
        shape=lump["shape"],
        center=tuple(float(value) for value in lump["center"]),
        radii=tuple(float(value) for value in lump["radii"]),
        stiffness_multiplier=float(lump["stiffness_multiplier"]),
        yaw=float(lump.get("yaw", 0.0)),
    )
    for lump in gt["lumps"]
]

with np.load(sample_path, allow_pickle=False) as sample_npz:
    sample = {key: sample_npz[key] for key in sample_npz.files}

write_scan_animation_html(out_path, phantom, scan, sample, lumps)
print(out_path)
PY
```

生成后建议用本地 HTTP 服务打开。直接 `xdg-open /path/to/file.html` 在某些环境里会打不开，或因为 `file://` 下的 ES module/import map 行为不稳定而显示空白。

```bash
cd /home/goodmansun/custom_simulation/data/newton_4lump_33x33_32steps_40_8/train

python3 -m http.server 8895 --bind 127.0.0.1
```

然后在浏览器打开：

```text
http://127.0.0.1:8895/sample_0004_scan_animation.html
```
