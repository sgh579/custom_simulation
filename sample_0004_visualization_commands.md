# sample_0004 可视化打开指令

样本前缀：

```bash
/home/goodmansun/custom_simulation/data/newton_4lump_33x33_32steps_40_8/train/sample_0004
```

## 0. Native 可视化路线（当前优先）

先更新环境，安装 PyVista/VTK/PySide6/PyVistaQt：

```bash
cd /home/goodmansun/custom_simulation
conda env update -f environment.yml --prune
conda activate palpation
```

查看连续解析 inclusion 形状，而不是 tet 离散后的近似：

```bash
python scripts/view_phantom_native.py \
  /home/goodmansun/custom_simulation/data/newton_4lump_33x33_32steps_40_8/train/sample_0004_gt.json \
  --resolution 72
```

导出给 ParaView / VTK 使用的大规模 mesh 文件：

```bash
python scripts/export_native_vtk.py \
  /home/goodmansun/custom_simulation/data/newton_4lump_33x33_32steps_40_8/train/sample_0004.npz \
  --out-dir /home/goodmansun/custom_simulation/data/newton_4lump_33x33_32steps_40_8/train/sample_0004_vtk \
  --extract-surface \
  --timeseries-row 16 \
  --timeseries-col 16 \
  --timeseries-stride 2
```

原生桌面按压过程播放器：

```bash
python scripts/play_press_native.py \
  /home/goodmansun/custom_simulation/data/newton_4lump_33x33_32steps_40_8/train/sample_0004.npz \
  --surface-resolution 128
```

FEM-style 离散 mesh 按压播放器：

```bash
python scripts/play_press_native.py \
  /home/goodmansun/custom_simulation/data/newton_4lump_33x33_32steps_40_8/train/sample_0004.npz \
  --mesh-style discrete \
  --tet-stride 64 \
  --vertex-stride 16
```
