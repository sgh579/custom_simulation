## Native Visualization

```bash
conda activate palpation
```

Continuous / analytic press player:

```bash
python scripts/play_press_native.py \
  runs/fixed_four_cylinder_poc/newton_poc/fixed_four_cylinder_sample.npz \
  --surface-resolution 96
```

FEM-style discrete press player:

```bash
python scripts/play_press_native.py \
  runs/fixed_four_cylinder_poc/newton_poc/fixed_four_cylinder_sample.npz \
  --mesh-style discrete \
  --tet-stride 64 \
  --vertex-stride 16
```

Use smaller stride values for denser FEM display:

```bash
python scripts/play_press_native.py \
  runs/fixed_four_cylinder_poc/newton_poc/fixed_four_cylinder_sample.npz \
  --mesh-style discrete \
  --tet-stride 16 \
  --vertex-stride 4
```

`--mesh-style both` overlays the continuous proxy/analytic surfaces with the FEM layers.
