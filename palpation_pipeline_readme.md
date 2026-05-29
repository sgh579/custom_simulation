# Newton/VBD Palpation Pipeline

This workspace now contains a minimal end-to-end pipeline for synthetic palpation data:

1. Build a tetrahedral soft phantom with per-tet material variation.
2. Randomly embed four lumps with randomized position, shape, size, yaw, and stiffness.
3. Press the phantom with a kinematic spherical probe on an `(x, y)` scan grid.
4. Save process data as `presses[H, W, T, 2]`, where channel 0 is indentation depth and channel 1 is total probe reaction `Fz`.
5. Train a small validation U-Net that outputs a `[H, W]` 0/1 inclusion projection grid.

## Data Generation

Fast smoke data without Newton:

```bash
/home/goodmansun/miniconda3/envs/torchnightly/bin/python scripts/generate_palpation_dataset.py \
  --backend analytic \
  --out-dir data/palpation_analytic \
  --num-train 80 \
  --num-val 20 \
  --grid-h 9 \
  --grid-w 9 \
  --press-steps 16 \
  --save-features
```

Richer analytic data with a larger phantom and 4 randomized z-separated lumps per sample:

```bash
/home/goodmansun/miniconda3/envs/torchnightly/bin/python scripts/generate_palpation_dataset.py \
  --backend analytic \
  --out-dir data/palpation_4lump_32x32x12 \
  --num-train 1000 \
  --num-val 200 \
  --size-x 0.18 \
  --size-y 0.18 \
  --height 0.08 \
  --cells-x 32 \
  --cells-y 32 \
  --cells-z 12 \
  --grid-h 33 \
  --grid-w 33 \
  --press-steps 32 \
  --max-indentation 0.032 \
  --lumps-min 4 \
  --lumps-max 4 \
  --lump-shapes sphere,ellipsoid,box,cylinder,capsule \
  --max-lump-radius-fraction 0.2 \
  --save-features
```

With `cells-x=32`, `cells-y=32`, and `cells-z=12`, the structured mesh has 14,157 particles and 61,440 tetrahedra. The default sampler keeps each inclusion geometry complete and only enforces that occupied z intervals do not overlap unless `--allow-z-overlap` is passed; x/y projections may overlap.

## Runtime Notes

Measured on this machine with the older 4-lump 32x32x12 analytic setup at 17x17 scan resolution, 100 train + 20 validation samples without per-press plot export took about 3.8 seconds to generate. The default generator now writes per-press CSV/PNG records for traceability, so large training-only batches can pass `--no-save-press-records` when those sidecars are not needed. Higher scan resolutions such as 33x33 improve GT/output granularity but increase per-sample presses quadratically.

Newton/VBD is much slower at this resolution. A tiny 3x3 scan with 4 press depths, 1 substep, and 2 VBD iterations took about 7.2 seconds on `cuda:0`; a full 13x13 scan with 16 depths, 3 substeps, and 5 VBD iterations should be treated as tens of minutes to around an hour per sample unless the simulator loop is further optimized.

Newton/VBD data:

```bash
/home/goodmansun/newton/.venv/bin/python scripts/generate_palpation_dataset.py \
  --backend newton \
  --out-dir data/palpation_newton \
  --num-train 8 \
  --num-val 2 \
  --grid-h 9 \
  --grid-w 9 \
  --press-steps 16 \
  --substeps-per-depth 3 \
  --vbd-iterations 5 \
  --device cpu
```

Use `--device cuda:0` for GPU if the current Newton/Warp build supports it.

Each `.npz` sample contains:

```text
presses:            [H, W, T, 2]  indentation depth and Fz
mask:               [H, W]        0/1 inclusion projection label
xy:                 [H, W, 2]     scan grid point coordinates
probe_pose:         [H, W, T, 7]  x, y, z, qx, qy, qz, qw
indentation_depth:  [H, W, T]
fz:                 [H, W, T]
contact_features:   [H, W, T, 5]  count, mean penetration, max penetration, mean contact z, mean normal z
lump_json:          JSON metadata for the first lump, kept for older scripts
lumps_json:         JSON list with every lump, including center/top depth from the top surface
num_lumps:          number of inclusions in the phantom
phantom_json/material_json/scan_json: generation config
```

For each generated phantom, the generator also writes sidecar files next to the `.npz`:

```text
sample_XXXX_gt.json:       GT metadata with phantom/material/scan config, all inclusion geometry, depth, stiffness, mask coverage, and stored array shapes
sample_XXXX_phantom.gltf:  3D preview of the actual tetrahedral material assignment, with semi-transparent normal tissue and colored lump tet surfaces/wireframes
sample_XXXX_scan_animation.html:
  interactive 3D scan animation with moving probe, deforming tet-mesh vertices/wireframes, play/speed controls, and auto-rotating camera view
sample_XXXX_press_records/:
  manifest.json:           press-record schema and grid summary
  index.csv:               one row per scan point, linking CSV and plot files
  press_rRRR_cCCC.csv:     per-step x/y, z displacement, probe z, Fz, and backend-specific contact features
  press_rRRR_cCCC_fz.png:  per-press F-z curve visualization
```

Open a specific phantom in the 3D viewer:

```bash
# By sample stem in a dataset directory
/home/goodmansun/miniconda3/envs/torchnightly/bin/python scripts/view_phantom_3d.py \
  sample_0001 \
  --data-dir data/acceptance_deep_phantoms/train

# Or by direct sidecar path
/home/goodmansun/miniconda3/envs/torchnightly/bin/python scripts/view_phantom_3d.py \
  data/acceptance_deep_phantoms/train/sample_0001_gt.json
```

## Validation U-Net

Use the `torchnightly` conda environment by default for generation, training, inference, and evaluation:

```bash
/home/goodmansun/miniconda3/envs/torchnightly/bin/python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(0))"
```

Train:

```bash
/home/goodmansun/miniconda3/envs/torchnightly/bin/python scripts/train_validation_unet.py \
  --data-dir data/palpation_4lump_32x32x12/train \
  --val-dir data/palpation_4lump_32x32x12/val \
  --out-dir runs/validation_unet_4lump \
  --epochs 30 \
  --batch-size 16 \
  --device cuda
```

Infer a final 0/1 grid:

```bash
/home/goodmansun/miniconda3/envs/torchnightly/bin/python scripts/infer_validation_unet.py \
  --checkpoint runs/validation_unet_4lump/best.pt \
  --input data/palpation_4lump_32x32x12/val/sample_0000.npz \
  --output runs/validation_unet_4lump/sample_0000_mask.npy \
  --prob-output runs/validation_unet_4lump/sample_0000_prob.npy \
  --device cuda
```

Evaluate a validation split and save visual comparisons:

```bash
/home/goodmansun/miniconda3/envs/torchnightly/bin/python scripts/evaluate_validation_unet.py \
  --checkpoint runs/validation_unet_4lump/best.pt \
  --data-dir data/palpation_4lump_32x32x12/val \
  --out-dir runs/validation_unet_4lump/eval \
  --threshold 0.5 \
  --device cuda
```

Each visualized eval sample writes `sample_XXXX_comparison.png` with four panels: GT, network probability, thresholded prediction, and a baseline equivalent-stiffness map. The baseline map is also saved as `sample_XXXX_baseline_stiffness.png` and `.npy`, using `k = (F_peak - F_start) / (disp_peak - disp_start)` at each scan point.

The U-Net follows the reference `scheme1_feature_unet` pattern: raw press curves are converted to mechanical feature maps, normalized per sample, then segmented with a compact 2D U-Net.
