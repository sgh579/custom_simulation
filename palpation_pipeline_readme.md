# Palpation Simulation Pipeline

This workspace contains an end-to-end pipeline for synthetic palpation data:

1. Build a tetrahedral soft phantom with per-tet material variation.
2. Randomly embed lumps with randomized position, shape, size, yaw, and stiffness.
3. Press the phantom with a kinematic spherical probe on an `(x, y)` scan grid.
4. Save process data as `presses[H, W, T, 2]`, where channel 0 is indentation depth and channel 1 is total probe reaction `Fz`.
5. Train validation models against the generated inclusion projection masks.

## Environment Setup

Create or update the pinned conda environment from the tracked environment file:

```bash
conda env create -f environment.yml
conda activate palpation
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

All workflow commands are expected to run in this environment. When the shell is not already activated, use `conda run -n palpation python ...`.

For native visualization, make sure the optional VTK/PySide stack is installed:

```bash
conda env update -f environment.yml --prune
conda activate palpation
python -c "import pyvista, vtk, pyvistaqt, PySide6, pyqtgraph; print('native visualization ready')"
```

## Data Generation

The main generator currently supports two backends:

```text
newton    Newton/VBD soft-body simulation with tetrahedral Neo-Hookean material
analytic  fast smoke-test surrogate
```

Fast smoke data without Newton:

```bash
python scripts/generate_palpation_dataset.py \
  --backend analytic \
  --out-dir data/palpation_analytic \
  --num-train 80 \
  --num-val 20 \
  --grid-h 9 \
  --grid-w 9 \
  --press-steps 16 \
  --save-features
```

Newton/VBD data:

```bash
python scripts/generate_palpation_dataset.py \
  --backend newton \
  --out-dir data/palpation_newton \
  --num-train 8 \
  --num-val 2 \
  --grid-h 9 \
  --grid-w 9 \
  --press-steps 16 \
  --substeps-per-depth 3 \
  --vbd-iterations 5 \
  --device cuda:0
```

Newton/VBD simulation is intentionally GPU-only in this workflow. The pinned device is `cuda:0`; if Warp cannot see that GPU, the script stops instead of falling back to CPU.

With `cells-x=32`, `cells-y=32`, and `cells-z=12`, the structured mesh has 14,157 particles and 61,440 tetrahedra. The default sampler keeps each inclusion geometry complete and only enforces that occupied z intervals do not overlap unless `--allow-z-overlap` is passed; x/y projections may overlap.

Each `.npz` sample contains:

```text
presses:            [H, W, T, 2]  indentation depth and Fz
mask:               [H, W]        0/1 inclusion projection label
xy:                 [H, W, 2]     scan grid point coordinates
probe_pose:         [H, W, T, 7]  x, y, z, qx, qy, qz, qw
indentation_depth:  [H, W, T]
fz:                 [H, W, T]
contact_features:   [H, W, T, 5]  backend-specific press diagnostics
nonlinearity_ratio:  [H, W]        late/early F-z slope ratio, present for Newton/VBD samples
lump_json:          JSON metadata for the first lump, kept for older scripts
lumps_json:         JSON list with every lump, including center/top depth from the top surface
num_lumps:          number of inclusions in the phantom
phantom_json/material_json/scan_json: generation config
```

For each generated phantom, the generator also writes sidecar files next to the `.npz` unless disabled by flags:

```text
metadata.json:              dataset-level manifest at the dataset root
sample_XXXX_gt.json:       GT metadata with phantom/material/scan config, inclusion geometry, mask coverage, and stored array shapes
sample_XXXX_phantom.gltf:  3D preview of the tetrahedral material assignment
sample_XXXX_press_records/:
  manifest.json:           press-record schema and grid summary
  index.csv:               one row per scan point, linking CSV and plot files
  press_rRRR_cCCC.csv:     per-step x/y, z displacement, probe z, Fz, and backend-specific contact features
  press_rRRR_cCCC_fz.png:  per-press F-z curve visualization
```

## Fixed Four-Cylinder POC

Run the deterministic Newton/VBD proof of concept:

```bash
python scripts/run_fixed_four_cylinder_poc.py \
  --out-dir runs/fixed_four_cylinder_poc
```

This builds an `80 x 80 x 25 mm` phantom with four vertical `20 mm` diameter, `5 mm` high cylinder inclusions at `100x` normal-tissue stiffness. The default output is written under `runs/fixed_four_cylinder_poc/newton_poc` using a `32 x 32 x 10` mesh, `5 x 5` scan grid, `64` press steps, `16 mm` indentation, and an `8 mm` probe diameter.

For a quick script check without the full output cost:

```bash
python scripts/run_fixed_four_cylinder_poc.py \
  --smoke \
  --out-dir runs/fixed_four_cylinder_poc
```

Newton/VBD is much slower than the analytic smoke backend. A tiny 3x3 scan with 4 press depths, 1 substep, and 2 VBD iterations took about 7.2 seconds on `cuda:0`; a full 13x13 scan with 16 depths, 3 substeps, and 5 VBD iterations should be treated as tens of minutes to around an hour per sample unless the simulator loop is further optimized.

## Native Visualization

The active visualization path is native Python/VTK rather than the removed browser phantom viewer.

The metadata/data contract is documented in `docs/data_contract.md`; the fixed sample metadata shape is templated in `docs/metadata_template.json`.

Analytic phantom attribute viewer:

```bash
python scripts/view_phantom_native.py \
  runs/fixed_four_cylinder_poc/newton_poc/metadata.json \
  --resolution 72
```

This viewer displays the continuous analytic inclusion shapes from metadata: sphere, ellipsoid, box, cylinder, and capsule. It is intended for inspecting the true phantom attributes, not the discretized tet material assignment.

Native press process player:

```bash
python scripts/play_press_native.py \
  runs/fixed_four_cylinder_poc/newton_poc/fixed_four_cylinder_sample.npz \
  --surface-resolution 128
```

The player uses a PySide6 window with a PyVista viewport and a live F-z plot. It shows analytic inclusions, scan-point peak-force map, probe position, and a deforming top-surface proxy for the selected `(row, col, step)`.

FEM-style discrete press player:

```bash
python scripts/play_press_native.py \
  runs/fixed_four_cylinder_poc/newton_poc/fixed_four_cylinder_sample.npz \
  --mesh-style discrete \
  --tet-stride 64 \
  --vertex-stride 16
```

Discrete mode adds UI toggles for normal tissue, lump tet surfaces, sampled tet wireframe, sampled vertices, scan map, and probe. Lower `--tet-stride` and `--vertex-stride` values show denser mesh detail at higher rendering cost; use `1` only when you really want all tets/vertices. `--mesh-style both` overlays the continuous proxy/analytic surfaces with the FEM layers.

VTK / ParaView export for large-scale mesh inspection:

```bash
python scripts/export_native_vtk.py \
  runs/fixed_four_cylinder_poc/newton_poc/fixed_four_cylinder_sample.npz \
  --out-dir runs/fixed_four_cylinder_poc/newton_poc/vtk \
  --extract-surface \
  --timeseries-row 2 \
  --timeseries-col 2 \
  --timeseries-stride 2
```

Key outputs:

```text
*_tet_mesh.vtu                 full tetrahedral unstructured grid with lump/stiffness cell data
*_tet_surface.vtp              optional extracted tet boundary surface
*_scan_points.vtp              scan grid point cloud with peak Fz/mask/nonlinearity fields
analytic_lumps/*.vtp           continuous inclusion surfaces
*_press_rRRR_cCCC.pvd          ParaView time-series for one selected press
```

For very large samples, use the `.vtu` and `.pvd` files as the primary mesh/time-series artifacts. The current `.npz` files are compressed, so NumPy cannot memory-map their mesh arrays; avoid exporting all optional products unless you need them.

## Validation U-Net

Use the `palpation` conda environment for generation, training, inference, and evaluation:

```bash
conda activate palpation
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(0))"
```

Train:

```bash
python scripts/train_validation_unet.py \
  --data-dir data/palpation_newton/train \
  --val-dir data/palpation_newton/val \
  --out-dir runs/validation_unet_newton \
  --input-mode fz \
  --epochs 30 \
  --batch-size 16 \
  --device cuda
```

Infer a final 0/1 grid:

```bash
python scripts/infer_validation_unet.py \
  --checkpoint runs/validation_unet_newton/best.pt \
  --input data/palpation_newton/val/sample_0000.npz \
  --output runs/validation_unet_newton/sample_0000_mask.npy \
  --prob-output runs/validation_unet_newton/sample_0000_prob.npy \
  --device cuda
```

Evaluate a validation split and save visual comparisons:

```bash
python scripts/evaluate_validation_unet.py \
  --checkpoint runs/validation_unet_newton/best.pt \
  --data-dir data/palpation_newton/val \
  --out-dir runs/validation_unet_newton/eval \
  --threshold 0.5 \
  --device cuda
```

Each visualized eval sample writes `sample_XXXX_comparison.png` with four panels: GT, network probability, thresholded prediction, and a baseline equivalent-stiffness map. The baseline map is also saved as `sample_XXXX_baseline_stiffness.png` and `.npy`, using `k = (F_peak - F_start) / (disp_peak - disp_start)` at each scan point.

By default the U-Net uses only the raw `Fz` trajectory as input: sample arrays `fz[H, W, T]` are rearranged into `T` image channels `[T, H, W]`, normalized per sample, then segmented with a compact 2D U-Net. The older engineered mechanical feature path remains available only through `--input-mode features`.
