# Palpation Data Contract

This file defines the stable workflow contract for generated palpation data.

## Runtime

All repository scripts are pinned to the conda environment named `palpation`.
Run commands through either:

```bash
conda activate palpation
python <script> ...
```

or:

```bash
conda run -n palpation python <script> ...
```

Newton is loaded from `/home/guoheng/newton`. Newton/VBD simulation is pinned to GPU device `cuda:0`; scripts must fail early if that device is not visible to Warp.

## Run Directory Naming

Fresh outputs written under `runs/` must use a date prefix on the run directory leaf:

```text
runs/<yyyymmdd-hhmmss>-<run_name>/
```

For nested run groups, prefix the leaf directory:

```text
runs/newton_fz_sweeps/<yyyymmdd-hhmmss>-fixed_four/
```

The prefix is local wall-clock time in `yyyymmdd-hhmmss` format. Resume or assemble-only workflows may target an existing directory and should not add a new prefix.

## Simulator Boundary

Phantom-specific scripts may define geometry, lump layout, scan grids, and material parameters. They must not fork or reimplement solver stepping. Newton simulation logic lives in:

```text
palpation_sim.newton_vbd.NewtonVBDPalpationSimulator
```

Changing simulator behavior should happen in `palpation_sim/newton_vbd.py` so every phantom design inherits the same logic.

## Required Output Files

Single-run phantom scripts write a run directory with:

```text
metadata.json                    stable sample metadata, following docs/metadata_template.json
*.npz                            numeric arrays
curve_summary.csv                optional F-z summary table
press_records/                   optional per-press CSV/plot records
phantom_mesh.gltf                optional mesh preview
scan_animation.html              optional browser animation
```

Dataset generation writes:

```text
metadata.json                    dataset-level manifest
train/sample_XXXX.npz            numeric arrays
train/sample_XXXX_gt.json        sample-level metadata with the same schema as metadata.json
val/sample_XXXX.npz
val/sample_XXXX_gt.json
```

## Core NPZ Arrays

```text
presses:            [H, W, T, 2] float32, channel 0 indentation depth [m], channel 1 total reaction Fz [N]
mask:               [H, W] float32 or uint8, inclusion projection label
xy:                 [H, W, 2] float32, scan coordinates [m]
probe_pose:         [H, W, T, 7] float32, x/y/z/qx/qy/qz/qw
indentation_depth:  [H, W, T] float32, indentation depth [m]
fz:                 [H, W, T] float32, total probe reaction force [N]
contact_features:   [H, W, T, 5] float32, Newton contact diagnostics
nonlinearity_ratio:  [H, W] float32, late/early F-z slope ratio
tet_lump_mask:      [num_tets] uint8, whether each tet belongs to an inclusion
tet_lump_id:        [num_tets] int32, inclusion id per tet or -1
mesh_vertices:      [num_vertices, 3] float32, optional mesh vertices [m]
mesh_tets:          [num_tets, 4] int32, optional tet indices
features:           optional engineered feature map for ML experiments
```

String JSON arrays in the NPZ:

```text
phantom_json
material_json
scan_json
lump_json
lumps_json
mesh_json
backend
```

## Metadata JSON

All sample metadata must follow `docs/metadata_template.json`.

Important fields:

```text
runtime.python.environment_name   must be palpation
runtime.newton.root               must be /home/guoheng/newton
runtime.device.required           must be cuda:0
run.date_prefix                    run directory prefix in yyyymmdd-hhmmss format when under runs/
run.output_dir                     output directory used by the writer
resource_usage.elapsed_seconds     elapsed wall-clock seconds for the sample/case/dataset
resource_usage.gpu_memory          nvidia-smi memory.used observations in bytes
resource_usage.disk_usage          final saved file sizes in bytes and allocated bytes
files.npz                         primary numeric sample file
files.phantom_3d                  glTF preview if generated
files.press_records               press-record directory if generated
phantom/material/scan             full configuration used by the run
lumps                             analytic lump definitions
arrays                            shape and dtype of every NPZ array
```

`resource_usage.gpu_memory` records full-device `memory.used` from `nvidia-smi` before, during, and after the run. The field includes `peak_delta_from_start_bytes` as a best-effort per-run delta, but full-device usage can include other processes on the same GPU.

Every generated case or sample metadata file must include `resource_usage`. Multi-case dataset manifests must also include dataset-level `resource_usage`; when the dataset stores per-case records in one top-level manifest, each case entry must carry its own `resource_usage`.

## Native 3D Player Inputs

Use the native Python tools as the primary visualization path:

```bash
python scripts/play_press_native.py <sample.npz | metadata.json | run_dir>
python scripts/view_phantom_native.py <sample.npz | metadata.json | run_dir>
python scripts/export_native_vtk.py <sample.npz | metadata.json | run_dir>
```

The preferred input for a completed single-run directory is the run directory itself or its `metadata.json`. The metadata `files.npz` entry resolves the numeric sample used by the player.
