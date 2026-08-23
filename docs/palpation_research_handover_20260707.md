# Palpation Research Handover Notes

Last verified: 2026-07-07, Australia/Sydney.

These notes preserve the current operational state for the project sometimes
called `palpation_research`. No local directory with that exact name was found;
the runnable codebase is this repository, `custom_simulation`.

## Project Identity

Use these paths as the current map:

```text
Local Mac code:      /Users/goodmansun/code_base/projects/custom_simulation
Paper workspace:    /Users/goodmansun/text_editing_wks/palpation_paper
lab204 code/data:   /home/guoheng/custom_simulation
Katana code:        /srv/scratch/z5511722/projects/custom_simulation
Katana Newton:      /srv/scratch/z5511722/src/newton
Katana venv:        /srv/scratch/z5511722/envs/palpation
Katana jobs:        /srv/scratch/z5511722/projects/custom_simulation/katana_jobs
```

Important aliases:

```bash
ssh lab204
ssh katana
ssh kdm
```

`katana` is for login, job submission, and light inspection. `kdm` is for data
transfer to or from Katana storage.

## Repository State

The local branch is `dev`. At the time these notes were written, the local
worktree was not clean. There were modified core files and many untracked
research scripts, including the high-resolution sample-0147 scripts.

Do not assume GitHub alone contains the complete runnable state for the
high-resolution sample-0147 work. Inspect the real worktree before reproducing
or publishing:

```bash
cd /Users/goodmansun/code_base/projects/custom_simulation
git status --short
git diff --stat
```

One important local code change is in `palpation_sim/workflow.py`:

```python
DEFAULT_NEWTON_ROOT = Path(os.environ.get("PALPATION_NEWTON_ROOT", "/home/guoheng/newton")).expanduser()
```

This is needed because lab204 uses `/home/guoheng/newton`, while Katana uses
`/srv/scratch/z5511722/src/newton`.

## Data Ownership

The local Mac currently does not have `custom_simulation/data`.

The authoritative source data for the high-resolution sample-0147 run is on
lab204:

```text
/home/guoheng/custom_simulation/data
```

That data tree was about `7.1G` on 2026-07-07. The minimal metadata file needed
for sample 0147 is:

```text
/home/guoheng/custom_simulation/data/palpation_synthetic_depth_modes_1000train_400val_400test_20260622/random_trajectory/data/test/sample_0147_gt.json
```

Within scripts, the relative path is:

```text
data/palpation_synthetic_depth_modes_1000train_400val_400test_20260622/random_trajectory/data/test/sample_0147_gt.json
```

Keep generated datasets, run directories, transfer packages, and large binaries
out of Git. See `docs/data_and_model_artifacts.md`.

## High-Resolution Sample 0147 Configuration

The successful Katana and lab204/RTX-5090 runs used this configuration:

```text
mesh: 96 x 96 x 32
scan grid: 20 x 20
probe diameter: 8 mm
max indentation: 10 mm
press steps: 160
substeps: 16
VBD iterations: 32
soft contact margin: 1 mm
normal k mu: 10000
normal k lambda: 10000
normal k damp: 0.0001
soft contact ke: 2000000
rigid stiffness multiplier: 10000
force estimator: bottom_support_reaction
device: cuda:0
```

Primary entrypoint:

```text
scripts/run_real_phantom_sample_newton.py
```

Convenience launchers:

```text
scripts/launch_real_phantom_sample0147_highres.sh
scripts/supervise_real_phantom_sample0147_highres.sh
```

## Katana Setup That Worked

Katana remote layout:

```text
Project: /srv/scratch/z5511722/projects/custom_simulation
Newton:  /srv/scratch/z5511722/src/newton
Env:     /srv/scratch/z5511722/envs/palpation
Run:     /srv/scratch/z5511722/projects/custom_simulation/runs/katana_sample0147_highres_reaction_v2_t160_s16_i32
Jobs:    /srv/scratch/z5511722/projects/custom_simulation/katana_jobs
```

The working PBS job scripts were placed in:

```text
/srv/scratch/z5511722/projects/custom_simulation/katana_jobs
```

They were mirrored locally at:

```text
/Users/goodmansun/code_base/projects/study_katana/palpation_katana_jobs
```

The job environment used module Python and a venv rather than conda:

```bash
module purge
module load python/3.11.3 cuda/12.8.0 gcc/12.2.0
source /srv/scratch/$USER/envs/palpation/bin/activate
export PALPATION_NEWTON_ROOT=/srv/scratch/$USER/src/newton
export MPLBACKEND=Agg
export XDG_CACHE_HOME=/srv/scratch/$USER/cache
export WARP_CACHE_PATH=/srv/scratch/$USER/cache/warp
```

The minimal installed Python packages were:

```text
numpy
matplotlib
warp-lang
```

Katana smoke jobs completed successfully:

| Job | ID | Result |
|---|---|---|
| CPU PBS smoke | `8480291.kman.restech.unsw.edu.au` | success, ran on `k177` |
| GPU smoke | `8480675.kman.restech.unsw.edu.au` | success, ran on `k105`, V100 |
| Tiny Newton smoke | `8480689.kman.restech.unsw.edu.au` | success, ran on `k105`, V100 |

The Katana `qsub` wrapper printed this warning during early tests while still
submitting jobs successfully:

```text
/opt/pbs/bin/qsub: line 32: allvars[$lastindex]: bad array subscript
```

Treat it as a wrapper warning unless it starts blocking submissions.

## Katana Sample-0147 Result

The completed Katana run is:

```text
/srv/scratch/z5511722/projects/custom_simulation/runs/katana_sample0147_highres_reaction_v2_t160_s16_i32
```

Generated files:

```text
real_phantom_sample0147_rigid_ecoflex0010_probe8_depth10_highres.npz
metadata.json
summary.json
curve_summary.csv
fz_curves.png
```

Final output details from `summary.json`:

```text
chunk_count: 20
curve_shape: [20, 20, 160]
total_chunk_elapsed_seconds: 66707.20947265625
total_chunk_hms: 18:31:47
peak_force_max_n: 1.4215730428695679
peak_force_median_n: 1.294863224029541
npz size: 10301712 bytes
```

PBS job history:

| Stage | Job ID | Result | Notes |
|---|---|---|---|
| row 0 pilot | `8480840.kman.restech.unsw.edu.au` | success | `k172`, NVIDIA L40S, about `41:47` PBS walltime |
| rows 1-19 array | `8483674[].kman.restech.unsw.edu.au` | success | submitted as indices `2-20`, chunks `20 / 20` after pilot |
| assemble | `8490410.kman.restech.unsw.edu.au` | success | `k211`, Exit Status 0, about `15s` PBS walltime |

Timing interpretations:

```text
Strict wall-clock from row0 qsub to assemble finish: 3:00:35
Wall-clock excluding the manual idle gap before assemble: 2:43:39
Execution-only critical path: 2:39:29
Simplified planning number used in discussion: about 2 hours
PBS cumulative GPU/CPU walltime: about 18:33:02
Summary chunk compute total: 18:31:47
```

The cumulative time is high because 20 row chunks consumed separate job walltime
in parallel. It is not the user's waiting time.

Monitor or re-check on Katana:

```bash
ssh katana
qstat -u "$USER"
find /srv/scratch/$USER/projects/custom_simulation/runs/katana_sample0147_highres_reaction_v2_t160_s16_i32/chunks \
  -maxdepth 1 -name 'rows_*.npz' | wc -l
cat /srv/scratch/$USER/projects/custom_simulation/runs/katana_sample0147_highres_reaction_v2_t160_s16_i32/summary.json
```

Katana scratch is not the long-term archive. Pull the run through KDM if it must
survive scratch cleanup:

```bash
rsync -avh --progress \
  kdm:/srv/scratch/z5511722/projects/custom_simulation/runs/katana_sample0147_highres_reaction_v2_t160_s16_i32/ \
  /Users/goodmansun/code_base/projects/custom_simulation/runs/katana_sample0147_highres_reaction_v2_t160_s16_i32/
```

## RTX 5090 Baseline

The comparable single-workstation record on lab204 is:

```text
/home/guoheng/custom_simulation/runs/20260626-143417-synthetic_random_sample0147_rigid_ecoflex0010_probe8mm_indent10mm_mesh96x96x32_reaction_seq_t160_s16_i32
```

The machine was:

```text
NVIDIA GeForce RTX 5090, driver 575.64.03, 32607 MiB VRAM
```

This run used one worker over all 20 rows, so the rows ran sequentially.

Summary:

```text
chunk_count: 20
curve_shape: [20, 20, 160]
total_chunk_elapsed_seconds: 38668.439208984375
total_chunk_hms: 10:44:28
average row chunk: about 1933s, or 32:13
wall time from init log to final npz/assemble: about 10:44:32
peak_force_max_n: 1.4235953092575073
peak_force_median_n: 1.293324589729309
```

The Katana run was faster in user waiting time because it parallelized rows
across GPU jobs. Against the strict 5090 sequential baseline, using the
simplified Katana planning number of about 2 hours gives roughly a `5.3x`
wall-clock speedup.

Do not compare Katana's cumulative PBS walltime to 5090 wall-clock as if they
were the same metric. The fair comparison depends on the question:

```text
User waiting time: Katana wall-clock vs 5090 wall-clock.
Resource accounting: Katana cumulative job walltime vs 5090 GPU-hours.
```

## Reproduction Checklist

Before launching a future high-resolution run:

1. Check the local code state and confirm which copy is authoritative.
2. Confirm the required metadata exists on the target machine.
3. Confirm Newton is reachable and `PALPATION_NEWTON_ROOT` is set correctly.
4. Run a tiny Newton/VBD smoke job on the target GPU.
5. For Katana, run one row pilot before submitting the whole row array.
6. Count chunk files before assemble.
7. Validate the final `.npz` with `numpy.load` and inspect `summary.json`.
8. Copy important Katana scratch outputs back through KDM.

Useful validation snippet:

```bash
python - <<'PY'
import numpy as np
path = "real_phantom_sample0147_rigid_ecoflex0010_probe8_depth10_highres.npz"
data = np.load(path)
for key in ["fz", "xy", "mask", "probe_wrench", "contact_features", "indentation_depth"]:
    arr = data[key]
    print(key, arr.shape, arr.dtype, float(np.nanmin(arr)), float(np.nanmax(arr)))
PY
```

Expected core shapes for the high-resolution sample:

```text
fz:                (20, 20, 160)
xy:                (20, 20, 2)
mask:              (20, 20)
probe_wrench:      (20, 20, 160, 6)
contact_features:  (20, 20, 160, 5)
indentation_depth: (20, 20, 160)
```

## Known Pitfalls

- The name `palpation_research` has been used conversationally, but the real
  repository is `custom_simulation`.
- The local Mac may not have the data tree; lab204 is still the primary compute
  and data host.
- GitHub may lag the runnable research state if local scripts are untracked.
- Katana scratch is not a permanent archive.
- Katana login nodes should not run real compute; always use PBS jobs.
- Requesting a specific scarce GPU model can increase queue time. Generic
  `ngpus=1` worked for the sample-0147 row jobs.
- The old OpenOnDemand `AADSTS50105` symptom meant Katana app provisioning, not
  a local SSH syntax problem. Current SSH aliases were later verified.
- Keep sparse-context, full-context, Dice, stiffness-map, raw-Fz, and
  constitutive-vs-apparent-stiffness claims separate when reusing results for
  manuscript text.

