# Data and Model Artifact Workflow

The repository has three copies with different roles:

- GitHub: public code and lightweight metadata.
- Local Mac: writing, inspection, and selected packaged results.
- `lab204`: compute host and primary source for generated data and runs.

## Data and Run Outputs

Do not commit generated datasets, run directories, checkpoints, or large plots directly to Git.

Use these paths consistently:

- `data/`: unpacked generated datasets.
- `runs/`: full experiment outputs.
- `artifacts/transfer_packages/`: compressed bundles for explicit transfer.
- `artifacts/manifests/`: small JSON manifests tracked by Git.

Create a manifest for selected artifacts:

```bash
python scripts/artifact_sync.py manifest \
  --name nonlinear_trajectory_20260620 \
  --paths runs/nonlinear_trajectory_20x_repeats10_seed20260618 \
  --out artifacts/manifests/nonlinear_trajectory_20260620.json
```

Sync the manifest paths from `lab204` to the local Mac:

```bash
python scripts/artifact_sync.py pull \
  --manifest artifacts/manifests/nonlinear_trajectory_20260620.json \
  --remote lab204 \
  --remote-root /home/guoheng/custom_simulation
```

Verify the copied paths against the manifest after every transfer:

```bash
python scripts/artifact_sync.py verify \
  --manifest artifacts/manifests/nonlinear_trajectory_20260620.json \
  --target local
```

Sync selected local packages to `lab204`:

```bash
python scripts/artifact_sync.py push \
  --manifest artifacts/manifests/local_transfer_packages.json \
  --remote lab204 \
  --remote-root /home/guoheng/custom_simulation
```

Verify remote paths from the Mac when `lab204` is the target:

```bash
python scripts/artifact_sync.py verify \
  --manifest artifacts/manifests/local_transfer_packages.json \
  --target remote \
  --remote lab204 \
  --remote-root /home/guoheng/custom_simulation
```

Use deletion only as a deliberate cleanup step. First run `--dry-run`; the real command must also include `--confirm-delete` with the exact manifest name:

```bash
python scripts/artifact_sync.py pull \
  --manifest artifacts/manifests/nonlinear_trajectory_20260620.json \
  --remote lab204 \
  --remote-root /home/guoheng/custom_simulation \
  --delete \
  --confirm-delete nonlinear_trajectory_20260620
```

The local cleanup backup from 2026-06-22 is recorded in `artifacts/manifests/mac_cleanup_backup_20260622_1537.json`; the actual files live on `lab204` under `artifacts/mac_cleanup_backup_20260622_1537/`.

Before committing, check that generated artifacts have not leaked into Git:

```bash
python scripts/check_git_artifact_policy.py
```

## Model Versioning With DVC

DVC is reserved for curated model artifacts, not whole `runs/` directories.

Install DVC with SSH support on each machine:

```bash
python -m pip install -r requirements-dev.txt
```

The committed default remote is:

```text
ssh://lab204/home/guoheng/dvc-remotes/custom_simulation-models
```

This is an internal remote for the three-copy workflow. Public users can read the `.dvc` pointers, but `dvc pull` requires access to `lab204` or a separately published DVC remote.

On `lab204`, use a local override:

```bash
mkdir -p /home/guoheng/dvc-remotes/custom_simulation-models
dvc remote modify --local lab204-models url /home/guoheng/dvc-remotes/custom_simulation-models
```

Promote a selected checkpoint into DVC:

```bash
python scripts/register_model_artifact.py \
  --source runs/segmentation_accuracy_sweep_20x_4seed/unet_fz_features_aug_focal/best.pt \
  --name unet_fz_features_aug_focal_20x4seed \
  --run-dir runs/segmentation_accuracy_sweep_20x_4seed/unet_fz_features_aug_focal
```

The registration script refuses a dirty Git worktree by default and records checkpoint size and SHA-256 in metadata. Use `--allow-dirty` only for intentionally preserved legacy or emergency artifacts, and explain that in `--description`.

Then push model content to the DVC remote:

```bash
dvc push
git add .dvc .dvcignore models docs/data_and_model_artifacts.md .gitignore
git status
```
