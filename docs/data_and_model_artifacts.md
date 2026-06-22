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

Sync selected local packages to `lab204`:

```bash
python scripts/artifact_sync.py push \
  --manifest artifacts/manifests/local_transfer_packages.json \
  --remote lab204 \
  --remote-root /home/guoheng/custom_simulation
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

Then push model content to the DVC remote:

```bash
dvc push
git add .dvc .dvcignore models docs/data_and_model_artifacts.md .gitignore
git status
```
