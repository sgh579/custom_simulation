# Artifact Layout

This repository keeps generated data and experiment outputs out of Git history.

- `data/`: unpacked/generated datasets used locally or on `lab204`.
- `runs/`: training, evaluation, logs, plots, and intermediate reports.
- `artifacts/transfer_packages/`: compressed bundles intended for explicit `rsync` transfer.
- `artifacts/manifests/`: small JSON manifests that describe transfer packages or selected run outputs.
- `models/`: curated model artifacts tracked with DVC, not a dump of every checkpoint in `runs/`.

Use `scripts/artifact_sync.py` to create manifests and sync selected files or directories between machines.
Use `scripts/register_model_artifact.py` to promote a selected checkpoint from `runs/` into `models/` and DVC.
