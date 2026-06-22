# DVC-Managed Models

Only curated model artifacts belong here. Raw training outputs remain in `runs/`.

Recommended flow:

1. Select a checkpoint and its run directory.
2. Register it with `scripts/register_model_artifact.py`.
3. Review the generated metadata.
4. Run `dvc push` to store the model in the configured DVC remote.
5. Commit the small `.dvc` pointer and metadata, not the model binary.

The default DVC remote is `lab204-models`, configured as:

```text
ssh://lab204/home/guoheng/dvc-remotes/custom_simulation-models
```

On `lab204`, you can override that URL locally so pushes use the local filesystem:

```bash
dvc remote modify --local lab204-models url /home/guoheng/dvc-remotes/custom_simulation-models
```
