from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_PREFIXES = (
    "data/",
    "runs/",
    "outputs/",
    "checkpoints/",
    "artifacts/transfer_packages/",
)
FORBIDDEN_SUFFIXES = (
    ".ckpt",
    ".glb",
    ".gltf",
    ".h5",
    ".hdf5",
    ".npy",
    ".npz",
    ".obj",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
    ".stl",
)
MODEL_ALLOWED_NAMES = {"metadata.json", ".gitignore", "README.md"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail if generated artifacts are tracked by Git.")
    parser.add_argument("--max-bytes", type=int, default=20 * 1024 * 1024, help="Maximum size for a Git-tracked file.")
    args = parser.parse_args()

    tracked = git_ls_files()
    errors = []
    for rel_path in tracked:
        path = PROJECT_ROOT / rel_path
        if is_forbidden_artifact_path(rel_path):
            errors.append(f"tracked generated artifact path: {rel_path}")
        if rel_path.endswith(FORBIDDEN_SUFFIXES) and not rel_path.endswith(".dvc"):
            errors.append(f"tracked binary artifact suffix: {rel_path}")
        if path.is_file() and path.stat().st_size > args.max_bytes:
            errors.append(f"tracked file exceeds {args.max_bytes} bytes: {rel_path}")

    if errors:
        print("Git artifact policy failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print(f"Git artifact policy OK ({len(tracked)} tracked paths checked).")


def git_ls_files() -> list[str]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=PROJECT_ROOT)
    return [item.decode("utf-8") for item in output.split(b"\0") if item]


def is_forbidden_artifact_path(rel_path: str) -> bool:
    if rel_path.endswith("/.gitkeep"):
        return False
    if rel_path.startswith("artifacts/manifests/"):
        return False
    if rel_path.startswith("models/"):
        name = Path(rel_path).name
        return not (name in MODEL_ALLOWED_NAMES or rel_path.endswith(".dvc"))
    return rel_path.startswith(FORBIDDEN_PREFIXES)


if __name__ == "__main__":
    main()
