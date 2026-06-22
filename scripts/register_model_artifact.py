from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path("models")


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a selected checkpoint into the DVC-managed model registry.")
    parser.add_argument("--source", type=Path, required=True, help="Checkpoint path relative to the repo root.")
    parser.add_argument("--name", required=True, help="Stable model artifact name under models/.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory that produced the checkpoint.")
    parser.add_argument("--description", default="", help="Short human-readable model note.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned registration without copying files.")
    parser.add_argument("--skip-dvc-add", action="store_true", help="Copy metadata only; do not call dvc add.")
    args = parser.parse_args()

    source = checked_repo_path(args.source)
    model_dir = PROJECT_ROOT / MODEL_ROOT / args.name
    destination = model_dir / source.name
    if destination.exists():
        raise SystemExit(f"Destination already exists: {destination.relative_to(PROJECT_ROOT)}")

    metadata = build_metadata(args, source, destination)
    if args.dry_run:
        print(json.dumps(metadata, indent=2, sort_keys=True))
        print(f"would copy {source.relative_to(PROJECT_ROOT)} -> {destination.relative_to(PROJECT_ROOT)}")
        return

    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"registered {destination.relative_to(PROJECT_ROOT)}")

    if not args.skip_dvc_add:
        ensure_dvc()
        dvc_target = destination.relative_to(PROJECT_ROOT).as_posix()
        subprocess.run(["dvc", "add", dvc_target], cwd=PROJECT_ROOT, check=True)
        print(f"DVC pointer ready for {dvc_target}.dvc")


def checked_repo_path(path: Path) -> Path:
    if path.is_absolute():
        raise SystemExit(f"Use a repo-relative path, not an absolute path: {path}")
    full = (PROJECT_ROOT / path).resolve()
    try:
        full.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise SystemExit(f"Path escapes the repo: {path}") from exc
    if not full.is_file():
        raise SystemExit(f"Checkpoint does not exist: {path}")
    return full


def build_metadata(args: argparse.Namespace, source: Path, destination: Path) -> dict[str, Any]:
    run_dir = None if args.run_dir is None else checked_run_dir(args.run_dir)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "name": args.name,
        "description": args.description,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": source.relative_to(PROJECT_ROOT).as_posix(),
        "model_file": destination.relative_to(PROJECT_ROOT).as_posix(),
        "git": git_state(),
    }
    if run_dir is not None:
        metadata["run_dir"] = run_dir.relative_to(PROJECT_ROOT).as_posix()
        for filename in ("metrics_summary.json", "run_config.json", "leaderboard_row.json"):
            candidate = run_dir / filename
            if candidate.exists():
                metadata[filename.removesuffix(".json")] = json.loads(candidate.read_text(encoding="utf-8"))
    return metadata


def checked_run_dir(path: Path) -> Path:
    if path.is_absolute():
        raise SystemExit(f"Use a repo-relative run directory: {path}")
    full = (PROJECT_ROOT / path).resolve()
    try:
        full.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise SystemExit(f"Run directory escapes the repo: {path}") from exc
    if not full.is_dir():
        raise SystemExit(f"Run directory does not exist: {path}")
    return full


def git_state() -> dict[str, Any]:
    def run(*cmd: str) -> str:
        return subprocess.check_output(cmd, cwd=PROJECT_ROOT, text=True).strip()

    try:
        return {
            "available": True,
            "commit": run("git", "rev-parse", "HEAD"),
            "branch": run("git", "branch", "--show-current"),
            "dirty": bool(run("git", "status", "--porcelain")),
        }
    except Exception:
        return {"available": False}


def ensure_dvc() -> None:
    if shutil.which("dvc") is None:
        raise SystemExit('DVC is not installed. Install it with: python -m pip install "dvc[ssh]"')


if __name__ == "__main__":
    main()
