from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_DIR = Path("artifacts/manifests")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create manifests and rsync selected generated artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="Write a JSON manifest for selected paths.")
    manifest.add_argument("--name", required=True, help="Stable manifest name without extension.")
    manifest.add_argument("--paths", nargs="+", required=True, help="Files or directories relative to the repo root.")
    manifest.add_argument("--out", type=Path, default=None, help="Output JSON path.")
    manifest.add_argument("--deep", action="store_true", help="Hash every file inside directories.")
    manifest.add_argument("--no-checksum", action="store_true", help="Skip SHA-256 checksums.")
    manifest.set_defaults(func=write_manifest)

    for command in ("push", "pull"):
        sync = subparsers.add_parser(command, help=f"Rsync manifest paths {'to' if command == 'push' else 'from'} a remote.")
        sync.add_argument("--manifest", type=Path, required=True)
        sync.add_argument("--remote", default="lab204", help="SSH host alias.")
        sync.add_argument("--remote-root", default="/home/guoheng/custom_simulation")
        sync.add_argument("--dry-run", action="store_true")
        sync.add_argument("--delete", action="store_true", help="Delete files absent from the source path.")
        sync.set_defaults(func=sync_manifest)

    args = parser.parse_args()
    args.func(args)


def write_manifest(args: argparse.Namespace) -> None:
    out = args.out or DEFAULT_MANIFEST_DIR / f"{args.name}.json"
    entries = [describe_path(Path(path), deep=args.deep, checksum=not args.no_checksum) for path in args.paths]
    manifest = {
        "schema_version": 1,
        "name": args.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "repo_root": str(PROJECT_ROOT),
        "git": git_state(),
        "entries": entries,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out)


def sync_manifest(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = manifest.get("entries", [])
    if not entries:
        raise SystemExit(f"No entries found in {args.manifest}")
    ensure_rsync()
    remote_root = args.remote_root.rstrip("/")
    if args.command == "push":
        subprocess.run(["ssh", args.remote, "mkdir", "-p", remote_root], check=True)
    else:
        Path(".").mkdir(exist_ok=True)

    for entry in entries:
        rel_path = entry["path"]
        if rel_path.startswith("/") or ".." in Path(rel_path).parts:
            raise SystemExit(f"Refusing unsafe manifest path: {rel_path}")
        if args.command == "push":
            source = rel_path
            destination = f"{args.remote}:{remote_root}/"
            cmd = ["rsync", "-aR"]
            if args.delete:
                cmd.append("--delete")
            if args.dry_run:
                cmd.append("--dry-run")
            cmd.extend([source, destination])
        else:
            source = f"{args.remote}:{remote_root}/./{rel_path}"
            destination = "."
            cmd = ["rsync", "-aR"]
            if args.delete:
                cmd.append("--delete")
            if args.dry_run:
                cmd.append("--dry-run")
            cmd.extend([source, destination])
        print("+ " + " ".join(shlex.quote(part) for part in cmd))
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def describe_path(rel_path: Path, *, deep: bool, checksum: bool) -> dict[str, Any]:
    if rel_path.is_absolute():
        raise SystemExit(f"Use repo-relative paths, not absolute paths: {rel_path}")
    path = PROJECT_ROOT / rel_path
    if not path.exists():
        raise SystemExit(f"Path does not exist: {rel_path}")
    if path.is_file():
        return {
            "path": rel_path.as_posix(),
            "kind": "file",
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path) if checksum else None,
        }
    files = sorted(child for child in path.rglob("*") if child.is_file())
    entry: dict[str, Any] = {
        "path": rel_path.as_posix(),
        "kind": "directory",
        "file_count": len(files),
        "size_bytes": sum(child.stat().st_size for child in files),
    }
    if deep and checksum:
        entry["files"] = [
            {
                "path": child.relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": child.stat().st_size,
                "sha256": sha256(child),
            }
            for child in files
        ]
    return entry


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> dict[str, Any]:
    def run(*cmd: str) -> str:
        return subprocess.check_output(cmd, cwd=PROJECT_ROOT, text=True).strip()

    try:
        commit = run("git", "rev-parse", "HEAD")
        branch = run("git", "branch", "--show-current")
        dirty = bool(run("git", "status", "--porcelain"))
    except Exception:
        return {"available": False}
    return {"available": True, "commit": commit, "branch": branch, "dirty": dirty}


def ensure_rsync() -> None:
    try:
        subprocess.run(["rsync", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception as exc:
        raise SystemExit("rsync is required for artifact sync.") from exc


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
