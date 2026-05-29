#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIEWER_DIR = PROJECT_ROOT / "tools"


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the 3D phantom viewer for a selected generated phantom.")
    parser.add_argument(
        "phantom",
        type=Path,
        help=(
            "Phantom selector: a *_phantom.gltf path, *_gt.json path, .npz sample path, "
            "or sample stem such as sample_0001."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/acceptance_deep_phantoms/train"),
        help="Directory used when the selector is a sample stem. Default: data/acceptance_deep_phantoms/train",
    )
    parser.add_argument("--viewer-dir", type=Path, default=DEFAULT_VIEWER_DIR, help="Directory containing phantom_viewer.html.")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host.")
    parser.add_argument("--port", type=int, default=8765, help="Preferred HTTP port.")
    parser.add_argument("--no-open", action="store_true", help="Print the URL without opening a browser.")
    args = parser.parse_args()

    gltf_path = resolve_phantom_gltf(args.phantom, args.data_dir)
    viewer_dir = args.viewer_dir.resolve()
    viewer_html = viewer_dir / "phantom_viewer.html"
    if not viewer_html.exists():
        raise SystemExit(f"Viewer HTML not found: {viewer_html}")

    model_ref = expose_model_to_viewer(gltf_path, viewer_dir)
    port = ensure_viewer_server(viewer_dir, args.host, args.port)
    url = f"http://{args.host}:{port}/phantom_viewer.html?model={urllib.parse.quote(model_ref)}"

    print(f"phantom: {gltf_path}")
    print(f"url: {url}")
    if not args.no_open:
        webbrowser.open(url)


def resolve_phantom_gltf(selector: Path, data_dir: Path) -> Path:
    selector_text = str(selector)
    candidates: list[Path] = []
    raw = Path(selector_text).expanduser()
    if raw.exists():
        candidates.append(raw)
    if not raw.is_absolute():
        candidates.append((PROJECT_ROOT / raw).resolve())
        candidates.append((PROJECT_ROOT / data_dir / raw).resolve())
        if raw.suffix == "":
            candidates.append((PROJECT_ROOT / data_dir / f"{raw.name}_phantom.gltf").resolve())
            candidates.append((PROJECT_ROOT / data_dir / f"{raw.name}_gt.json").resolve())
            candidates.append((PROJECT_ROOT / data_dir / f"{raw.name}.npz").resolve())

    for candidate in candidates:
        if candidate.exists():
            return _gltf_from_existing_path(candidate)

    searched = "\n  ".join(str(path) for path in candidates)
    raise SystemExit(f"Could not resolve phantom selector '{selector_text}'. Searched:\n  {searched}")


def _gltf_from_existing_path(path: Path) -> Path:
    path = path.resolve()
    if path.suffix.lower() == ".gltf":
        return path
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        files = data.get("files", {})
        if not isinstance(files, dict) or not files.get("phantom_3d"):
            raise SystemExit(f"GT JSON does not contain files.phantom_3d: {path}")
        gltf_path = (path.parent / str(files["phantom_3d"])).resolve()
        if not gltf_path.exists():
            raise SystemExit(f"Referenced glTF does not exist: {gltf_path}")
        return gltf_path
    if path.suffix.lower() == ".npz":
        gltf_path = path.with_name(f"{path.stem}_phantom.gltf")
        if not gltf_path.exists():
            raise SystemExit(f"Expected glTF next to sample NPZ: {gltf_path}")
        return gltf_path.resolve()
    raise SystemExit(f"Unsupported phantom selector path: {path}")


def expose_model_to_viewer(gltf_path: Path, viewer_dir: Path) -> str:
    model_dir = viewer_dir / "_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(gltf_path.resolve()).encode("utf-8")).hexdigest()[:10]
    link_path = model_dir / f"{gltf_path.stem}_{digest}.gltf"
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    try:
        link_path.symlink_to(gltf_path.resolve())
    except OSError:
        shutil.copy2(gltf_path, link_path)
    return f"_models/{link_path.name}"


def ensure_viewer_server(viewer_dir: Path, host: str, preferred_port: int) -> int:
    if viewer_is_available(host, preferred_port):
        return preferred_port
    if port_is_free(host, preferred_port):
        start_server(viewer_dir, host, preferred_port)
        wait_for_viewer(host, preferred_port)
        return preferred_port

    for port in range(preferred_port + 1, preferred_port + 51):
        if port_is_free(host, port):
            start_server(viewer_dir, host, port)
            wait_for_viewer(host, port)
            return port
    raise SystemExit(f"Could not find a free viewer port near {preferred_port}.")


def viewer_is_available(host: str, port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/phantom_viewer.html", timeout=0.4) as response:
            return response.status == 200
    except Exception:
        return False


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) != 0


def start_server(viewer_dir: Path, host: str, port: int) -> None:
    log_path = viewer_dir / f".phantom_viewer_server_{port}.log"
    log_file = log_path.open("ab")
    subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", host],
        cwd=viewer_dir,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def wait_for_viewer(host: str, port: int) -> None:
    for _ in range(30):
        if viewer_is_available(host, port):
            return
        time.sleep(0.1)
    raise SystemExit(f"Viewer server did not start on http://{host}:{port}")


if __name__ == "__main__":
    main()
