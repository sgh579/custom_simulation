#!/usr/bin/env python3
"""Build self-contained 1000/400/400 synthetic-data packages by depth mode."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_SPLIT_PACKAGE = Path(
    "runs/dataset_packages/"
    "palpation_random_shapes_20x_1800pool_randomsplit_1000train_400val_400test_seed20260617"
)
DEFAULT_TRAJECTORY_PACKAGE = Path(
    "runs/dataset_packages/"
    "palpation_nonlinear_trajectory_20x_100p40p40_repeats10_seed20260618"
)
DEFAULT_OUT_ROOT = Path(
    "data/"
    "palpation_synthetic_depth_modes_1000train_400val_400test_20260622"
)

SPLITS = ("train", "val", "test")
SPLIT_COUNTS = {"train": 1000, "val": 400, "test": 400}
LIMITED_TRAJECTORY_SEED = 20260618
TRAJECTORY_INPUT_STEPS = 20


@dataclass(frozen=True)
class SplitRecord:
    split: str
    split_index: int
    sample: str
    source_npz: Path
    source_gt_json: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-package", type=Path, default=DEFAULT_SPLIT_PACKAGE)
    parser.add_argument("--trajectory-package", type=Path, default=DEFAULT_TRAJECTORY_PACKAGE)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--cleanup-clutter",
        action="store_true",
        help="Remove superseded source/smoke dataset records after the new package is complete.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = args.out_root
    if out_root.exists():
        if not args.replace:
            raise FileExistsError(f"{out_root} already exists; pass --replace to rebuild it")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    records = load_split_manifest(args.split_package / "manifest.csv")
    validate_split_records(records)

    mode_summaries: list[dict[str, object]] = []
    global_manifest_rows: list[dict[str, object]] = []

    fixed_summary, fixed_rows = build_fixed_depth_package(out_root / "fixed_depth", records, args.split_package)
    mode_summaries.append(fixed_summary)
    global_manifest_rows.extend(fixed_rows)

    random_summary, random_rows = build_random_depth_package(out_root / "random_depth", records, args.split_package)
    mode_summaries.append(random_summary)
    global_manifest_rows.extend(random_rows)

    traj_summary, traj_rows = build_random_trajectory_package(out_root / "random_trajectory", args.trajectory_package)
    mode_summaries.append(traj_summary)
    global_manifest_rows.extend(traj_rows)

    cleanup_rows: list[dict[str, str]] = []
    if args.cleanup_clutter:
        cleanup_rows = cleanup_clutter(out_root)
    write_cleanup_manifest(out_root, cleanup_rows, enabled=args.cleanup_clutter)

    write_global_manifest(out_root / "manifest.csv", global_manifest_rows)
    write_root_summary(out_root, mode_summaries, cleanup_rows, args)
    write_root_readme(out_root, mode_summaries)
    write_checksums(out_root)
    print(f"packaged synthetic depth modes: {out_root}")


def load_split_manifest(path: Path) -> list[SplitRecord]:
    if not path.exists():
        raise FileNotFoundError(path)
    records: list[SplitRecord] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            records.append(
                SplitRecord(
                    split=row["split"],
                    split_index=int(row["split_index"]),
                    sample=row["sample"],
                    source_npz=Path(row["source_path"]),
                    source_gt_json=Path(row["metadata_path"]),
                )
            )
    return records


def validate_split_records(records: list[SplitRecord]) -> None:
    counts = {split: 0 for split in SPLITS}
    for record in records:
        if record.split not in counts:
            raise ValueError(f"Unexpected split {record.split!r}")
        counts[record.split] += 1
        if not record.source_npz.exists():
            raise FileNotFoundError(record.source_npz)
        if not record.source_gt_json.exists():
            raise FileNotFoundError(record.source_gt_json)
    if counts != SPLIT_COUNTS:
        raise ValueError(f"Expected split counts {SPLIT_COUNTS}, got {counts}")


def build_fixed_depth_package(package_dir: Path, records: list[SplitRecord], source_package: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    package_dir.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for record in records:
        split_dir = package_dir / "data" / record.split
        split_dir.mkdir(parents=True, exist_ok=True)
        target_npz = split_dir / record.sample
        target_gt = split_dir / gt_name(record.sample)
        shutil.copy2(record.source_npz, target_npz)
        shutil.copy2(record.source_gt_json, target_gt)
        target_viz = copy_optional_viz(record.source_npz, split_dir / viz_name(record.sample))
        rows.append(manifest_row("fixed_depth", record, target_npz, target_gt, target_viz, derived=False))
    summary = {
        "mode": "fixed_depth",
        "description": "Original fixed maximum-depth straight presses copied as self-contained raw samples.",
        "split_counts": dict(SPLIT_COUNTS),
        "source_split_package": str(source_package),
        "sample_contract": "Raw .npz with indentation_depth/fz/presses/probe_pose on a 20x20 scan grid and 20 press steps.",
    }
    write_mode_files(package_dir, summary, rows)
    return summary, rows


def build_random_depth_package(package_dir: Path, records: list[SplitRecord], source_package: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    package_dir.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for record in records:
        split_dir = package_dir / "data" / record.split
        split_dir.mkdir(parents=True, exist_ok=True)
        target_npz = split_dir / record.sample
        target_gt = split_dir / gt_name(record.sample)
        target_viz = split_dir / viz_name(record.sample)
        seed = stable_sample_seed(LIMITED_TRAJECTORY_SEED, record.split, record.sample)
        derive_random_depth_npz(record.source_npz, target_npz, seed=seed)
        shutil.copy2(record.source_gt_json, target_gt)
        write_random_depth_viz(target_viz, target_npz)
        rows.append(manifest_row("random_depth", record, target_npz, target_gt, target_viz, derived=True))
    summary = {
        "mode": "random_depth",
        "description": "Raw samples derived from the fixed-depth split by deterministic per-scan-point random endpoint depth resampling.",
        "split_counts": dict(SPLIT_COUNTS),
        "source_split_package": str(source_package),
        "limited_trajectory_seed": LIMITED_TRAJECTORY_SEED,
        "trajectory_input_steps": TRAJECTORY_INPUT_STEPS,
        "random_endpoint_policy": "For each sample and scan point, choose endpoint_idx in [1, source_steps-1], then linearly resample 20 points from step 0 to that endpoint.",
        "sample_contract": "Raw .npz with updated indentation_depth/fz/presses/probe_pose/contact_features plus random_depth_endpoint_index.",
    }
    write_mode_files(package_dir, summary, rows)
    return summary, rows


def build_random_trajectory_package(package_dir: Path, source_package: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    source_data = source_package / "data"
    if not source_data.exists():
        raise FileNotFoundError(source_data)
    package_dir.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for split in SPLITS:
        source_split = source_data / split
        split_dir = package_dir / "data" / split
        split_dir.mkdir(parents=True, exist_ok=True)
        source_files = sorted(source_split.glob("*.npz"))
        if len(source_files) != SPLIT_COUNTS[split]:
            raise ValueError(f"{source_split} has {len(source_files)} npz files, expected {SPLIT_COUNTS[split]}")
        for idx, source_npz in enumerate(source_files):
            source_gt = source_npz.with_name(f"{source_npz.stem}_gt.json")
            if not source_gt.exists():
                raise FileNotFoundError(source_gt)
            sample = source_npz.name
            target_npz = split_dir / sample
            target_gt = split_dir / gt_name(sample)
            shutil.copy2(source_npz, target_npz)
            shutil.copy2(source_gt, target_gt)
            target_viz = copy_optional_viz(source_npz, split_dir / viz_name(sample))
            record = SplitRecord(split=split, split_index=idx, sample=sample, source_npz=source_npz, source_gt_json=source_gt)
            rows.append(manifest_row("random_trajectory", record, target_npz, target_gt, target_viz, derived=False))
    metadata = source_data / "metadata.json"
    if metadata.exists():
        shutil.copy2(metadata, package_dir / "source_metadata.json")
    summary = {
        "mode": "random_trajectory",
        "description": "Original nonlinear/random-trajectory raw samples with 6D probe_wrench and trajectory_xy_offset.",
        "split_counts": dict(SPLIT_COUNTS),
        "source_package": str(source_package),
        "sample_contract": "Raw .npz with indentation_depth/fz/probe_wrench/trajectory_xy_offset on a 20x20 scan grid and 20 press steps.",
    }
    write_mode_files(package_dir, summary, rows)
    return summary, rows


def derive_random_depth_npz(source_npz: Path, target_npz: Path, *, seed: int) -> None:
    with np.load(source_npz, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    fz = np.asarray(arrays["fz"], dtype=np.float32)
    depth = np.asarray(arrays["indentation_depth"], dtype=np.float32)
    if fz.shape != depth.shape or fz.ndim != 3:
        raise ValueError(f"Expected fz/depth [H,W,T], got {fz.shape} and {depth.shape}")
    endpoint_idx, lo, hi, alpha = limited_positions(depth.shape, seed=seed)
    for key in ("indentation_depth", "fz"):
        arrays[key] = resample_hwt(np.asarray(arrays[key], dtype=np.float32), lo, hi, alpha)
    for key in ("presses", "probe_pose", "contact_features", "probe_force", "probe_torque", "probe_wrench"):
        if key in arrays:
            arrays[key] = resample_hwt(np.asarray(arrays[key], dtype=np.float32), lo, hi, alpha)
    if "presses" in arrays:
        presses = np.asarray(arrays["presses"], dtype=np.float32).copy()
        if presses.shape[:3] == arrays["fz"].shape and presses.shape[-1] >= 2:
            presses[..., 0] = arrays["indentation_depth"]
            presses[..., 1] = arrays["fz"]
            arrays["presses"] = presses
    arrays["nonlinearity_ratio"] = nonlinearity_ratio_map(arrays["indentation_depth"], arrays["fz"])
    arrays["random_depth_endpoint_index"] = endpoint_idx.astype(np.int16)
    arrays["depth_policy_json"] = np.asarray(
        json.dumps(
            {
                "mode": "random_depth",
                "source_sample": str(source_npz),
                "limited_trajectory_seed": LIMITED_TRAJECTORY_SEED,
                "sample_seed": int(seed),
                "trajectory_input_steps": TRAJECTORY_INPUT_STEPS,
                "endpoint_index_policy": "integers(1, source_steps, endpoint=False) per scan point",
            },
            sort_keys=True,
        )
    )
    np.savez_compressed(target_npz, **arrays)


def limited_positions(shape: tuple[int, int, int], *, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w, source_steps = shape
    rng = np.random.default_rng(seed)
    endpoints = rng.integers(1, source_steps, size=(h, w), endpoint=False).astype(np.int64)
    fractions = np.linspace(0.0, 1.0, TRAJECTORY_INPUT_STEPS, dtype=np.float32)
    positions = endpoints[None, :, :].astype(np.float32) * fractions[:, None, None]
    lo = np.floor(positions).astype(np.int64)
    hi = np.minimum(lo + 1, source_steps - 1)
    alpha = (positions - lo.astype(np.float32)).astype(np.float32)
    return endpoints, lo, hi, alpha


def resample_hwt(array: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    if array.ndim < 3:
        return array
    if array.shape[:3] != (lo.shape[1], lo.shape[2], int(np.max(hi)) + 1):
        if array.shape[0] == lo.shape[1] and array.shape[1] == lo.shape[2] and array.shape[2] >= int(np.max(hi)) + 1:
            pass
        else:
            return array
    transposed = np.moveaxis(array, 2, 0)
    rows = np.arange(lo.shape[1])[None, :, None]
    cols = np.arange(lo.shape[2])[None, None, :]
    low = transposed[lo, rows, cols]
    high = transposed[hi, rows, cols]
    while alpha.ndim < low.ndim:
        alpha = alpha[..., None]
    out = low + alpha * (high - low)
    return np.moveaxis(np.nan_to_num(out).astype(np.float32), 0, 2)


def nonlinearity_ratio_map(depth: np.ndarray, fz: np.ndarray) -> np.ndarray:
    h, w, _t = depth.shape
    out = np.zeros((h, w), dtype=np.float32)
    for row in range(h):
        for col in range(w):
            early = segment_slope(depth[row, col], fz[row, col], 0.10, 0.35)
            late = segment_slope(depth[row, col], fz[row, col], 0.65, 0.90)
            if np.isfinite(early) and abs(early) > 1.0e-9 and np.isfinite(late):
                out[row, col] = np.float32(late / early)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def segment_slope(depth: np.ndarray, force: np.ndarray, lo: float, hi: float) -> float:
    z = np.asarray(depth, dtype=np.float64)
    f = np.asarray(force, dtype=np.float64)
    span = float(np.max(z) - np.min(z))
    if span <= 1.0e-12:
        return 0.0
    keep = (z >= float(np.min(z)) + lo * span) & (z <= float(np.min(z)) + hi * span)
    if int(keep.sum()) < 2:
        return 0.0
    coeffs = np.polyfit(z[keep], f[keep], deg=1)
    return float(coeffs[0])


def stable_sample_seed(base_seed: int, split_name: str, sample_name: str) -> int:
    payload = f"{int(base_seed)}|{split_name}|{sample_name}".encode("utf-8")
    digest = hashlib.blake2s(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**32)


def manifest_row(
    mode: str,
    record: SplitRecord,
    target_npz: Path,
    target_gt: Path,
    target_viz: Path | None,
    *,
    derived: bool,
) -> dict[str, object]:
    return {
        "mode": mode,
        "split": record.split,
        "split_index": record.split_index,
        "sample": target_npz.name,
        "npz_path": str(target_npz),
        "gt_json_path": str(target_gt),
        "visualization_command_path": "" if target_viz is None else str(target_viz),
        "source_npz": str(record.source_npz),
        "source_gt_json": str(record.source_gt_json),
        "derived": str(bool(derived)).lower(),
    }


def copy_optional_viz(source_npz: Path, target_viz: Path) -> Path | None:
    source_viz = source_npz.with_name(f"{source_npz.stem}_visualization_command.md")
    if not source_viz.exists():
        return None
    shutil.copy2(source_viz, target_viz)
    return target_viz


def write_random_depth_viz(path: Path, sample_npz: Path) -> None:
    path.write_text(
        "# Random-depth sample\n\n"
        f"Derived raw sample: `{sample_npz.name}`\n\n"
        "The indentation/Fz time series was deterministically resampled to a random endpoint depth per scan point.\n",
        encoding="utf-8",
    )


def gt_name(sample_name: str) -> str:
    return f"{Path(sample_name).stem}_gt.json"


def viz_name(sample_name: str) -> str:
    return f"{Path(sample_name).stem}_visualization_command.md"


def write_mode_files(package_dir: Path, summary: dict[str, object], rows: list[dict[str, object]]) -> None:
    write_global_manifest(package_dir / "manifest.csv", rows)
    (package_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (package_dir / "README.md").write_text(mode_readme(summary), encoding="utf-8")


def mode_readme(summary: dict[str, object]) -> str:
    mode = str(summary["mode"])
    lines = [
        f"# {mode}",
        "",
        str(summary["description"]),
        "",
        "- train: 1000",
        "- val: 400",
        "- test: 400",
        "",
        f"Sample contract: {summary['sample_contract']}",
        "",
    ]
    return "\n".join(lines)


def write_global_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_root_summary(out_root: Path, mode_summaries: list[dict[str, object]], cleanup_rows: list[dict[str, str]], args: argparse.Namespace) -> None:
    summary = {
        "package_name": out_root.name,
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "split_counts_per_mode": dict(SPLIT_COUNTS),
        "modes": [item["mode"] for item in mode_summaries],
        "mode_summaries": mode_summaries,
        "source_split_package": str(args.split_package),
        "source_random_trajectory_package": str(args.trajectory_package),
        "cleanup_removed_paths": len([row for row in cleanup_rows if row.get("status") == "removed"]),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_root_readme(out_root: Path, mode_summaries: list[dict[str, object]]) -> None:
    lines = [
        "# Synthetic Depth Modes 1000/400/400",
        "",
        "Self-contained raw synthetic-data package organized by depth mode.",
        "",
        "Modes:",
    ]
    for summary in mode_summaries:
        lines.append(f"- `{summary['mode']}`: {summary['description']}")
    lines.extend(
        [
            "",
            "Each mode contains `data/train`, `data/val`, and `data/test` with 1000/400/400 `.npz` samples and matching `_gt.json` files.",
            "`random_depth` is derived deterministically from the fixed-depth split using the recorded seed and endpoint policy.",
            "",
        ]
    )
    (out_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_cleanup_manifest(out_root: Path, rows: list[dict[str, str]], *, enabled: bool) -> None:
    path = out_root / "cleanup_removed_paths.csv"
    fieldnames = ["path", "kind", "status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if enabled:
            writer.writerows(rows)


def write_checksums(root: Path) -> None:
    checksum_path = root / "checksums_sha256.txt"
    with checksum_path.open("w", encoding="utf-8") as handle:
        for path in sorted(iter_package_files(root)):
            rel = path.relative_to(root).as_posix()
            digest = sha256_file(path)
            handle.write(f"{digest}  {rel}\n")


def iter_package_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.name != "checksums_sha256.txt":
            yield path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cleanup_clutter(out_root: Path) -> list[dict[str, str]]:
    clutter = [
        Path("data/palpation_random_shape_20x_pipeline_smoke"),
        Path("data/palpation_newton_random_shapes_20x_scan20_extra1000_seed83_launcher_smoke"),
        Path("data/palpation_newton_random_shapes_20x_scan20_combined_seed11_seed23"),
        Path("data/palpation_newton_random_shapes_20x_scan20_combined_seed11_seed23_seed37"),
        Path("data/palpation_newton_random_shapes_20x_scan20_combined_seed11_seed23_seed37_seed53"),
        Path("data/palpation_newton_random_shapes_20x_scan20_combined_seed11_seed23_seed37_seed53_seed71"),
        Path("data/palpation_newton_random_shapes_20x_scan20_combined_seed11_seed23_seed37_seed53_extra1000_seed83"),
        Path("data/palpation_newton_random_shapes_20x_scan20_v1"),
        Path("data/palpation_newton_random_shapes_20x_scan20_seed23"),
        Path("data/palpation_newton_random_shapes_20x_scan20_seed37"),
        Path("data/palpation_newton_random_shapes_20x_scan20_seed53"),
        Path("data/palpation_newton_random_shapes_20x_scan20_seed71"),
        Path("data/palpation_newton_random_shapes_20x_scan20_extra1000_seed83"),
        Path("runs/dataset_packages/palpation_random_shapes_20x_800train_80val_80test_20260616"),
        Path("runs/dataset_packages/palpation_random_shapes_20x_800train_80val_80test_20260616.tar.gz"),
        Path("runs/dataset_packages/palpation_random_shapes_20x_800train_80val_80test_20260616.tar.gz.sha256"),
        Path("runs/dataset_packages/palpation_random_shapes_20x_1800pool_randomsplit_1000train_400val_400test_seed20260617"),
        Path("runs/dataset_packages/palpation_nonlinear_trajectory_20x_100p40p40_repeats10_seed20260618"),
        Path("runs/dataset_packages/palpation_nonlinear_trajectory_20x_100p40p40_repeats10_seed20260618_smoke"),
        Path("runs/dataset_packages/nonlinear_traj_newton_interface_smoke"),
        Path("runs/dataset_packages/nonlinear_traj_wrench_analytic_smoke"),
        Path("runs/dataset_packages/nonlinear_traj_wrench_newton_smoke"),
    ]
    rows: list[dict[str, str]] = []
    resolved_out = out_root.resolve()
    for path in clutter:
        if not path.exists() and not path.is_symlink():
            rows.append({"path": str(path), "kind": "missing", "status": "skipped"})
            continue
        if resolved_out == path.resolve() or resolved_out in path.resolve().parents:
            rows.append({"path": str(path), "kind": "protected", "status": "skipped"})
            continue
        kind = "dir" if path.is_dir() else "file"
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
        rows.append({"path": str(path), "kind": kind, "status": "removed"})
    return rows


if __name__ == "__main__":
    main()
