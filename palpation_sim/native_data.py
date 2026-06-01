from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from .config import MaterialConfig, PhantomConfig, ScanConfig
from .phantom import LumpSpec

T = TypeVar("T")


def resolve_sample_or_metadata(selector: Path, data_dir: Path | None = None) -> tuple[Path | None, Path | None]:
    """Resolve a sample selector into optional NPZ and metadata/GT JSON paths."""
    raw = selector.expanduser()
    candidates: list[Path] = []
    if raw.exists():
        candidates.append(raw.resolve())
    if not raw.is_absolute():
        roots = [Path.cwd()]
        if data_dir is not None:
            roots.append(data_dir)
        for root in roots:
            base = (root / raw).resolve()
            candidates.append(base)
            if raw.suffix == "":
                candidates.extend(
                    [
                        base.with_suffix(".npz"),
                        base.with_name(f"{base.name}_gt.json"),
                        base / "metadata.json",
                    ]
                )

    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix.lower() == ".npz":
            return candidate, _metadata_for_npz(candidate)
        if candidate.suffix.lower() == ".json":
            sample = _sample_for_metadata(candidate)
            return sample, candidate
        if candidate.is_dir():
            metadata = candidate / "metadata.json"
            samples = sorted(candidate.glob("*.npz"))
            return (samples[0] if samples else None), (metadata if metadata.exists() else None)

    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not resolve '{selector}'. Searched:\n  {searched}")


def load_sample_arrays(path: Path, keys: set[str] | None = None) -> dict[str, np.ndarray]:
    """Load selected arrays from an NPZ sample.

    The current pipeline writes compressed NPZ files, so NumPy cannot memory-map
    these arrays. Callers should pass ``keys`` when inspecting large samples.
    """
    with np.load(path, allow_pickle=False) as data:
        selected = data.files if keys is None else [key for key in data.files if key in keys]
        return {key: data[key] for key in selected}


def load_metadata(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_phantom_scan_material_lumps(
    *,
    sample_path: Path | None = None,
    metadata_path: Path | None = None,
) -> tuple[PhantomConfig, ScanConfig, MaterialConfig, list[LumpSpec], dict[str, Any]]:
    """Load core simulation metadata from a sample NPZ and/or sidecar JSON."""
    metadata = load_metadata(metadata_path)
    sample_meta: dict[str, Any] = {}
    if sample_path is not None and sample_path.exists():
        keys = {"phantom_json", "scan_json", "material_json", "lumps_json"}
        sample = load_sample_arrays(sample_path, keys)
        sample_meta = {
            "phantom": _json_from_array(sample.get("phantom_json")),
            "scan": _json_from_array(sample.get("scan_json")),
            "material": _json_from_array(sample.get("material_json")),
            "lumps": _json_from_array(sample.get("lumps_json")),
        }

    phantom_data = _first_mapping(sample_meta.get("phantom"), metadata.get("phantom"))
    scan_data = _first_mapping(sample_meta.get("scan"), metadata.get("scan"))
    material_data = _first_mapping(sample_meta.get("material"), metadata.get("material"))
    lump_records = _first_sequence(sample_meta.get("lumps"), metadata.get("lumps"))

    phantom = _dataclass_from_mapping(PhantomConfig, phantom_data)
    scan = _dataclass_from_mapping(ScanConfig, scan_data)
    material = _dataclass_from_mapping(MaterialConfig, material_data)
    lumps = [_lump_from_mapping(record) for record in lump_records]
    return phantom, scan, material, lumps, metadata


def sample_id_from_paths(sample_path: Path | None, metadata_path: Path | None) -> str:
    if sample_path is not None:
        return sample_path.stem
    if metadata_path is not None:
        return metadata_path.stem
    return "sample"


def _metadata_for_npz(path: Path) -> Path | None:
    gt_path = path.with_name(f"{path.stem}_gt.json")
    if gt_path.exists():
        return gt_path
    metadata_path = path.with_name("metadata.json")
    if metadata_path.exists():
        return metadata_path
    return None


def _sample_for_metadata(path: Path) -> Path | None:
    files = load_metadata(path).get("files", {})
    if isinstance(files, dict) and files.get("npz"):
        candidate = path.parent / str(files["npz"])
        if candidate.exists():
            return candidate.resolve()
    sample_name = path.name.replace("_gt.json", ".npz")
    candidate = path.with_name(sample_name)
    if candidate.exists():
        return candidate.resolve()
    samples = sorted(path.parent.glob("*.npz"))
    return samples[0].resolve() if samples else None


def _json_from_array(value: np.ndarray | None) -> Any:
    if value is None:
        return None
    raw = value.item() if value.shape == () else value.tolist()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def _first_mapping(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def _first_sequence(*values: Any) -> list[dict[str, Any]]:
    for value in values:
        if isinstance(value, list):
            return [record for record in value if isinstance(record, dict)]
    return []


def _dataclass_from_mapping(cls: type[T], data: dict[str, Any]) -> T:
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: data[key] for key in allowed if key in data})


def _lump_from_mapping(record: dict[str, Any]) -> LumpSpec:
    return LumpSpec(
        shape=str(record["shape"]),  # type: ignore[arg-type]
        center=tuple(float(value) for value in record["center"]),  # type: ignore[arg-type]
        radii=tuple(float(value) for value in record["radii"]),  # type: ignore[arg-type]
        stiffness_multiplier=float(record["stiffness_multiplier"]),
        yaw=float(record.get("yaw", 0.0)),
    )
