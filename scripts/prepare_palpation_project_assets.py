#!/usr/bin/env python3
"""Export curated research assets to the palpation project page."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paper-root",
        type=Path,
        required=True,
        help="Research workspace containing paper-latex-editing/ and release/.",
    )
    parser.add_argument(
        "--site-root",
        type=Path,
        required=True,
        help="Destination checkout of the personal website.",
    )
    parser.add_argument(
        "--vbd-replay",
        type=Path,
        required=True,
        help="NPZ containing particle-position frames from the lab204 VBD rerun.",
    )
    return parser.parse_args()


def rounded(values: np.ndarray, digits: int) -> list:
    return np.round(np.asarray(values, dtype=np.float64), digits).tolist()


def downsample_mean(values: np.ndarray, target: int = 32) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    factor = array.shape[0] // target
    return array.reshape(target, factor, target, factor).mean(axis=(1, 3))


def downsample_binary(values: np.ndarray, target: int = 32) -> np.ndarray:
    array = np.asarray(values, dtype=np.uint8)
    factor = array.shape[0] // target
    return array.reshape(target, factor, target, factor).max(axis=(1, 3))


def highres_ground_truth(metadata: dict, target: int = 128) -> np.ndarray:
    scan = metadata["scan"]
    xs = np.linspace(
        float(scan["x_values"][0]),
        float(scan["x_values"][-1]),
        target,
        dtype=np.float32,
    )
    ys = np.linspace(
        float(scan["y_values"][0]),
        float(scan["y_values"][-1]),
        target,
        dtype=np.float32,
    )
    xv, yv = np.meshgrid(xs, ys)
    points = np.stack([xv, yv, np.zeros_like(xv)], axis=-1)
    mask = np.zeros((target, target), dtype=bool)

    for lump in metadata["lumps"]:
        center = np.asarray(lump["center"], dtype=np.float32)
        radii = np.asarray(lump["radii"], dtype=np.float32)
        relative = points - center
        yaw = -float(lump.get("yaw", 0.0))
        cosine = np.cos(yaw)
        sine = np.sin(yaw)
        x_values = relative[..., 0].copy()
        y_values = relative[..., 1].copy()
        relative[..., 0] = cosine * x_values - sine * y_values
        relative[..., 1] = sine * x_values + cosine * y_values
        relative_xy = relative[..., :2]

        if lump["shape"] == "box":
            inside = np.all(np.abs(relative_xy) <= radii[:2], axis=-1)
        elif lump["shape"] in {
            "sphere",
            "ellipsoid",
            "cylinder",
            "capsule",
        }:
            normalized = relative_xy / radii[:2]
            inside = np.sum(normalized**2, axis=-1) <= 1.0
        else:
            raise ValueError(f"Unsupported lump shape: {lump['shape']}")
        mask |= inside

    return mask.astype(np.uint8)


def dice_score(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    prediction_bool = np.asarray(prediction, dtype=bool)
    ground_truth_bool = np.asarray(ground_truth, dtype=bool)
    intersection = int(np.logical_and(prediction_bool, ground_truth_bool).sum())
    denominator = int(prediction_bool.sum() + ground_truth_bool.sum())
    return 1.0 if denominator == 0 else 2.0 * intersection / denominator


def result_directory(stem: str, figure_root: Path) -> Path:
    matches = list((figure_root / "individual").glob(f"*_{stem}"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one result directory for {stem}, got {matches}")
    return matches[0]


def build_v1_example(
    item: dict[str, object],
    *,
    rank: int,
    stiffness_threshold: float,
    curve_threshold: float,
    figure_root: Path,
) -> dict[str, object]:
    stem = str(item["stem"])
    metadata = json.loads(
        (figure_root / "data" / f"{stem}_gt.json").read_text(encoding="utf-8")
    )
    result_root = result_directory(stem, figure_root)
    stiffness_map = np.load(
        result_root / "equivalent_stiffness_map_20x20_N_per_m.npy",
        allow_pickle=False,
    )
    stiffness_probability = np.load(
        result_root / "stiffness_unet_probability_128.npy",
        allow_pickle=False,
    )
    curve_probability = np.load(
        result_root / "curve_unet_probability_128.npy",
        allow_pickle=False,
    )
    ground_truth = highres_ground_truth(metadata)

    stiffness_dice = dice_score(
        stiffness_probability >= stiffness_threshold,
        ground_truth,
    )
    curve_dice = dice_score(
        curve_probability >= curve_threshold,
        ground_truth,
    )
    expected_stiffness = float(item["stiffness_dice"])
    expected_curve = float(item["curve_dice"])
    if not np.isclose(stiffness_dice, expected_stiffness, atol=1.0e-12):
        raise RuntimeError(
            f"{stem}: stiffness Dice {stiffness_dice} != {expected_stiffness}"
        )
    if not np.isclose(curve_dice, expected_curve, atol=1.0e-12):
        raise RuntimeError(f"{stem}: curve Dice {curve_dice} != {expected_curve}")

    return {
        "rank": rank,
        "sampleId": stem,
        "basePhantomId": str(item["base_phantom_id"]),
        "numLumps": int(item["num_lumps"]),
        "shapes": [str(shape) for shape in item["shapes"]],
        "stiffnessNPerM": rounded(stiffness_map, 1),
        "stiffnessProbability": rounded(
            downsample_mean(stiffness_probability),
            4,
        ),
        "proposedProbability": rounded(
            downsample_mean(curve_probability),
            4,
        ),
        "groundTruth": downsample_binary(ground_truth).tolist(),
    }


def unique_edges(tets: np.ndarray) -> np.ndarray:
    pairs = np.asarray(
        [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
        ],
        dtype=np.int64,
    )
    edges = np.asarray(tets, dtype=np.int64)[:, pairs].reshape(-1, 2)
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def compact_edges(edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices, inverse = np.unique(edges.reshape(-1), return_inverse=True)
    return vertices, inverse.reshape(-1, 2)


def build_mesh_replay(vbd_path: Path) -> dict[str, object]:
    with np.load(vbd_path, allow_pickle=False) as replay:
        positions = np.asarray(replay["positions"], dtype=np.float32)
        base_vertices = np.asarray(replay["mesh_vertices"], dtype=np.float32)
        tets = np.asarray(replay["mesh_tets"], dtype=np.int64)
        tet_lump_id = np.asarray(replay["tet_lump_id"], dtype=np.int32)
        probe_positions = np.asarray(replay["probe_positions"], dtype=np.float32)
        depth = np.asarray(replay["indentation_depth"], dtype=np.float32)
        force = np.asarray(replay["recorded_fz"], dtype=np.float32)
        row = int(replay["press_row"])
        col = int(replay["press_col"])

    probe_xy = probe_positions[-1, :2]
    centroids = base_vertices[tets].mean(axis=1)
    radial = np.linalg.norm(centroids[:, :2] - probe_xy[None, :], axis=1)
    cell_y = 0.18 / 48.0
    local = radial <= 0.040
    cross_section = np.abs(centroids[:, 1] - probe_xy[1]) <= 0.85 * cell_y
    top_layer = centroids[:, 2] >= 0.074
    inclusion = tet_lump_id >= 0

    tissue_ids = np.flatnonzero(local & (cross_section | top_layer) & ~inclusion)
    inclusion_ids = np.flatnonzero(local & inclusion)
    tissue_edges_global = unique_edges(tets[tissue_ids])
    inclusion_edges_global = unique_edges(tets[inclusion_ids])
    all_edges = np.vstack((tissue_edges_global, inclusion_edges_global))
    selected_vertices, compact_all_edges = compact_edges(all_edges)
    tissue_count = tissue_edges_global.shape[0]

    frames_mm = positions[:, selected_vertices] * 1000.0
    probe_mm = probe_positions * 1000.0
    center_mm = np.asarray(
        [probe_xy[0] * 1000.0, probe_xy[1] * 1000.0, 40.0],
        dtype=np.float32,
    )
    frames_mm -= center_mm[None, None, :]
    probe_mm -= center_mm[None, :]

    return {
        "kind": "Newton/VBD particle-position rerun",
        "press": {"row": row, "col": col},
        "fullMesh": {
            "vertices": int(base_vertices.shape[0]),
            "tetrahedra": int(tets.shape[0]),
        },
        "displayMesh": {
            "vertices": int(selected_vertices.shape[0]),
            "tissueEdges": int(tissue_count),
            "inclusionEdges": int(compact_all_edges.shape[0] - tissue_count),
        },
        "tissueEdges": compact_all_edges[:tissue_count].reshape(-1).tolist(),
        "inclusionEdges": compact_all_edges[tissue_count:].reshape(-1).tolist(),
        "framesMm": rounded(frames_mm, 2),
        "probeMm": rounded(probe_mm, 2),
        "depthMm": rounded(depth * 1000.0, 3),
        "forceN": rounded(force, 2),
    }


def main() -> None:
    args = parse_args()
    paper_root = args.paper_root.expanduser().resolve()
    site_root = args.site_root.expanduser().resolve()
    figure_root = (
        paper_root
        / "paper-latex-editing"
        / "figures"
        / "high_gap10_curve_vs_stiffness"
    )
    sample_path = figure_root / "data" / "sample_0299.npz"
    selection_path = figure_root / "selection_manifest.json"

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    sample_metadata = json.loads(
        (figure_root / "data" / "sample_0299_gt.json").read_text(
            encoding="utf-8"
        )
    )
    stiffness_threshold = float(selection["stiffness_threshold"])
    curve_threshold = float(selection["curve_threshold"])
    examples = [
        build_v1_example(
            item,
            rank=index,
            stiffness_threshold=stiffness_threshold,
            curve_threshold=curve_threshold,
            figure_root=figure_root,
        )
        for index, item in enumerate(selection["selected"], start=1)
    ]
    selected = next(
        example for example in examples if example["sampleId"] == "sample_0299"
    )

    with np.load(sample_path, allow_pickle=False) as sample:
        fz = np.asarray(sample["fz"], dtype=np.float32)
        depth = np.asarray(sample["indentation_depth"], dtype=np.float32)
        stiffness = (
            fz[..., -1] - fz[..., 0]
        ) / np.maximum(depth[..., -1] - depth[..., 0], 1e-9)

    payload = {
        "sampleId": "sample_0299",
        "source": {
            "backend": str(sample_metadata["backend"]),
            "split": "held-out test sample",
            "scope": "simulation only",
        },
        "scan": {
            "grid": [20, 20],
            "steps": 20,
            "depthMm": rounded(depth[0, 0] * 1000.0, 3),
            "fzN": rounded(fz, 2),
            "stiffnessNPerM": rounded(stiffness, 1),
        },
        "output": {
            "curveProbability": selected["proposedProbability"],
            "stiffnessProbability": selected["stiffnessProbability"],
            "groundTruth": selected["groundTruth"],
            "curveThreshold": curve_threshold,
            "stiffnessThreshold": stiffness_threshold,
        },
        "meshReplay": build_mesh_replay(args.vbd_replay.resolve()),
    }

    data_path = site_root / "assets" / "data" / "palpation" / "sample-0299.json"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(
        json.dumps(payload, separators=(",", ":")), encoding="utf-8"
    )

    stiffness_values = np.concatenate(
        [
            np.asarray(example["stiffnessNPerM"], dtype=np.float32).reshape(-1)
            for example in examples
        ]
    )
    evidence_payload = {
        "dataset": "V1",
        "scope": "held-out simulation test split",
        "selection": {
            "stiffnessThreshold": stiffness_threshold,
            "proposedThreshold": curve_threshold,
        },
        "stiffnessRangeNPerM": rounded(
            np.percentile(stiffness_values, [2.0, 98.0]),
            1,
        ),
        "examples": examples,
    }
    evidence_path = (
        site_root
        / "assets"
        / "data"
        / "palpation"
        / "v1-high-gap-10.json"
    )
    evidence_path.write_text(
        json.dumps(evidence_payload, separators=(",", ":")),
        encoding="utf-8",
    )

    paper_path = (
        site_root / "files" / "palpation" / "is-stiffness-sufficient.pdf"
    )
    paper_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paper_root / "release" / "paper_template.pdf", paper_path)

    print(
        json.dumps(
            {
                "data": str(data_path),
                "bytes": data_path.stat().st_size,
                "evidence": str(evidence_path),
                "evidenceBytes": evidence_path.stat().st_size,
                "evidenceSamples": [
                    example["sampleId"] for example in examples
                ],
                "paper": str(paper_path),
                "displayMesh": payload["meshReplay"]["displayMesh"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
