from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from palpation_sim.native_data import load_phantom_scan_material_lumps  # noqa: E402
from palpation_sim.config import PhantomConfig, ScanConfig  # noqa: E402


DEFAULT_RUN_DIR = Path(
    "runs/nonlinear_trajectory_20x_repeats10_seed20260618/"
    "highres_fz_temporal_variants/r128_fz_features_aug_focal_unet"
)
DEFAULT_DATA_ROOT = Path(
    "data/palpation_synthetic_depth_modes_1000train_400val_400test_20260622/"
    "random_trajectory/data"
)
DEFAULT_OUT_DIR = Path(
    "runs/projection_prior_pilot_r128_fz_features_aug_focal_20260625"
)
DEFAULT_THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
DEFAULT_SCALES = (0.8, 0.95, 1.1, 1.25)
FUSION_ALPHAS = tuple(round(v, 2) for v in np.linspace(0.0, 1.0, 11))
FUSION_THRESHOLDS = tuple(round(v, 3) for v in np.linspace(0.1, 0.9, 33))


@dataclass(frozen=True)
class Primitive2D:
    shape: str
    cx: float
    cy: float
    cz: float
    rx: float
    ry: float
    rz: float
    yaw: float
    stiffness_multiplier: float
    source: str


@dataclass
class ProjectionResult:
    mask: np.ndarray
    objective: float
    primitives: list[Primitive2D]
    num_candidates: int


@dataclass
class PrimitivePrior:
    shape_probs: dict[str, float]
    count_probs: dict[int, float]
    rx_bounds: tuple[float, float]
    ry_bounds: tuple[float, float]
    rz_bounds: tuple[float, float]
    z_bounds: tuple[float, float]
    stiffness_median: float
    rx_samples: np.ndarray
    ry_samples: np.ndarray
    rz_samples: np.ndarray


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pilot a generated-3D-primitive -> 2D-projection prior as a post-process "
            "for existing high-resolution 2D segmentation probability maps."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--test-data-dir", type=Path, default=DEFAULT_DATA_ROOT / "test")
    parser.add_argument("--val-data-dir", type=Path, default=DEFAULT_DATA_ROOT / "val")
    parser.add_argument("--test-score-path", type=Path, default=None)
    parser.add_argument("--val-score-path", type=Path, default=None)
    parser.add_argument("--baseline-summary", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--label-size", type=int, default=0, help="0 infers from the score map shape.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional debug cap per split; 0 means full split.")
    parser.add_argument("--component-thresholds", type=str, default=",".join(str(v) for v in DEFAULT_THRESHOLDS))
    parser.add_argument("--candidate-scales", type=str, default=",".join(str(v) for v in DEFAULT_SCALES))
    parser.add_argument("--max-components", type=int, default=4)
    parser.add_argument("--min-component-area", type=int, default=10)
    parser.add_argument("--random-candidates", type=int, default=48)
    parser.add_argument("--area-weight", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--max-visual-samples", type=int, default=10)
    args = parser.parse_args()

    started = time.perf_counter()
    run_dir = args.run_dir
    test_score_path = args.test_score_path or run_dir / "test_scores.npy"
    val_score_path = args.val_score_path or run_dir / "val_scores.npy"
    baseline_summary_path = args.baseline_summary or run_dir / "test_metrics_summary.json"
    thresholds = parse_float_list(args.component_thresholds)
    scales = parse_float_list(args.candidate_scales)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("loading scores and data splits", flush=True)
    test_scores = load_scores(test_score_path, max_samples=args.max_samples)
    val_scores = load_scores(val_score_path, max_samples=args.max_samples)
    label_size = int(args.label_size or test_scores.shape[-1])
    if test_scores.shape[-2:] != (label_size, label_size):
        raise SystemExit(f"Expected square score maps matching label_size={label_size}, got {test_scores.shape}")

    prior_dirs = default_prior_dirs(args.test_data_dir, args.val_data_dir)
    prior = estimate_prior(prior_dirs)
    val_names, val_masks, val_meta = load_split_masks(args.val_data_dir, label_size, max_samples=args.max_samples)
    test_names, test_masks, test_meta = load_split_masks(args.test_data_dir, label_size, max_samples=args.max_samples)
    ensure_split_alignment("val", val_scores, val_names, val_masks)
    ensure_split_alignment("test", test_scores, test_names, test_masks)

    baseline_threshold = read_baseline_threshold(baseline_summary_path, default=0.5)
    print(f"baseline val-selected threshold: {baseline_threshold:.3f}", flush=True)

    print("fitting projection-prior masks on validation split", flush=True)
    val_projection = fit_projection_split(
        val_scores,
        val_meta,
        prior,
        thresholds=thresholds,
        scales=scales,
        max_components=args.max_components,
        min_component_area=args.min_component_area,
        random_candidates=args.random_candidates,
        area_weight=args.area_weight,
        rng=rng,
    )
    print("fitting projection-prior masks on test split", flush=True)
    test_projection = fit_projection_split(
        test_scores,
        test_meta,
        prior,
        thresholds=thresholds,
        scales=scales,
        max_components=args.max_components,
        min_component_area=args.min_component_area,
        random_candidates=args.random_candidates,
        area_weight=args.area_weight,
        rng=rng,
    )

    val_projection_masks = np.stack([item.mask for item in val_projection]).astype(np.uint8)
    test_projection_masks = np.stack([item.mask for item in test_projection]).astype(np.uint8)

    fusion_choice = tune_fusion(val_scores, val_projection_masks, val_masks)
    fusion_alpha = float(fusion_choice["alpha"])
    fusion_threshold = float(fusion_choice["threshold"])
    print(
        f"validation-selected fusion: alpha={fusion_alpha:.2f}, threshold={fusion_threshold:.3f}, "
        f"val_dice={fusion_choice['dice']:.4f}",
        flush=True,
    )

    baseline_test = aggregate_metrics(test_scores >= baseline_threshold, test_masks)
    baseline_fixed_test = aggregate_metrics(test_scores >= 0.5, test_masks)
    baseline_test_oracle = best_threshold_metrics(test_scores, test_masks)
    projection_test = aggregate_metrics(test_projection_masks, test_masks)
    fused_test_scores = fuse_scores(test_scores, test_projection_masks, fusion_alpha)
    fused_test = aggregate_metrics(fused_test_scores >= fusion_threshold, test_masks)
    fused_test_oracle = best_fusion_oracle(test_scores, test_projection_masks, test_masks)

    val_baseline = aggregate_metrics(val_scores >= baseline_threshold, val_masks)
    val_projection_metrics = aggregate_metrics(val_projection_masks, val_masks)
    val_fused = aggregate_metrics(fuse_scores(val_scores, val_projection_masks, fusion_alpha) >= fusion_threshold, val_masks)

    per_sample_rows = write_per_sample_metrics(
        args.out_dir / "metrics_per_sample.csv",
        test_names,
        test_scores,
        test_projection_masks,
        fused_test_scores,
        test_masks,
        baseline_threshold=baseline_threshold,
        fusion_threshold=fusion_threshold,
        projection_results=test_projection,
    )

    candidate_json = [
        {
            "sample": name,
            "objective": float(result.objective),
            "num_candidates": int(result.num_candidates),
            "primitives": [asdict(primitive) for primitive in result.primitives],
        }
        for name, result in zip(test_names, test_projection, strict=True)
    ]
    write_json(args.out_dir / "projection_primitives_test.json", candidate_json)

    summary = {
        "experiment": "projection_prior_postprocess_pilot",
        "interpretation": (
            "This is a fast viability test for a generated 3D primitive prior projected into 2D. "
            "It is not a trained diffusion model; it tests whether a 3D-to-2D projection constraint "
            "can improve existing 2D probability maps before investing in a diffusion generator."
        ),
        "run_dir": str(run_dir),
        "test_score_path": str(test_score_path),
        "val_score_path": str(val_score_path),
        "test_data_dir": str(args.test_data_dir),
        "val_data_dir": str(args.val_data_dir),
        "label_size": label_size,
        "num_val_samples": int(val_masks.shape[0]),
        "num_test_samples": int(test_masks.shape[0]),
        "baseline_threshold": baseline_threshold,
        "fusion_selection": fusion_choice,
        "test_metrics": {
            "baseline_fixed_0p5": baseline_fixed_test,
            "baseline_val_selected": baseline_test,
            "baseline_test_oracle_threshold": baseline_test_oracle,
            "projection_prior_only": projection_test,
            "projection_fusion_val_selected": fused_test,
            "projection_fusion_test_oracle": fused_test_oracle,
        },
        "val_metrics": {
            "baseline_val_selected": val_baseline,
            "projection_prior_only": val_projection_metrics,
            "projection_fusion_val_selected": val_fused,
        },
        "delta_vs_baseline_val_selected": {
            "projection_prior_only_dice": projection_test["dice"] - baseline_test["dice"],
            "projection_fusion_val_selected_dice": fused_test["dice"] - baseline_test["dice"],
            "projection_fusion_test_oracle_dice": fused_test_oracle["dice"] - baseline_test["dice"],
        },
        "candidate_generation": {
            "component_thresholds": thresholds,
            "candidate_scales": scales,
            "max_components": int(args.max_components),
            "min_component_area": int(args.min_component_area),
            "random_candidates_per_sample": int(args.random_candidates),
            "area_weight": float(args.area_weight),
            "seed": int(args.seed),
            "prior": prior_to_json(prior),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(args.out_dir / "metrics_summary.json", summary)
    write_report(args.out_dir / "REPORT.md", summary, per_sample_rows)
    write_visuals(
        args.out_dir,
        test_names,
        test_scores,
        test_projection_masks,
        fused_test_scores >= fusion_threshold,
        test_masks,
        per_sample_rows,
        baseline_threshold=baseline_threshold,
        max_visual_samples=args.max_visual_samples,
    )
    print(json.dumps(summary["test_metrics"], indent=2), flush=True)
    print(f"projection-prior pilot complete: {args.out_dir}", flush=True)


def load_scores(path: Path, *, max_samples: int) -> np.ndarray:
    scores = np.asarray(np.load(path), dtype=np.float32)
    if scores.ndim == 4 and scores.shape[1] == 1:
        scores = scores[:, 0]
    if scores.ndim != 3:
        raise ValueError(f"Expected score array [N,H,W], got {scores.shape} from {path}")
    scores = np.clip(np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    return scores[:max_samples] if max_samples and max_samples > 0 else scores


def default_prior_dirs(test_data_dir: Path, val_data_dir: Path) -> list[Path]:
    root = test_data_dir.parent
    candidates = [root / "train", val_data_dir, test_data_dir]
    return [path for path in candidates if path.exists()]


def estimate_prior(split_dirs: Sequence[Path]) -> PrimitivePrior:
    counts: list[int] = []
    shapes: list[str] = []
    rx: list[float] = []
    ry: list[float] = []
    rz: list[float] = []
    z: list[float] = []
    stiffness: list[float] = []
    files: list[Path] = []
    for split_dir in split_dirs:
        files.extend(sorted(split_dir.glob("*.npz")))
    if not files:
        raise FileNotFoundError(f"No prior .npz files found under {split_dirs}")
    for path in files:
        phantom, _scan, _material, lumps, _metadata = load_phantom_scan_material_lumps(
            sample_path=path,
            metadata_path=path.with_name(f"{path.stem}_gt.json"),
        )
        del phantom
        counts.append(len(lumps))
        for lump in lumps:
            shapes.append(str(lump.shape))
            rx.append(float(lump.radii[0]))
            ry.append(float(lump.radii[1]))
            rz.append(max(float(lump.radii[2]), 1e-4))
            z.append(float(lump.center[2]))
            stiffness.append(float(lump.stiffness_multiplier))

    if not rx:
        raise ValueError("Prior estimation found no lumps.")
    return PrimitivePrior(
        shape_probs=probabilities(shapes),
        count_probs={int(k): float(v) for k, v in probabilities(counts).items()},
        rx_bounds=quantile_bounds(rx, 0.02, 0.98, lo=0.002, hi=0.06),
        ry_bounds=quantile_bounds(ry, 0.02, 0.98, lo=0.002, hi=0.06),
        rz_bounds=quantile_bounds(rz, 0.02, 0.98, lo=0.0005, hi=0.04),
        z_bounds=quantile_bounds(z, 0.02, 0.98, lo=0.001, hi=0.079),
        stiffness_median=float(np.median(stiffness)),
        rx_samples=np.asarray(rx, dtype=np.float32),
        ry_samples=np.asarray(ry, dtype=np.float32),
        rz_samples=np.asarray(rz, dtype=np.float32),
    )


def probabilities(values: Sequence[Any]) -> dict[Any, float]:
    counts: dict[Any, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    total = float(sum(counts.values()))
    return {key: value / total for key, value in sorted(counts.items(), key=lambda item: str(item[0]))}


def quantile_bounds(values: Sequence[float], q_lo: float, q_hi: float, *, lo: float, hi: float) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float32)
    low = max(float(np.quantile(arr, q_lo)), lo)
    high = min(float(np.quantile(arr, q_hi)), hi)
    if high <= low:
        high = low * 1.5
    return low, high


def load_split_masks(
    data_dir: Path,
    label_size: int,
    *,
    max_samples: int,
) -> tuple[list[str], np.ndarray, list[dict[str, Any]]]:
    files = sorted(data_dir.glob("*.npz"))
    if max_samples and max_samples > 0:
        files = files[:max_samples]
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    names: list[str] = []
    masks: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    for path in files:
        phantom, scan, _material, lumps, _metadata = load_phantom_scan_material_lumps(
            sample_path=path,
            metadata_path=path.with_name(f"{path.stem}_gt.json"),
        )
        x_grid, y_grid = scan_area_grid(scan, phantom, label_size)
        mask = project_primitives_mask(x_grid, y_grid, lumps_to_primitives(lumps, source="ground_truth"))
        names.append(path.name)
        masks.append(mask.astype(np.uint8))
        meta.append({"path": path, "phantom": phantom, "scan": scan, "x_grid": x_grid, "y_grid": y_grid})
    return names, np.stack(masks).astype(np.uint8), meta


def scan_area_grid(scan: ScanConfig, phantom: PhantomConfig, label_size: int) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.asarray(scan.x_values(phantom), dtype=np.float32)
    y_values = np.asarray(scan.y_values(phantom), dtype=np.float32)
    xs = np.linspace(float(x_values[0]), float(x_values[-1]), int(label_size), dtype=np.float32)
    ys = np.linspace(float(y_values[0]), float(y_values[-1]), int(label_size), dtype=np.float32)
    return np.meshgrid(xs, ys)


def lumps_to_primitives(lumps: Iterable[Any], *, source: str) -> list[Primitive2D]:
    primitives: list[Primitive2D] = []
    for lump in lumps:
        primitives.append(
            Primitive2D(
                shape=str(lump.shape),
                cx=float(lump.center[0]),
                cy=float(lump.center[1]),
                cz=float(lump.center[2]),
                rx=float(lump.radii[0]),
                ry=float(lump.radii[1]),
                rz=float(lump.radii[2]),
                yaw=float(lump.yaw),
                stiffness_multiplier=float(lump.stiffness_multiplier),
                source=source,
            )
        )
    return primitives


def ensure_split_alignment(split: str, scores: np.ndarray, names: Sequence[str], masks: np.ndarray) -> None:
    if len(scores) != len(names) or len(scores) != len(masks):
        raise ValueError(f"{split} alignment mismatch: scores={len(scores)}, names={len(names)}, masks={len(masks)}")
    if scores.shape[-2:] != masks.shape[-2:]:
        raise ValueError(f"{split} shape mismatch: scores={scores.shape[-2:]}, masks={masks.shape[-2:]}")


def read_baseline_threshold(path: Path, *, default: float) -> float:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    return float(summary.get("val_selected_threshold", {}).get("threshold", default))


def fit_projection_split(
    scores: np.ndarray,
    meta: Sequence[dict[str, Any]],
    prior: PrimitivePrior,
    *,
    thresholds: Sequence[float],
    scales: Sequence[float],
    max_components: int,
    min_component_area: int,
    random_candidates: int,
    area_weight: float,
    rng: np.random.Generator,
) -> list[ProjectionResult]:
    results: list[ProjectionResult] = []
    total = int(scores.shape[0])
    for idx, prob in enumerate(scores):
        result = fit_projection_mask(
            prob,
            meta[idx],
            prior,
            thresholds=thresholds,
            scales=scales,
            max_components=max_components,
            min_component_area=min_component_area,
            random_candidates=random_candidates,
            area_weight=area_weight,
            rng=rng,
        )
        results.append(result)
        if (idx + 1) % 50 == 0 or idx + 1 == total:
            print(f"  {idx + 1}/{total} projection masks", flush=True)
    return results


def fit_projection_mask(
    prob: np.ndarray,
    meta: dict[str, Any],
    prior: PrimitivePrior,
    *,
    thresholds: Sequence[float],
    scales: Sequence[float],
    max_components: int,
    min_component_area: int,
    random_candidates: int,
    area_weight: float,
    rng: np.random.Generator,
) -> ProjectionResult:
    x_grid = np.asarray(meta["x_grid"], dtype=np.float32)
    y_grid = np.asarray(meta["y_grid"], dtype=np.float32)
    phantom: PhantomConfig = meta["phantom"]
    candidates = component_candidates(
        prob,
        x_grid,
        y_grid,
        phantom,
        prior,
        thresholds=thresholds,
        scales=scales,
        max_components=max_components,
        min_component_area=min_component_area,
    )
    candidates.extend(
        random_primitive_candidates(
            prob,
            x_grid,
            y_grid,
            phantom,
            prior,
            count=random_candidates,
            max_components=max_components,
            rng=rng,
        )
    )
    if not candidates:
        candidates.append([])

    target_area = float((prob >= 0.5).mean())
    log_gain = np.log(np.clip(prob, 1e-5, 1.0 - 1e-5)) - np.log(np.clip(1.0 - prob, 1e-5, 1.0))
    best_score = -float("inf")
    best_mask = np.zeros_like(prob, dtype=np.uint8)
    best_primitives: list[Primitive2D] = []
    for primitives in candidates:
        mask = project_primitives_mask(x_grid, y_grid, primitives)
        objective = float(np.mean(mask.astype(np.float32) * log_gain))
        objective -= float(area_weight) * abs(float(mask.mean()) - target_area)
        if objective > best_score:
            best_score = objective
            best_mask = mask.astype(np.uint8)
            best_primitives = primitives
    return ProjectionResult(best_mask, best_score, best_primitives, len(candidates))


def component_candidates(
    prob: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    phantom: PhantomConfig,
    prior: PrimitivePrior,
    *,
    thresholds: Sequence[float],
    scales: Sequence[float],
    max_components: int,
    min_component_area: int,
) -> list[list[Primitive2D]]:
    from scipy import ndimage

    candidates: list[list[Primitive2D]] = []
    structure = np.ones((3, 3), dtype=np.uint8)
    for threshold in thresholds:
        labels, count = ndimage.label(prob >= float(threshold), structure=structure)
        if count <= 0:
            continue
        objects = ndimage.find_objects(labels)
        components: list[tuple[float, Primitive2D]] = []
        for label_id, slices in enumerate(objects, start=1):
            if slices is None:
                continue
            component = labels[slices] == label_id
            area = int(component.sum())
            if area < min_component_area:
                continue
            local_prob = prob[slices]
            local_x = x_grid[slices]
            local_y = y_grid[slices]
            primitive = fit_component_primitive(
                local_prob,
                component,
                local_x,
                local_y,
                phantom,
                prior,
                threshold=threshold,
            )
            components.append((float(local_prob[component].sum()), primitive))
        components.sort(key=lambda item: item[0], reverse=True)
        selected = [primitive for _score, primitive in components[:max_components]]
        if not selected:
            continue
        for n_components in range(1, min(max_components, len(selected)) + 1):
            base = selected[:n_components]
            for scale in scales:
                candidates.append([scale_primitive(item, scale, phantom, prior) for item in base])
            candidates.append([box_from_primitive(item, phantom, prior) for item in base])
    return candidates


def fit_component_primitive(
    prob: np.ndarray,
    component: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    phantom: PhantomConfig,
    prior: PrimitivePrior,
    *,
    threshold: float,
) -> Primitive2D:
    weights = np.where(component, np.maximum(prob, 1e-4), 0.0).astype(np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        weights = component.astype(np.float64)
        total = float(weights.sum())
    cx = float((weights * x_grid).sum() / total)
    cy = float((weights * y_grid).sum() / total)
    dx = x_grid - cx
    dy = y_grid - cy
    cov_xx = float((weights * dx * dx).sum() / total)
    cov_yy = float((weights * dy * dy).sum() / total)
    cov_xy = float((weights * dx * dy).sum() / total)
    cov = np.asarray([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 1e-10)
    eigvecs = eigvecs[:, order]
    yaw = float(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))
    rx = float(2.25 * math.sqrt(float(eigvals[0])))
    ry = float(2.25 * math.sqrt(float(eigvals[1])))
    if rx < ry:
        rx, ry = ry, rx
        yaw += math.pi / 2.0
    rx, ry = clip_xy_radii(rx, ry, prior)
    cx, cy = clamp_center(cx, cy, rx, ry, phantom)
    return Primitive2D(
        shape="ellipsoid",
        cx=cx,
        cy=cy,
        cz=float(np.mean(prior.z_bounds)),
        rx=rx,
        ry=ry,
        rz=float(np.mean(prior.rz_bounds)),
        yaw=yaw,
        stiffness_multiplier=prior.stiffness_median,
        source=f"component_t{threshold:.2f}",
    )


def random_primitive_candidates(
    prob: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    phantom: PhantomConfig,
    prior: PrimitivePrior,
    *,
    count: int,
    max_components: int,
    rng: np.random.Generator,
) -> list[list[Primitive2D]]:
    if count <= 0:
        return []
    flat_prob = np.asarray(prob, dtype=np.float64).reshape(-1)
    weights = np.maximum(flat_prob, 1e-4) ** 2.0
    weights = weights / weights.sum()
    shape_keys = list(prior.shape_probs)
    shape_p = np.asarray([prior.shape_probs[key] for key in shape_keys], dtype=np.float64)
    shape_p = shape_p / shape_p.sum()
    count_keys = np.asarray(list(prior.count_probs), dtype=np.int64)
    count_p = np.asarray([prior.count_probs[int(key)] for key in count_keys], dtype=np.float64)
    count_p = count_p / count_p.sum()
    candidates: list[list[Primitive2D]] = []
    for _ in range(count):
        n_primitives = int(rng.choice(count_keys, p=count_p))
        n_primitives = max(1, min(n_primitives, max_components))
        flat_indices = rng.choice(flat_prob.size, size=n_primitives, replace=True, p=weights)
        primitives: list[Primitive2D] = []
        for flat_idx in np.atleast_1d(flat_indices):
            shape = str(rng.choice(shape_keys, p=shape_p))
            rx = float(rng.choice(prior.rx_samples) * rng.uniform(0.75, 1.35))
            ry = float(rng.choice(prior.ry_samples) * rng.uniform(0.75, 1.35))
            if shape in {"sphere", "capsule"}:
                r = 0.5 * (rx + ry)
                rx = r
                ry = r
            rx, ry = clip_xy_radii(rx, ry, prior)
            cx = float(x_grid.reshape(-1)[flat_idx] + rng.normal(0.0, 0.25 * rx))
            cy = float(y_grid.reshape(-1)[flat_idx] + rng.normal(0.0, 0.25 * ry))
            cx, cy = clamp_center(cx, cy, rx, ry, phantom)
            primitives.append(
                Primitive2D(
                    shape=shape,
                    cx=cx,
                    cy=cy,
                    cz=float(rng.uniform(*prior.z_bounds)),
                    rx=rx,
                    ry=ry,
                    rz=float(rng.choice(prior.rz_samples)),
                    yaw=float(rng.uniform(-math.pi, math.pi)),
                    stiffness_multiplier=prior.stiffness_median,
                    source="random_prior",
                )
            )
        candidates.append(primitives)
    return candidates


def scale_primitive(
    primitive: Primitive2D,
    scale: float,
    phantom: PhantomConfig,
    prior: PrimitivePrior,
) -> Primitive2D:
    rx, ry = clip_xy_radii(primitive.rx * float(scale), primitive.ry * float(scale), prior)
    cx, cy = clamp_center(primitive.cx, primitive.cy, rx, ry, phantom)
    return Primitive2D(
        shape=primitive.shape,
        cx=cx,
        cy=cy,
        cz=primitive.cz,
        rx=rx,
        ry=ry,
        rz=primitive.rz,
        yaw=primitive.yaw,
        stiffness_multiplier=primitive.stiffness_multiplier,
        source=f"{primitive.source}_scale{scale:.2f}",
    )


def box_from_primitive(
    primitive: Primitive2D,
    phantom: PhantomConfig,
    prior: PrimitivePrior,
) -> Primitive2D:
    rx, ry = clip_xy_radii(primitive.rx * 0.9, primitive.ry * 0.9, prior)
    cx, cy = clamp_center(primitive.cx, primitive.cy, rx, ry, phantom)
    return Primitive2D(
        shape="box",
        cx=cx,
        cy=cy,
        cz=primitive.cz,
        rx=rx,
        ry=ry,
        rz=primitive.rz,
        yaw=primitive.yaw,
        stiffness_multiplier=primitive.stiffness_multiplier,
        source=f"{primitive.source}_box",
    )


def clip_xy_radii(rx: float, ry: float, prior: PrimitivePrior) -> tuple[float, float]:
    rx_low, rx_high = prior.rx_bounds
    ry_low, ry_high = prior.ry_bounds
    return float(np.clip(rx, rx_low, rx_high * 1.6)), float(np.clip(ry, ry_low, ry_high * 1.6))


def clamp_center(cx: float, cy: float, rx: float, ry: float, phantom: PhantomConfig) -> tuple[float, float]:
    margin_x = min(max(rx, ry) + 0.002, 0.48 * phantom.size_x)
    margin_y = min(max(rx, ry) + 0.002, 0.48 * phantom.size_y)
    lo_x = -0.5 * phantom.size_x + margin_x
    hi_x = 0.5 * phantom.size_x - margin_x
    lo_y = -0.5 * phantom.size_y + margin_y
    hi_y = 0.5 * phantom.size_y - margin_y
    if lo_x >= hi_x:
        lo_x, hi_x = -0.45 * phantom.size_x, 0.45 * phantom.size_x
    if lo_y >= hi_y:
        lo_y, hi_y = -0.45 * phantom.size_y, 0.45 * phantom.size_y
    return float(np.clip(cx, lo_x, hi_x)), float(np.clip(cy, lo_y, hi_y))


def project_primitives_mask(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    primitives: Sequence[Primitive2D],
) -> np.ndarray:
    if not primitives:
        return np.zeros_like(x_grid, dtype=np.uint8)
    mask = np.zeros_like(x_grid, dtype=bool)
    for primitive in primitives:
        rel_x = x_grid - float(primitive.cx)
        rel_y = y_grid - float(primitive.cy)
        c = math.cos(-float(primitive.yaw))
        s = math.sin(-float(primitive.yaw))
        rot_x = c * rel_x - s * rel_y
        rot_y = s * rel_x + c * rel_y
        rx = max(float(primitive.rx), 1e-6)
        ry = max(float(primitive.ry), 1e-6)
        if primitive.shape == "box":
            current = (np.abs(rot_x) <= rx) & (np.abs(rot_y) <= ry)
        else:
            current = (rot_x / rx) ** 2 + (rot_y / ry) ** 2 <= 1.0
        mask |= current
    return mask.astype(np.uint8)


def tune_fusion(scores: np.ndarray, projection_masks: np.ndarray, masks: np.ndarray) -> dict[str, float | int]:
    best: dict[str, float | int] | None = None
    for alpha in FUSION_ALPHAS:
        fused = fuse_scores(scores, projection_masks, float(alpha))
        for threshold in FUSION_THRESHOLDS:
            row = aggregate_metrics(fused >= float(threshold), masks)
            item = {"alpha": float(alpha), "threshold": float(threshold), **row}
            if best is None or float(item["dice"]) > float(best["dice"]):
                best = item
    assert best is not None
    return best


def best_fusion_oracle(scores: np.ndarray, projection_masks: np.ndarray, masks: np.ndarray) -> dict[str, float | int]:
    best: dict[str, float | int] | None = None
    for alpha in FUSION_ALPHAS:
        fused = fuse_scores(scores, projection_masks, float(alpha))
        row = best_threshold_metrics(fused, masks)
        item = {"alpha": float(alpha), **row}
        if best is None or float(item["dice"]) > float(best["dice"]):
            best = item
    assert best is not None
    return best


def fuse_scores(scores: np.ndarray, projection_masks: np.ndarray, alpha: float) -> np.ndarray:
    return ((1.0 - float(alpha)) * scores + float(alpha) * projection_masks.astype(np.float32)).astype(np.float32)


def best_threshold_metrics(scores: np.ndarray, masks: np.ndarray) -> dict[str, float | int]:
    best: dict[str, float | int] | None = None
    for threshold in FUSION_THRESHOLDS:
        row = aggregate_metrics(scores >= float(threshold), masks)
        row = {"threshold": float(threshold), **row}
        if best is None or float(row["dice"]) > float(best["dice"]):
            best = row
    assert best is not None
    return best


def aggregate_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    pred_bool = np.asarray(pred).astype(bool)
    gt_bool = np.asarray(gt).astype(bool)
    tp = int(np.logical_and(pred_bool, gt_bool).sum())
    tn = int(np.logical_and(~pred_bool, ~gt_bool).sum())
    fp = int(np.logical_and(pred_bool, ~gt_bool).sum())
    fn = int(np.logical_and(~pred_bool, gt_bool).sum())
    eps = 1e-8
    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    dice = 2.0 * tp / max(2 * tp + fp + fn, eps)
    iou = tp / max(tp + fp + fn, eps)
    pixel_accuracy = (tp + tn) / max(tp + tn + fp + fn, eps)
    return {
        "pixel_accuracy": float(pixel_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "dice": float(dice),
        "iou": float(iou),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def write_per_sample_metrics(
    path: Path,
    names: Sequence[str],
    scores: np.ndarray,
    projection_masks: np.ndarray,
    fused_scores: np.ndarray,
    masks: np.ndarray,
    *,
    baseline_threshold: float,
    fusion_threshold: float,
    projection_results: Sequence[ProjectionResult],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, name in enumerate(names):
        baseline = aggregate_metrics(scores[idx] >= baseline_threshold, masks[idx])
        projection = aggregate_metrics(projection_masks[idx], masks[idx])
        fused = aggregate_metrics(fused_scores[idx] >= fusion_threshold, masks[idx])
        rows.append(
            {
                "sample": name,
                "baseline_dice": baseline["dice"],
                "projection_dice": projection["dice"],
                "fusion_dice": fused["dice"],
                "projection_delta_dice": projection["dice"] - baseline["dice"],
                "fusion_delta_dice": fused["dice"] - baseline["dice"],
                "baseline_iou": baseline["iou"],
                "projection_iou": projection["iou"],
                "fusion_iou": fused["iou"],
                "gt_positive": int(masks[idx].sum()),
                "projection_positive": int(projection_masks[idx].sum()),
                "baseline_positive": int((scores[idx] >= baseline_threshold).sum()),
                "fusion_positive": int((fused_scores[idx] >= fusion_threshold).sum()),
                "projection_objective": float(projection_results[idx].objective),
                "projection_candidates": int(projection_results[idx].num_candidates),
                "projection_primitives": int(len(projection_results[idx].primitives)),
            }
        )
    write_csv(path, rows)
    return rows


def write_visuals(
    out_dir: Path,
    names: Sequence[str],
    scores: np.ndarray,
    projection_masks: np.ndarray,
    fused_pred: np.ndarray,
    masks: np.ndarray,
    rows: Sequence[dict[str, Any]],
    *,
    baseline_threshold: float,
    max_visual_samples: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        "baseline",
        "projection",
        "fusion",
    ]
    dice_values = [
        aggregate_metrics(scores >= baseline_threshold, masks)["dice"],
        aggregate_metrics(projection_masks, masks)["dice"],
        aggregate_metrics(fused_pred, masks)["dice"],
    ]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.bar(labels, dice_values, color=["#3b82f6", "#14b8a6", "#f97316"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Dice")
    ax.set_title("Projection Prior Pilot Test Dice")
    for idx, value in enumerate(dice_values):
        ax.text(idx, value + 0.015, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_dir / "dice_bar.png", dpi=180)
    plt.close(fig)

    deltas = np.asarray([float(row["fusion_delta_dice"]) for row in rows], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.hist(deltas, bins=30, color="#64748b", edgecolor="white")
    ax.axvline(0.0, color="#111827", linewidth=1.2)
    ax.set_xlabel("Fusion Dice Delta vs Baseline")
    ax.set_ylabel("Samples")
    ax.set_title("Per-Sample Projection Fusion Delta")
    fig.tight_layout()
    fig.savefig(out_dir / "fusion_delta_hist.png", dpi=180)
    plt.close(fig)

    selected = select_visual_rows(rows, max_visual_samples)
    if not selected:
        return
    cols = ["probability", "baseline", "projection", "fusion", "ground truth"]
    fig, axes = plt.subplots(len(selected), len(cols), figsize=(2.3 * len(cols), 2.15 * len(selected)))
    if len(selected) == 1:
        axes = np.asarray([axes])
    for row_idx, sample_idx in enumerate(selected):
        baseline_pred = scores[sample_idx] >= baseline_threshold
        images = [scores[sample_idx], baseline_pred, projection_masks[sample_idx], fused_pred[sample_idx], masks[sample_idx]]
        for col_idx, image in enumerate(images):
            ax = axes[row_idx, col_idx]
            ax.imshow(image, cmap="viridis" if col_idx == 0 else "gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(cols[col_idx], fontsize=9)
            if col_idx == 0:
                row = rows[sample_idx]
                ax.set_ylabel(
                    f"{names[sample_idx]}\n"
                    f"B {float(row['baseline_dice']):.3f}  F {float(row['fusion_dice']):.3f}",
                    fontsize=8,
                )
    fig.tight_layout()
    fig.savefig(out_dir / "prediction_contact_sheet.png", dpi=180)
    plt.close(fig)


def select_visual_rows(rows: Sequence[dict[str, Any]], max_rows: int) -> list[int]:
    if max_rows <= 0 or not rows:
        return []
    order = np.argsort([float(row["fusion_delta_dice"]) for row in rows])
    picks: list[int] = []
    picks.extend(order[-max_rows // 2 :][::-1].tolist())
    picks.extend(order[: max_rows - len(picks)].tolist())
    seen: set[int] = set()
    unique: list[int] = []
    for idx in picks:
        if int(idx) not in seen:
            seen.add(int(idx))
            unique.append(int(idx))
    return unique[:max_rows]


def write_report(path: Path, summary: dict[str, Any], rows: Sequence[dict[str, Any]]) -> None:
    test = summary["test_metrics"]
    delta = summary["delta_vs_baseline_val_selected"]
    best_rows = sorted(rows, key=lambda row: float(row["fusion_delta_dice"]), reverse=True)[:5]
    worst_rows = sorted(rows, key=lambda row: float(row["fusion_delta_dice"]))[:5]
    lines = [
        "# Projection-Prior Pilot",
        "",
        "This is a fast post-processing viability test for the route: generated 3D primitive model -> 2D projection -> segmentation prior.",
        "It does not train a diffusion model; it checks whether the projection constraint has measurable value on the existing U-Net probability maps.",
        "",
        "## Test Dice",
        "",
        f"- Baseline val-selected: {test['baseline_val_selected']['dice']:.6f}",
        f"- Projection prior only: {test['projection_prior_only']['dice']:.6f} "
        f"({delta['projection_prior_only_dice']:+.6f})",
        f"- Projection fusion, validation-selected: {test['projection_fusion_val_selected']['dice']:.6f} "
        f"({delta['projection_fusion_val_selected_dice']:+.6f})",
        f"- Projection fusion, test oracle: {test['projection_fusion_test_oracle']['dice']:.6f} "
        f"({delta['projection_fusion_test_oracle_dice']:+.6f})",
        "",
        "## Validation Selection",
        "",
        f"- alpha: {summary['fusion_selection']['alpha']}",
        f"- threshold: {summary['fusion_selection']['threshold']}",
        f"- validation Dice: {summary['fusion_selection']['dice']:.6f}",
        "",
        "## Largest Fusion Improvements",
        "",
    ]
    lines.extend(format_rows(best_rows))
    lines.extend(["", "## Largest Fusion Regressions", ""])
    lines.extend(format_rows(worst_rows))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_rows(rows: Sequence[dict[str, Any]]) -> list[str]:
    return [
        f"- {row['sample']}: baseline={float(row['baseline_dice']):.4f}, "
        f"fusion={float(row['fusion_dice']):.4f}, delta={float(row['fusion_delta_dice']):+.4f}"
        for row in rows
    ]


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def prior_to_json(prior: PrimitivePrior) -> dict[str, Any]:
    return {
        "shape_probs": prior.shape_probs,
        "count_probs": {str(key): value for key, value in prior.count_probs.items()},
        "rx_bounds": list(prior.rx_bounds),
        "ry_bounds": list(prior.ry_bounds),
        "rz_bounds": list(prior.rz_bounds),
        "z_bounds": list(prior.z_bounds),
        "stiffness_median": prior.stiffness_median,
        "num_lump_samples": int(prior.rx_samples.size),
    }


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
