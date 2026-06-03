from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from palpation_sim.workflow import require_runtime_environment, resolve_required_torch_cuda_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Xiao et al. 2020-style LSTM depth classifier.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/xiao_lstm_eval"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--normalize", choices=["auto", "none", "sample", "dataset"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", help="Required CUDA device, e.g. cuda or cuda:0.")
    args = parser.parse_args()
    require_runtime_environment()

    _load_ml_dependencies()

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    sequence_length = int(args.sequence_length or checkpoint.get("sequence_length", 50))
    normalize = str(checkpoint.get("normalize", "sample")) if args.normalize == "auto" else args.normalize
    class_depths_mm = tuple(float(value) for value in checkpoint.get("class_depths_mm", (0, 5, 8, 10)))
    sequence_mean = checkpoint.get("sequence_mean")
    sequence_std = checkpoint.get("sequence_std")

    dataset = XiaoDepthSequenceDataset(
        args.data_dir,
        sequence_length=sequence_length,
        normalize=normalize,
        class_depths_mm=class_depths_mm,
        sequence_mean=sequence_mean,
        sequence_std=sequence_std,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = Xiao2020DepthLSTM(
        input_size=int(checkpoint.get("input_size", 3)),
        hidden_size=int(checkpoint.get("hidden_size", 70)),
        num_layers=int(checkpoint.get("num_layers", 2)),
        num_classes=int(checkpoint.get("num_classes", len(class_depths_mm))),
        dropout=float(checkpoint.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    confusion = [[0 for _ in class_depths_mm] for _ in class_depths_mm]
    rows: list[dict[str, object]] = []
    total_correct = 0
    total_count = 0
    file_index = 0
    with torch.no_grad():
        for sequences, targets in loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            logits = model(sequences)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            batch_size = int(targets.shape[0])
            total_correct += int((preds == targets).sum().detach().cpu())
            total_count += batch_size

            preds_cpu = preds.detach().cpu().tolist()
            targets_cpu = targets.detach().cpu().tolist()
            probs_cpu = probs.detach().cpu().tolist()
            for local_idx in range(batch_size):
                truth = int(targets_cpu[local_idx])
                pred = int(preds_cpu[local_idx])
                confusion[truth][pred] += 1
                probability = float(probs_cpu[local_idx][pred])
                rows.append(
                    {
                        "sample": dataset.files[file_index].name,
                        "truth_label": truth,
                        "truth_depth_mm": class_depths_mm[truth],
                        "pred_label": pred,
                        "pred_depth_mm": class_depths_mm[pred],
                        "pred_probability": probability,
                        "correct": int(truth == pred),
                    }
                )
                file_index += 1

    per_class = _per_class_metrics(confusion, class_depths_mm)
    summary = {
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "num_samples": total_count,
        "accuracy": total_correct / max(total_count, 1),
        "class_depths_mm": class_depths_mm,
        "normalize": normalize,
        "sequence_length": sequence_length,
        "confusion_matrix_rows_truth_cols_pred": confusion,
        "per_class": per_class,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "metrics_summary.json").open("w") as summary_file:
        json.dump(summary, summary_file, indent=2)
    with (args.out_dir / "predictions.csv").open("w", newline="") as csv_file:
        fieldnames = [
            "sample",
            "truth_label",
            "truth_depth_mm",
            "pred_label",
            "pred_depth_mm",
            "pred_probability",
            "correct",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (args.out_dir / "confusion_matrix.csv").open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["truth\\pred", *[f"{depth:g}mm" for depth in class_depths_mm]])
        for depth, row in zip(class_depths_mm, confusion):
            writer.writerow([f"{depth:g}mm", *row])
    print(json.dumps(summary, indent=2))


def _per_class_metrics(confusion: list[list[int]], class_depths_mm: tuple[float, ...]) -> list[dict[str, float]]:
    total = sum(sum(row) for row in confusion)
    metrics = []
    for idx, depth in enumerate(class_depths_mm):
        tp = confusion[idx][idx]
        fn = sum(confusion[idx]) - tp
        fp = sum(row[idx] for row in confusion) - tp
        tn = total - tp - fn - fp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        metrics.append(
            {
                "label": float(idx),
                "depth_mm": float(depth),
                "precision": float(precision),
                "recall": float(recall),
                "support": float(sum(confusion[idx])),
                "tp": float(tp),
                "tn": float(tn),
                "fp": float(fp),
                "fn": float(fn),
            }
        )
    return metrics


def resolve_device(name: str):
    return resolve_required_torch_cuda_device(torch, name)


def _load_ml_dependencies() -> None:
    global DataLoader, Xiao2020DepthLSTM, XiaoDepthSequenceDataset, torch
    try:
        import torch as torch_mod
        from torch.utils.data import DataLoader as data_loader_cls

        from palpation_sim.dataset import XiaoDepthSequenceDataset as dataset_cls
        from palpation_sim.models import Xiao2020DepthLSTM as model_cls
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch/Numpy dependencies are required for evaluation. Create the conda env from "
            "environment.yml, activate it, then rerun this script."
        ) from exc

    torch = torch_mod
    DataLoader = data_loader_cls
    XiaoDepthSequenceDataset = dataset_cls
    Xiao2020DepthLSTM = model_cls


if __name__ == "__main__":
    main()
