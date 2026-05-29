from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_features(path: Path) -> np.ndarray:
    with np.load(path) as sample:
        if "features" in sample:
            features = ensure_chw(sample["features"])
        elif "presses" in sample:
            features = extract_feature_map(sample["presses"])
        else:
            raise KeyError(f"{path} must contain either 'features' or 'presses'")
    return normalize_feature_map(features)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validation U-Net inference.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Input .npz containing features or presses.")
    parser.add_argument("--output", type=Path, required=True, help="Output .npy 0/1 mask.")
    parser.add_argument("--prob-output", type=Path, default=None, help="Optional .npy probability map.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    _load_ml_dependencies()

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = ValidationUNet(
        in_channels=int(checkpoint["in_channels"]),
        base_channels=int(checkpoint.get("base_channels", 24)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    features = torch.from_numpy(load_features(args.input)).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(features))[0, 0].cpu().numpy()
    binary = (prob >= args.threshold).astype(np.uint8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, binary)
    if args.prob_output is not None:
        args.prob_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.prob_output, prob.astype(np.float32))

def _load_ml_dependencies() -> None:
    global ValidationUNet, ensure_chw, extract_feature_map, normalize_feature_map, np, torch
    try:
        import numpy as np_mod
        import torch as torch_mod

        from palpation_sim.features import ensure_chw as ensure_chw_fn
        from palpation_sim.features import extract_feature_map as extract_feature_map_fn
        from palpation_sim.features import normalize_feature_map as normalize_feature_map_fn
        from palpation_sim.models import ValidationUNet as model_cls
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch/Numpy dependencies are required for inference. Install them with: "
            "python -m pip install -r requirements-ml.txt"
        ) from exc

    np = np_mod
    torch = torch_mod
    ensure_chw = ensure_chw_fn
    extract_feature_map = extract_feature_map_fn
    normalize_feature_map = normalize_feature_map_fn
    ValidationUNet = model_cls


if __name__ == "__main__":
    main()
