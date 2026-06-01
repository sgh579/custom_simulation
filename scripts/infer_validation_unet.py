from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_model_input(path: Path, input_mode: str) -> np.ndarray:
    with np.load(path) as sample:
        if input_mode == "fz":
            features = _load_fz_channels(sample, path)
        elif input_mode == "presses":
            if "presses" not in sample:
                raise KeyError(f"{path} must contain 'presses' when input_mode='presses'")
            features = presses_to_channel_map(sample["presses"])
        elif input_mode == "features" and "features" in sample:
            features = ensure_chw(sample["features"])
        elif input_mode == "features" and "presses" in sample:
            features = extract_feature_map(sample["presses"])
        elif input_mode == "auto" and ("fz" in sample or "presses" in sample):
            features = _load_fz_channels(sample, path)
        elif input_mode == "auto" and "features" in sample:
            features = ensure_chw(sample["features"])
        else:
            raise KeyError(f"{path} must contain data compatible with input_mode='{input_mode}'")
    return normalize_feature_map(features)


def _load_fz_channels(sample: np.lib.npyio.NpzFile, path: Path) -> np.ndarray:
    if "fz" in sample:
        return fz_to_channel_map(sample["fz"])
    if "presses" in sample:
        return fz_to_channel_map(sample["presses"])
    raise KeyError(f"{path} must contain 'fz' or 'presses' when input_mode='fz'")


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
    parser.add_argument("--input", type=Path, required=True, help="Input .npz containing Fz, presses, or features.")
    parser.add_argument("--output", type=Path, required=True, help="Output .npy 0/1 mask.")
    parser.add_argument("--prob-output", type=Path, default=None, help="Optional .npy probability map.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    _load_ml_dependencies()

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    input_mode = str(checkpoint.get("input_mode", "features"))
    model = ValidationUNet(
        in_channels=int(checkpoint["in_channels"]),
        base_channels=int(checkpoint.get("base_channels", 24)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    features = torch.from_numpy(load_model_input(args.input, input_mode)).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(features))[0, 0].cpu().numpy()
    binary = (prob >= args.threshold).astype(np.uint8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, binary)
    if args.prob_output is not None:
        args.prob_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.prob_output, prob.astype(np.float32))

def _load_ml_dependencies() -> None:
    global ValidationUNet, ensure_chw, extract_feature_map, fz_to_channel_map, normalize_feature_map, presses_to_channel_map, np, torch
    try:
        import numpy as np_mod
        import torch as torch_mod

        from palpation_sim.features import ensure_chw as ensure_chw_fn
        from palpation_sim.features import extract_feature_map as extract_feature_map_fn
        from palpation_sim.features import fz_to_channel_map as fz_to_channel_map_fn
        from palpation_sim.features import normalize_feature_map as normalize_feature_map_fn
        from palpation_sim.features import presses_to_channel_map as presses_to_channel_map_fn
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
    fz_to_channel_map = fz_to_channel_map_fn
    normalize_feature_map = normalize_feature_map_fn
    presses_to_channel_map = presses_to_channel_map_fn
    ValidationUNet = model_cls


if __name__ == "__main__":
    main()
