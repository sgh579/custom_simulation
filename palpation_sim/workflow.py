from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


CONDA_ENV_NAME = "palpation"
DEFAULT_NEWTON_ROOT = Path("/home/guoheng/newton")
REQUIRED_NEWTON_DEVICE = "cuda:0"
SIMULATOR_ENTRYPOINT = "palpation_sim.newton_vbd.NewtonVBDPalpationSimulator"
DATA_CONTRACT_VERSION = 1
SAMPLE_METADATA_SCHEMA_VERSION = 2
DATASET_METADATA_SCHEMA_VERSION = 1
DATA_CONTRACT_PATH = "docs/data_contract.md"
METADATA_TEMPLATE_PATH = "docs/metadata_template.json"


def require_runtime_environment(*, require_newton: bool = False, newton_root: str | Path | None = None) -> None:
    """Fail early when a workflow script is launched outside the pinned runtime."""
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    executable = Path(sys.executable)
    executable_looks_right = any(part == CONDA_ENV_NAME for part in executable.parts)
    if conda_env != CONDA_ENV_NAME and not executable_looks_right:
        raise RuntimeError(
            "This workflow is pinned to the conda environment 'palpation'. "
            "Run `conda activate palpation` first, or use `conda run -n palpation python ...`."
        )

    if require_newton:
        root = Path(newton_root) if newton_root is not None else DEFAULT_NEWTON_ROOT
        root = root.expanduser()
        if root != DEFAULT_NEWTON_ROOT:
            raise RuntimeError(f"This workflow is pinned to Newton root {DEFAULT_NEWTON_ROOT}; got {root}")
        if not root.exists():
            raise RuntimeError(f"Newton root not found: {root}")


def runtime_metadata(*, newton_root: str | Path | None = None, device: str | None = None) -> dict[str, object]:
    root = Path(newton_root) if newton_root is not None else DEFAULT_NEWTON_ROOT
    return {
        "python": {
            "environment_manager": "conda",
            "environment_name": CONDA_ENV_NAME,
            "executable": sys.executable,
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        },
        "newton": {
            "root": str(root.expanduser()),
            "simulator_entrypoint": SIMULATOR_ENTRYPOINT,
        },
        "device": {
            "required": REQUIRED_NEWTON_DEVICE,
            "requested": device or REQUIRED_NEWTON_DEVICE,
        },
    }


def metadata_contract() -> dict[str, object]:
    return {
        "version": DATA_CONTRACT_VERSION,
        "definition": DATA_CONTRACT_PATH,
        "template": METADATA_TEMPLATE_PATH,
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def resolve_required_torch_cuda_device(torch_module: Any, requested: str | None = "cuda") -> Any:
    device_name = "cuda" if requested is None or str(requested).strip() == "" else str(requested).strip()
    if device_name.lower() == "auto":
        raise RuntimeError("Device 'auto' is disabled. Use a CUDA device such as 'cuda' or 'cuda:0'.")
    if device_name == "cuda":
        device_name = REQUIRED_NEWTON_DEVICE
    if device_name != REQUIRED_NEWTON_DEVICE:
        raise RuntimeError(f"This workflow is pinned to CUDA device '{REQUIRED_NEWTON_DEVICE}'; got '{device_name}'.")
    if not device_name.startswith("cuda"):
        raise RuntimeError(f"This workflow requires a CUDA GPU device; got '{device_name}'.")
    if not torch_module.cuda.is_available():
        raise RuntimeError(f"CUDA is not available, but this workflow requires '{device_name}'.")
    if ":" in device_name:
        try:
            index = int(device_name.split(":", 1)[1])
        except ValueError as exc:
            raise RuntimeError(f"Invalid CUDA device name: {device_name}") from exc
        if index >= int(torch_module.cuda.device_count()):
            raise RuntimeError(
                f"Requested CUDA device '{device_name}' is not visible. "
                f"torch.cuda.device_count()={torch_module.cuda.device_count()}"
            )
    return torch_module.device(device_name)
