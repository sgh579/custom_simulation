#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Print CUDA/NVIDIA device visibility diagnostics.")
    parser.add_argument(
        "--newton-root",
        type=Path,
        default=Path("/home/guoheng/newton"),
        help="Optional Newton source root to prepend before importing warp/newton.",
    )
    parser.add_argument("--skip-nvidia-smi", action="store_true", help="Do not run nvidia-smi.")
    args = parser.parse_args()

    print_section("Python")
    print(f"executable: {sys.executable}")
    print(f"version: {sys.version.replace(os.linesep, ' ')}")
    print(f"conda env: {os.environ.get('CONDA_DEFAULT_ENV')}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")

    if not args.skip_nvidia_smi:
        print_section("nvidia-smi")
        run_command(["nvidia-smi", "-L"])
        run_command(["nvidia-smi"])

    print_section("CUDA Driver API")
    check_cuda_driver()

    print_section("Torch")
    check_torch()

    print_section("Warp")
    check_warp(args.newton_root)


def print_section(title: str) -> None:
    print()
    print(f"==== {title} ====")


def run_command(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=False, text=True, capture_output=True)
    except FileNotFoundError as exc:
        print(f"not found: {exc}")
        return
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip())
    print(f"exit code: {result.returncode}")


def check_cuda_driver() -> None:
    found = ctypes.util.find_library("cuda")
    print(f"ctypes.find_library('cuda'): {found}")
    try:
        lib = ctypes.CDLL("libcuda.so.1")
    except OSError as exc:
        print(f"load libcuda.so.1 failed: {type(exc).__name__}: {exc}")
        return

    version = ctypes.c_int()
    ret_version = lib.cuDriverGetVersion(ctypes.byref(version))
    print(f"cuDriverGetVersion ret={ret_version} version={version.value}")

    ret_init = lib.cuInit(0)
    print(f"cuInit ret={ret_init} ({cuda_error_name(ret_init)})")

    count = ctypes.c_int(-1)
    ret_count = lib.cuDeviceGetCount(ctypes.byref(count))
    print(f"cuDeviceGetCount ret={ret_count} ({cuda_error_name(ret_count)}) count={count.value}")

    for index in range(max(count.value, 0)):
        device = ctypes.c_int()
        ret_device = lib.cuDeviceGet(ctypes.byref(device), index)
        name = ctypes.create_string_buffer(256)
        ret_name = lib.cuDeviceGetName(name, len(name), device)
        print(
            "device "
            f"{index}: cuDeviceGet ret={ret_device}, "
            f"cuDeviceGetName ret={ret_name}, name={name.value.decode(errors='replace')}"
        )


def cuda_error_name(code: int) -> str:
    names = {
        0: "CUDA_SUCCESS",
        1: "CUDA_ERROR_INVALID_VALUE",
        2: "CUDA_ERROR_OUT_OF_MEMORY",
        3: "CUDA_ERROR_NOT_INITIALIZED",
        4: "CUDA_ERROR_DEINITIALIZED",
        100: "CUDA_ERROR_NO_DEVICE",
        101: "CUDA_ERROR_INVALID_DEVICE",
        201: "CUDA_ERROR_INVALID_CONTEXT",
        209: "CUDA_ERROR_NO_BINARY_FOR_GPU",
        222: "CUDA_ERROR_UNSUPPORTED_PTX_VERSION",
        700: "CUDA_ERROR_ILLEGAL_ADDRESS",
        702: "CUDA_ERROR_LAUNCH_TIMEOUT",
        719: "CUDA_ERROR_LAUNCH_FAILED",
        801: "CUDA_ERROR_NOT_SUPPORTED",
        802: "CUDA_ERROR_SYSTEM_NOT_READY",
        803: "CUDA_ERROR_SYSTEM_DRIVER_MISMATCH",
        804: "CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE",
        999: "CUDA_ERROR_UNKNOWN",
    }
    return names.get(int(code), "unknown CUDA error")


def check_torch() -> None:
    try:
        import torch
    except Exception as exc:
        print(f"import torch failed: {type(exc).__name__}: {exc}")
        return

    print(f"torch version: {getattr(torch, '__version__', 'unknown')}")
    print(f"torch CUDA build: {getattr(torch.version, 'cuda', None)}")
    try:
        available = torch.cuda.is_available()
        count = torch.cuda.device_count()
        print(f"torch.cuda.is_available: {available}")
        print(f"torch.cuda.device_count: {count}")
        for index in range(count):
            try:
                print(f"torch device {index}: {torch.cuda.get_device_name(index)}")
            except Exception as exc:
                print(f"torch device {index} name failed: {type(exc).__name__}: {exc}")
    except Exception as exc:
        print(f"torch CUDA query failed: {type(exc).__name__}: {exc}")


def check_warp(newton_root: Path) -> None:
    if newton_root.exists():
        sys.path.insert(0, str(newton_root))
        print(f"prepended newton root: {newton_root}")
    else:
        print(f"newton root not found: {newton_root}")

    try:
        import warp as wp
    except Exception as exc:
        print(f"import warp failed: {type(exc).__name__}: {exc}")
        return

    print(f"warp version: {getattr(wp, '__version__', 'unknown')}")
    try:
        devices = [str(device) for device in wp.get_devices()]
        print(f"warp devices: {devices}")
    except Exception as exc:
        print(f"warp get_devices failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
