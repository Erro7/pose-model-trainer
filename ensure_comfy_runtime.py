#!/usr/bin/env python3
"""Prepare an isolated ComfyUI runtime with a CUDA or ROCm PyTorch build."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HOME = Path(os.environ.get("TRAINER_HOME", "/mnt/workspace/pose-model-trainer-data")).expanduser().resolve()
COMFY_ROOT = Path(os.environ.get("TRAINER_COMFY_DIR", str(HOME / "repos" / "ComfyUI"))).expanduser().resolve()
PYPI_INDEX = os.environ.get("PIP_INDEX_URL", "https://mirrors.aliyun.com/pypi/simple")
RUNTIME_PROFILE = os.environ.get("TRAINER_RUNTIME_PROFILE", "").strip().lower()
GPU_VENDOR = os.environ.get("TRAINER_GPU_VENDOR", "").strip().lower()


def gpu_vendor() -> str:
    if GPU_VENDOR in {"nvidia", "rocm"}:
        return GPU_VENDOR
    if shutil.which("nvidia-smi"):
        return "nvidia"
    if shutil.which("rocm-smi") or Path("/opt/rocm").is_dir():
        return "rocm"
    return "unknown"


def torch_profile() -> tuple[str, str, str]:
    if RUNTIME_PROFILE == "cu130":
        return "2.13.0", "0.28.0", "https://download.pytorch.org/whl/cu130"
    if RUNTIME_PROFILE == "cu128":
        return "2.10.0", "0.25.0", "https://download.pytorch.org/whl/cu128"
    return "2.10.0", "0.25.0", "https://download.pytorch.org/whl/cu128"


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("[comfy] $ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def pip_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    env.setdefault("PIP_DEFAULT_TIMEOUT", "120")
    env.setdefault("PIP_RETRIES", "5")
    return env


def torch_ready() -> bool:
    code = r"""
import sys
import torch
hip = str(getattr(torch.version, "hip", "") or "")
cuda = str(torch.version.cuda or "")
available = bool(torch.cuda.is_available())
print(f"torch={torch.__version__} cuda={cuda or 'none'} hip={hip or 'none'} available={available}")
if not available:
    raise SystemExit(1)
if sys.argv[1] == "rocm" and not hip:
    raise SystemExit(2)
if sys.argv[1] == "nvidia" and not cuda:
    raise SystemExit(3)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, gpu_vendor()],
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stdout + result.stderr).strip()
    if output:
        print(f"[comfy] torch check:\n{output}", flush=True)
    return result.returncode == 0


def install_nvidia_torch() -> None:
    torch_version, torchvision_version, index = torch_profile()
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--progress-bar",
            "raw",
            "--index-url",
            index,
            f"torch=={torch_version}",
            f"torchvision=={torchvision_version}",
        ],
        env=pip_env(),
    )


def ensure_comfy_source() -> None:
    if (COMFY_ROOT / "main.py").is_file():
        if (COMFY_ROOT / ".git").is_dir():
            run(["git", "-C", str(COMFY_ROOT), "pull", "--ff-only"])
        return
    COMFY_ROOT.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", "https://github.com/comfyanonymous/ComfyUI.git", str(COMFY_ROOT)])


def install_comfy_requirements() -> None:
    requirements = COMFY_ROOT / "requirements.txt"
    if not requirements.is_file():
        raise SystemExit(f"[comfy] requirements.txt not found: {requirements}")
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--progress-bar",
            "raw",
            "--index-url",
            PYPI_INDEX,
            "--extra-index-url",
            "https://pypi.org/simple",
            "-r",
            str(requirements),
        ],
        env=pip_env(),
    )


def main() -> None:
    if sys.prefix == sys.base_prefix:
        raise SystemExit("[comfy] run this installer inside the isolated .venv-comfy environment")
    vendor = gpu_vendor()
    print(f"[comfy] GPU vendor: {vendor}", flush=True)
    print(f"[comfy] runtime profile: {RUNTIME_PROFILE or 'auto'}", flush=True)
    print(f"[comfy] python: {sys.executable}", flush=True)
    if vendor == "unknown":
        raise SystemExit("[comfy] NVIDIA CUDA or AMD ROCm is required")

    if vendor == "nvidia" and not torch_ready():
        install_nvidia_torch()
        if not torch_ready():
            raise SystemExit("[comfy] PyTorch CUDA remains unavailable after installation")
    elif vendor == "rocm" and not torch_ready():
        raise SystemExit(
            "[comfy] ROCm PyTorch is not visible in this environment. "
            "Recreate .venv-comfy with --system-site-packages from a ROCm container that has torch HIP installed."
        )

    ensure_comfy_source()
    install_comfy_requirements()
    if not torch_ready():
        raise SystemExit("[comfy] PyTorch validation failed after ComfyUI dependency installation")
    print("[comfy] ComfyUI runtime is ready", flush=True)


if __name__ == "__main__":
    main()

