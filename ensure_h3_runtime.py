#!/usr/bin/env python3
"""Ensure the isolated H3 runtime matches the selected DSW image profile."""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import fcntl
except ImportError:
    fcntl = None

HOME = Path(os.environ.get("TRAINER_HOME", "/mnt/workspace/pose-model-trainer-data")).expanduser().resolve()
DIFFSYNTH = HOME / "repos" / "DiffSynth-Studio"
REQUIRED = ("torch", "torchvision", "diffsynth", "entropack", "cupy", "accelerate")
RECOMMENDED_IMAGE = "modelscope:1.40.1-pytorch2.13.0-gpu-py312-cu130-ubuntu22.04"
BASE_IMAGE = os.environ.get(
    "TRAINER_BASE_IMAGE",
    RECOMMENDED_IMAGE,
)

RUNTIME_PROFILE = os.environ.get("TRAINER_RUNTIME_PROFILE", "").lower()

if RUNTIME_PROFILE == "rocm" or (not RUNTIME_PROFILE and "rocm" in BASE_IMAGE.lower()):
    PROFILE = "rocm"
    DEFAULT_TORCH = ""
    DEFAULT_TORCHVISION = ""
    DEFAULT_INDEX = ""
    ENTROPACK_EXTRA = ""
    EXPECTED_CUDA = ""
elif RUNTIME_PROFILE == "cu130" or (not RUNTIME_PROFILE and "cu130" in BASE_IMAGE.lower()):
    PROFILE = "cu130"
    DEFAULT_TORCH = "2.13.0"
    DEFAULT_TORCHVISION = "0.28.0"
    DEFAULT_INDEX = "https://download.pytorch.org/whl/cu130"
    ENTROPACK_EXTRA = "cuda13"
    EXPECTED_CUDA = "13.0"
else:
    PROFILE = "cu128"
    DEFAULT_TORCH = "2.10.0"
    DEFAULT_TORCHVISION = "0.25.0"
    DEFAULT_INDEX = "https://download.pytorch.org/whl/cu128"
    ENTROPACK_EXTRA = "cuda12"
    EXPECTED_CUDA = "12.8"

TORCH_VERSION = os.environ.get("H3_TORCH_VERSION", DEFAULT_TORCH)
TORCHVISION_VERSION = os.environ.get("H3_TORCHVISION_VERSION", DEFAULT_TORCHVISION)
TORCH_INDEX = os.environ.get("H3_TORCH_INDEX", DEFAULT_INDEX)
PYPI_INDEX = os.environ.get(
    "PIP_INDEX_URL",
    "https://mirrors.aliyun.com/pypi/simple",
)
PROFILE_MARKER = Path(sys.prefix) / f".h3-runtime-{PROFILE}-torch{TORCH_VERSION}"
LOCK_FILE = HOME / "logs" / "h3-runtime.lock"
PIP_CACHE_DIR = Path(
    os.environ.get("TRAINER_PIP_CACHE_DIR", str(HOME / "pip-cache"))
).expanduser().resolve()

CUDA_CHECK = f"""
import torch
print(f'torch={{torch.__version__}} built_cuda={{torch.version.cuda}}')
installed = torch.__version__.split('+')[0]
if installed != '{TORCH_VERSION}':
    raise SystemExit(f'expected torch {TORCH_VERSION}, found {{installed}}')
if not str(torch.version.cuda).startswith('{EXPECTED_CUDA}'):
    raise SystemExit(f'expected CUDA {EXPECTED_CUDA}, found {{torch.version.cuda}}')
if not torch.cuda.is_available():
    raise SystemExit('torch.cuda.is_available() is false')
free, total = torch.cuda.mem_get_info()
print(f'cuda_ready=true gpu={{torch.cuda.get_device_name(0)}} vram_gb={{total / 2**30:.1f}}')
"""


def missing_modules() -> list[str]:
    return [name for name in REQUIRED if importlib.util.find_spec(name) is None]


def rocm_runtime_info() -> str:
    markers = []
    if "rocm" in BASE_IMAGE.lower():
        markers.append(f"image {BASE_IMAGE}")

    try:
        import torch

        hip = getattr(torch.version, "hip", None)
        if hip:
            markers.append(f"torch HIP {hip}")
    except Exception:
        pass

    rocm_root = Path(os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH") or "/opt/rocm")
    if rocm_root.is_dir():
        version = ""
        try:
            version = (rocm_root / ".info/version").read_text(encoding="utf-8").strip()
        except OSError:
            pass
        markers.append(f"ROCm {version}" if version else f"ROCm at {rocm_root}")
    elif shutil.which("rocm-smi") or shutil.which("hipconfig"):
        markers.append("ROCm tools detected")

    return ", ".join(dict.fromkeys(markers))


def gpu_support_error(rocm_info: str, nvidia_visible: bool) -> str | None:
    if nvidia_visible:
        return None
    if rocm_info:
        return (
            "The H3 workflow is not supported on AMD ROCm. "
            "H3-Image EntroPack is NVIDIA CUDA-only, and H3 video FL2VA/Ref2VA is not integrated in this launcher. "
            f"Detected: {rocm_info}. "
            f"Use an NVIDIA CUDA GPU Notebook with {RECOMMENDED_IMAGE}."
        )
    return (
        "NVIDIA GPU is not visible and AMD ROCm was not detected. "
        f"Use an NVIDIA CUDA GPU Notebook with {RECOMMENDED_IMAGE}."
    )


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("[runtime] $ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def pip_env() -> dict[str, str]:
    PIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PIP_CACHE_DIR", str(PIP_CACHE_DIR))
    env.setdefault("PIP_DEFAULT_TIMEOUT", "120")
    env.setdefault("PIP_RETRIES", "5")
    return env


def cuda_status() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-c", CUDA_CHECK],
        text=True,
        capture_output=True,
    )
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def acquire_lock():
    if fcntl is None:
        return None
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    handle = LOCK_FILE.open("w", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise SystemExit("[runtime] another H3 runtime setup is already running")
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def install_compatible_torch() -> None:
    print(
        f"[runtime] installing {PROFILE}: "
        f"torch={TORCH_VERSION} torchvision={TORCHVISION_VERSION}",
        flush=True,
    )
    run([
        sys.executable, "-m", "pip", "install",
        "--cache-dir", str(PIP_CACHE_DIR), "--force-reinstall",
        "--progress-bar", "raw", "--retries", "5", "--timeout", "120",
        f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}",
        "--index-url", TORCH_INDEX,
        "--extra-index-url", PYPI_INDEX,
    ], env=pip_env())


def main() -> None:
    print(f"[runtime] base_image={BASE_IMAGE}", flush=True)
    print(f"[runtime] profile={PROFILE} entropack={ENTROPACK_EXTRA}", flush=True)
    print(f"[runtime] python={sys.executable}", flush=True)
    print(f"[runtime] pypi_index={PYPI_INDEX}", flush=True)
    print(f"[runtime] torch_index={TORCH_INDEX}", flush=True)
    print(f"[runtime] pip_cache={PIP_CACHE_DIR}", flush=True)
    print(f"[runtime] prefix={sys.prefix} base_prefix={sys.base_prefix}", flush=True)
    if sys.prefix == sys.base_prefix:
        raise SystemExit(
            "H3 runtime must run inside the project virtual environment; "
            f"received {sys.executable}"
        )

    rocm_info = rocm_runtime_info()
    nvidia_visible = shutil.which("nvidia-smi") is not None
    if rocm_info and nvidia_visible:
        print(
            f"[runtime] warning: AMD ROCm also detected ({rocm_info}); NVIDIA CUDA will be used",
            flush=True,
        )
    support_error = gpu_support_error(rocm_info, nvidia_visible)
    if support_error:
        raise SystemExit(f"[runtime] {support_error}")

    lock = acquire_lock()
    if lock is not None:
        print("[runtime] acquired setup lock", flush=True)

    cuda_ok, cuda_output = cuda_status()
    if cuda_output:
        print(f"[runtime] CUDA check:\n{cuda_output}", flush=True)
    if not cuda_ok:
        install_compatible_torch()
        cuda_ok, cuda_output = cuda_status()
        print(f"[runtime] CUDA recheck:\n{cuda_output}", flush=True)
        if not cuda_ok:
            raise SystemExit(
                f"CUDA remains unavailable after installing the {PROFILE} PyTorch build."
            )

    missing = missing_modules()
    profile_incomplete = not PROFILE_MARKER.is_file()
    if missing or profile_incomplete:
        if missing:
            print(f"[runtime] missing: {', '.join(missing)}", flush=True)
        if profile_incomplete:
            print(f"[runtime] applying profile: {PROFILE}", flush=True)
        if not (DIFFSYNTH / "pyproject.toml").is_file():
            DIFFSYNTH.parent.mkdir(parents=True, exist_ok=True)
            run([
                "git", "clone", "--depth", "1",
                "https://github.com/modelscope/DiffSynth-Studio.git",
                str(DIFFSYNTH),
            ])
        run([
            sys.executable, "-m", "pip", "install",
            "--cache-dir", str(PIP_CACHE_DIR), "--progress-bar", "raw",
            "--retries", "5", "--timeout", "120",
            "numpy>=2.0,<2.6", "opencv-python<4.12",
            f"entropack[{ENTROPACK_EXTRA}]>=0.1.7",
            str(DIFFSYNTH), "bitsandbytes", "accelerate",
        ], env=pip_env())

    remaining = missing_modules()
    if remaining:
        raise SystemExit("H3 runtime is still incomplete: " + ", ".join(remaining))

    cuda_ok, cuda_output = cuda_status()
    print(f"[runtime] final CUDA check:\n{cuda_output}", flush=True)
    if not cuda_ok:
        raise SystemExit("H3 dependencies are installed, but the pinned CUDA profile failed.")
    PROFILE_MARKER.touch()
    print("[runtime] H3 dependencies are ready", flush=True)


if __name__ == "__main__":
    main()
