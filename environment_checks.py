from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

GIB = 1024**3
DEFAULT_STORAGE_SETTINGS = {
    "total_limit_gb": 100.0,
    "model_limit_gb": 80.0,
    "warn_percent": 80.0,
    "stop_percent": 95.0,
    "system_reserve_gb": 10.0,
}


@dataclass(frozen=True)
class GpuRuntime:
    vendor: str = "none"
    name: str = ""
    vram_gb: float | None = None
    driver_version: str = ""
    cuda_driver_version: str = ""
    rocm_version: str = ""
    detail: str = ""


@dataclass(frozen=True)
class TorchRuntime:
    interpreter: str
    version: str = ""
    cuda_version: str = ""
    hip_version: str = ""
    available: bool = False
    device: str = ""
    vram_gb: float | None = None
    error: str = ""


@dataclass(frozen=True)
class ContainerSnapshot:
    image: str
    os_name: str
    python_version: str
    gpu: GpuRuntime
    torch: TorchRuntime
    ram_gb: float | None
    home: str
    disk_total_gb: float
    disk_free_gb: float


@dataclass(frozen=True)
class ModelRequirement:
    key: str
    label: str
    kind: str
    vendor: str
    min_cuda: str = ""
    min_rocm: str = ""
    min_torch: str = ""
    min_vram_gb: float | None = None
    min_ram_gb: float | None = None
    min_storage_gb: float | None = None
    note: str = ""


@dataclass(frozen=True)
class CompatibilityCheck:
    label: str
    current: str
    required: str
    status: str
    detail: str = ""


@dataclass(frozen=True)
class ModelCompatibility:
    model: ModelRequirement
    status: str
    checks: tuple[CompatibilityCheck, ...]


@dataclass(frozen=True)
class StorageSettings:
    total_limit_gb: float
    model_limit_gb: float
    warn_percent: float
    stop_percent: float
    system_reserve_gb: float


MODEL_REQUIREMENTS = (
    ModelRequirement(
        key="h3-image",
        label="MiniMax H3 图片 LoRA",
        kind="image",
        vendor="nvidia",
        min_cuda="12.8",
        min_torch="2.10",
        min_vram_gb=22.0,
        min_ram_gb=24.0,
        min_storage_gb=80.0,
        note="EntroPack 仅支持 NVIDIA CUDA；AMD ROCm 不可用。",
    ),
    ModelRequirement(
        key="sdxl",
        label="SDXL 图片 LoRA",
        kind="image",
        vendor="any",
        min_torch="2.0",
        min_vram_gb=12.0,
        min_ram_gb=16.0,
        min_storage_gb=20.0,
        note="使用当前加速训练环境；显存要求随分辨率与优化器变化。",
    ),
    ModelRequirement(
        key="flux",
        label="FLUX.1 图片 LoRA",
        kind="image",
        vendor="nvidia",
        min_cuda="12.8",
        min_torch="2.4",
        min_vram_gb=20.0,
        min_ram_gb=24.0,
        min_storage_gb=40.0,
        note="需要 CLIP-L、T5XXL 与 AE；当前主路径按 NVIDIA CUDA 校验。",
    ),
    ModelRequirement(
        key="wan-1.3b",
        label="Wan 2.1 T2V 1.3B LoRA",
        kind="video",
        vendor="any",
        min_torch="2.0",
        min_vram_gb=12.0,
        min_ram_gb=16.0,
        min_storage_gb=20.0,
        note="AMD ROCm 为实验路径，自动关闭 FP8、T5 FP8、cuDNN 与 adamw8bit。",
    ),
    ModelRequirement(
        key="wan-14b",
        label="Wan 2.1 T2V 14B LoRA",
        kind="video",
        vendor="any",
        min_torch="2.0",
        min_vram_gb=24.0,
        min_ram_gb=32.0,
        min_storage_gb=60.0,
        note="AMD ROCm 为实验路径，建议先使用小分辨率与少帧数验证。",
    ),
    ModelRequirement(
        key="wan-2.2-a14b",
        label="Wan 2.2 T2V A14B LoRA",
        kind="video",
        vendor="any",
        min_torch="2.0",
        min_vram_gb=24.0,
        min_ram_gb=32.0,
        min_storage_gb=60.0,
        note="AMD ROCm 为实验路径，建议先使用小分辨率与少帧数验证。",
    ),
)


def _run_text(command: list[str], timeout: int = 15) -> str:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return ""
    return (result.stdout or result.stderr or "").strip()


def _version_tuple(value: str) -> tuple[int, ...]:
    match = re.search(r"\d+(?:\.\d+)*", str(value or ""))
    if not match:
        return ()
    return tuple(int(part) for part in match.group(0).split("."))


def _version_at_least(value: str, minimum: str) -> bool:
    actual = _version_tuple(value)
    required = _version_tuple(minimum)
    if not actual or not required:
        return False
    length = max(len(actual), len(required))
    return actual + (0,) * (length - len(actual)) >= required + (0,) * (length - len(required))


def _read_os_name() -> str:
    release = Path("/etc/os-release")
    try:
        values = {}
        for line in release.read_text(encoding="utf-8", errors="replace").splitlines():
            key, separator, value = line.partition("=")
            if separator:
                values[key] = value.strip().strip('"')
        return values.get("PRETTY_NAME") or values.get("NAME") or platform.platform()
    except OSError:
        return platform.platform()


def _read_ram_gb() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().total / GIB
    except (ImportError, OSError):
        pass
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024 / GIB
    except (OSError, ValueError, IndexError):
        pass
    return None


def detect_nvidia_gpu() -> GpuRuntime | None:
    query = _run_text(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"]
    )
    if not query:
        return None
    first_line = query.splitlines()[0]
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) < 3:
        return GpuRuntime(vendor="nvidia", detail=f"无法解析 nvidia-smi 输出：{first_line}")
    name, memory_text, driver = parts[-3], parts[-2], parts[-1]
    try:
        vram_gb = float(memory_text) / 1024
    except (TypeError, ValueError):
        vram_gb = None
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", _run_text(["nvidia-smi"]))
    return GpuRuntime(
        vendor="nvidia",
        name=name.strip(),
        vram_gb=vram_gb,
        driver_version=driver.strip(),
        cuda_driver_version=cuda_match.group(1) if cuda_match else "",
        detail="NVIDIA CUDA",
    )


def detect_rocm_gpu() -> GpuRuntime | None:
    markers = []
    configured = os.environ.get("TRAINER_ROCM_INFO", "").strip()
    if configured:
        markers.append(configured)
    root = Path(os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH") or "/opt/rocm")
    version = ""
    try:
        version = (root / ".info/version").read_text(encoding="utf-8").strip()
    except OSError:
        pass
    if version:
        markers.append(f"ROCm {version}")
    elif root.is_dir():
        markers.append(f"ROCm 位于 {root}")
    elif shutil.which("rocm-smi") or shutil.which("hipconfig"):
        markers.append("已检测到 ROCm 工具")
    if not markers:
        return None
    name = ""
    rocminfo = _run_text(["rocminfo"])
    for line in rocminfo.splitlines():
        if "Marketing Name:" in line:
            name = line.split("Marketing Name:", 1)[1].strip()
            break
    return GpuRuntime(
        vendor="rocm",
        name=name or "AMD ROCm GPU",
        rocm_version=version,
        detail="；".join(dict.fromkeys(markers)),
    )


def detect_gpu() -> GpuRuntime:
    nvidia = detect_nvidia_gpu()
    rocm = detect_rocm_gpu()
    if nvidia:
        if rocm:
            return GpuRuntime(
                vendor=nvidia.vendor,
                name=nvidia.name,
                vram_gb=nvidia.vram_gb,
                driver_version=nvidia.driver_version,
                cuda_driver_version=nvidia.cuda_driver_version,
                detail=f"NVIDIA CUDA；同时检测到 AMD {rocm.detail}",
            )
        return nvidia
    if rocm:
        return rocm
    return GpuRuntime(detail="未检测到 NVIDIA CUDA 或 AMD ROCm")


TORCH_PROBE = """
import json
payload = {}
try:
    import torch
    payload.update({
        "version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda or ""),
        "hip_version": str(getattr(torch.version, "hip", "") or ""),
    })
    try:
        payload["available"] = bool(torch.cuda.is_available())
        if payload["available"]:
            payload["device"] = str(torch.cuda.get_device_name(0))
            _, total = torch.cuda.mem_get_info()
            payload["vram_gb"] = round(float(total) / (1024 ** 3), 2)
    except Exception as exc:
        payload["available"] = False
        payload["error"] = f"{type(exc).__name__}: {exc}"
except Exception as exc:
    payload["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(payload, ensure_ascii=False))
"""


def inspect_torch(interpreter: Path | str) -> TorchRuntime:
    executable = Path(interpreter).expanduser()
    if not executable.is_file():
        return TorchRuntime(interpreter=str(executable), error="Python 解释器不存在")
    raw = _run_text([str(executable), "-c", TORCH_PROBE], timeout=30)
    try:
        payload = json.loads(raw.splitlines()[-1]) if raw else {}
    except (json.JSONDecodeError, IndexError):
        payload = {"error": raw or "无法读取 PyTorch 运行时信息"}
    return TorchRuntime(
        interpreter=str(executable),
        version=str(payload.get("version") or ""),
        cuda_version=str(payload.get("cuda_version") or ""),
        hip_version=str(payload.get("hip_version") or ""),
        available=bool(payload.get("available")),
        device=str(payload.get("device") or ""),
        vram_gb=float(payload["vram_gb"]) if payload.get("vram_gb") is not None else None,
        error=str(payload.get("error") or ""),
    )


def inspect_modules(interpreter: Path | str, modules: Iterable[str]) -> dict[str, bool]:
    executable = Path(interpreter).expanduser()
    names = sorted(dict.fromkeys(str(name) for name in modules))
    if not executable.is_file() or not names:
        return {name: False for name in names}
    code = (
        "import importlib.util, json; "
        f"names={names!r}; "
        "print(json.dumps({name: importlib.util.find_spec(name) is not None for name in names}))"
    )
    raw = _run_text([str(executable), "-c", code], timeout=30)
    try:
        payload = json.loads(raw.splitlines()[-1]) if raw else {}
    except (json.JSONDecodeError, IndexError):
        payload = {}
    return {name: bool(payload.get(name)) for name in names}


def inspect_container(
    home: Path,
    task_python: Path | str,
    base_image: str,
) -> ContainerSnapshot:
    disk = shutil.disk_usage(str(home))
    return ContainerSnapshot(
        image=base_image,
        os_name=_read_os_name(),
        python_version=platform.python_version(),
        gpu=detect_gpu(),
        torch=inspect_torch(task_python),
        ram_gb=_read_ram_gb(),
        home=str(home),
        disk_total_gb=disk.total / GIB,
        disk_free_gb=disk.free / GIB,
    )


def _check(
    label: str,
    current: str,
    required: str,
    status: str,
    detail: str = "",
) -> CompatibilityCheck:
    return CompatibilityCheck(label, current, required, status, detail)


def evaluate_model(
    snapshot: ContainerSnapshot,
    model: ModelRequirement,
    available_storage_gb: float | None = None,
) -> ModelCompatibility:
    checks: list[CompatibilityCheck] = []
    gpu = snapshot.gpu
    torch = snapshot.torch

    vendor_ok = (
        (model.vendor == "any" and gpu.vendor in {"nvidia", "rocm"})
        or (model.vendor == "nvidia" and gpu.vendor == "nvidia")
        or (model.vendor == "rocm" and gpu.vendor == "rocm")
    )
    checks.append(
        _check(
            "GPU 类型",
            "NVIDIA CUDA" if gpu.vendor == "nvidia" else "AMD ROCm" if gpu.vendor == "rocm" else "未检测到",
            "NVIDIA" if model.vendor == "nvidia" else "AMD" if model.vendor == "rocm" else "NVIDIA / AMD",
            "pass" if vendor_ok else "block",
            model.note if not vendor_ok else "",
        )
    )

    if gpu.vendor == "nvidia" and model.min_cuda:
        cuda_status = "pass" if _version_at_least(gpu.cuda_driver_version, model.min_cuda) else "block"
        if not gpu.cuda_driver_version:
            cuda_status = "warn"
        checks.append(
            _check("CUDA 驱动", gpu.cuda_driver_version or "未检测到", f">= {model.min_cuda}", cuda_status)
        )
    elif gpu.vendor == "rocm" and model.min_rocm:
        rocm_status = "pass" if _version_at_least(gpu.rocm_version, model.min_rocm) else "warn"
        checks.append(
            _check("ROCm", gpu.rocm_version or gpu.detail or "未检测到版本", f">= {model.min_rocm}", rocm_status)
        )

    if model.min_torch:
        torch_status = "pass" if _version_at_least(torch.version, model.min_torch) else "block"
        if not torch.version:
            torch_status = "warn"
        checks.append(
            _check("PyTorch", torch.version or torch.error or "未安装", f">= {model.min_torch}", torch_status)
        )

    built_for_gpu = bool(torch.cuda_version) if gpu.vendor == "nvidia" else bool(torch.hip_version)
    runtime_status = "pass" if torch.available and built_for_gpu else "block"
    runtime_label = "CUDA" if gpu.vendor == "nvidia" else "HIP"
    runtime_version = torch.cuda_version if gpu.vendor == "nvidia" else torch.hip_version
    checks.append(
        _check(
            f"PyTorch {runtime_label}",
            (runtime_version or "无") + (" · 可用" if torch.available else " · 不可用"),
            f"与 {runtime_label} 匹配",
            runtime_status,
            torch.error,
        )
    )

    if model.min_vram_gb is not None:
        if gpu.vram_gb is not None:
            vram_status = "pass" if gpu.vram_gb >= model.min_vram_gb else "block"
        elif torch.vram_gb is not None:
            vram_status = "pass" if torch.vram_gb >= model.min_vram_gb else "block"
        else:
            vram_status = "warn"
        vram_value = gpu.vram_gb if gpu.vram_gb is not None else torch.vram_gb
        checks.append(
            _check(
                "显存",
                f"{vram_value:.1f} GB" if vram_value is not None else "未知",
                f">= {model.min_vram_gb:.0f} GB",
                vram_status,
            )
        )

    if model.min_ram_gb is not None:
        if snapshot.ram_gb is None:
            ram_status = "warn"
            ram_value = "未知"
        else:
            ram_status = "pass" if snapshot.ram_gb >= model.min_ram_gb else "block"
            ram_value = f"{snapshot.ram_gb:.1f} GB"
        checks.append(_check("系统内存", ram_value, f">= {model.min_ram_gb:.0f} GB", ram_status))

    if model.min_storage_gb is not None:
        if available_storage_gb is None:
            storage_status = "warn"
            storage_value = "未知"
        else:
            storage_status = "pass" if available_storage_gb >= model.min_storage_gb else "block"
            storage_value = f"{available_storage_gb:.1f} GB"
        checks.append(
            _check("可用存储", storage_value, f">= {model.min_storage_gb:.0f} GB", storage_status)
        )

    if any(check.status == "block" for check in checks):
        overall = "block"
    elif any(check.status == "warn" for check in checks):
        overall = "warn"
    else:
        overall = "pass"
    if gpu.vendor == "rocm" and model.vendor == "any" and overall == "pass":
        overall = "warn"
    return ModelCompatibility(model=model, status=overall, checks=tuple(checks))


def evaluate_models(
    snapshot: ContainerSnapshot,
    models: Iterable[ModelRequirement] = MODEL_REQUIREMENTS,
    available_storage_gb: float | None = None,
) -> tuple[ModelCompatibility, ...]:
    return tuple(evaluate_model(snapshot, model, available_storage_gb) for model in models)


def normalize_storage_settings(
    values: Mapping[str, object],
    defaults: Mapping[str, object] | None = None,
) -> StorageSettings:
    merged = dict(DEFAULT_STORAGE_SETTINGS)
    if defaults:
        merged.update(defaults)
    merged.update({key: value for key, value in values.items() if value is not None})
    try:
        total = float(merged["total_limit_gb"])
        model = float(merged["model_limit_gb"])
        warn = float(merged["warn_percent"])
        stop = float(merged["stop_percent"])
        reserve = float(merged["system_reserve_gb"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("存储配置必须是有效数字") from exc
    if total <= 0:
        raise ValueError("环境总存储上限必须大于 0")
    if model <= 0:
        raise ValueError("模型存储上限必须大于 0")
    if model > total:
        raise ValueError("模型存储上限不能大于环境总存储上限")
    if reserve < 0:
        raise ValueError("系统安全预留不能为负数")
    if not 0 < warn < stop <= 100:
        raise ValueError("警告阈值必须小于停止阈值，且两者都在 0–100 之间")
    return StorageSettings(
        total_limit_gb=round(total, 2),
        model_limit_gb=round(model, 2),
        warn_percent=round(warn, 2),
        stop_percent=round(stop, 2),
        system_reserve_gb=round(reserve, 2),
    )


def load_storage_settings(
    path: Path,
    defaults: Mapping[str, object] | None = None,
) -> StorageSettings:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("配置根节点必须是对象")
        return normalize_storage_settings(payload, defaults)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        try:
            return normalize_storage_settings({}, defaults)
        except ValueError:
            return normalize_storage_settings({})


def save_storage_settings(
    path: Path,
    values: Mapping[str, object],
    defaults: Mapping[str, object] | None = None,
) -> StorageSettings:
    settings = normalize_storage_settings(values, defaults)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "total_limit_gb": settings.total_limit_gb,
        "model_limit_gb": settings.model_limit_gb,
        "warn_percent": settings.warn_percent,
        "stop_percent": settings.stop_percent,
        "system_reserve_gb": settings.system_reserve_gb,
    }
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, path)
    return settings


def effective_storage_limit(
    settings: StorageSettings,
    used_gb: float,
    disk_free_gb: float,
) -> float:
    safe_remaining = max(0.0, disk_free_gb - settings.system_reserve_gb)
    return max(0.0, min(settings.total_limit_gb, used_gb + safe_remaining))


def storage_status(
    settings: StorageSettings,
    used_gb: float,
    effective_limit_gb: float,
) -> str:
    if effective_limit_gb <= 0:
        return "block"
    percent = used_gb / effective_limit_gb * 100
    if percent >= settings.stop_percent:
        return "block"
    if percent >= settings.warn_percent:
        return "warn"
    return "pass"

