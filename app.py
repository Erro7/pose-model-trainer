from __future__ import annotations

import hashlib
import inspect
import json
import math
import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import random
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
import warnings
import zipfile
from collections import Counter, deque
from pathlib import Path

try:
    import gradio as gr
    from loguru import logger
    from PIL import Image
    import psutil
    import toml
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"WebUI 依赖缺失：{exc.name}。请使用 `bash start.sh` 启动，"
        "启动器会自动安装 requirements.txt。"
    ) from exc

from environment_checks import (
    DEFAULT_STORAGE_SETTINGS,
    effective_storage_limit,
    evaluate_models,
    inspect_container,
    inspect_modules,
    load_storage_settings,
    save_storage_settings,
    storage_status,
)
from workflow_schema import (
    BODY_NATURAL,
    BODY_OPTIONS,
    CUSTOM_GENERATION_SIZE,
    DEFAULT_GENERATION_BATCH_PRESET,
    DEFAULT_GENERATION_JOB_NAME,
    DEFAULT_GENERATION_SEED,
    DEFAULT_IMAGE_MODEL,
    GENERATION_BATCH_PRESETS,
    GALLERY_STRATEGY_BALANCED,
    GALLERY_STRATEGY_FIXED_PERSON,
    GALLERY_STRATEGY_MULTI_REF,
    GALLERY_STRATEGY_OPTIONS,
    GALLERY_STRATEGY_RANDOM_PER_ITEM,
    IMAGE_MODEL_CHOICES,
    STYLE_OPTIONS,
    STYLE_PRESERVE,
    body_prompt,
    image_model_spec,
    normalize_creative_profile,
    style_prompt,
    uses_reference_body,
)

ROOT = Path(__file__).resolve().parent
UI_BUILD_SOURCES = (
    Path(__file__),
    Path(__file__).with_name("environment_checks.py"),
    Path(__file__).with_name("workflow_schema.py"),
)
UI_BUILD_ID = hashlib.sha256(
    b"".join(path.read_bytes() for path in UI_BUILD_SOURCES if path.is_file())
).hexdigest()[:12]
HOME = Path(os.environ.get("TRAINER_HOME", Path.home() / "modelscope-trainer-data")).expanduser().resolve()
REPOS = HOME / "repos"
MODELS = HOME / "models"
CONFIGS = HOME / "configs"
OUTPUTS = HOME / "outputs"
LOGS = HOME / "logs"
DATASETS = HOME / "datasets"
POSE_GALLERY = REPOS / "ai-pose-gallery-zh"
LEGACY_POSE_GALLERY = DATASETS / "pose_gallery"
EXPORTS = HOME / "exports"
ENVIRONMENT_CONFIG = CONFIGS / "environment.json"
STORAGE_DEFAULTS = {
    **DEFAULT_STORAGE_SETTINGS,
    "total_limit_gb": float(os.environ.get("TRAINER_TOTAL_BUDGET_GB", "100")),
    "model_limit_gb": float(os.environ.get("TRAINER_MODEL_BUDGET_GB", "80")),
}
ENVIRONMENT_SETTINGS = load_storage_settings(ENVIRONMENT_CONFIG, STORAGE_DEFAULTS)
TOTAL_BUDGET_GB = ENVIRONMENT_SETTINGS.total_limit_gb
MODEL_BUDGET_GB = ENVIRONMENT_SETTINGS.model_limit_gb
DATA_RESERVE_GB = max(0.0, TOTAL_BUDGET_GB - MODEL_BUDGET_GB)
STORAGE_WARN_PERCENT = ENVIRONMENT_SETTINGS.warn_percent
STORAGE_STOP_PERCENT = ENVIRONMENT_SETTINGS.stop_percent
STORAGE_SYSTEM_RESERVE_GB = ENVIRONMENT_SETTINGS.system_reserve_gb
os.environ["TRAINER_TOTAL_BUDGET_GB"] = str(TOTAL_BUDGET_GB)
os.environ["TRAINER_MODEL_BUDGET_GB"] = str(MODEL_BUDGET_GB)
os.environ["TRAINER_STORAGE_RESERVE_GB"] = str(STORAGE_SYSTEM_RESERVE_GB)
RECOMMENDED_IMAGE = "modelscope:1.40.1-pytorch2.13.0-gpu-py312-cu130-ubuntu22.04"
BASE_IMAGE = os.environ.get(
    "TRAINER_BASE_IMAGE",
    RECOMMENDED_IMAGE,
)
RUNTIME_PROFILE = os.environ.get("TRAINER_RUNTIME_PROFILE", "cu130")
GPU_VENDOR = os.environ.get("TRAINER_GPU_VENDOR", "").strip().lower()
for folder in (REPOS, MODELS, CONFIGS, OUTPUTS, LOGS, DATASETS, EXPORTS):
    folder.mkdir(parents=True, exist_ok=True)
UI_BUILD_FILE = LOGS / "ui-build.json"
UI_BUILD_FILE.write_text(json.dumps({"build": UI_BUILD_ID}), encoding="utf-8")
LATEST_LOG = LOGS / "latest.log"
LATEST_LOG.touch(exist_ok=True)
LOG_BUFFER: deque[str] = deque(maxlen=5000)
LOG_LOCK = threading.Lock()
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level} | [webui] {message}"


def _ui_log_sink(message) -> None:
    with LOG_LOCK:
        LOG_BUFFER.append(str(message).rstrip("\n"))


def _seed_log_buffer() -> None:
    try:
        lines = LATEST_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    with LOG_LOCK:
        LOG_BUFFER.extend(lines[-LOG_BUFFER.maxlen:])


logger.remove()
logger.add(sys.stderr, level="INFO", format=LOG_FORMAT)
logger.add(_ui_log_sink, level="INFO", format=LOG_FORMAT)
_seed_log_buffer()
logger.add(
    LATEST_LOG,
    level="INFO",
    format=LOG_FORMAT,
    rotation="20 MB",
    retention=3,
    encoding="utf-8",
    enqueue=True,
    backtrace=False,
    diagnose=False,
)

STATE = {"process": None, "log": None, "task": None, "lock": threading.Lock()}
TASK_STATE_FILE = LOGS / "task-state.json"
TASK_LABELS = {
    "h3-generate": "H3 目标图生成",
    "h3-train": "H3 LoRA 训练",
    "train": "LoRA 训练",
}
INSTALL_LOG = LOGS / "install-backends.log"
INSTALL_STATE = {"running": False, "step": "尚未开始", "returncode": None, "lock": threading.Lock()}

PERSON_MODE_BATCH = GALLERY_STRATEGY_FIXED_PERSON
PERSON_MODE_PER_ITEM = GALLERY_STRATEGY_RANDOM_PER_ITEM
PERSON_MODE_BALANCED = GALLERY_STRATEGY_BALANCED
PERSON_MODE_MULTI_REF = GALLERY_STRATEGY_MULTI_REF
PERSON_MODES = GALLERY_STRATEGY_OPTIONS
OUTFIT_MODE_MULTI_REF = "多图作为同一套服装"
OUTFIT_MODE_RANDOM = "每张作为独立服装随机"
OUTFIT_MODES = (OUTFIT_MODE_MULTI_REF, OUTFIT_MODE_RANDOM)
H3_DEFAULT_OUTPUT_NAME = "h3-character-lora"
H3_QUANTIZED_PROFILE = "量化版 <80GB（推荐）"
H3_BF16_PROFILE = "原始 BF16 两阶段（约 130GB）"
H3_MODEL_PROFILES = (H3_QUANTIZED_PROFILE, H3_BF16_PROFILE)
H3_BF16_MIN_MODEL_GB = 130
H3_BF16_MIN_TOTAL_GB = H3_BF16_MIN_MODEL_GB + 20
H3_AVAILABLE_MODEL_PROFILES = (
    H3_MODEL_PROFILES
    if MODEL_BUDGET_GB >= H3_BF16_MIN_MODEL_GB and TOTAL_BUDGET_GB >= H3_BF16_MIN_TOTAL_GB
    else (H3_QUANTIZED_PROFILE,)
)
H3_DEFAULT_MODEL_PROFILE = H3_MODEL_PROFILES[0]
SUPPORTED_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})
COMMON_GENERATION_SIZES = (
    (512, 512, "1:1"),
    (512, 768, "2:3"),
    (768, 512, "3:2"),
    (576, 768, "3:4"),
    (768, 576, "4:3"),
    (768, 768, "1:1"),
    (576, 1024, "9:16"),
    (1024, 576, "16:9"),
    (768, 1024, "3:4"),
    (1024, 768, "4:3"),
    (720, 1280, "9:16"),
    (1280, 720, "16:9"),
    (832, 1216, "2:3"),
    (1216, 832, "3:2"),
    (1024, 1024, "1:1"),
    (1216, 1216, "1:1"),
)


def q(value: object) -> str:
    return shlex.quote(str(value))


def task_python_path() -> Path:
    configured = os.environ.get("TRAINER_TASK_PYTHON", "").strip()
    candidate = Path(configured) if configured else ROOT / ".venv-h3/bin/python"
    return candidate if candidate.is_file() else Path(sys.executable)


def detect_rocm_runtime() -> str:
    markers = []
    if "rocm" in BASE_IMAGE.lower():
        markers.append(f"镜像 {BASE_IMAGE}")
    rocm_root = Path(os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH") or "/opt/rocm")
    if rocm_root.is_dir():
        version = ""
        try:
            version = (rocm_root / ".info/version").read_text(encoding="utf-8").strip()
        except OSError:
            pass
        markers.append(f"ROCm {version}" if version else f"ROCm 位于 {rocm_root}")
    elif shutil.which("rocm-smi") or shutil.which("hipconfig"):
        markers.append("已检测到 ROCm 工具")
    return "；".join(dict.fromkeys(markers))


ROCM_INFO = os.environ.get("TRAINER_ROCM_INFO", "").strip() or detect_rocm_runtime()


def rocm_video_mode() -> bool:
    return GPU_VENDOR == "rocm" or (bool(ROCM_INFO) and shutil.which("nvidia-smi") is None)


def video_support_info() -> str:
    if rocm_video_mode():
        return "Wan 2.1 / 2.2（AMD ROCm 实验路径，关闭 FP8/cuDNN 加速）；H3 视频未接入"
    return "Wan 2.1 / 2.2（NVIDIA CUDA）；H3 视频 FL2VA/Ref2VA 未接入本启动器"


def h3_gpu_support_error() -> str | None:
    if GPU_VENDOR != "rocm" and shutil.which("nvidia-smi"):
        return None
    if ROCM_INFO:
        return (
            f"AMD ROCm（{ROCM_INFO}）不能运行 H3 Image EntroPack，H3 视频也未接入本启动器。"
            "可在“训练”页使用 Wan 2.1 / 2.2 视频 LoRA（AMD ROCm 实验路径）。"
            f"请改用 {RECOMMENDED_IMAGE}。"
        )
    return f"未检测到 NVIDIA GPU 或 AMD ROCm；请改用 {RECOMMENDED_IMAGE}。"


def generation_hardware() -> dict:
    gpu_name, vram_gib = "", None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        first_line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        if first_line:
            gpu_name, _, memory = first_line.rpartition(",")
            vram_gib = float(memory.strip()) / 1024
    except (FileNotFoundError, ValueError, IndexError):
        pass

    ram_gib = psutil.virtual_memory().total / 2**30
    vram_budget = 0.0
    for threshold, megapixels in ((40, 1.5), (28, 1.05), (18, 0.59), (12, 0.39), (6, 0.26)):
        if vram_gib is not None and vram_gib >= threshold:
            vram_budget = megapixels
            break
    ram_budget = 0.0
    for threshold, megapixels in ((64, 1.5), (40, 1.05), (24, 0.79), (16, 0.59), (12, 0.39), (8, 0.26)):
        if ram_gib >= threshold:
            ram_budget = megapixels
            break
    if vram_gib is None:
        safe_megapixels = 0.26
    else:
        safe_megapixels = min(vram_budget, ram_budget) or 0.26
    return {
        "gpu_name": gpu_name or ("AMD ROCm（H3 不支持）" if ROCM_INFO else "未检测到 NVIDIA GPU"),
        "vram_gib": vram_gib,
        "ram_gib": ram_gib,
        "safe_megapixels": safe_megapixels,
        "rocm_info": ROCM_INFO,
    }


def generation_size_options() -> tuple[list[str], str, str]:
    hardware = generation_hardware()
    safe_megapixels = hardware["safe_megapixels"]
    eligible = [
        (width, height, aspect)
        for width, height, aspect in COMMON_GENERATION_SIZES
        if width * height <= safe_megapixels * 1_000_000
    ]
    if not eligible:
        eligible = [COMMON_GENERATION_SIZES[0]]
    preferred = ((768, 768), (512, 768), (512, 512))
    default_size = next(
        (size for size in preferred if any((size[0], size[1]) == (item[0], item[1]) for item in eligible)),
        (eligible[-1][0], eligible[-1][1]),
    )
    choices = []
    for width, height, aspect in eligible:
        label = f"{aspect} · {width} × {height}"
        if (width, height) == default_size:
            label += " · 推荐"
        choices.append(label)
    choices.append(CUSTOM_GENERATION_SIZE)
    default_choice = next(
        (choice for choice in choices if f"{default_size[0]} × {default_size[1]}" in choice),
        choices[0],
    )

    if hardware["vram_gib"] is None and hardware["rocm_info"]:
        gpu_text = (
            f"AMD ROCm 环境（{hardware['rocm_info']}）；"
            "H3 EntroPack 仅支持 NVIDIA CUDA 12.8+"
        )
    elif hardware["vram_gib"] is None:
        gpu_text = "未检测到 NVIDIA GPU；当前尺寸仅作预览，请先修复环境"
    else:
        gpu_text = f"{hardware['gpu_name']} · {hardware['vram_gib']:.1f} GiB 显存"
    safe_text = f"≤ {safe_megapixels:.2f} MP（约 {int(safe_megapixels * 1_000_000):,} 像素）"
    available = "、".join(f"{width}×{height}" for width, height, _ in eligible)
    guidance = (
        f"{gpu_text} · 内存 {hardware['ram_gib']:.1f} GiB · 安全上限 {safe_text}\n"
        f"可用尺寸：{available}"
    )
    if hardware["rocm_info"] and hardware["vram_gib"] is None:
        guidance += f"\n当前为 AMD ROCm，尺寸仅作预览，无法运行 H3；请改用 {RECOMMENDED_IMAGE}。"
    return choices, default_choice, guidance


def resolve_generation_size(size_choice: str, width: int, height: int) -> tuple[int, int]:
    if size_choice != CUSTOM_GENERATION_SIZE:
        match = re.search(r"(\d+)\s*×\s*(\d+)", size_choice or "")
        if match:
            return int(match.group(1)), int(match.group(2))
    width, height = int(width or 0), int(height or 0)
    if width < 256 or height < 256 or width > 2048 or height > 2048:
        raise gr.Error("自定义宽高需在 256–2048 之间")
    if width % 16 or height % 16:
        raise gr.Error("自定义宽高需为 16 的倍数，例如 768 × 768")
    safe_pixels = int(generation_hardware()["safe_megapixels"] * 1_000_000)
    if width * height > safe_pixels:
        raise gr.Error(
            f"自定义尺寸 {width}×{height} 超出当前硬件的安全预算 {safe_pixels:,} 像素。"
            "请选择列表中的常用尺寸，或降低宽高。"
        )
    return width, height


def refresh_generation_size_ui(current_choice: str):
    choices, default, guidance = generation_size_options()
    value = current_choice if current_choice in choices else default
    return gr.update(choices=choices, value=value), guidance


def custom_generation_size_visibility(size_choice: str):
    return gr.update(visible=size_choice == CUSTOM_GENERATION_SIZE)


def custom_generation_count_visibility(batch_preset: str):
    return gr.update(visible=GENERATION_BATCH_PRESETS.get(batch_preset) is None)


def _image_model_status_markdown(model_label: str) -> str:
    spec = image_model_spec(model_label)
    status = "已接入" if spec.available else "待接入"
    lines = [f"### {spec.label} · {status}"]
    if spec.backend == "h3":
        lines.extend([
            f"模型预算 {MODEL_BUDGET_GB:.0f}GB · 素材/输出预留 {DATA_RESERVE_GB:.0f}GB",
        ])
    else:
        lines.append("批量生成待接入")
    return "\n\n".join(lines)


def update_image_model_ui(model_label: str):
    spec = image_model_spec(model_label)
    task = current_task_snapshot()
    task_active = _task_active(task)
    button_text = "一键生成训练图片" if spec.available else "当前模型批量生成未接入"
    h3_training_available = spec.backend == "h3" and spec.available
    return (
        spec.label,
        gr.update(visible=spec.backend == "h3"),
        gr.update(value=_image_model_status_markdown(spec.label)),
        gr.update(value=button_text, interactive=spec.available and not task_active),
        gr.update(interactive=h3_training_available and not task_active),
        gr.update(interactive=h3_training_available and not task_active),
    )


def py() -> str:
    return q(task_python_path())


def accelerate() -> str:
    return f"{py()} -m accelerate.commands.launch"


def h3_accelerate() -> str:
    return (
        f"{accelerate()} --num_processes 1 --num_machines 1 "
        "--mixed_precision no --dynamo_backend no"
    )


def h3_gpu_lock_command(command: str) -> str:
    lock = LOGS / "h3-gpu.lock"
    return (
        f"mkdir -p {q(lock.parent)} && "
        f"flock -n -E 75 {q(lock)} bash -c {q(command)} || "
        "{ rc=$?; if [ \"$rc\" -eq 75 ]; then "
        "echo '[h3] 已有 H3 GPU 任务正在运行，请等待当前任务结束' >&2; "
        "fi; exit \"$rc\"; }"
    )


def _tree_bytes(paths: list[Path]) -> int:
    total = 0
    seen = set()
    for root in paths:
        if not root.exists():
            continue
        for current, _, files in os.walk(root):
            for filename in files:
                path = Path(current) / filename
                try:
                    stat = path.stat()
                except OSError:
                    continue
                identity = (stat.st_dev, stat.st_ino)
                if identity in seen:
                    continue
                seen.add(identity)
                total += stat.st_size
    return total


def storage_info() -> str:
    gib = 2 ** 30
    model_bytes = _tree_bytes([MODELS])
    total_bytes = _tree_bytes([HOME])
    data_bytes = max(0, total_bytes - model_bytes)
    model_gb = model_bytes / gib
    data_gb = data_bytes / gib
    total_gb = total_bytes / gib
    model_left = max(0.0, MODEL_BUDGET_GB - model_gb)
    data_left = max(0.0, DATA_RESERVE_GB - data_gb)
    status = "正常"
    if model_gb >= MODEL_BUDGET_GB:
        status = "模型预算已满，已阻止继续下载"
    elif total_gb >= TOTAL_BUDGET_GB:
        status = "总预算已满，请清理数据"
    return (
        f"持久数据：{HOME}\n"
        f"WebUI 环境：{ROOT / '.venv'}\n"
        f"H3 任务环境：{ROOT / '.venv-h3'}\n"
        f"状态：{status}\n"
        f"模型：{model_gb:.1f} / {MODEL_BUDGET_GB:.0f} GB（剩余 {model_left:.1f} GB）\n"
        f"生成与输出：{data_gb:.1f} / {DATA_RESERVE_GB:.0f} GB（预留剩余 {data_left:.1f} GB）\n"
        f"项目数据合计：{total_gb:.1f} / {TOTAL_BUDGET_GB:.0f} GB"
    )


def system_info() -> str:
    try:
        gpu_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        gpu = gpu_result.stdout.strip()
    except FileNotFoundError:
        gpu = ""
    if not gpu and ROCM_INFO:
        gpu = f"AMD ROCm（{ROCM_INFO}）；H3 禁用，Wan 视频为实验支持"
    elif gpu and ROCM_INFO:
        gpu = f"{gpu}；同时检测到 AMD ROCm（{ROCM_INFO}），H3 将使用 NVIDIA CUDA"
    else:
        gpu = gpu or "未检测到 NVIDIA GPU"
    disk = psutil.disk_usage(str(HOME))
    venv = "是" if sys.prefix != sys.base_prefix else "否"
    return (
        f"宿主镜像: {BASE_IMAGE}\n"
        f"推荐 H3 镜像: {RECOMMENDED_IMAGE}\n"
        f"H3 运行配置: {RUNTIME_PROFILE}\n"
        f"WebUI Python: {sys.executable}\n"
        f"H3 任务 Python: {task_python_path()}\n"
        f"WebUI 虚拟环境: {venv}\n"
        f"GPU: {gpu}\n"
        f"视频训练: {video_support_info()}\n"
        f"RAM: {psutil.virtual_memory().total / 2**30:.1f} GiB\n"
        f"可用磁盘: {disk.free / 2**30:.1f} GiB\n"
        f"工作目录: {HOME}"
    )


def run_visible(command: str) -> str:
    proc = subprocess.run(["bash", "-c", command], cwd=HOME, text=True, capture_output=True)
    text = (proc.stdout + "\n" + proc.stderr).strip()
    if proc.returncode:
        raise gr.Error(text[-5000:] or f"命令失败，退出码 {proc.returncode}")
    return text[-5000:] or "完成"


def _install_log_tail(limit: int = 500) -> str:
    if not INSTALL_LOG.is_file():
        return "尚无安装日志"
    return "\n".join(INSTALL_LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:])


def read_install_log() -> tuple[str, str]:
    with INSTALL_STATE["lock"]:
        running = INSTALL_STATE["running"]
        step = INSTALL_STATE["step"]
        returncode = INSTALL_STATE["returncode"]
    if running:
        status = f"安装中 · {step}"
    elif returncode is None:
        status = step
    elif returncode == 0:
        status = "安装完成"
    else:
        status = f"安装失败 · 退出码 {returncode} · {step}"
    return status, _install_log_tail()


def install_backends():
    with INSTALL_STATE["lock"]:
        if INSTALL_STATE["running"]:
            yield "已有安装任务正在运行", _install_log_tail()
            return
        INSTALL_STATE.update(running=True, step="准备安装", returncode=None)

    sd = REPOS / "sd-scripts"
    musubi = REPOS / "musubi-tuner"
    diffsynth = REPOS / "DiffSynth-Studio"
    steps = [
        ("1/7 更新 sd-scripts", f"if test -d {q(sd / '.git')}; then git -C {q(sd)} pull --ff-only; else git clone --depth 1 https://github.com/kohya-ss/sd-scripts.git {q(sd)}; fi"),
        ("2/7 更新 musubi-tuner", f"if test -d {q(musubi / '.git')}; then git -C {q(musubi)} pull --ff-only; else git clone --depth 1 https://github.com/kohya-ss/musubi-tuner.git {q(musubi)}; fi"),
        ("3/7 更新 DiffSynth-Studio", f"if test -d {q(diffsynth / '.git')}; then git -C {q(diffsynth)} pull --ff-only; else git clone --depth 1 https://github.com/modelscope/DiffSynth-Studio.git {q(diffsynth)}; fi"),
        ("4/7 安装 sd-scripts 依赖", f"cd {q(sd)} && {py()} -m pip install --progress-bar off -r requirements.txt"),
        ("5/7 安装 musubi-tuner", f"{py()} -m pip install --progress-bar off -e {q(musubi)}"),
        ("6/7 安装 DiffSynth-Studio", f"{py()} -m pip install --progress-bar off {q(diffsynth)}"),
        ("7/7 修复并验证运行依赖", f"{py()} -m pip install --progress-bar off 'numpy>=2.0,<2.6' 'opencv-python<4.12' bitsandbytes accelerate && {py()} -c \"import numpy, pandas, pyarrow, modelscope; print('dependency check OK: python=' + __import__('sys').executable + ' numpy=' + numpy.__version__)\""),
    ]
    INSTALL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with INSTALL_LOG.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n{'=' * 72}\n{time.strftime('%Y-%m-%d %H:%M:%S')} 开始安装 / 更新训练后端\n")
        log_file.flush()
        try:
            for label, command in steps:
                with INSTALL_STATE["lock"]:
                    INSTALL_STATE["step"] = label
                log_file.write(f"\n>>> {label}\n$ {command}\n")
                log_file.flush()
                yield f"安装中 · {label}", _install_log_tail()
                env = os.environ.copy()
                env.update(PYTHONUNBUFFERED="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
                proc = subprocess.Popen(
                    ["bash", "-c", command],
                    cwd=HOME,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    env=env,
                )
                assert proc.stdout is not None
                for line in iter(proc.stdout.readline, ""):
                    log_file.write(line)
                    log_file.flush()
                    yield f"安装中 · {label}", _install_log_tail()
                proc.stdout.close()
                returncode = proc.wait()
                if returncode:
                    with INSTALL_STATE["lock"]:
                        INSTALL_STATE.update(step=label, returncode=returncode)
                    log_file.write(f"\n!!! {label} 失败，退出码 {returncode}\n")
                    log_file.flush()
                    yield f"安装失败 · {label} · 退出码 {returncode}", _install_log_tail()
                    return
                log_file.write(f"<<< {label} 完成\n")
                log_file.flush()
            with INSTALL_STATE["lock"]:
                INSTALL_STATE.update(step="全部步骤完成", returncode=0)
            log_file.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} 全部安装完成\n")
            log_file.flush()
            yield "安装完成 · 7/7", _install_log_tail()
        except Exception as exc:
            with INSTALL_STATE["lock"]:
                INSTALL_STATE.update(step=f"异常：{exc}", returncode=-1)
            log_file.write(f"\n!!! 安装异常：{type(exc).__name__}: {exc}\n")
            log_file.flush()
            yield f"安装异常 · {exc}", _install_log_tail()
        finally:
            with INSTALL_STATE["lock"]:
                INSTALL_STATE["running"] = False


def pose_gallery_root(require_data: bool = False) -> Path:
    configured = os.environ.get("POSE_GALLERY_DIR", "").strip()
    gallery = Path(configured).expanduser() if configured else POSE_GALLERY
    gallery = gallery.resolve()
    if not require_data or (gallery / "data/pose-zh.json").is_file():
        return gallery
    raise gr.Error(
        "未找到 Pose 数据文件。\n"
        f"标准位置：{POSE_GALLERY / 'data/pose-zh.json'}\n"
        f"兼容入口：{LEGACY_POSE_GALLERY}\n"
        "如需使用其他位置，请在启动前设置："
        "export POSE_GALLERY_DIR=/你的/ai-pose-gallery-zh"
    )


def ensure_pose_gallery_location() -> str:
    gallery = pose_gallery_root()
    if not (gallery / "data/pose-zh.json").is_file():
        return f"Pose 图库尚未就绪：{gallery}"

    legacy = LEGACY_POSE_GALLERY
    legacy_path = legacy.absolute()
    if gallery == legacy_path or legacy_path in gallery.parents:
        return f"Pose 图库未创建兼容链接：标准位置与旧位置重叠（{legacy}）"

    try:
        legacy.parent.mkdir(parents=True, exist_ok=True)
        if legacy.is_symlink():
            if legacy.resolve() == gallery:
                return f"Pose 图库兼容链接已就绪：{legacy} -> {gallery}"
            legacy.unlink()
        elif legacy.exists():
            stamp = time.strftime("%Y%m%d-%H%M%S")
            backup = legacy.with_name(f"{legacy.name}.legacy-{stamp}")
            suffix = 1
            while backup.exists() or backup.is_symlink():
                backup = legacy.with_name(f"{legacy.name}.legacy-{stamp}-{suffix}")
                suffix += 1
            legacy.rename(backup)
            logger.info(f"Pose 图库旧目录已备份到 {backup}")
        legacy.symlink_to(gallery, target_is_directory=True)
    except OSError as exc:
        logger.warning(f"无法统一 Pose 图库位置：{legacy} -> {gallery}：{exc}")
        return f"Pose 图库兼容链接创建失败：{exc}"
    return f"Pose 图库已统一：{legacy} -> {gallery}"


POSE_GALLERY_LINK_STATUS = ensure_pose_gallery_location()


def _has_cached_file(filename: str) -> bool:
    if not MODELS.is_dir():
        return False
    try:
        return next(MODELS.rglob(filename), None) is not None
    except OSError:
        return False


ENV_SETUP_LOG = LOGS / "h3-environment.log"
ENV_SETUP_PID = LOGS / "h3-environment.pid"
ENV_SETUP_RESULT = LOGS / "h3-environment-result.txt"


def _environment_pid() -> int | None:
    try:
        return int(ENV_SETUP_PID.read_text(encoding="utf-8").strip())
    except (OSError, TypeError, ValueError):
        return None


def _environment_process_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        process = psutil.Process(pid)
        command = " ".join(process.cmdline())
        return process.is_running() and "ensure_h3_runtime.py" in command
    except (psutil.Error, OSError):
        return False


def _environment_log_lines(limit: int = 40) -> list[str]:
    if not ENV_SETUP_LOG.is_file():
        return []
    text = ENV_SETUP_LOG.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    return [line.strip() for line in text.splitlines()[-limit:] if line.strip()]


def _environment_progress(lines: list[str]) -> str:
    joined = "\n".join(lines)
    if "CUDA recheck:" in joined:
        return "正在验证 CUDA 与 GPU 驱动兼容性…"
    if "[runtime] missing:" in joined:
        return "正在补齐当前图片后端的训练依赖…"
    if "installing cu" in joined:
        return "正在安装匹配 GPU 驱动的 PyTorch…"
    profiles = [line for line in lines if "[runtime] applying profile:" in line]
    if profiles:
        return f"正在应用运行配置：{profiles[-1].rsplit(':', 1)[-1].strip()}…"
    return "正在检查并准备训练环境…"


def start_training_environment_setup() -> str:
    gpu_error = h3_gpu_support_error()
    if gpu_error:
        result = f"[失败] {gpu_error}"
        ENV_SETUP_RESULT.write_text(result, encoding="utf-8")
        return result
    interpreter = task_python_path()
    script = ROOT / "ensure_h3_runtime.py"
    if not interpreter.is_file():
        return "[失败] 找不到训练任务环境。请重新运行一键启动脚本。"
    pid = _environment_pid()
    if _environment_process_running(pid):
        return f"环境准备正在后台运行（PID {pid}）。页面会自动更新，重连或刷新不会中断。"
    ENV_SETUP_RESULT.unlink(missing_ok=True)
    ENV_SETUP_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_file = ENV_SETUP_LOG.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [str(interpreter), str(script)],
            cwd=HOME,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    finally:
        log_file.close()
    ENV_SETUP_PID.write_text(str(process.pid), encoding="utf-8")
    return (
        f"训练环境准备任务已启动（PID {process.pid}）。\n"
        "页面每 3 秒自动更新；DSW 重连或刷新页面不会中断安装。"
    )


def poll_training_environment_setup():
    if ENV_SETUP_RESULT.is_file():
        return ENV_SETUP_RESULT.read_text(encoding="utf-8", errors="replace")
    pid = _environment_pid()
    lines = _environment_log_lines()
    if _environment_process_running(pid):
        visible = [
            line for line in lines[-14:]
            if not line.startswith("[runtime] $ ")
        ]
        live_log = "\n".join(visible) or "等待下载器输出…"
        return (
            f"{_environment_progress(lines)}\n"
            f"后台 PID：{pid} · 每 3 秒自动刷新\n\n"
            "实时下载日志\n"
            f"{live_log}\n\n"
            "可以安全刷新页面，任务会继续运行。"
        )
    if not pid:
        return gr.skip()
    joined = "\n".join(lines)
    if "H3 dependencies are ready" in joined:
        result = health_check()
        ENV_SETUP_RESULT.write_text(result, encoding="utf-8")
        ENV_SETUP_PID.unlink(missing_ok=True)
        return result
    details = "\n".join(lines[-8:]) or "后台进程已退出，但没有生成日志。"
    result = f"[失败] 训练环境准备未完成。\n\n{details}"
    ENV_SETUP_RESULT.write_text(result, encoding="utf-8")
    ENV_SETUP_PID.unlink(missing_ok=True)
    return result


def health_check() -> str:
    passed = "[通过]"
    missing = "[缺失]"
    warning = "[待完成]"
    blocked = False
    lines = [
        "统一环境检查",
        "",
        system_info(),
        "",
        storage_info(),
        "",
        "组件检查",
    ]

    venv_ok = sys.prefix != sys.base_prefix and Path(sys.executable).parent == ROOT / ".venv/bin"
    task_venv_ok = task_python_path().parent == ROOT / ".venv-h3/bin"
    lines.append(f"{passed if venv_ok else missing} WebUI 环境：{sys.executable}")
    lines.append(f"{passed if task_venv_ok else missing} H3 任务环境：{task_python_path()}")
    blocked = blocked or not venv_ok or not task_venv_ok

    gpu_error = h3_gpu_support_error()
    lines.append(f"{missing if gpu_error else passed} H3 GPU：{gpu_error or 'NVIDIA CUDA'}")
    blocked = blocked or bool(gpu_error)
    video_detail = video_support_info()
    lines.append(f"{warning if rocm_video_mode() else passed} 视频训练：{video_detail}")

    writable = HOME.is_dir() and os.access(HOME, os.W_OK)
    lines.append(f"{passed if writable else missing} 持久目录：{HOME}")
    blocked = blocked or not writable

    runtime_code = (
        "import torch, diffsynth, entropack; "
        "assert torch.cuda.is_available(); "
        "free,total=torch.cuda.mem_get_info(); "
        "print(f'torch={torch.__version__} cuda={torch.version.cuda} vram={total/2**30:.1f}GB')"
    )
    runtime = subprocess.run(
        [str(task_python_path()), "-c", runtime_code],
        capture_output=True,
        text=True,
    )
    runtime_text = (runtime.stdout + runtime.stderr).strip().splitlines()
    runtime_detail = runtime_text[-1] if runtime_text else "未安装或 CUDA 不可用"
    runtime_ok = runtime.returncode == 0
    lines.append(f"{passed if runtime_ok else missing} H3 + CUDA：{runtime_detail}")
    blocked = blocked or not runtime_ok

    try:
        gallery = pose_gallery_root(require_data=True)
        gallery_files = [
            gallery / "data/pose-zh.json",
            gallery / "data/costume-zh.json",
            gallery / "data/portrait-zh.json",
        ]
        gallery_ok = all(path.is_file() for path in gallery_files)
        lines.append(f"{passed if gallery_ok else missing} 三合一图库：{gallery}")
    except Exception as exc:
        gallery_ok = False
        lines.append(f"{missing} 三合一图库：{str(exc).splitlines()[0]}")
    blocked = blocked or not gallery_ok

    required_weights = [
        "dit_7bit.safetensors",
        "vae_int8.safetensors",
        "text_encoder_4bpp.safetensors",
        "pdd_fused_heads.safetensors",
    ]
    available = sum(_has_cached_file(name) for name in required_weights)
    weights_ok = available == len(required_weights)
    lines.append(
        f"{passed if weights_ok else warning} H3 权重：{available}/{len(required_weights)}"
        + ("" if weights_ok else "（生成时自动下载）")
    )

    gib = 2 ** 30
    model_gb = _tree_bytes([MODELS]) / gib
    total_gb = _tree_bytes([HOME]) / gib
    storage_ok = model_gb < MODEL_BUDGET_GB and total_gb < TOTAL_BUDGET_GB
    lines.append(
        f"{passed if storage_ok else missing} 存储：模型 {model_gb:.1f}/{MODEL_BUDGET_GB:.0f}GB，"
        f"总计 {total_gb:.1f}/{TOTAL_BUDGET_GB:.0f}GB"
    )
    blocked = blocked or not storage_ok

    candidates = sum(
        1 for path in DATASETS.glob("*/targets/*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    finalized = len(list(DATASETS.glob("*/metadata.json")))
    loras = len(list(OUTPUTS.rglob("*.safetensors")))
    lines.extend([
        "",
        "流程进度",
        f"{passed if candidates else warning} 目标图：{candidates} 张",
        f"{passed if finalized else warning} 已确认训练集：{finalized} 个",
        f"{passed if loras else warning} LoRA 输出：{loras} 个",
    ])

    if gpu_error and rocm_video_mode():
        next_step = "H3 不可用；可在“训练”页使用 Wan 2.1 / 2.2 视频 LoRA（AMD ROCm 实验路径）。"
    elif gpu_error:
        next_step = gpu_error
    elif blocked:
        next_step = "先修复上方标记为“缺失”的环境项目。"
    elif not candidates:
        next_step = "图片 → 1 · 批量生成素材。"
    elif not finalized:
        next_step = "图片 → 2 · 审核与建集。"
    elif not loras:
        next_step = "图片 → 3 · LoRA 训练。"
    else:
        next_step = "环境和完整流程均已就绪，可继续生成或训练新批次。"
    overall = "阻塞" if blocked else "可用"
    lines.extend(["", f"总体状态：{overall}", f"下一步：{next_step}"])
    return "\n".join(lines)


COMFY_ROOT = REPOS / "ComfyUI"
COMFY_ENV = ROOT / ".venv-comfy"
COMFY_SETUP_LOG = LOGS / "comfy-environment.log"
COMFY_SETUP_PID = LOGS / "comfy-environment.pid"
COMFY_SETUP_RESULT = LOGS / "comfy-environment-result.txt"


def _comfy_setup_pid() -> int | None:
    try:
        return int(COMFY_SETUP_PID.read_text(encoding="utf-8").strip())
    except (OSError, TypeError, ValueError):
        return None


def _comfy_setup_running(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        process = psutil.Process(pid)
        command = " ".join(process.cmdline())
        return process.is_running() and "ensure_comfy_runtime.py" in command
    except (psutil.Error, OSError):
        return False


def _comfy_log_tail(limit: int = 30) -> str:
    try:
        lines = COMFY_SETUP_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "等待 ComfyUI 安装器输出…"
    return "\n".join(lines[-limit:]) or "等待 ComfyUI 安装器输出…"


def start_comfy_environment_setup() -> str:
    pid = _comfy_setup_pid()
    if _comfy_setup_running(pid):
        return f"ComfyUI 运行环境正在后台安装（PID {pid}），页面会自动刷新。"
    base_python = Path(getattr(sys, "_base_executable", sys.executable)).resolve()
    if not base_python.is_file():
        base_python = Path(shutil.which("python3") or sys.executable).resolve()
    venv_flags = "--system-site-packages" if rocm_video_mode() else ""
    command = (
        f"if [ ! -x {q(COMFY_ENV / 'bin/python')} ]; then "
        f"{q(base_python)} -m venv {venv_flags} {q(COMFY_ENV)}; fi && "
        f"exec {q(COMFY_ENV / 'bin/python')} {q(ROOT / 'ensure_comfy_runtime.py')}"
    )
    COMFY_SETUP_RESULT.unlink(missing_ok=True)
    COMFY_SETUP_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_file = COMFY_SETUP_LOG.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            ["bash", "-c", command],
            cwd=HOME,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    finally:
        log_file.close()
    COMFY_SETUP_PID.write_text(str(process.pid), encoding="utf-8")
    return (
        f"ComfyUI 运行环境安装已启动（PID {process.pid}）。\n"
        "会创建独立环境、安装匹配 GPU 的 PyTorch、拉取 ComfyUI 并执行验证；页面每 3 秒自动刷新。"
    )


def poll_comfy_environment_setup():
    if COMFY_SETUP_RESULT.is_file():
        return COMFY_SETUP_RESULT.read_text(encoding="utf-8", errors="replace")
    pid = _comfy_setup_pid()
    if _comfy_setup_running(pid):
        return f"ComfyUI 运行环境安装中 · PID {pid}\n\n{_comfy_log_tail()}"
    if not pid:
        return gr.skip()
    joined = COMFY_SETUP_LOG.read_text(encoding="utf-8", errors="replace") if COMFY_SETUP_LOG.is_file() else ""
    if "ComfyUI runtime is ready" in joined:
        result = "ComfyUI 运行环境已就绪。\n\n" + _comfy_log_tail(20)
    else:
        result = (
            "ComfyUI 运行环境安装未完成，请检查日志。\n\n"
            + "\n".join(joined.splitlines()[-30:])
        )
    COMFY_SETUP_RESULT.write_text(result, encoding="utf-8")
    COMFY_SETUP_PID.unlink(missing_ok=True)
    return result


def _markdown_cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip() or "-"


def _status_text(status: str) -> str:
    return {"pass": "通过", "warn": "警告", "block": "阻止"}.get(status, status)


def _storage_usage() -> tuple[float, float, float]:
    gib = 2 ** 30
    model_gb = _tree_bytes([MODELS]) / gib
    total_gb = _tree_bytes([HOME]) / gib
    return total_gb, model_gb, max(0.0, total_gb - model_gb)


def _environment_overview_markdown(snapshot) -> str:
    gpu = snapshot.gpu
    torch = snapshot.torch
    if gpu.vendor == "nvidia":
        gpu_line = f"{gpu.name or 'NVIDIA GPU'} · CUDA 驱动 {gpu.cuda_driver_version or '未知'}"
        if gpu.vram_gb is not None:
            gpu_line += f" · {gpu.vram_gb:.1f} GB"
        if "AMD" in gpu.detail:
            gpu_line += f" · {gpu.detail}"
    elif gpu.vendor == "rocm":
        gpu_line = f"{gpu.name or 'AMD ROCm GPU'} · {gpu.rocm_version or gpu.detail or 'ROCm'}"
    else:
        gpu_line = gpu.detail
    torch_line = torch.version or torch.error or "未安装"
    if torch.version:
        runtime = torch.cuda_version or torch.hip_version
        runtime_name = "CUDA" if torch.cuda_version else "HIP" if torch.hip_version else "无"
        torch_line = f"{torch.version} · {runtime_name} {runtime or '无'} · {'可用' if torch.available else '不可用'}"
    ram_line = f"{snapshot.ram_gb:.1f} GB" if snapshot.ram_gb is not None else "未知"
    return (
        "### 当前容器\n"
        "| 项目 | 当前值 |\n"
        "| --- | --- |\n"
        f"| 镜像 | {_markdown_cell(snapshot.image)} |\n"
        f"| 系统 | {_markdown_cell(snapshot.os_name)} |\n"
        f"| Python | {_markdown_cell(snapshot.python_version)} |\n"
        f"| GPU | {_markdown_cell(gpu_line)} |\n"
        f"| 训练 PyTorch | {_markdown_cell(torch_line)} |\n"
        f"| 系统内存 | {_markdown_cell(ram_line)} |\n"
        f"| 持久目录 | `{_markdown_cell(snapshot.home)}` |\n"
        f"| 磁盘剩余 | {snapshot.disk_free_gb:.1f} / {snapshot.disk_total_gb:.1f} GB |"
    )


def _environment_compatibility_markdown(compatible, available_storage_gb: float) -> str:
    rows = []
    details = []
    for result in compatible:
        model = result.model
        failed = [check for check in result.checks if check.status == "block"]
        warnings = [check for check in result.checks if check.status == "warn"]
        reasons = []
        if failed:
            reasons.extend(f"{check.label}：{check.current}，要求 {check.required}" for check in failed)
        if warnings:
            reasons.extend(f"{check.label}：{check.current}，建议 {check.required}" for check in warnings)
        if not reasons and model.note:
            reasons.append(model.note)
        rows.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(model.label),
                    "图片" if model.kind == "image" else "视频",
                    _status_text(result.status),
                    _markdown_cell("；".join(reasons) or "参数匹配"),
                ]
            )
            + " |"
        )
        if result.status != "pass":
            details.append(f"- **{model.label}**：{model.note}")
    detail_text = "\n".join(details)
    return (
        "### 模型兼容性\n"
        f"用于兼容判断的可用存储：{available_storage_gb:.1f} GB\n\n"
        "| 模型 | 类型 | 结果 | 关键限制 |\n"
        "| --- | --- | --- | --- |\n"
        + "\n".join(rows)
        + ("\n\n" + detail_text if detail_text else "")
    )


def _environment_install_markdown(snapshot, modules: dict[str, bool]) -> str:
    gpu = snapshot.gpu
    torch = snapshot.torch
    driver_status = "pass" if gpu.vendor in {"nvidia", "rocm"} else "block"
    driver_value = (
        f"NVIDIA {gpu.driver_version or '未知驱动'}"
        if gpu.vendor == "nvidia"
        else f"AMD {gpu.rocm_version or gpu.detail}"
        if gpu.vendor == "rocm"
        else "未检测到"
    )
    torch_status = "pass" if torch.version and torch.available else "block"
    torch_value = torch.version or torch.error or "未安装"
    if torch.version:
        torch_value += f" · {'CUDA' if torch.cuda_version else 'HIP' if torch.hip_version else 'CPU'}"
        torch_value += f" {torch.cuda_version or torch.hip_version or ''}".rstrip()
    module_names = [
        ("accelerate", "accelerate"),
        ("diffsynth", "DiffSynth-Studio"),
        ("entropack", "EntroPack"),
    ]
    rows = [
        ("GPU 驱动", driver_value, driver_status, "NVIDIA 驱动只检查；容器内不安装宿主驱动。"),
        ("PyTorch", torch_value, torch_status, "CUDA/HIP 必须与当前 GPU 类型匹配。"),
    ]
    for module, label in module_names:
        installed = modules.get(module, False)
        rows.append(
            (
                label,
                "已安装" if installed else "未安装",
                "pass" if installed else "warn",
                f"模块 `{module}`",
            )
        )
    comfy_python_ok = (COMFY_ENV / "bin/python").is_file()
    comfy_source_ok = (COMFY_ROOT / "main.py").is_file()
    comfy_status = "pass" if comfy_python_ok and comfy_source_ok else "warn"
    rows.append(
        (
            "ComfyUI",
            f"环境{'已就绪' if comfy_python_ok else '缺失'} · 源码{'已就绪' if comfy_source_ok else '缺失'}",
            comfy_status,
            "独立 `.venv-comfy` 与 `repos/ComfyUI`。",
        )
    )
    table = "\n".join(
        f"| {_markdown_cell(label)} | {_markdown_cell(value)} | {_status_text(status)} | {_markdown_cell(detail)} |"
        for label, value, status, detail in rows
    )
    return (
        "### 安装检查\n"
        "| 组件 | 当前状态 | 结果 | 说明 |\n"
        "| --- | --- | --- | --- |\n"
        + table
    )


def _environment_storage_markdown(settings, used_gb: float, model_gb: float, data_gb: float, effective_gb: float, status: str) -> str:
    percent = (used_gb / effective_gb * 100) if effective_gb > 0 else 0.0
    return (
        "### 存储上限\n"
        "| 项目 | 当前配置 |\n"
        "| --- | --- |\n"
        f"| 环境总上限 | {settings.total_limit_gb:.1f} GB |\n"
        f"| 模型上限 | {settings.model_limit_gb:.1f} GB |\n"
        f"| 系统安全预留 | {settings.system_reserve_gb:.1f} GB |\n"
        f"| 警告 / 停止阈值 | {settings.warn_percent:.0f}% / {settings.stop_percent:.0f}% |\n"
        f"| 当前项目数据 | {used_gb:.1f} GB |\n"
        f"| 其中模型 | {model_gb:.1f} GB |\n"
        f"| 其中素材与输出 | {data_gb:.1f} GB |\n"
        f"| 当前有效上限 | {effective_gb:.1f} GB |\n"
        f"| 使用率 | {percent:.1f}% · {_status_text(status)} |"
    )


def refresh_environment_dashboard():
    settings = load_storage_settings(ENVIRONMENT_CONFIG, STORAGE_DEFAULTS)
    snapshot = inspect_container(HOME, task_python_path(), BASE_IMAGE)
    used_gb, model_gb, data_gb = _storage_usage()
    effective_gb = effective_storage_limit(settings, used_gb, snapshot.disk_free_gb)
    status = storage_status(settings, used_gb, effective_gb)
    available_storage_gb = max(0.0, effective_gb - used_gb)
    compatible = evaluate_models(snapshot, available_storage_gb=available_storage_gb)
    modules = inspect_modules(
        task_python_path(),
        ("accelerate", "diffsynth", "entropack"),
    )
    return (
        _environment_overview_markdown(snapshot),
        _environment_compatibility_markdown(compatible, available_storage_gb),
        _environment_install_markdown(snapshot, modules),
        _environment_storage_markdown(settings, used_gb, model_gb, data_gb, effective_gb, status),
    )


def save_environment_storage(total, model, warn, stop, reserve):
    global TOTAL_BUDGET_GB, MODEL_BUDGET_GB, DATA_RESERVE_GB
    global STORAGE_WARN_PERCENT, STORAGE_STOP_PERCENT, STORAGE_SYSTEM_RESERVE_GB
    try:
        settings = save_storage_settings(
            ENVIRONMENT_CONFIG,
            {
                "total_limit_gb": total,
                "model_limit_gb": model,
                "warn_percent": warn,
                "stop_percent": stop,
                "system_reserve_gb": reserve,
            },
            STORAGE_DEFAULTS,
        )
    except (OSError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc
    TOTAL_BUDGET_GB = settings.total_limit_gb
    MODEL_BUDGET_GB = settings.model_limit_gb
    DATA_RESERVE_GB = max(0.0, TOTAL_BUDGET_GB - MODEL_BUDGET_GB)
    STORAGE_WARN_PERCENT = settings.warn_percent
    STORAGE_STOP_PERCENT = settings.stop_percent
    STORAGE_SYSTEM_RESERVE_GB = settings.system_reserve_gb
    os.environ["TRAINER_TOTAL_BUDGET_GB"] = str(TOTAL_BUDGET_GB)
    os.environ["TRAINER_MODEL_BUDGET_GB"] = str(MODEL_BUDGET_GB)
    os.environ["TRAINER_STORAGE_RESERVE_GB"] = str(STORAGE_SYSTEM_RESERVE_GB)
    used_gb, model_gb, data_gb = _storage_usage()
    disk_free_gb = shutil.disk_usage(str(HOME)).free / 2 ** 30
    effective_gb = effective_storage_limit(settings, used_gb, disk_free_gb)
    status = storage_status(settings, used_gb, effective_gb)
    return (
        "存储上限已保存。新任务立即使用当前预算；H3 模型配置下拉在刷新页面后更新。\n\n"
        + _environment_storage_markdown(settings, used_gb, model_gb, data_gb, effective_gb, status)
    )


def sync_pose_gallery() -> str:
    gallery = pose_gallery_root()
    if (gallery / ".git").is_dir():
        result = run_visible(f"git -C {q(gallery)} pull --ff-only")
    else:
        if gallery.exists() and (not gallery.is_dir() or any(gallery.iterdir())):
            raise gr.Error(
                f"Pose 图库目录已存在但不是完整 Git 仓库：{gallery}\n"
                "请先移走该目录，或设置 POSE_GALLERY_DIR 指向已有完整图库。"
            )
        gallery.parent.mkdir(parents=True, exist_ok=True)
        result = run_visible(f"git clone --depth 1 https://github.com/anhao/ai-pose-gallery-zh.git {q(gallery)}")
    link_status = ensure_pose_gallery_location()
    return f"标准资源目录：{gallery}\n{result}\n{link_status}"


def ensure_pose_gallery_ready() -> tuple[Path, str]:
    try:
        return pose_gallery_root(require_data=True), ""
    except gr.Error as first_error:
        try:
            sync_message = sync_pose_gallery()
            return pose_gallery_root(require_data=True), f"Pose 图库已自动同步。\n{sync_message}"
        except Exception as exc:
            raise gr.Error(
                "Pose 图库缺失且自动同步失败。请检查网络，或在图片页的“高级设置”中手动同步。\n"
                f"{first_error}\n{exc}"
            ) from exc


def _upload_paths(items) -> list[Path]:
    paths = []
    for item in items or []:
        path = Path(str(getattr(item, "name", item))).expanduser().resolve()
        if path.is_file():
            paths.append(path)
    return paths


def _validate_image_files(paths: list[Path], label: str) -> None:
    for path in paths:
        if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise gr.Error(f"{label}包含不支持的格式：{path}；请转换为 PNG、JPG、WebP 或 BMP")
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise gr.Error(f"{label}图片无法读取：{path}") from exc


def _copy_named(source: Path, destination: Path, prefix: str, index: int) -> str:
    suffix = source.suffix.lower() or ".png"
    filename = f"{prefix}_{index:03d}{suffix}"
    shutil.copy2(source, destination / filename)
    return filename


def _new_dataset_root(safe_name: str) -> Path:
    root = DATASETS / f"{safe_name}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=False)
    return root


def prepare_h3_character_dataset(
    targets, people, outfits, poses, pose_local_paths: str,
    job_name: str, trigger_word: str, instruction: str,
) -> tuple[str, str, str]:
    target_files = _upload_paths(targets)
    people_files = _upload_paths(people)
    outfit_files = _upload_paths(outfits)
    pose_files = _upload_paths(poses)
    for raw in re.split(r"[,\n]", pose_local_paths or ""):
        if raw.strip():
            path = Path(raw.strip()).expanduser().resolve()
            if path.is_file():
                pose_files.append(path)
    if not target_files:
        raise gr.Error("至少需要一张目标成品图作为监督输出")
    _validate_image_files(target_files, "目标成品图")
    _validate_image_files(people_files, "人物参考图")
    _validate_image_files(outfit_files, "服装参考图")
    _validate_image_files(pose_files, "Pose 参考图")
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", job_name.strip()).strip("-.") or "h3-character"
    root = _new_dataset_root(safe_name)
    image_dir, ref_dir = root / "targets", root / "references"
    image_dir.mkdir()
    ref_dir.mkdir()
    reference_names = []
    for prefix, files in (("person", people_files), ("outfit", outfit_files), ("pose", pose_files)):
        for index, source in enumerate(files, 1):
            reference_names.append(f"references/{_copy_named(source, ref_dir, prefix, index)}")
    role_hints = []
    picture = 1
    for label, files in (
        ("identity", people_files), ("outfit", outfit_files), ("pose", pose_files),
    ):
        if not files:
            continue
        end = picture + len(files) - 1
        subject = f"Picture {picture}" if picture == end else f"Pictures {picture}-{end}"
        role_hints.append(f"{subject} {'is an' if picture == end else 'are'} {label} reference{'s' if picture != end else ''}.")
        picture = end + 1
    role_hint = " ".join(role_hints)
    prompt = " ".join(x for x in (trigger_word.strip(), role_hint, instruction.strip()) if x)
    if not prompt:
        raise gr.Error("请填写人物触发词或训练指令，不能创建空 prompt 的训练集")
    records = []
    for index, source in enumerate(target_files, 1):
        target_name = f"targets/{_copy_named(source, image_dir, 'target', index)}"
        record = {"image": target_name, "prompt": prompt}
        if reference_names:
            record["edit_image"] = reference_names
        records.append(record)
    metadata = root / "metadata.json"
    metadata.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = (
        f"任务目录：{root}\n目标图：{len(target_files)}\n人物参考：{len(people_files)}\n"
        f"服装参考：{len(outfit_files)}\nPose 参考：{len(pose_files)}\n"
        f"每个样本参考图总数：{len(reference_names)}\n"
        f"训练模式：{'参考图编辑' if reference_names else '仅目标图（文生图）'}\nMetadata：{metadata}"
    )
    return str(root), summary, str(root)


def balanced_people_assignments(people: list, count: int, seed: int) -> list:
    if not people or count <= 0:
        return []
    rng = random.Random(int(seed))
    base, remainder = divmod(count, len(people))
    assignments = [person for person in people for _ in range(base)]
    if remainder:
        assignments.extend(rng.sample(people, remainder))
    rng.shuffle(assignments)
    return assignments


def random_sample(items: list, count: int, rng: random.Random) -> list:
    if not items or count <= 0:
        return []
    pool = list(items)
    selected = []
    while len(selected) < count:
        rng.shuffle(pool)
        selected.extend(pool[: count - len(selected)])
    return selected


def parse_keywords(raw: str) -> list[str]:
    terms = [term.strip().lower() for term in re.split(r"[,，;；|\n]+", raw or "")]
    return list(dict.fromkeys(term for term in terms if term))


def matches_keywords(searchable: str, keywords: list[str]) -> bool:
    return not keywords or any(keyword in searchable for keyword in keywords)


BODY_PROPORTION_MARKERS = (
    "全身", "身材", "体型", "体态", "身体比例", "肢体比例", "骨骼", "身高", "腿长", "腿型",
    "full body", "body proportion", "body type", "skeleton", "height", "leg length",
)
SAFE_FILTER_MARKERS = (
    "性感", "内衣", "泳装", "双人缠绵", "裸体", "裸露",
    "sexy", "lingerie", "bikini", "swimsuit", "nsfw", "nude", "topless",
)
ASIAN_FEMALE_ANATOMY_GUIDANCE = (
    "a healthy natural adult East Asian female build: balanced head-to-body relationships, "
    "moderate shoulder width, a soft natural waist-to-hip curve, proportionate torso and limbs, "
    "realistic joint placement, hands and feet, and a gently contoured silhouette with a soft "
    "natural presence, without exaggerated runway-model elongation."
)
FACIAL_IDENTITY_GUIDANCE = (
    "Treat the person reference as the sole identity and beauty reference. Preserve the person's "
    "facial identity, face shape, facial proportions, eyes, eyebrows, nose, mouth, skin tone, "
    "hairstyle and recognizable features, and keep the face strongly recognizable. Render a real "
    "adult face with clear refined features, lively eyes and polished natural beauty makeup: clean "
    "translucent base, natural brow shape, soft detailed eye makeup, delicate eyeliner and lashes, "
    "subtle blush, natural lip color and gentle highlights. Keep realistic skin texture; avoid "
    "waxy plastic skin, a mask-like face, a generic mannequin face or a generic influencer face."
)
NO_TEXT_GUIDANCE = (
    "The output must be a clean photograph with no text of any kind: no captions, titles, labels, "
    "letters, numbers, logos, signatures, watermarks, UI overlays, borders or typography. "
    "Do not transcribe or visualize any description, metadata or prompt wording."
)
POSE_ONLY_GUIDANCE = (
    "The pose reference supplies motion and camera guidance only: joint placement, limb direction, "
    "hand placement, body orientation, framing and camera angle. It supplies no identity or "
    "appearance. Do not copy or blend its face, head shape, facial proportions, hairstyle, skin "
    "tone, makeup, body shape, body proportions, clothing, mannequin or white-model surface, or "
    "background lighting. The face, hair, makeup and skin appearance must continue to come from "
    "the person reference; body proportions follow the separate body guidance."
)
PROMPT_MODE_DEFAULT = "默认模板（推荐）"
PROMPT_MODE_ORIGINAL = "原版 Prompt（兼容旧效果）"
PROMPT_MODES = (PROMPT_MODE_DEFAULT, PROMPT_MODE_ORIGINAL)
DEFAULT_GENERATION_INSTRUCTION = (
    "Keep the person's real facial identity and natural beauty, with refined camera-ready makeup, "
    "clear lively eyes, realistic skin texture and flattering portrait lighting."
)
DEFAULT_PROMPT_TEMPLATE = (
    "A polished photorealistic portrait photograph of {subject}, using {person_range} as the "
    "identity, beauty and hairstyle reference. Reference metadata: {portrait_text}. "
    "{facial_identity_guidance} "
    "{body_guidance} "
    "Dress the same person using only the garment design, colors, materials and accessories shown "
    "in {outfit_range}. Outfit metadata: {costume_text}. "
    "Pose metadata: {pose_text}. Reproduce that body action, limb placement, body orientation, "
    "framing and camera angle from {pose_reference}. "
    "{pose_guidance} "
    "Use soft flattering portrait lighting with natural skin highlights, realistic fabric, coherent "
    "anatomy and a clean uncluttered background. "
    "{instruction} {no_text_guidance}"
)
ORIGINAL_PROMPT_TEMPLATE = (
    "Preserve the identity, face, hairstyle and body proportions from {person_range}. "
    "Dress the same person in the clothing from {outfit_range}. "
    "Match the body pose, limb placement and camera viewpoint in {pose_reference} exactly. "
    "Outfit description: {costume_text}. Pose description: {pose_text}. {instruction}"
)
PROMPT_TEMPLATE_HELP = (
    "可用占位符：`{subject}`、`{person_range}`、`{outfit_range}`、`{pose_reference}`、"
    "`{facial_identity_guidance}`、`{body_guidance}`、`{pose_guidance}`、`{instruction}`、"
    "`{no_text_guidance}`、`{portrait_text}`、`{costume_text}`、`{pose_text}`。"
)


def prompt_template_for_mode(mode: str) -> str:
    return ORIGINAL_PROMPT_TEMPLATE if mode == PROMPT_MODE_ORIGINAL else DEFAULT_PROMPT_TEMPLATE


def build_pose_prompt_text(item: dict | None) -> str:
    item = item or {}
    title = str(item.get("姿势名称") or item.get("标题") or "").strip()
    system = str(item.get("体系") or "").strip()
    category = str(item.get("分类") or "").strip()
    angle = str(item.get("角度") or "").strip()
    if not title or angle.lower() in {"", "reference"} or title.lower().startswith("uploaded pose"):
        return "动作、肢体方向、构图与镜头角度以 Pose 参考图为准"
    details = "·".join(part for part in (system, category) if part)
    suffix = f"（{details}）" if details else ""
    return f"{angle}视角，动作：{title}{suffix}"


def build_costume_prompt_text(item: dict | None) -> str:
    item = item or {}
    title = str(item.get("服装名称") or item.get("标题") or "").strip()
    if not title:
        return ""
    category = str(item.get("分类") or "").strip()
    tags = "、".join(str(tag).strip() for tag in item.get("标签", [])[:8] if str(tag).strip())
    description = f"身穿{category}：{title}" if category else title
    if tags:
        description += f"（细节：{tags}）"
    keywords = str(item.get("关键词") or "").strip()
    return f"{description}；{keywords}" if keywords else description


def build_portrait_prompt_text(people: list[dict]) -> str:
    if len(people) > 1:
        return "人物面部与发型以多张人物参考图为准，保持同一身份与妆造"
    if not people:
        return "人物面部与发型以人物参考图为准"
    person = people[0]
    item = person.get("gallery_item") or {}
    hair_len = str(item.get("发长") or "").strip()
    hair_style = str(item.get("发型名称") or item.get("发型") or "").strip()
    section = str(item.get("分区") or "").strip()
    title = str(item.get("标题") or "").strip()
    hair = "、".join(part for part in (hair_len, hair_style) if part)
    details = "，".join(part for part in (section, title) if part)
    if hair or details:
        suffix = f"（{details}）" if details else ""
        return f"人物面部与发型：{hair}{suffix}"
    fallback = str(person.get("title") or person.get("id") or "").strip()
    return f"人物面部与发型：{fallback}" if fallback else "人物面部与发型以人物参考图为准"


def portrait_has_body_proportions(item: dict | None) -> bool | None:
    if not item:
        return None
    searchable = " ".join(str(value) for value in item.values()).lower()
    return any(marker.lower() in searchable for marker in BODY_PROPORTION_MARKERS)


def gallery_item_is_safe(item: dict, fields: tuple[str, ...]) -> bool:
    searchable = " ".join(str(item.get(field, "")) for field in fields).lower()
    return not any(marker in searchable for marker in SAFE_FILTER_MARKERS)


def build_h3_generation_prompt(
    person_range: str, outfit_range: str, pose_number: int,
    body_proportions: bool | None, instruction: str,
    prompt_mode: str = PROMPT_MODE_DEFAULT, prompt_template: str = "",
    costume_text: str = "", pose_text: str = "",
    creative_style: str = STYLE_PRESERVE, creative_body: str = BODY_NATURAL,
    portrait_text: str = "",
) -> str:
    profile = normalize_creative_profile(creative_style, creative_body, PERSON_MODE_BATCH)
    creative_style, creative_body = profile.style, profile.body
    explicit_body_guidance = body_prompt(creative_body)
    if uses_reference_body(creative_body):
        body_proportions = True
    elif explicit_body_guidance:
        body_proportions = False
    if explicit_body_guidance:
        body_guidance = (
            f"{explicit_body_guidance} Keep body proportions independent of the pose reference "
            "and preserve anatomically coherent joints, hands and feet."
        )
    elif body_proportions is False:
        body_guidance = (
            "Do not infer body proportions from the person reference or the pose reference. "
            f"Construct the body as {ASIAN_FEMALE_ANATOMY_GUIDANCE}"
        )
    elif body_proportions is True:
        body_guidance = (
            f"Preserve the recognizable body type from {person_range}. "
            "Keep clearly visible body proportions consistent with that person, but harmonize them into "
            f"{ASIAN_FEMALE_ANATOMY_GUIDANCE} "
            "Body proportions must remain independent of the pose reference."
        )
    else:
        body_guidance = (
            f"Preserve any recognizable body type from {person_range}. "
            "Preserve any clearly visible body cues from that person, but harmonize them into "
            f"{ASIAN_FEMALE_ANATOMY_GUIDANCE} "
            "If the body is not visible, construct it directly; body proportions must remain independent of the pose reference."
        )
    template = (prompt_template or "").strip() or prompt_template_for_mode(prompt_mode)
    substitutions = {
        "subject": "an adult East Asian woman" if body_proportions is False else "an adult woman",
        "person_range": person_range,
        "outfit_range": outfit_range,
        "pose_reference": f"<Picture {pose_number}>",
        "facial_identity_guidance": FACIAL_IDENTITY_GUIDANCE,
        "body_guidance": body_guidance,
        "pose_guidance": POSE_ONLY_GUIDANCE,
        "instruction": (instruction or "").strip(),
        "no_text_guidance": NO_TEXT_GUIDANCE,
        "portrait_text": portrait_text,
        "costume_text": costume_text,
        "pose_text": pose_text,
    }
    try:
        rendered = template.format_map(substitutions)
        return " ".join(f"{style_prompt(creative_style)} {rendered}".split())
    except KeyError as exc:
        raise gr.Error(f"提示词模板包含未知占位符：{exc}") from exc
    except ValueError as exc:
        raise gr.Error(f"提示词模板格式错误：{exc}") from exc


def prepare_h3_generation_job(
    people, outfits, poses, person_mode, outfit_mode,
    portrait_keyword, keyword, costume_keyword,
    max_items, safe_only, job_name, instruction, prompt_mode, prompt_template,
    creative_style, creative_body, image_model, seed, width, height,
):
    count = max(1, int(max_items))
    seed = int(seed)
    rng = random.Random(seed)
    profile = normalize_creative_profile(creative_style, creative_body, person_mode)
    creative_style, creative_body, person_mode = (
        profile.style,
        profile.body,
        profile.gallery_strategy,
    )
    people_files = _upload_paths(people)
    outfit_files = _upload_paths(outfits)
    pose_files = _upload_paths(poses)
    _validate_image_files(people_files, "自定义人物图")
    _validate_image_files(outfit_files, "自定义服装图")
    _validate_image_files(pose_files, "自定义 Pose 图")
    pose_keywords = parse_keywords(keyword)
    gallery_cache: Path | None = None
    gallery_note = ""
    if not people_files or not outfit_files or not pose_files:
        gallery_cache, gallery_note = ensure_pose_gallery_ready()

    def gallery() -> Path:
        nonlocal gallery_cache
        if gallery_cache is None:
            gallery_cache = pose_gallery_root(require_data=True)
        return gallery_cache

    person_candidates = []
    if people_files:
        for index, path in enumerate(people_files, 1):
            person_candidates.append({
                "id": f"uploaded-person-{index}",
                "title": path.stem or f"person {index}",
                "path": path,
                "gallery_item": None,
                "body_proportions": None,
            })
    else:
        portrait_index = gallery() / "data/portrait-zh.json"
        if not portrait_index.is_file():
            raise gr.Error("Pose 仓库缺少 data/portrait-zh.json，请重新同步完整图库")
        portrait_keywords = parse_keywords(portrait_keyword)
        for item in json.loads(portrait_index.read_text(encoding="utf-8")):
            searchable = " ".join(str(item.get(k, "")) for k in ("id", "分区", "发长", "发型名称", "标题", "检索"))
            if not matches_keywords(searchable.lower(), portrait_keywords):
                continue
            if safe_only and not gallery_item_is_safe(
                item, ("id", "分区", "发长", "发型名称", "标题", "检索", "原图"),
            ):
                continue
            source = gallery() / item.get("原图", "")
            if source.is_file():
                person_candidates.append({
                    "id": str(item.get("id", "")),
                    "title": str(item.get("标题", "")),
                    "path": source,
                    "gallery_item": item,
                    "body_proportions": portrait_has_body_proportions(item),
                })
        if not person_candidates:
            raise gr.Error("人像库中没有符合条件且已下载的人物；可清空关键词或重新同步图库")

    if pose_files:
        pose_pool = [
            ({
                "id": f"uploaded-pose-{index}", "体系": "uploaded", "分类": "custom",
                "姿势名称": f"Uploaded pose {index}", "角度": "reference",
            }, path)
            for index, path in enumerate(pose_files, 1)
        ]
    else:
        index = gallery() / "data/pose-zh.json"
        if not index.is_file():
            raise gr.Error("请先在“姿势图库”页同步 Pose 仓库，或上传自定义 Pose 参考图")
        pose_pool = []
        for item in json.loads(index.read_text(encoding="utf-8")):
            searchable = " ".join(str(item.get(k, "")) for k in ("体系", "分类", "姿势名称", "角度", "检索"))
            if not matches_keywords(searchable.lower(), pose_keywords):
                continue
            if safe_only and not gallery_item_is_safe(
                item, ("id", "体系", "分类", "姿势名称", "角度", "检索", "原图"),
            ):
                continue
            pose = gallery() / item.get("原图", "")
            if pose.is_file():
                pose_pool.append((item, pose))
    if not pose_pool:
        raise gr.Error("没有可用的 Pose 参考图；请调整筛选条件或上传自定义 Pose")
    selected = random_sample(pose_pool, count, rng)

    if outfit_files:
        if outfit_mode == OUTFIT_MODE_RANDOM:
            costume_pool = [
                ({
                    "id": f"uploaded-outfit-{index}",
                    "服装名称": f"the uploaded outfit {index}",
                    "关键词": "",
                }, [path])
                for index, path in enumerate(outfit_files, 1)
            ]
        else:
            costume_pool = [({
                "id": "uploaded-outfit-set", "服装名称": "the uploaded outfit set", "关键词": "",
            }, list(outfit_files))]
    else:
        costume_index = gallery() / "data/costume-zh.json"
        if not costume_index.is_file():
            raise gr.Error("Pose 仓库缺少 data/costume-zh.json，请重新同步图库")
        costume_keywords = parse_keywords(costume_keyword)
        costume_pool = []
        for item in json.loads(costume_index.read_text(encoding="utf-8")):
            searchable = " ".join(str(item.get(k, "")) for k in ("分类", "服装名称", "标签", "关键词", "检索"))
            if not matches_keywords(searchable.lower(), costume_keywords):
                continue
            if safe_only and not gallery_item_is_safe(
                item, ("id", "分类", "服装名称", "标签", "关键词", "检索", "原图"),
            ):
                continue
            source = gallery() / item.get("原图", "")
            if source.is_file():
                costume_pool.append((item, [source]))
        if not costume_pool:
            raise gr.Error("服装库中没有符合条件且已下载的服装；可换关键词或重新同步图库")
    costume_sets = random_sample(costume_pool, len(selected), rng)
    safe_name = (
        re.sub(r"[^A-Za-z0-9._-]+", "-", job_name.strip()).strip("-.")
        or DEFAULT_GENERATION_JOB_NAME
    )
    root = _new_dataset_root(safe_name)
    refs = root / "references"
    refs.mkdir()

    if person_mode == PERSON_MODE_MULTI_REF and len(person_candidates) > 1:
        person_assignments = [list(person_candidates) for _ in selected]
    elif person_mode == PERSON_MODE_PER_ITEM:
        person_assignments = [
            [candidate] for candidate in random_sample(person_candidates, len(selected), rng)
        ]
    elif person_mode == PERSON_MODE_BALANCED:
        person_assignments = [
            [candidate] for candidate in balanced_people_assignments(person_candidates, len(selected), seed)
        ]
    else:
        fixed_person = rng.choice(person_candidates)
        person_assignments = [[fixed_person] for _ in selected]

    person_names_by_source = {}
    for assignment in person_assignments:
        for candidate in assignment:
            path = candidate["path"]
            if path not in person_names_by_source:
                person_names_by_source[path] = (
                    f"references/{_copy_named(path, refs, 'person', len(person_names_by_source) + 1)}"
                )
    outfit_names_by_source = {}
    manifest = []
    for i, (pose_item, pose_path) in enumerate(selected, 1):
        assigned_people = person_assignments[i - 1]
        person_names = [person_names_by_source[candidate["path"]] for candidate in assigned_people]
        costume_item, costume_paths = costume_sets[i - 1]
        outfit_names = []
        for costume_path in costume_paths:
            if costume_path not in outfit_names_by_source:
                outfit_names_by_source[costume_path] = (
                    f"references/{_copy_named(costume_path, refs, 'costume', len(outfit_names_by_source) + 1)}"
                )
            outfit_names.append(outfit_names_by_source[costume_path])
        pose_name = f"references/{_copy_named(pose_path, refs, 'pose', i)}"
        all_refs = person_names + outfit_names + [pose_name]
        person_range = "<Picture 1>" if len(person_names) == 1 else f"<Picture 1>-<Picture {len(person_names)}>"
        outfit_start = len(person_names) + 1
        outfit_range = f"<Picture {outfit_start}>" if len(outfit_names) == 1 else f"<Picture {outfit_start}>-<Picture {outfit_start + len(outfit_names) - 1}>"
        pose_number = len(all_refs)
        portrait_body_proportions = (
            assigned_people[0]["body_proportions"] if len(assigned_people) == 1 else None
        )
        pose_text = build_pose_prompt_text(pose_item)
        costume_text = build_costume_prompt_text(costume_item)
        portrait_text = build_portrait_prompt_text(assigned_people)
        prompt = build_h3_generation_prompt(
            person_range, outfit_range, pose_number,
            portrait_body_proportions, instruction, prompt_mode, prompt_template,
            costume_text=costume_text, pose_text=pose_text,
            creative_style=creative_style, creative_body=creative_body,
            portrait_text=portrait_text,
        )
        if len(assigned_people) > 1:
            portrait_id = "person-pool"
            portrait_title = f"person pool ({len(assigned_people)} refs)"
            portrait_mode = PERSON_MODE_MULTI_REF
        else:
            portrait_id = assigned_people[0]["id"]
            portrait_title = assigned_people[0]["title"]
            if not people_files and person_mode == PERSON_MODE_PER_ITEM:
                portrait_mode = "gallery-random-each"
            elif not people_files and person_mode == PERSON_MODE_BALANCED:
                portrait_mode = "gallery-balanced"
            elif not people_files:
                portrait_mode = "gallery-random-batch"
            else:
                portrait_mode = person_mode
        manifest.append({
            "output": f"target_{i:04d}.png", "pose_id": pose_item.get("id", ""),
            "pose_name": pose_item.get("姿势名称", ""), "costume_id": costume_item.get("id", ""),
            "costume_name": costume_item.get("服装名称", ""),
            "portrait_id": portrait_id, "portrait_title": portrait_title,
            "portrait_mode": portrait_mode, "selection_seed": int(seed),
            "person_strategy": person_mode, "gallery_strategy": person_mode,
            "outfit_mode": outfit_mode,
            "portrait_body_proportions": portrait_body_proportions,
            "prompt_mode": prompt_mode,
            "image_model": image_model,
            "model_backend": "h3",
            "width": int(width),
            "height": int(height),
            "creative_style": creative_style,
            "creative_body": creative_body,
            "prompt": prompt, "edit_image": all_refs,
        })
    (root / "generation.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    person_counts = Counter(candidate["id"] for assignment in person_assignments for candidate in assignment)
    person_origin = "上传人物" if people_files else "图库随机"
    if len(person_counts) == 1:
        only_person = next(iter(person_counts))
        person_source = f"{person_origin} · {only_person} · 本批全部使用同一人物"
    else:
        distribution = "、".join(f"{person_id}×{amount}" for person_id, amount in person_counts.most_common(6))
        if len(person_counts) > 6:
            distribution += "…"
        person_source = f"{person_origin} · {person_mode} · 本批 {len(person_counts)} 人：{distribution}"

    pose_ids = [str(item.get("id", "")) for item, _ in selected]
    pose_source = (
        f"{'上传' if pose_files else '仓库'} Pose · {len(pose_pool)} 个候选 · 随机抽取 {len(selected)} 条"
        f"（{ '、'.join(pose_ids[:6]) }{'…' if len(pose_ids) > 6 else ''}）"
    )
    costume_ids = [str(item.get("id", "")) for item, _ in costume_sets]
    outfit_source = (
        f"{'上传' if outfit_files else '仓库'}服装 · {len(costume_pool)} 个候选 · "
        f"随机抽取 {len(costume_sets)} 套（{ '、'.join(costume_ids[:6]) }{'…' if len(costume_ids) > 6 else ''}）"
    )
    proportion_values = {
        candidate["body_proportions"] for assignment in person_assignments for candidate in assignment
    }
    if not uses_reference_body(creative_body):
        proportion_source = f"{creative_body}；不沿用人物或 Pose 的体型比例，Pose 仅控制姿势"
    elif proportion_values == {False}:
        proportion_source = "人物参考不提供身材比例；使用自然的亚洲女性比例，Pose 仅控制姿势"
    elif proportion_values == {None}:
        proportion_source = "人物参考仅保留身份和可见特征；比例收敛为自然的亚洲女性比例，Pose 仅控制姿势"
    elif len(proportion_values) == 1:
        proportion_source = "以人物参考的可见体型比例为准并收敛为自然的亚洲女性比例；Pose 仅控制姿势"
    else:
        proportion_source = "逐条按所分配的人物参考处理比例，并统一收敛为自然的亚洲女性比例；Pose 仅控制姿势"
    gallery_summary = f"{gallery_note}\n" if gallery_note else ""
    return str(root), (
        f"已构造 {len(manifest)} 个生成任务。\n人物：{person_source}\n"
        f"服装：{outfit_source}\nPose：{pose_source}\n身材与比例：{proportion_source}\n"
        f"图片模型：{image_model}\n"
        f"人物风格：{creative_style}\n身材选择：{creative_body}\n"
        f"提示词：{prompt_mode}\n素材 Seed：{seed}\n"
        f"{gallery_summary}"
        "提示词由人物、服装与 Pose 字段自动组合；人物分配记录在 generation.json。"
    )


def build_h3_generation_command(job_dir, width, height, seed):
    job = Path(job_dir).expanduser().resolve()
    if not (job / "generation.json").is_file():
        raise gr.Error("请先构造批量生成任务")
    model_cache = MODELS / ".modelscope-cache"
    model_dir = MODELS
    ensure = f"{py()} {q(ROOT / 'ensure_h3_runtime.py')}"
    download = f"{py()} {q(ROOT / 'download_h3.py')} --profile quantized --models-dir {q(model_dir)}"
    generate = f"{py()} {q(ROOT / 'generate_h3_targets.py')} --job {q(job)} --width {int(width)} --height {int(height)} --seed {int(seed)}"
    command = (
        f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MODELSCOPE_CACHE={q(model_cache)} DIFFSYNTH_MODEL_BASE_PATH={q(model_dir)} && "
        f"{ensure} && {download} && DIFFSYNTH_SKIP_DOWNLOAD=True {generate}"
    )
    return h3_gpu_lock_command(command)


def start_h3_generation(
    image_model, generation_preset, people, outfits, poses, person_mode, outfit_mode,
    portrait_keyword, pose_keyword, costume_keyword,
    max_items, safe_only, job_name, instruction, prompt_mode, prompt_template,
    creative_style, creative_body, size_choice, width, height, seed,
):
    model_spec = image_model_spec(image_model)
    if not model_spec.available:
        raise gr.Error(f"{model_spec.label} 的批量图片生成适配器尚未接入；请切换到已接入的模型。")
    if model_spec.backend != "h3":
        raise gr.Error(f"当前生成入口只实现了 H3 后端，收到模型：{model_spec.label}")
    gpu_error = h3_gpu_support_error()
    if gpu_error:
        raise gr.Error(gpu_error)
    preset_count = GENERATION_BATCH_PRESETS.get(generation_preset)
    if preset_count is not None:
        max_items = preset_count
    width, height = resolve_generation_size(size_choice, width, height)
    try:
        resolved_seed = int(seed)
    except (TypeError, ValueError) as exc:
        raise gr.Error("Seed 必须是整数；留 -1 表示每批自动随机") from exc
    if resolved_seed < 0:
        resolved_seed = random.SystemRandom().randrange(1, 2**31)
    job_dir, summary = prepare_h3_generation_job(
        people, outfits, poses, person_mode, outfit_mode,
        portrait_keyword, pose_keyword, costume_keyword,
        max_items, safe_only, job_name, instruction, prompt_mode, prompt_template,
        creative_style, creative_body, model_spec.label, resolved_seed, width, height,
    )
    command = build_h3_generation_command(job_dir, width, height, resolved_seed)
    summary = (
        f"{summary}\n训练规模：{generation_preset}\n"
        f"画面尺寸：{width} × {height}\n本次 Seed：{resolved_seed}"
    )
    status = start_training(command, "h3-generate", {"job_dir": job_dir, "summary": summary, "command": command})
    return job_dir, summary, command, status, job_dir


def start_h3_lora(
    image_model, dataset_dir, output_name, model_profile, repeats, epochs,
    rank, lr, max_pixels, adapter_path, low_vram,
):
    model_spec = image_model_spec(image_model)
    if model_spec.backend != "h3" or not model_spec.available:
        raise gr.Error(f"{model_spec.label} 尚未接入 H3 LoRA 训练；请切换到 H3 模型")
    gpu_error = h3_gpu_support_error()
    if gpu_error:
        raise gr.Error(gpu_error)
    command, output_dir = build_h3_character_command(
        dataset_dir, output_name, model_profile, repeats, epochs,
        rank, lr, max_pixels, adapter_path, low_vram,
    )
    status = start_training(
        command,
        "h3-train",
        {"dataset_dir": str(Path(dataset_dir).expanduser().resolve()), "output_dir": output_dir, "command": command},
    )
    return command, output_dir, status


def _review_target_entries(job: Path) -> list[tuple[int, Path, dict]]:
    manifest_file = job / "generation.json"
    if manifest_file.is_file():
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise gr.Error(f"generation.json 读取失败：{exc}") from exc
        entries = []
        for number, item in enumerate(manifest, 1):
            target = job / "targets" / str(item.get("output", ""))
            if target.is_file():
                entries.append((number, target, item))
        return entries
    metadata_file = job / "metadata.json"
    if metadata_file.is_file():
        try:
            records = json.loads(metadata_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise gr.Error(f"metadata.json 读取失败：{exc}") from exc
        entries = []
        for number, item in enumerate(records, 1):
            target = job / str(item.get("image", ""))
            if target.is_file():
                entries.append((number, target, item))
        if entries:
            return entries
    return [(number, target, {}) for number, target in enumerate(_job_target_files(job), 1)]


def _format_number_set(numbers: set[int]) -> str:
    values = sorted(number for number in numbers if number > 0)
    parts: list[str] = []
    index = 0
    while index < len(values):
        start = end = values[index]
        while index + 1 < len(values) and values[index + 1] == end + 1:
            index += 1
            end = values[index]
        if start == end:
            parts.append(str(start))
        elif end - start == 1:
            parts.append(f"{start},{end}")
        else:
            parts.append(f"{start}-{end}")
        index += 1
    return ",".join(parts)


def _resolved_review_job_dir(job_dir) -> str:
    value = str(job_dir or "").strip()
    return value or _latest_generation_job_dir()


def _review_state_file(job_dir) -> Path | None:
    value = _resolved_review_job_dir(job_dir)
    if not value:
        return None
    job = Path(value).expanduser().resolve()
    return job / "review.json" if job.is_dir() else None


def _load_review_exclusions(job_dir) -> str:
    path = _review_state_file(job_dir)
    if path is None or not path.is_file():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return ""
    value = data.get("exclusions", "") if isinstance(data, dict) else ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        try:
            return _format_number_set({int(item) for item in value if int(item) > 0})
        except (TypeError, ValueError):
            return ""
    return ""


def _save_review_exclusions(job_dir, exclusions: str) -> None:
    path = _review_state_file(job_dir)
    if path is None:
        return
    temp_path = path.with_name(f"{path.name}.tmp")
    try:
        temp_path.write_text(
            json.dumps({"exclusions": exclusions or "", "updated_at": _now_text()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temp_path, path)
    except OSError:
        pass


def _render_generation_review(job_dir, exclusions="") -> tuple[list[tuple[str, str]], str]:
    if not str(job_dir or "").strip():
        return [], "请先生成目标图，或从成果库载入一个批次。"
    job = Path(job_dir).expanduser().resolve()
    entries = _review_target_entries(job)
    if not entries:
        return [], "当前批次还没有可预览的目标图；先完成生成，或从成果库载入一个批次。"
    excluded = _number_set(exclusions) & {number for number, _, _ in entries}
    gallery = []
    for number, path, item in entries:
        details = " · ".join(
            part for part in (
                str(item.get("portrait_title") or item.get("portrait_id") or ""),
                str(item.get("costume_id") or ""),
                str(item.get("pose_id") or ""),
                str(item.get("pose_name") or ""),
            ) if part
        )
        caption = f"#{number:04d}"
        if number in excluded:
            caption = f"{caption} · 已排除"
        if details:
            caption = f"{caption} · {details}"
        gallery.append((str(path), caption))
    retained = len(entries) - len(excluded)
    status = (
        f"候选图 {len(entries)} 张 · 已排除 {len(excluded)} 张 · 将保留 {retained} 张。"
        "点选图片后可用“排除选中/恢复选中”操作；也可以直接修改排除序号。"
    )
    return gallery, status


def _latest_generation_job_dir() -> str:
    task = current_task_snapshot()
    if task and task.get("job_dir"):
        return str(task["job_dir"])
    jobs = [path.parent for path in DATASETS.glob("*/generation.json")]
    if not jobs:
        return ""
    return str(max(jobs, key=lambda path: path.stat().st_mtime))


def refresh_generated_targets(job_dir, exclusions=""):
    if not str(job_dir or "").strip():
        job_dir = _latest_generation_job_dir()
    if not str(exclusions or "").strip():
        exclusions = _load_review_exclusions(job_dir)
    gallery, status = _render_generation_review(job_dir, exclusions)
    return gallery, None, status, exclusions, exclusions


def generation_results_signature(job_dir) -> str:
    value = _resolved_review_job_dir(job_dir)
    if not value:
        return ""
    job = Path(value).expanduser().resolve()
    if not job.is_dir():
        return ""
    try:
        entries = _review_target_entries(job)
    except gr.Error:
        return ""
    if not entries:
        return f"{job}:0"
    try:
        latest = max(path.stat().st_mtime_ns for _, path, _ in entries)
    except OSError:
        return f"{job}:{len(entries)}:unknown"
    return f"{job}:{len(entries)}:{latest}"


def refresh_generation_results_live(job_dir, exclusions, previous_signature):
    current_signature = generation_results_signature(job_dir)
    if not current_signature or current_signature == previous_signature:
        return gr.skip(), gr.skip(), previous_signature
    gallery, status = _render_generation_review(job_dir, exclusions)
    return gallery, status, current_signature


def mirror_text_state(value):
    return value


def select_review_target(job_dir, exclusions, evt: gr.SelectData):
    job_dir = _resolved_review_job_dir(job_dir)
    if not job_dir:
        return None, "没有可用的生成批次，请先生成目标图或从成果库载入批次。"
    entries = _review_target_entries(Path(job_dir).expanduser().resolve())
    if not isinstance(evt.index, int) or not 0 <= evt.index < len(entries):
        return None, "没有识别到选中的候选图，请刷新后重试。"
    number = entries[evt.index][0]
    state = "已排除" if number in _number_set(exclusions) else "保留"
    return number, f"已选中 #{number:04d} · 当前状态：{state}。点击下方按钮排除或恢复。"


def _set_selected_target_excluded(job_dir, exclusions, selected_number, exclude: bool):
    if selected_number is None:
        raise gr.Error("请先在候选图中点选一张图片")
    job_dir = _resolved_review_job_dir(job_dir)
    if not job_dir:
        raise gr.Error("没有可用的生成批次，请先生成目标图或从成果库载入批次")
    numbers = _number_set(exclusions)
    if exclude:
        numbers.add(int(selected_number))
    else:
        numbers.discard(int(selected_number))
    formatted = _format_number_set(numbers)
    _save_review_exclusions(job_dir, formatted)
    gallery, status = _render_generation_review(job_dir, formatted)
    action = "已排除" if exclude else "已恢复"
    return (
        gallery, formatted, int(selected_number),
        f"{action} #{int(selected_number):04d}。\n{status}", formatted,
    )


def exclude_selected_target(job_dir, exclusions, selected_number):
    return _set_selected_target_excluded(job_dir, exclusions, selected_number, True)


def restore_selected_target(job_dir, exclusions, selected_number):
    return _set_selected_target_excluded(job_dir, exclusions, selected_number, False)


def clear_review_exclusions(job_dir):
    _save_review_exclusions(job_dir, "")
    gallery, status = _render_generation_review(job_dir, "")
    return gallery, "", None, status, ""


def apply_review_exclusions(job_dir, exclusions):
    _save_review_exclusions(job_dir, exclusions)
    gallery, status = _render_generation_review(job_dir, exclusions)
    return gallery, None, status, exclusions


def _number_set(raw: str) -> set[int]:
    result = set()
    for part in re.split(r"[,，\s]+", raw or ""):
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            if start.isdigit() and end.isdigit():
                first, last = int(start), int(end)
                if last < first:
                    raise gr.Error(f"排除序号范围无效：{part}")
                if last - first > 20000:
                    raise gr.Error(f"排除序号范围过大：{part}")
                result.update(range(first, last + 1))
        elif part.isdigit():
            result.add(int(part))
    return result


def finalize_generated_dataset(job_dir, trigger_word, exclusions, refined_targets):
    job = Path(job_dir).expanduser().resolve()
    if not job.is_dir():
        raise gr.Error("生成任务不存在")
    task = current_task_snapshot()
    if task and _task_active(task):
        raise gr.Error("仍有生成或训练任务运行；请等待结束或先停止后再整理训练集")
    entries = _review_target_entries(job)
    if not entries:
        raise gr.Error("还没有可用的目标图")
    if not str(exclusions or "").strip():
        exclusions = _load_review_exclusions(job)
    refined = _upload_paths(refined_targets)
    if len(refined) > len(entries):
        raise gr.Error(
            f"精修替换图有 {len(refined)} 张，但当前只有 {len(entries)} 张候选图；"
            "请删除多余的替换图或先恢复被排除/缺失的候选图"
        )
    for index, source in enumerate(refined, 1):
        shutil.copy2(source, entries[index - 1][1])
    trigger = (trigger_word or "").strip()
    _save_review_exclusions(job, exclusions)
    excluded, excluded_numbers, records = _number_set(exclusions), [], []
    for number, target, item in entries:
        if number in excluded:
            excluded_numbers.append(number)
        else:
            base_prompt = str(item.get("prompt", "")).strip()
            if trigger and not base_prompt.lower().startswith(f"{trigger.lower()} "):
                base_prompt = " ".join((trigger, base_prompt)).strip()
            if not base_prompt:
                raise gr.Error(f"第 {number} 条候选图缺少提示词；请填写人物触发词或使用带 prompt 的生成批次")
            record = {
                "image": target.relative_to(job).as_posix(),
                "prompt": base_prompt,
            }
            refs = item.get("edit_image") or []
            if isinstance(refs, str):
                refs = [refs]
            if refs:
                record["edit_image"] = list(refs)
            records.append(record)
    if not records:
        raise gr.Error("排除后没有剩余目标图；请恢复至少一张候选图")
    _validate_h3_training_files(job, records)
    metadata = job / "metadata.json"
    metadata.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    excluded_text = _format_number_set(set(excluded_numbers)) or "无"
    message = (
        f"已应用 {len(refined)} 张顺序精修图，排除 {len(excluded_numbers)} 张（{excluded_text}），"
        f"保留 {len(records)} 张并重建 metadata.json。"
    )
    return str(job), message, str(job)


def suggest_h3_output_name(job_dir) -> tuple[str, str]:
    if not job_dir:
        return H3_DEFAULT_OUTPUT_NAME, H3_DEFAULT_OUTPUT_NAME
    name = Path(job_dir).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
    output_name = f"{(name or 'h3-character')[:96].rstrip('-.')}-lora"
    return output_name, output_name


def finalize_and_start_h3_training(
    image_model, job_dir, trigger_word, exclusions, refined_targets,
    output_name, model_profile, repeats, epochs, rank, lr, max_pixels, adapter_path, low_vram,
):
    task = current_task_snapshot()
    if task and _task_active(task):
        raise gr.Error("仍有后台任务运行；请等待当前生成或训练结束后再一键训练")
    dataset_dir, review_summary, _ = finalize_generated_dataset(
        job_dir, trigger_word, exclusions, refined_targets,
    )
    try:
        command, output_dir, status = start_h3_lora(
            image_model, dataset_dir, output_name, model_profile, repeats, epochs,
            rank, lr, max_pixels, adapter_path, low_vram,
        )
    except gr.Error as exc:
        return (
            dataset_dir,
            f"{review_summary}\n训练未启动：{exc}",
            gr.update(),
            gr.update(),
            f"训练未启动：{exc}",
            dataset_dir,
        )
    return (
        dataset_dir,
        f"{review_summary}\n已使用默认训练参数启动 LoRA 训练。",
        command,
        output_dir,
        status,
        dataset_dir,
    )


def _positive_int(label: str, value, minimum: int = 1) -> int:
    try:
        number_value = int(value)
    except (TypeError, ValueError) as exc:
        raise gr.Error(f"{label} 必须是整数") from exc
    if number_value < minimum:
        raise gr.Error(f"{label} 必须大于或等于 {minimum}")
    return number_value


def _positive_float(label: str, value) -> float:
    try:
        number_value = float(value)
    except (TypeError, ValueError) as exc:
        raise gr.Error(f"{label} 必须是数字") from exc
    if not math.isfinite(number_value) or number_value <= 0:
        raise gr.Error(f"{label} 必须是大于 0 的有限数字")
    return number_value


def _validate_h3_training_files(dataset: Path, records: list) -> None:
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise gr.Error(f"metadata.json 第 {index} 条不是对象")
        prompt = record.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise gr.Error(f"metadata.json 第 {index} 条缺少有效 prompt")
        for key, required in (("image", True), ("edit_image", False)):
            raw = record.get(key)
            if not raw:
                if required:
                    raise gr.Error(f"metadata.json 第 {index} 条缺少 {key}")
                continue
            values = raw if isinstance(raw, list) else [raw]
            for value in values:
                path = Path(str(value)).expanduser()
                if not path.is_absolute():
                    path = dataset / path
                if not path.is_file():
                    raise gr.Error(f"第 {index} 条样本缺少文件：{path}")
                if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                    raise gr.Error(f"第 {index} 条样本不是支持的图片格式：{path}")
                try:
                    with Image.open(path) as image:
                        image.verify()
                except (OSError, ValueError, Image.DecompressionBombError) as exc:
                    raise gr.Error(f"第 {index} 条样本图片无法读取：{path}") from exc


def build_h3_character_command(dataset_dir, output_name, model_profile, repeats, epochs, rank, lr, max_pixels, adapter_path, low_vram):
    dataset = Path(dataset_dir).expanduser().resolve()
    metadata = dataset / "metadata.json"
    if not metadata.is_file():
        raise gr.Error("找不到 metadata.json，请先生成训练数据集")
    try:
        records = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise gr.Error(f"metadata.json 读取失败：{exc}") from exc
    if not records:
        raise gr.Error("metadata.json 为空，请重新创建训练数据集")
    if not isinstance(records, list):
        raise gr.Error("metadata.json 必须是样本列表")
    _validate_h3_training_files(dataset, records)
    if model_profile not in H3_MODEL_PROFILES:
        raise gr.Error(f"未知模型配置：{model_profile}")
    if model_profile == H3_BF16_PROFILE and model_profile not in H3_AVAILABLE_MODEL_PROFILES:
        raise gr.Error(
            f"原始 BF16 两阶段需要约 {H3_BF16_MIN_MODEL_GB}GB 模型空间和至少 "
            f"{H3_BF16_MIN_TOTAL_GB}GB 总预算；当前为模型 {MODEL_BUDGET_GB:.0f}GB、"
            f"总计 {TOTAL_BUDGET_GB:.0f}GB。请改用量化版，或调高 "
            "TRAINER_MODEL_BUDGET_GB 与 TRAINER_TOTAL_BUDGET_GB 后重启。"
        )
    repeats = _positive_int("数据重复", repeats)
    epochs = _positive_int("Epoch", epochs)
    rank = _positive_int("LoRA Rank", rank)
    max_pixels = _positive_int("最大像素", max_pixels, 65536)
    lr = _positive_float("学习率", lr)
    total_steps = len(records) * repeats * epochs
    save_steps = max(1, math.ceil(total_steps / 5))
    edit_flags = [bool(record.get("edit_image")) for record in records]
    if any(edit_flags) and not all(edit_flags):
        raise gr.Error("数据集中的 edit_image 不一致；请统一为全部带参考图或全部仅目标图")
    data_inputs = "--data_file_keys image,edit_image --extra_inputs edit_image" if all(edit_flags) else "--data_file_keys image"
    script = REPOS / "DiffSynth-Studio/examples/minimax_h3_image/model_training/train.py"
    if not script.is_file():
        raise gr.Error("尚未安装 DiffSynth-Studio，请先在“环境”页安装训练后端")
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", output_name.strip()).strip("-.") or "h3-character-lora"
    dataset_digest = hashlib.sha256(
        (json.dumps(records, ensure_ascii=False, sort_keys=True) + f":{int(max_pixels)}").encode("utf-8")
    ).hexdigest()[:8]
    cache_dir = OUTPUTS / f"{name}-split-cache-{dataset_digest}"
    lora_dir = OUTPUTS / name
    if lora_dir.is_dir() and any(
        path.is_file() and path.suffix.lower() == ".safetensors"
        for path in lora_dir.rglob("*")
    ):
        raise gr.Error(
            f"输出目录已有 LoRA 权重：{lora_dir}。请更换训练输出名称，避免覆盖已有结果。"
        )
    model_cache = MODELS / ".modelscope-cache"
    model_dir = MODELS
    bf16_models = (
        "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,"
        "MiniMax/MiniMax-H3-Image:transformer/model*.safetensors,"
        "MiniMax/MiniMax-H3-Image:pdd_fused_heads.safetensors,"
        "MiniMax/MiniMax-H3-Image:vae/diffusion_pytorch_model.safetensors"
    )
    targets = "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2"
    training_options = (
        f"--max_pixels {int(max_pixels)} "
        f"--learning_rate {float(lr)} --remove_prefix_in_ckpt pipe.dit. --lora_base_model dit "
        f"--lora_target_modules {q(targets)} --lora_rank {int(rank)} --use_gradient_checkpointing "
        f"--save_steps {int(save_steps)} --enable_csv_log"
    )
    common_tail = f"{data_inputs} {training_options}"
    cached_tail = f"--data_file_keys image {training_options}"
    adapter_value = str(adapter_path or "").strip()
    adapter = Path(adapter_value).expanduser() if adapter_value else None
    if adapter and not adapter.is_file():
        raise gr.Error("训练适配器路径不存在")
    if model_profile == H3_QUANTIZED_PROFILE:
        quant_models = (
            "DiffSynth-Studio/MiniMax-H3-Image-EntroPack:models/dit_7bit.safetensors,"
            "DiffSynth-Studio/EntroPackPreQuants:models/MiniMax-H3/text_encoder_4bpp.safetensors,"
            "MiniMax/MiniMax-H3-Image:pdd_fused_heads.safetensors,"
            "DiffSynth-Studio/MiniMax-H3-Image-EntroPack:models/vae_int8.safetensors"
        )
        stage1 = (
            f"{h3_accelerate()} {q(ROOT / 'h3_train.py')} --dataset_base_path {q(dataset)} "
            f"--dataset_metadata_path {q(metadata)} --dataset_repeat 1 "
            f"--num_epochs 1 --model_id_with_origin_paths {q(quant_models)} "
            f"{common_tail} --task sft:data_process --output_path {q(cache_dir)} "
            f"--offload_models {q('DiffSynth-Studio/MiniMax-H3-Image-EntroPack:models/dit_7bit.safetensors,MiniMax/MiniMax-H3-Image:pdd_fused_heads.safetensors')} "
            f"--dataset_num_workers 4"
        )
        stage2_models = (
            "DiffSynth-Studio/MiniMax-H3-Image-EntroPack:models/dit_7bit.safetensors,"
            "MiniMax/MiniMax-H3-Image:pdd_fused_heads.safetensors"
        )
        stage2 = (
            f"{h3_accelerate()} {q(ROOT / 'h3_train.py')} --dataset_base_path {q(cache_dir)} "
            f"--dataset_repeat {int(repeats)} --num_epochs {int(epochs)} "
            f"--model_id_with_origin_paths {q(stage2_models)} {cached_tail} "
            f"--task sft:train --output_path {q(lora_dir)} --dataset_num_workers 4 --find_unused_parameters"
        )
        if low_vram:
            stage2 += " --enable_model_cpu_offload"
        if adapter:
            stage2 += f" --preset_lora_path {q(adapter)} --preset_lora_model dit"
        ensure = f"{py()} {q(ROOT / 'ensure_h3_runtime.py')}"
        download = f"{py()} {q(ROOT / 'download_h3.py')} --profile quantized --models-dir {q(model_dir)}"
        command = (
            f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MODELSCOPE_CACHE={q(model_cache)} DIFFSYNTH_MODEL_BASE_PATH={q(model_dir)} && "
            f"{ensure} && {download} && export DIFFSYNTH_SKIP_DOWNLOAD=True && {stage1} && {stage2}"
        )
        return h3_gpu_lock_command(command), str(lora_dir)
    if low_vram:
        common_tail += (
            " --use_gradient_checkpointing_offload --enable_model_cpu_offload "
            "--enable_optimizer_cpu_offload --initialize_model_on_cpu"
        )
    common = f"{common_tail} --model_id_with_origin_paths {q(bf16_models)}"
    cached_common = f"{cached_tail} --model_id_with_origin_paths {q(bf16_models)}"
    stage1 = (
        f"{h3_accelerate()} {q(script)} --dataset_base_path {q(dataset)} "
        f"--dataset_metadata_path {q(metadata)} --dataset_repeat 1 --num_epochs 1 {common} "
        f"--output_path {q(cache_dir)} --task sft:data_process "
        f"--offload_models {q('MiniMax/MiniMax-H3-Image:transformer/model*.safetensors,MiniMax/MiniMax-H3-Image:pdd_fused_heads.safetensors')}"
    )
    stage2 = (
        f"{h3_accelerate()} {q(script)} --dataset_base_path {q(cache_dir)} "
        f"--dataset_repeat {int(repeats)} --num_epochs {int(epochs)} {cached_common} "
        f"--output_path {q(lora_dir)} --task sft:train "
        f"--offload_models {q('MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3-Image:vae/diffusion_pytorch_model.safetensors')}"
    )
    if adapter:
        stage2 += f" --preset_lora_path {q(adapter)} --preset_lora_model dit"
    ensure = f"{py()} {q(ROOT / 'ensure_h3_runtime.py')}"
    download = f"{py()} {q(ROOT / 'download_h3.py')} --profile bf16 --models-dir {q(model_dir)}"
    command = (
        f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MODELSCOPE_CACHE={q(model_cache)} DIFFSYNTH_MODEL_BASE_PATH={q(model_dir)} && "
        f"{ensure} && {download} && export DIFFSYNTH_SKIP_DOWNLOAD=True && {stage1} && {stage2}"
    )
    return h3_gpu_lock_command(command), str(lora_dir)


def write_dataset_config(kind: str, data_dir: str, width: int, height: int, frames: int, repeats: int) -> Path:
    data = Path(data_dir).expanduser().resolve()
    if not data.is_dir():
        raise gr.Error(f"数据目录不存在：{data}")
    general = {"caption_extension": ".txt", "enable_bucket": True}
    if kind == "image":
        config = {"general": general, "datasets": [{
            "resolution": [int(width), int(height)], "batch_size": 1,
            "subsets": [{"image_dir": str(data), "num_repeats": int(repeats)}],
        }]}
    else:
        cache = HOME / "cache" / data.name
        cache.mkdir(parents=True, exist_ok=True)
        config = {"general": general, "datasets": [{
            "resolution": [int(width), int(height)], "batch_size": 1,
            "num_repeats": int(repeats), "video_directory": str(data),
            "cache_directory": str(cache), "target_frames": [int(frames)],
            "frame_extraction": "head",
        }]}
    path = CONFIGS / f"dataset_{kind}.toml"
    path.write_text(toml.dumps(config), encoding="utf-8")
    return path


def build_command(task, data_dir, model_path, component1, component2, component3, output_name, width, height, frames, repeats, epochs, rank, lr, extra):
    if not output_name.strip():
        raise gr.Error("请填写输出名称")
    model = Path(model_path).expanduser().resolve()
    if not model.exists():
        raise gr.Error(f"模型路径不存在：{model}")
    image = task in ("SDXL LoRA", "FLUX.1 LoRA")
    rocm_video = not image and rocm_video_mode()
    dataset = write_dataset_config("image" if image else "video", data_dir, width, height, frames, repeats)
    output = OUTPUTS / output_name.strip()
    output.mkdir(parents=True, exist_ok=True)
    optimizer = "adamw" if rocm_video else "adamw8bit"
    common = (
        f"--dataset_config {q(dataset)} --output_dir {q(output)} --output_name {q(output_name.strip())} "
        f"--mixed_precision bf16 --save_precision bf16 --gradient_checkpointing "
        f"--optimizer_type {optimizer} --learning_rate {float(lr)} --network_dim {int(rank)} "
        f"--max_train_epochs {int(epochs)} --save_every_n_epochs 1 --seed 42"
    )
    if task == "SDXL LoRA":
        script = REPOS / "sd-scripts/sdxl_train_network.py"
        cmd = f"{accelerate()} --num_cpu_threads_per_process 1 {q(script)} --pretrained_model_name_or_path {q(model)} --network_module networks.lora {common} --cache_latents"
    elif task == "FLUX.1 LoRA":
        if not all(Path(x).expanduser().exists() for x in (component1, component2, component3)):
            raise gr.Error("FLUX.1 需要有效的 CLIP-L、T5XXL 和 AE/VAE 路径")
        script = REPOS / "sd-scripts/flux_train_network.py"
        cmd = f"{accelerate()} --num_cpu_threads_per_process 1 {q(script)} --pretrained_model_name_or_path {q(model)} --clip_l {q(Path(component1).expanduser())} --t5xxl {q(Path(component2).expanduser())} --ae {q(Path(component3).expanduser())} --network_module networks.lora_flux {common} --cache_latents --cache_text_encoder_outputs --network_train_unet_only"
    else:
        vae, t5 = Path(component1).expanduser(), Path(component2).expanduser()
        if not vae.exists() or not t5.exists():
            raise gr.Error("Wan 训练需要有效的 VAE 和 UMT5 编码器路径")
        base = REPOS / "musubi-tuner/src/musubi_tuner"
        wan_task = {"Wan 2.1 T2V 1.3B LoRA": "t2v-1.3B", "Wan 2.1 T2V 14B LoRA": "t2v-14B", "Wan 2.2 T2V A14B LoRA": "t2v-A14B"}[task]
        cache_latents = f"{py()} {q(base / 'wan_cache_latents.py')} --dataset_config {q(dataset)} --vae {q(vae)} --vae_cache_cpu"
        cache_text = f"{py()} {q(base / 'wan_cache_text_encoder_outputs.py')} --dataset_config {q(dataset)} --t5 {q(t5)} --batch_size 1"
        train = f"{accelerate()} --num_cpu_threads_per_process 1 --mixed_precision bf16 {q(base / 'wan_train_network.py')} --task {wan_task} --dit {q(model)} --network_module networks.lora_wan {common} --sdpa --blocks_to_swap 20"
        if rocm_video:
            cache_latents += " --disable_cudnn_backend"
        else:
            cache_text += " --fp8_t5"
            train += " --fp8_base"
        cmd = f"{cache_latents} && {cache_text} && {train}"
    if extra.strip():
        cmd += " " + extra.strip()
    return cmd


def _now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _elapsed_text(started_at: str) -> str:
    try:
        started = time.mktime(time.strptime(started_at, "%Y-%m-%d %H:%M:%S"))
    except (TypeError, ValueError, OverflowError):
        return "--:--:--"
    seconds = max(0, int(time.time() - started))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _pid_alive(pid) -> bool:
    try:
        process = psutil.Process(int(pid))
    except (psutil.Error, TypeError, ValueError):
        return False
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def _load_task_state() -> dict | None:
    try:
        data = json.loads(TASK_STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _write_task_state(task: dict) -> None:
    temp_path = TASK_STATE_FILE.with_name(f"{TASK_STATE_FILE.name}.tmp")
    try:
        temp_path.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp_path, TASK_STATE_FILE)
    except OSError:
        pass


def _reconcile_task_locked() -> dict | None:
    task = STATE.get("task") or _load_task_state()
    if task and task.get("status") in {"running", "paused"} and not _pid_alive(task.get("pid")):
        task.update(status="finished", finished_at=task.get("finished_at") or _now_text())
        _write_task_state(task)
    STATE["task"] = task
    return task


def current_task_snapshot() -> dict | None:
    with STATE["lock"]:
        task = _reconcile_task_locked()
        return dict(task) if task else None


def _task_running(task: dict | None) -> bool:
    return bool(task and task.get("status") == "running" and _pid_alive(task.get("pid")))


def _task_paused(task: dict | None) -> bool:
    return bool(task and task.get("status") == "paused" and _pid_alive(task.get("pid")))


def _task_active(task: dict | None) -> bool:
    return _task_running(task) or _task_paused(task)


def _task_is_running_locked() -> bool:
    proc = STATE.get("process")
    if proc and proc.poll() is None:
        return True
    return _task_active(_reconcile_task_locked())


def _task_label(task: dict | None) -> str:
    if not task:
        return "任务"
    return TASK_LABELS.get(task.get("kind", ""), task.get("kind") or "任务")


def _exit_signal_name(exit_code: int) -> str:
    try:
        return signal.Signals(-exit_code).name
    except ValueError:
        return f"信号 {-exit_code}"


def _task_status_text(task: dict | None, kind: str) -> str:
    if not task:
        return "空闲"
    active = _task_active(task)
    paused = _task_paused(task)
    label = _task_label(task)
    if task.get("kind") != kind:
        if active:
            state_text = "已暂停" if paused else "运行中"
            return f"其他任务{state_text}：{label} · PID {task.get('pid')}"
        return "空闲"
    if paused:
        return (
            f"已暂停 · {label} · PID {task.get('pid')} · "
            f"暂停于 {task.get('paused_at', '')} · 点击“继续”恢复"
        )
    if active:
        return f"运行中 · {label} · PID {task.get('pid')} · 已运行 {_elapsed_text(task.get('started_at', ''))}"
    if task.get("status") == "finished":
        exit_code = task.get("exit_code")
        if isinstance(exit_code, int) and exit_code < 0:
            signal_name = task.get("exit_signal") or _exit_signal_name(exit_code)
            return f"已结束 · {label} · 被 {signal_name} 终止 · {task.get('finished_at', '')}"
        return f"已结束 · {label} · 退出码 {exit_code if exit_code is not None else '未知'} · {task.get('finished_at', '')}"
    return f"{label} · {task.get('status', '未知')}"


def _task_statuses(task: dict | None) -> tuple[str, str, str]:
    return (
        _task_status_text(task, "h3-generate"),
        _task_status_text(task, "h3-train"),
        _task_status_text(task, "train"),
    )


def _task_button_updates(task: dict | None, image_model_label: str = DEFAULT_IMAGE_MODEL) -> tuple:
    active = _task_active(task)
    paused = _task_paused(task)
    kind = task.get("kind") if task else None
    model_spec = image_model_spec(image_model_label)
    generation_available = model_spec.available
    h3_training_available = model_spec.backend == "h3" and model_spec.available

    def pause_button(task_kind: str):
        enabled = active and kind == task_kind
        return gr.update(value="继续" if paused and enabled else "暂停", interactive=enabled)

    return (
        gr.update(interactive=not active and generation_available),
        gr.update(interactive=active and kind == "h3-generate"),
        pause_button("h3-generate"),
        gr.update(interactive=not active and h3_training_available),
        gr.update(interactive=active and kind == "h3-train"),
        pause_button("h3-train"),
        gr.update(interactive=not active),
        gr.update(interactive=active and kind == "train"),
        pause_button("train"),
    )


def poll_task_controls(image_model_label: str = DEFAULT_IMAGE_MODEL) -> tuple:
    task = current_task_snapshot()
    return (*_task_statuses(task), *_task_button_updates(task, image_model_label))


def restore_task_ui() -> tuple:
    task = current_task_snapshot()
    auto_job = gr.update()
    auto_job_state_value = ""
    auto_summary = gr.update()
    auto_command = gr.update()
    h3_dataset = gr.update()
    h3_dataset_state_value = ""
    h3_lora_path = gr.update()
    h3_command = gr.update()
    train_command = gr.update()
    if task:
        if task.get("job_dir"):
            auto_job = gr.update(value=task["job_dir"])
            auto_job_state_value = task["job_dir"]
        if task.get("kind") == "h3-generate":
            auto_summary = gr.update(value=task.get("summary", ""))
            auto_command = gr.update(value=task.get("command", ""))
        if task.get("dataset_dir"):
            h3_dataset = gr.update(value=task["dataset_dir"])
            h3_dataset_state_value = task["dataset_dir"]
        if task.get("output_dir"):
            h3_lora_path = gr.update(value=task["output_dir"])
        if task.get("kind") == "h3-train":
            h3_command = gr.update(value=task.get("command", ""))
        if task.get("kind") == "train":
            train_command = gr.update(value=task.get("command", ""))
    if not auto_job_state_value:
        auto_job_state_value = _latest_generation_job_dir()
        if auto_job_state_value:
            auto_job = gr.update(value=auto_job_state_value)
    artifacts = load_artifact_library()
    return (
        *_task_statuses(task),
        auto_job,
        auto_job_state_value,
        auto_summary,
        auto_command,
        h3_dataset,
        h3_dataset_state_value,
        h3_lora_path,
        h3_command,
        train_command,
        *_task_button_updates(task),
        *artifacts,
    )


def _pump_process_log(proc: subprocess.Popen) -> None:
    if proc.stdout is None:
        return
    try:
        for line in proc.stdout:
            logger.info(line.rstrip("\n"))
    finally:
        proc.stdout.close()
        returncode = proc.wait()
        exit_signal = None
        if returncode < 0:
            exit_signal = _exit_signal_name(returncode)
        with STATE["lock"]:
            if STATE.get("process") is proc:
                task = STATE.get("task") or {}
                task.update(
                    status="finished", exit_code=returncode, exit_signal=exit_signal,
                    finished_at=_now_text(),
                )
                STATE["task"] = task
                STATE["process"] = None
                _write_task_state(task)
                label = _task_label(task)
            else:
                task = {}
                label = "任务"
        if exit_signal:
            logger.warning(f"[进程结束] {label} signal={exit_signal} exit={returncode}")
        else:
            logger.info(f"[进程结束] {label} exit={returncode}")
        output_dir = task.get("output_dir") if isinstance(task, dict) else None
        if output_dir:
            output_root = Path(str(output_dir)).expanduser()
            weights = sorted(output_root.rglob("*.safetensors")) if output_root.is_dir() else []
            if weights:
                logger.info(f"[输出] 已生成 {len(weights)} 个 LoRA 权重：{output_root}")
            elif returncode == 0:
                logger.warning(f"[输出] 训练进程正常结束，但输出目录没有 LoRA 权重：{output_root}")
            else:
                logger.warning(f"[输出] 训练未完成，尚未生成 LoRA 权重：{output_root}")


def start_training(command: str, task_kind: str = "train", context: dict | None = None) -> str:
    if not command.strip():
        raise gr.Error("请先生成命令")
    context = context or {}
    with STATE["lock"]:
        if _task_is_running_locked():
            raise gr.Error("已有任务正在运行或暂停，请先停止或等待结束")
        label = TASK_LABELS.get(task_kind, task_kind)
        logger.info("=" * 72)
        logger.info(f"任务启动 · {label} · {_now_text()}")
        logger.info(f"$ {command}")
        process_env = os.environ.copy()
        process_env.setdefault("PYTHONUNBUFFERED", "1")
        process_env.setdefault("PYTHONFAULTHANDLER", "1")
        proc = subprocess.Popen(
            ["bash", "-c", command],
            cwd=HOME,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
            env=process_env,
        )
        task = {
            "pid": proc.pid,
            "kind": task_kind,
            "label": label,
            "command": command,
            "started_at": _now_text(),
            "finished_at": None,
            "exit_code": None,
            "status": "running",
        }
        task.update(context)
        STATE.update(process=proc, log=None, task=task)
        _write_task_state(task)
        threading.Thread(target=_pump_process_log, args=(proc,), daemon=True).start()
    return f"已启动 · {label} · PID {proc.pid}；日志：{LATEST_LOG}"


def start_general_training(command: str) -> str:
    return start_training(command, "train", {"command": command})


def pause_or_resume_training() -> str:
    with STATE["lock"]:
        task = _reconcile_task_locked()
        if not _task_active(task):
            return "当前没有可暂停或继续的任务"
        pid = int(task["pid"])
        label = _task_label(task)
        if _task_paused(task):
            try:
                os.killpg(pid, signal.SIGCONT)
            except ProcessLookupError:
                return f"PID {pid} 已退出"
            except PermissionError:
                return f"没有权限继续 PID {pid}"
            task.update(status="running", resumed_at=_now_text())
            STATE["task"] = task
            _write_task_state(task)
            logger.info(f"继续任务 · {label} · PID {pid}")
            return f"已继续 · {label} · PID {pid}"
        try:
            os.killpg(pid, signal.SIGSTOP)
        except ProcessLookupError:
            return f"PID {pid} 已退出"
        except PermissionError:
            return f"没有权限暂停 PID {pid}"
        task.update(status="paused", paused_at=_now_text())
        STATE["task"] = task
        _write_task_state(task)
        logger.info(f"暂停任务 · {label} · PID {pid}")
        return f"已暂停 · {label} · PID {pid}；点击“继续”恢复"


def stop_training() -> str:
    with STATE["lock"]:
        task = _reconcile_task_locked()
        proc = STATE.get("process")
        if proc and proc.poll() is None:
            pid = proc.pid
        elif _task_active(task):
            pid = int(task["pid"])
        else:
            return "当前没有运行中的任务"
        label = _task_label(task)
        logger.info(f"发送停止信号 · {label} · PID {pid}")
        try:
            if _task_paused(task):
                os.killpg(pid, signal.SIGCONT)
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return f"PID {pid} 已退出"
        except PermissionError:
            return f"没有权限停止 PID {pid}"
        return f"已向 {label} PID {pid} 发送停止信号"


def read_log(limit: int = 250) -> str:
    path = LATEST_LOG
    task = current_task_snapshot()
    status = _task_status_text(task, task.get("kind", "")) if task else "未启动/历史日志"
    if not path.exists():
        return status
    try:
        limit = max(50, min(int(limit), 5000))
    except (TypeError, ValueError):
        limit = 250
    with LOG_LOCK:
        buffered = list(LOG_BUFFER)
    if len(buffered) >= limit:
        lines = buffered[-limit:]
    else:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
    updated = time.strftime("%H:%M:%S", time.localtime(path.stat().st_mtime))
    return f"{status} · 更新 {updated} · 最近 {len(lines)} 行\n\n" + "\n".join(lines)


def list_lora_outputs(output_dir: str):
    if not output_dir.strip():
        return []
    root = Path(output_dir).expanduser().resolve()
    output_root = OUTPUTS.resolve()
    if not root.is_dir() or (root != output_root and output_root not in root.parents):
        return []
    return [str(path) for path in sorted(root.rglob("*.safetensors"), key=lambda item: item.stat().st_mtime, reverse=True)]


def _job_target_files(job: Path) -> list[Path]:
    paths: list[Path] = []
    manifest_file = job / "generation.json"
    if manifest_file.is_file():
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = []
        for item in manifest:
            path = job / "targets" / str(item.get("output", ""))
            if path.is_file():
                paths.append(path)
    if not paths:
        target_dir = job / "targets"
        if target_dir.is_dir():
            paths = sorted(
                path for path in target_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
            )
    return list(dict.fromkeys(paths))


def _artifact_job_choices(selected=None) -> tuple[list[tuple[str, str]], list[str]]:
    jobs = []
    for job in DATASETS.iterdir():
        if not job.is_dir():
            continue
        targets = _job_target_files(job)
        if not targets:
            continue
        try:
            updated = max(path.stat().st_mtime for path in targets)
        except OSError:
            updated = job.stat().st_mtime
        label = f"{job.name} · {len(targets)} 张 · {time.strftime('%m-%d %H:%M', time.localtime(updated))}"
        jobs.append((updated, label, job))
    jobs.sort(key=lambda item: item[0], reverse=True)
    choices = [(label, str(job)) for _, label, job in jobs]
    valid = {value for _, value in choices}
    current = selected if isinstance(selected, list) else ([selected] if selected else [])
    values = [value for value in current if value in valid]
    if not values and choices:
        values = [choices[0][1]]
    return choices, values


def _selected_artifact_jobs(job_dirs) -> list[Path]:
    values = job_dirs if isinstance(job_dirs, list) else ([job_dirs] if job_dirs else [])
    root = DATASETS.resolve()
    jobs = []
    for value in values:
        path = Path(str(value)).expanduser().resolve()
        if path == root or root not in path.parents or not path.is_dir():
            raise gr.Error(f"成果批次路径无效：{value}")
        jobs.append(path)
    if not jobs:
        raise gr.Error("请先选择至少一个成果批次")
    return jobs


def preview_artifact_jobs(job_dirs):
    jobs = _selected_artifact_jobs(job_dirs)
    gallery = []
    lines = []
    shown = 0
    for job in jobs:
        targets = _job_target_files(job)
        lines.append(f"{job.name}：{len(targets)} 张目标图 · {'已有 metadata' if (job / 'metadata.json').is_file() else '尚未整理训练集'}")
        for path in targets:
            if shown >= 240:
                break
            gallery.append((str(path), f"{job.name} · {path.name}"))
            shown += 1
    if shown >= 240 and sum(len(_job_target_files(job)) for job in jobs) > shown:
        lines.append(f"预览仅显示前 {shown} 张；下载 ZIP 会包含全部文件。")
    return gallery, "\n".join(lines)


def refresh_artifact_library(selected=None):
    choices, values = _artifact_job_choices(selected)
    gallery, status = preview_artifact_jobs(values) if values else ([], "还没有可用的生成批次。")
    return (
        gr.update(choices=choices, value=values, interactive=bool(choices)),
        gallery,
        status,
        list_all_lora_outputs(),
    )


def load_artifact_library():
    choices, values = _artifact_job_choices()
    gallery, status = preview_artifact_jobs(values) if values else ([], "还没有可用的生成批次。")
    return (
        gr.update(choices=choices, value=values, interactive=bool(choices)),
        gallery,
        status,
        list_all_lora_outputs(),
    )


def poll_artifact_job_choices(selected=None):
    choices, values = _artifact_job_choices(selected)
    return gr.update(choices=choices, value=values, interactive=bool(choices))


def load_artifact_preview(job_dirs):
    return preview_artifact_jobs(job_dirs)


def use_artifact_for_review(job_dirs):
    jobs = _selected_artifact_jobs(job_dirs)
    job = jobs[0]
    exclusions = _load_review_exclusions(str(job))
    gallery, review_status = _render_generation_review(str(job), exclusions)
    extra = f"；另有 {len(jobs) - 1} 个批次未载入" if len(jobs) > 1 else ""
    return (
        str(job), gallery, exclusions, None, review_status,
        f"已把 {job.name} 载入“图片 → 2 · 审核与建集”{extra}。",
        str(job),
    )


def use_artifact_as_dataset(job_dirs, trigger_word: str):
    jobs = _selected_artifact_jobs(job_dirs)
    job = jobs[0]
    metadata = job / "metadata.json"
    if metadata.is_file():
        return str(job), f"已载入训练数据集：{job.name}。", str(job)

    trigger = (trigger_word or "ohwx person").strip()
    records, skipped = [], 0
    manifest_file = job / "generation.json"
    if manifest_file.is_file():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        for item in manifest:
            target = job / "targets" / str(item.get("output", ""))
            if not target.is_file():
                skipped += 1
                continue
            record = {"image": f"targets/{target.name}", "prompt": " ".join((trigger, str(item.get("prompt", "")).strip())).strip()}
            refs = [job / str(path) for path in item.get("edit_image", [])]
            if refs and all(path.is_file() for path in refs):
                record["edit_image"] = [str(path.relative_to(job)) for path in refs]
            elif refs:
                skipped += 1
                continue
            records.append(record)
    else:
        for target in _job_target_files(job):
            records.append({"image": f"targets/{target.name}", "prompt": trigger})
    if not records:
        raise gr.Error("这个批次没有可用目标图或参考图，无法创建训练集")
    _validate_h3_training_files(job, records)
    metadata.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    note = f"，跳过 {skipped} 个缺失文件样本" if skipped else ""
    return str(job), f"已为 {job.name} 创建 metadata.json，共 {len(records)} 个样本{note}。", str(job)


def _write_artifact_zip(jobs: list[Path], filename: str) -> tuple[Path, int]:
    export = EXPORTS / filename
    if export.exists():
        export.unlink()
    file_count = 0
    with zipfile.ZipFile(export, "w", compression=zipfile.ZIP_STORED) as archive:
        for job in jobs:
            for path in sorted(job.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(job)
                include = relative.parts[0] in {"targets", "references"} or relative.name in {
                    "generation.json", "metadata.json",
                }
                if not include:
                    continue
                archive.write(path, arcname=f"{job.name}/{relative.as_posix()}")
                file_count += 1
    return export, file_count


def download_artifact_jobs(job_dirs):
    jobs = _selected_artifact_jobs(job_dirs)
    export, count = _write_artifact_zip(jobs, f"h3-materials-{time.strftime('%Y%m%d-%H%M%S')}.zip")
    names = "、".join(job.name for job in jobs)
    return gr.update(value=str(export), interactive=True), f"已打包 {len(jobs)} 个批次、{count} 个文件：{names}"


def list_all_lora_outputs():
    root = OUTPUTS.resolve()
    if not root.is_dir():
        return []
    return [
        str(path) for path in sorted(root.rglob("*.safetensors"), key=lambda item: item.stat().st_mtime, reverse=True)
    ]


def download_all_lora_outputs():
    outputs = [Path(path) for path in list_all_lora_outputs()]
    if not outputs:
        raise gr.Error("还没有可下载的 LoRA 输出")
    root = OUTPUTS.resolve()
    export = EXPORTS / f"h3-lora-outputs-{time.strftime('%Y%m%d-%H%M%S')}.zip"
    with zipfile.ZipFile(export, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in outputs:
            archive.write(path, arcname=f"outputs/{path.relative_to(root).as_posix()}")
    return gr.update(value=str(export), interactive=True), f"已打包 {len(outputs)} 个 LoRA 文件。"


generation_size_choices, generation_size_default, generation_size_guidance = generation_size_options()
initial_artifact_choices, initial_artifact_values = _artifact_job_choices()


def ui_reload_head() -> str:
    build = json.dumps(UI_BUILD_ID)
    build_file = json.dumps(UI_BUILD_FILE.as_posix())
    return f"""
<script>
(() => {{
  const build = {build};
  const buildFile = {build_file};
  const storageKey = "pose-trainer-ui-build";
  const endpoint = new URL(`gradio_api/file=${{encodeURIComponent(buildFile)}}`, document.baseURI).toString();
  let previous = null;
  try {{
    previous = window.sessionStorage.getItem(storageKey);
    window.sessionStorage.setItem(storageKey, build);
  }} catch (error) {{}}
  if (previous && previous !== build) {{
    window.location.reload();
    return;
  }}
  const checkBuild = async () => {{
    try {{
      const response = await fetch(`${{endpoint}}?t=${{Date.now()}}`, {{ cache: "no-store" }});
      if (!response.ok) return;
      const payload = await response.json();
      if (payload.build && payload.build !== build) {{
        try {{ window.sessionStorage.setItem(storageKey, payload.build); }} catch (error) {{}}
        window.location.reload();
      }}
    }} catch (error) {{}}
  }};
  window.addEventListener("load", () => {{
    window.setTimeout(checkBuild, 1200);
    window.setInterval(checkBuild, 8000);
  }});
}})();
</script>
"""


def launch_head_options(target) -> dict:
    script = ui_reload_head()
    if "head" in inspect.signature(target.launch).parameters:
        return {"head": script}
    target.head += script
    return {}


IMAGE_WORKBENCH_CSS = """
.image-workbench-tabs .gr-group .styler {
  --layout-gap: 12px !important;
  --form-gap-width: 12px !important;
}
.workbench-param-section {
  margin-top: 14px !important;
  padding: 0 12px 12px !important;
  border: 1px solid var(--border-color-primary) !important;
  border-left: 3px solid var(--color-accent) !important;
  border-radius: 10px !important;
  background: var(--background-fill-secondary) !important;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06) !important;
  overflow: hidden !important;
}
.workbench-param-section .label-wrap {
  margin: 0 -12px 10px !important;
  padding: 10px 12px !important;
  border-bottom: 1px solid var(--border-color-primary) !important;
  border-radius: 0 !important;
  background: var(--background-fill-primary) !important;
  color: var(--body-text-color) !important;
  font-weight: 600 !important;
}
.workbench-param-section .label-wrap:hover {
  background: var(--background-fill-secondary) !important;
}
.param-select > .container {
  display: flex !important;
  flex-direction: column !important;
  align-items: stretch !important;
  gap: 4px !important;
}
.param-select [data-testid="block-info"] {
  margin-bottom: 0 !important;
  min-width: 0 !important;
  line-height: 1.25 !important;
  white-space: normal !important;
}
.seed-field .md.prose,
.seed-field .md.prose * {
  white-space: nowrap !important;
}
.prompt-template-editor textarea {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;
  font-size: 12.5px !important;
  line-height: 1.5 !important;
}
.prompt-template-help {
  margin-top: 2px !important;
  font-size: 0.85rem !important;
  opacity: 0.72 !important;
}
"""


def build_demo_blocks():
    launch_accepts_css = "css" in inspect.signature(gr.Blocks.launch).parameters
    blocks_kwargs = {
        "title": "ModelScope LoRA Trainer",
        "analytics_enabled": False,
    }
    if not launch_accepts_css:
        blocks_kwargs["css"] = IMAGE_WORKBENCH_CSS
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The 'css' parameter in the Blocks constructor will be removed in Gradio 6.0.*",
            category=DeprecationWarning,
        )
        return gr.Blocks(**blocks_kwargs)


demo = build_demo_blocks()
with demo:
    gr.Markdown("# 魔搭 Notebook · 图片 / 视频 LoRA 快速训练器\n24GB 单卡默认配置；先安装环境，再选择数据与权重。")
    image_model_state = gr.State(DEFAULT_IMAGE_MODEL)
    h3_dataset_state = gr.State("")
    h3_output_state = gr.State(H3_DEFAULT_OUTPUT_NAME)
    h3_profile_state = gr.State(H3_DEFAULT_MODEL_PROFILE)
    h3_repeats_state = gr.State(5)
    h3_epochs_state = gr.State(1)
    h3_rank_state = gr.State(16)
    h3_lr_state = gr.State(0.0001)
    h3_pixels_state = gr.State(589824)
    h3_adapter_state = gr.State("")
    h3_low_vram_state = gr.State(True)
    with gr.Tab("图片"):
        gr.Markdown("## 批量素材 → 审核建集 → LoRA 训练")
        with gr.Accordion("模型选择与后端配置", open=True):
            with gr.Row():
                image_model = gr.Dropdown(
                    list(IMAGE_MODEL_CHOICES),
                    value=DEFAULT_IMAGE_MODEL,
                    label="图片模型",
                    elem_classes=["param-select"],
                )
                with gr.Column(visible=True) as h3_profile_column:
                    h3_profile = gr.Dropdown(
                        list(H3_AVAILABLE_MODEL_PROFILES),
                        value=H3_DEFAULT_MODEL_PROFILE,
                        label="H3 训练模型配置",
                        elem_classes=["param-select"],
                    )
            image_model_status = gr.Markdown(_image_model_status_markdown(DEFAULT_IMAGE_MODEL))
        with gr.Tabs(elem_classes=["image-workbench-tabs"]):
            with gr.Tab("生成"):
                with gr.Group():
                    gr.Markdown("## 批量生成素材")
                    gr.Markdown("### 随机训练参数")
                    with gr.Row():
                        auto_preset = gr.Dropdown(
                            choices=list(GENERATION_BATCH_PRESETS),
                            value=DEFAULT_GENERATION_BATCH_PRESET,
                            label="训练规模",
                            elem_classes=["param-select"],
                        )
                        auto_person_mode = gr.Dropdown(
                            choices=list(PERSON_MODES),
                            value=PERSON_MODE_BATCH,
                            label="图库随机策略",
                            elem_classes=["param-select"],
                        )
                        auto_seed = gr.Number(
                            value=DEFAULT_GENERATION_SEED,
                            precision=0,
                            label="Seed · -1 = 每批随机",
                            elem_classes=["param-select", "seed-field"],
                        )
                    with gr.Row(visible=False) as auto_custom_count_row:
                        auto_pose_count = gr.Number(
                            value=8, minimum=1, maximum=200, precision=0,
                            label="自定义生成数量",
                        )
                    gr.Markdown("### 通用参数")
                    with gr.Row():
                        auto_size = gr.Dropdown(
                            choices=generation_size_choices,
                            value=generation_size_default,
                            label="画面比例 / 尺寸",
                            elem_classes=["param-select"],
                        )
                        auto_creative_style = gr.Dropdown(
                            choices=list(STYLE_OPTIONS),
                            value=STYLE_PRESERVE,
                            label="人物风格",
                            elem_classes=["param-select"],
                        )
                        auto_creative_body = gr.Dropdown(
                            choices=list(BODY_OPTIONS),
                            value=BODY_NATURAL,
                            label="身材选择",
                            elem_classes=["param-select"],
                        )
                    auto_size_refresh = gr.Button("重新检测尺寸")
                    auto_size_info = gr.Textbox(
                        value=generation_size_guidance,
                        show_label=False,
                        lines=2,
                        interactive=False,
                    )
                    with gr.Row(visible=False) as auto_custom_size_row:
                        auto_width = gr.Number(value=768, precision=0, label="自定义宽度")
                        auto_height = gr.Number(value=768, precision=0, label="自定义高度")
                    with gr.Accordion(
                        "Prompt / 提示词",
                        open=True,
                        elem_classes=["workbench-param-section"],
                    ):
                        auto_prompt_mode = gr.Dropdown(
                            choices=list(PROMPT_MODES),
                            value=PROMPT_MODE_DEFAULT,
                            label="模板模式",
                            elem_classes=["param-select"],
                        )
                        with gr.Row():
                            auto_prompt_reset = gr.Button("应用模板")
                        auto_prompt_template = gr.Textbox(
                            value=DEFAULT_PROMPT_TEMPLATE,
                            label="模板内容（可编辑）",
                            lines=8,
                            max_lines=18,
                            elem_classes=["prompt-template-editor"],
                        )
                        gr.Markdown(PROMPT_TEMPLATE_HELP, elem_classes=["prompt-template-help"])
                        auto_instruction = gr.Textbox(
                            value=DEFAULT_GENERATION_INSTRUCTION,
                            label="附加画面要求",
                            placeholder="补充人物气质、妆容、光线或背景要求",
                            lines=4,
                            max_lines=8,
                        )
                    auto_start = gr.Button("一键生成训练图片", variant="primary")
                    with gr.Accordion(
                        "高级设置",
                        open=False,
                        elem_classes=["workbench-param-section"],
                    ):
                        with gr.Row():
                            auto_job_name = gr.Textbox(
                                value=DEFAULT_GENERATION_JOB_NAME,
                                label="批次名称",
                            )
                        gr.Markdown("**自定义素材覆盖**")
                        gr.Markdown("留空即从图库随机。")
                        with gr.Row():
                            auto_people = gr.Files(label="自定义人物（可选）", file_types=["image"])
                            auto_outfits = gr.Files(label="自定义服装（可选）", file_types=["image"])
                            auto_poses = gr.Files(label="自定义 Pose（可选）", file_types=["image"])
                        auto_outfit_mode = gr.Dropdown(
                            choices=list(OUTFIT_MODES),
                            value=OUTFIT_MODE_MULTI_REF,
                            label="多张自定义服装的处理方式",
                            elem_classes=["param-select"],
                        )
                        gr.Markdown(
                            f"**Pose 图库：** `{POSE_GALLERY}`"
                        )
                        h3_pose_sync = gr.Button("同步 / 更新 Pose 图库")
                        h3_pose_sync_result = gr.Textbox(label="Pose 图库同步结果", lines=4)
                        h3_pose_sync.click(sync_pose_gallery, outputs=h3_pose_sync_result)
                        gr.Markdown("**图库筛选：** 留空 = 全库；多个关键词用逗号分隔。")
                        with gr.Row():
                            auto_portrait_keyword = gr.Textbox(
                                label="人物筛选（可多个关键词）",
                                placeholder="短发，长发，正脸参考；留空 = 全库随机",
                            )
                            auto_pose_keyword = gr.Textbox(
                                label="Pose 筛选（可多个关键词）",
                                placeholder="站姿，坐姿，侧面；留空 = 全库随机",
                            )
                            auto_costume_keyword = gr.Textbox(
                                label="服装筛选（可多个关键词）",
                                placeholder="春季，礼服；留空 = 全库随机",
                            )
                        auto_pose_safe = gr.Checkbox(
                            value=True,
                            label="18+ 安全过滤（推荐开启）",
                        )
                    with gr.Accordion(
                        "当前图片后端设置（H3）",
                        open=False,
                        elem_classes=["workbench-param-section"],
                    ):
                        auto_command = gr.Textbox(label="自动命令", lines=4)
                    with gr.Row():
                        auto_stop = gr.Button("停止", interactive=False)
                        auto_pause = gr.Button("暂停", interactive=False)
                        auto_log_refresh = gr.Button("刷新日志")
                    with gr.Accordion(
                        "生成日志与调试",
                        open=False,
                        elem_classes=["workbench-param-section"],
                    ):
                        auto_log = gr.Textbox(
                            label="生成日志（自动刷新）",
                            lines=14,
                            max_lines=24,
                            show_copy_button=True,
                            autoscroll=True,
                        )
                        auto_log_file = gr.DownloadButton("下载完整日志", value=str(LATEST_LOG))

            with gr.Tab("结果") as image_results_tab:
                with gr.Group():
                    auto_selected_target = gr.State(None)
                    auto_job_state = gr.State("")
                    auto_exclusions_state = gr.State("")
                    auto_results_signature = gr.State("")
                    auto_gallery = gr.Gallery(
                        label="批量生成结果",
                        columns=6,
                        rows=2,
                        height="auto",
                        object_fit="cover",
                        allow_preview=True,
                        show_download_button=True,
                    )
                    auto_review_status = gr.Markdown("等待生成结果…")
                    with gr.Row():
                        auto_refresh = gr.Button("手动刷新结果", variant="primary")
                        auto_exclude_selected = gr.Button("排除选中")
                        auto_restore_selected = gr.Button("恢复选中")
                    auto_status = gr.Textbox(label="生成状态", interactive=False)
                    auto_job_dir = gr.Textbox(label="当前批次目录", interactive=False)
                    auto_summary = gr.Textbox(label="批次摘要", lines=3, interactive=False)
                    with gr.Accordion("高级审核", open=False):
                        with gr.Row():
                            auto_apply_exclusions = gr.Button("应用排除序号")
                            auto_clear_exclusions = gr.Button("清空排除")
                        with gr.Row():
                            auto_exclusions = gr.Textbox(
                                label="排除序号（支持范围）",
                                placeholder="例如：2,5-7；点选图片时会自动回填",
                            )
                            auto_refined = gr.Files(
                                label="精修替换图（可选）",
                                file_types=["image"],
                            )
                        auto_trigger = gr.Textbox(value="ohwx person", label="人物触发词（默认即可）")
                    with gr.Row():
                        auto_finalize = gr.Button("仅创建训练集")
                        auto_finalize_train = gr.Button("创建训练集并开始训练", variant="primary")
                    h3_dataset = gr.Textbox(label="训练数据集目录", interactive=False)
                    auto_finalize_summary = gr.Textbox(label="整理结果", lines=5, interactive=False)

            with gr.Tab("LoRA 训练"):
                with gr.Accordion("H3 LoRA 训练", open=True):
                    h3_train_dataset = gr.Textbox(label="当前训练数据集", interactive=False)
                    h3_start = gr.Button("使用推荐配置单独训练", variant="primary")
                    with gr.Accordion("训练参数", open=False):
                        with gr.Row():
                            h3_output = gr.Textbox(value=H3_DEFAULT_OUTPUT_NAME, label="训练输出名称")
                        with gr.Row():
                            h3_rank = gr.Number(value=16, precision=0, label="LoRA Rank")
                            h3_lr = gr.Number(value=0.0001, label="学习率")
                            h3_repeats = gr.Number(value=5, precision=0, label="数据重复")
                            h3_epochs = gr.Number(value=1, precision=0, label="Epoch")
                        with gr.Row():
                            h3_pixels = gr.Number(value=589824, precision=0, label="最大像素", info="24GB 建议 ≤589824；OOM 时降到 262144")
                            h3_low_vram = gr.Checkbox(value=True, label="低显存 CPU Offload")
                        h3_adapter = gr.Textbox(
                            label="训练适配器（可选）",
                            placeholder=".safetensors 路径",
                        )
                        h3_command = gr.Textbox(label="自动命令", lines=7)
                        h3_lora_path = gr.Textbox(label="输出目录")
                    with gr.Row():
                        h3_stop = gr.Button("停止", interactive=False)
                        h3_pause = gr.Button("暂停", interactive=False)
                        h3_log_refresh = gr.Button("刷新日志")
                        h3_outputs_refresh = gr.Button("刷新输出")
                    h3_status = gr.Textbox(label="训练状态", interactive=False)
                    with gr.Row():
                        h3_log_lines = gr.Dropdown(
                            [100, 250, 1000], value=250, label="日志显示行数",
                            elem_classes=["param-select"],
                        )
                        h3_log_file = gr.DownloadButton("下载完整训练日志", value=str(LATEST_LOG))
                    h3_log = gr.Textbox(
                        label="训练日志（2 秒自动刷新）",
                        lines=16,
                        max_lines=26,
                        show_copy_button=True,
                        autoscroll=True,
                    )
                    h3_outputs = gr.Files(label="LoRA 输出文件", interactive=False)

                with gr.Accordion("手动数据（高级）", open=False):
                    gr.Markdown("目标成品图必填；参考图可留空。")
                    with gr.Row():
                        h3_targets = gr.Files(label="目标成品图（必填）", file_types=["image"])
                        h3_people = gr.Files(label="人物参考图（可选）", file_types=["image"])
                    with gr.Row():
                        h3_outfits = gr.Files(label="服装参考图（可选）", file_types=["image"])
                        h3_poses = gr.Files(label="Pose 参考图（可选）", file_types=["image"])
                    h3_pose_paths = gr.Textbox(
                        label="Pose 本地路径（可选）",
                        placeholder="每行或逗号分隔",
                    )
                    with gr.Accordion("数据集描述", open=False):
                        with gr.Row():
                            h3_job = gr.Textbox(value="character-3in1", label="任务名称")
                            h3_trigger = gr.Textbox(value="ohwx person", label="人物触发词")
                        h3_instruction = gr.Textbox(
                            value=(
                                f"{DEFAULT_GENERATION_INSTRUCTION} Preserve identity, facial features, hairstyle "
                                "and skin tone from the person reference when provided. Use the Pose reference only "
                                "for body action, joint placement, limb direction, framing and camera angle, never "
                                "for identity, face, hair, makeup, skin tone, body shape or body proportions. Keep "
                                "natural, anatomically coherent proportions and follow the outfit design shown in "
                                "the outfit reference. Do not add captions, labels, logos, watermarks or other text."
                            ),
                            label="训练指令",
                            lines=5,
                        )
                    h3_prepare = gr.Button("创建手动训练集", variant="primary")
                    h3_dataset_summary = gr.Textbox(label="数据集摘要", lines=6)

        image_model.change(
            update_image_model_ui,
            image_model,
            [
                image_model_state, h3_profile_column, image_model_status,
                auto_start, auto_finalize_train, h3_start,
            ],
        )
        auto_start.click(
            start_h3_generation,
            [
                image_model, auto_preset, auto_people, auto_outfits, auto_poses,
                auto_person_mode, auto_outfit_mode,
                auto_portrait_keyword,
                auto_pose_keyword, auto_costume_keyword, auto_pose_count,
                auto_pose_safe, auto_job_name, auto_instruction, auto_prompt_mode, auto_prompt_template,
                auto_creative_style, auto_creative_body,
                auto_size, auto_width, auto_height, auto_seed,
            ],
            [auto_job_dir, auto_summary, auto_command, auto_status, auto_job_state],
        )
        auto_prompt_mode.change(prompt_template_for_mode, auto_prompt_mode, auto_prompt_template)
        auto_prompt_reset.click(prompt_template_for_mode, auto_prompt_mode, auto_prompt_template)
        auto_size_refresh.click(
            refresh_generation_size_ui,
            auto_size,
            [auto_size, auto_size_info],
        ).then(custom_generation_size_visibility, auto_size, auto_custom_size_row)
        auto_size.change(custom_generation_size_visibility, auto_size, auto_custom_size_row)
        auto_preset.change(custom_generation_count_visibility, auto_preset, auto_custom_count_row)
        auto_stop.click(stop_training, outputs=auto_status)
        auto_pause.click(pause_or_resume_training, outputs=auto_status)
        auto_log_timer = gr.Timer(value=2.0, active=True)
        auto_log_timer.tick(read_log, outputs=auto_log)
        auto_log_refresh.click(read_log, outputs=auto_log)
        auto_refresh.click(
            refresh_generated_targets,
            [auto_job_state, auto_exclusions_state],
            [auto_gallery, auto_selected_target, auto_review_status, auto_exclusions, auto_exclusions_state],
        ).then(
            generation_results_signature,
            auto_job_state,
            auto_results_signature,
        )
        image_results_tab.select(
            refresh_generated_targets,
            [auto_job_state, auto_exclusions_state],
            [auto_gallery, auto_selected_target, auto_review_status, auto_exclusions, auto_exclusions_state],
            show_progress="hidden",
        ).then(
            generation_results_signature,
            auto_job_state,
            auto_results_signature,
        )
        auto_results_timer = gr.Timer(value=4.0, active=True)
        auto_results_timer.tick(
            refresh_generation_results_live,
            [auto_job_state, auto_exclusions_state, auto_results_signature],
            [auto_gallery, auto_review_status, auto_results_signature],
            show_progress="hidden",
        )
        auto_apply_exclusions.click(
            apply_review_exclusions,
            [auto_job_state, auto_exclusions],
            [auto_gallery, auto_selected_target, auto_review_status, auto_exclusions_state],
        )
        auto_exclusions.submit(
            apply_review_exclusions,
            [auto_job_state, auto_exclusions],
            [auto_gallery, auto_selected_target, auto_review_status, auto_exclusions_state],
        )
        auto_clear_exclusions.click(
            clear_review_exclusions,
            auto_job_state,
            [auto_gallery, auto_exclusions, auto_selected_target, auto_review_status, auto_exclusions_state],
        )
        auto_gallery.select(
            select_review_target,
            [auto_job_state, auto_exclusions_state],
            [auto_selected_target, auto_review_status],
        )
        auto_exclude_selected.click(
            exclude_selected_target,
            [auto_job_state, auto_exclusions_state, auto_selected_target],
            [auto_gallery, auto_exclusions, auto_selected_target, auto_review_status, auto_exclusions_state],
        )
        auto_restore_selected.click(
            restore_selected_target,
            [auto_job_state, auto_exclusions_state, auto_selected_target],
            [auto_gallery, auto_exclusions, auto_selected_target, auto_review_status, auto_exclusions_state],
        )
        auto_finalize.click(
            finalize_generated_dataset,
            [auto_job_state, auto_trigger, auto_exclusions_state, auto_refined],
            [h3_dataset, auto_finalize_summary, h3_dataset_state],
        )
        auto_finalize_train.click(
            finalize_and_start_h3_training,
            [
                image_model_state, auto_job_state, auto_trigger, auto_exclusions_state, auto_refined,
                h3_output_state, h3_profile_state, h3_repeats_state, h3_epochs_state,
                h3_rank_state, h3_lr_state, h3_pixels_state, h3_adapter_state, h3_low_vram_state,
            ],
            [h3_dataset, auto_finalize_summary, h3_command, h3_lora_path, h3_status, h3_dataset_state],
        )
        auto_job_dir.change(mirror_text_state, auto_job_dir, auto_job_state)
        auto_exclusions.change(mirror_text_state, auto_exclusions, auto_exclusions_state)
        auto_job_dir.change(suggest_h3_output_name, auto_job_dir, [h3_output, h3_output_state])
        h3_dataset.change(mirror_text_state, h3_dataset, h3_dataset_state)
        h3_dataset.change(lambda value: value, h3_dataset, h3_train_dataset)
        h3_output.change(mirror_text_state, h3_output, h3_output_state)
        h3_profile.change(mirror_text_state, h3_profile, h3_profile_state)
        h3_repeats.change(mirror_text_state, h3_repeats, h3_repeats_state)
        h3_epochs.change(mirror_text_state, h3_epochs, h3_epochs_state)
        h3_rank.change(mirror_text_state, h3_rank, h3_rank_state)
        h3_lr.change(mirror_text_state, h3_lr, h3_lr_state)
        h3_pixels.change(mirror_text_state, h3_pixels, h3_pixels_state)
        h3_adapter.change(mirror_text_state, h3_adapter, h3_adapter_state)
        h3_low_vram.change(mirror_text_state, h3_low_vram, h3_low_vram_state)

        h3_start.click(
            start_h3_lora,
            [
                image_model_state, h3_dataset_state, h3_output_state, h3_profile_state,
                h3_repeats_state, h3_epochs_state,
                h3_rank_state, h3_lr_state, h3_pixels_state, h3_adapter_state, h3_low_vram_state,
            ],
            [h3_command, h3_lora_path, h3_status],
        )
        h3_stop.click(stop_training, outputs=h3_status)
        h3_pause.click(pause_or_resume_training, outputs=h3_status)
        h3_log_timer = gr.Timer(value=2.0, active=True)
        h3_log_timer.tick(read_log, inputs=h3_log_lines, outputs=h3_log)
        h3_log_refresh.click(read_log, inputs=h3_log_lines, outputs=h3_log)
        h3_outputs_refresh.click(list_lora_outputs, h3_lora_path, h3_outputs)
        h3_outputs_timer = gr.Timer(value=5.0, active=True)
        h3_outputs_timer.tick(list_lora_outputs, h3_lora_path, h3_outputs)

        h3_prepare.click(
            prepare_h3_character_dataset,
            [
                h3_targets, h3_people, h3_outfits, h3_poses,
                h3_pose_paths, h3_job, h3_trigger, h3_instruction,
            ],
            [h3_dataset, h3_dataset_summary, h3_dataset_state],
        )
    with gr.Tab("视频"):
        task = gr.Dropdown(["SDXL LoRA", "FLUX.1 LoRA", "Wan 2.1 T2V 1.3B LoRA", "Wan 2.1 T2V 14B LoRA", "Wan 2.2 T2V A14B LoRA"], value="Wan 2.1 T2V 1.3B LoRA", label="任务")
        gr.Markdown(
            f"视频支持：{video_support_info()}。AMD ROCm 自动使用 bf16 + AdamW，关闭 FP8、T5 FP8 和 cuDNN 加速。"
        )
        with gr.Row():
            data_dir = gr.Textbox(label="训练数据目录")
            model_path = gr.Textbox(label="模型或 DiT 权重路径")
            output_name = gr.Textbox(value="my_lora", label="输出名称")
        gr.Markdown("组件路径：FLUX 依次填写 CLIP-L / T5XXL / AE；Wan 依次填写 VAE / UMT5，第三项留空；SDXL 可全部留空。")
        with gr.Row():
            component1 = gr.Textbox(label="组件 1")
            component2 = gr.Textbox(label="组件 2")
            component3 = gr.Textbox(label="组件 3")
        with gr.Row():
            width = gr.Number(value=512, precision=0, label="宽")
            height = gr.Number(value=288, precision=0, label="高")
            frames = gr.Number(value=49, precision=0, label="视频帧数")
            repeats = gr.Number(value=1, precision=0, label="数据重复")
        with gr.Row():
            epochs = gr.Number(value=16, precision=0, label="Epoch")
            rank = gr.Number(value=32, precision=0, label="LoRA Rank")
            lr = gr.Number(value=0.0002, label="学习率")
        extra = gr.Textbox(label="额外参数（高级）", placeholder="例如 --resume /path/to/state")
        generate = gr.Button("生成训练命令")
        command = gr.Textbox(label="命令预览", lines=8)
        generate.click(build_command, [task, data_dir, model_path, component1, component2, component3, output_name, width, height, frames, repeats, epochs, rank, lr, extra], command)
        with gr.Row():
            start = gr.Button("开始训练", variant="primary")
            stop = gr.Button("停止", interactive=False)
            pause = gr.Button("暂停", interactive=False)
            update_log = gr.Button("刷新日志")
        status = gr.Textbox(label="状态")
        log = gr.Textbox(
            label="最新日志（自动刷新）",
            lines=20,
            max_lines=30,
            show_copy_button=True,
            autoscroll=True,
        )
        log_file = gr.DownloadButton("下载完整日志", value=str(LATEST_LOG))
        train_log_timer = gr.Timer(value=2.0, active=True)
        train_log_timer.tick(read_log, outputs=log)
        start.click(start_general_training, command, status)
        stop.click(stop_training, outputs=status)
        pause.click(pause_or_resume_training, outputs=status)
        update_log.click(read_log, outputs=log)

    with gr.Tab("结果"):
        gr.Markdown(
            "所有生成批次（含手动数据）都持久保存在 `TRAINER_HOME/datasets`，刷新页面或重启 WebUI 后仍可查看。"
            "可多选批次后批量下载 ZIP，或直接载入图片页的审核与训练流程。"
        )
        with gr.Row():
            artifact_refresh = gr.Button("刷新成果库", variant="primary")
            artifact_download = gr.DownloadButton("批量下载选中批次 ZIP", value=None)
        artifact_jobs = gr.Dropdown(
            choices=initial_artifact_choices,
            value=initial_artifact_values,
            multiselect=True,
            allow_custom_value=True,
            label="成果批次（可多选）",
        )
        artifact_status = gr.Textbox(label="成果摘要", lines=4, interactive=False)
        artifact_gallery = gr.Gallery(label="成果预览", columns=5, height="auto")
        with gr.Row():
            artifact_load_review = gr.Button("载入到图片审核")
            artifact_trigger = gr.Textbox(value="ohwx person", label="创建训练集时的人物触发词")
            artifact_load_train = gr.Button("作为训练集使用", variant="primary")
        gr.Markdown("### LoRA 训练成品\n`TRAINER_HOME/outputs` 中的 `.safetensors` 会自动收集，可在刷新页面后继续下载。")
        with gr.Row():
            artifact_outputs_refresh = gr.Button("刷新 LoRA 输出")
            artifact_outputs_download = gr.DownloadButton("打包下载全部 LoRA", value=None)
        artifact_outputs = gr.Files(label="历史 LoRA 输出文件", interactive=False)

        artifact_refresh.click(
            refresh_artifact_library,
            artifact_jobs,
            [artifact_jobs, artifact_gallery, artifact_status, artifact_outputs],
        )
        artifact_jobs.change(
            load_artifact_preview,
            artifact_jobs,
            [artifact_gallery, artifact_status],
        )
        artifact_download.click(
            download_artifact_jobs,
            artifact_jobs,
            [artifact_download, artifact_status],
        )
        artifact_load_review.click(
            use_artifact_for_review,
            artifact_jobs,
            [
                auto_job_dir, auto_gallery, auto_exclusions, auto_selected_target,
                auto_review_status, artifact_status, auto_job_state,
            ],
        )
        artifact_load_train.click(
            use_artifact_as_dataset,
            [artifact_jobs, artifact_trigger],
            [h3_dataset, artifact_status, h3_dataset_state],
        )
        artifact_outputs_refresh.click(list_all_lora_outputs, outputs=artifact_outputs)
        artifact_outputs_download.click(
            download_all_lora_outputs,
            outputs=[artifact_outputs_download, artifact_status],
        )
        artifact_jobs_timer = gr.Timer(value=5.0, active=True)
        artifact_jobs_timer.tick(
            poll_artifact_job_choices,
            artifact_jobs,
            artifact_jobs,
        )

    with gr.Tab("环境"):
        gr.Markdown("## 容器、模型与训练环境")
        with gr.Row():
            environment_refresh = gr.Button("刷新容器与运行时检查", variant="primary")
            health_run = gr.Button("检查并准备训练环境")
        environment_overview = gr.Markdown("正在读取当前容器信息…")
        with gr.Row():
            with gr.Column(scale=3):
                environment_compatibility = gr.Markdown("正在计算模型兼容性…")
                environment_install = gr.Markdown("正在检查安装状态…")
            with gr.Column(scale=2):
                gr.Markdown("### 存储上限配置")
                with gr.Row():
                    storage_total = gr.Number(
                        value=ENVIRONMENT_SETTINGS.total_limit_gb,
                        minimum=1,
                        precision=1,
                        label="环境总上限 GB",
                    )
                    storage_model = gr.Number(
                        value=ENVIRONMENT_SETTINGS.model_limit_gb,
                        minimum=1,
                        precision=1,
                        label="模型上限 GB",
                    )
                with gr.Row():
                    storage_warn = gr.Number(
                        value=ENVIRONMENT_SETTINGS.warn_percent,
                        minimum=1,
                        maximum=99,
                        precision=0,
                        label="警告阈值 %",
                    )
                    storage_stop = gr.Number(
                        value=ENVIRONMENT_SETTINGS.stop_percent,
                        minimum=2,
                        maximum=100,
                        precision=0,
                        label="停止阈值 %",
                    )
                storage_reserve = gr.Number(
                    value=ENVIRONMENT_SETTINGS.system_reserve_gb,
                    minimum=0,
                    precision=1,
                    label="磁盘系统安全预留 GB",
                    info="有效环境上限不会占用这部分磁盘空间",
                )
                storage_save = gr.Button("保存存储上限", variant="primary")
                environment_storage = gr.Markdown("正在统计存储占用…")

        health_result = gr.Textbox(
            value="通用训练环境检查与当前后端依赖准备输出会显示在这里。",
            label="训练环境准备结果",
            lines=18,
            interactive=False,
        )
        gr.Markdown(
            f"**宿主基线：** `{BASE_IMAGE}` ；训练后端与 ComfyUI 使用独立持久环境，不修改镜像自带运行时。\\n\\n"
            "**GPU 要求：** 环境检查会按 NVIDIA CUDA / AMD ROCm 与各模型参数逐项匹配；H3 需要 NVIDIA CUDA 12.8+，"
            "SDXL 与 Wan 视频可在匹配的 ROCm 环境运行，Wan ROCm 路径仍属实验支持。\\n\\n"
            f"**固定策略：模型最多 {MODEL_BUDGET_GB:.0f}GB，生成数据与 LoRA 输出至少预留 {DATA_RESERVE_GB:.0f}GB。** "
            "各训练后端会自动复用缓存；达到模型上限后停止新增下载。"
        )
        with gr.Accordion("安装 / 更新运行环境", open=False):
            gr.Markdown("训练后端会在首次运行时按需补齐依赖；只有需要修复或安装其他后端时才使用这里。")
            with gr.Row():
                install = gr.Button("安装 / 更新全部训练后端", variant="primary")
                install_log_refresh = gr.Button("刷新安装日志")
            install_status = gr.Textbox(value="尚未开始", label="安装状态", interactive=False)
            install_log = gr.Textbox(label="安装实时日志（最近 500 行）", lines=16, max_lines=26)
            gr.Markdown(f"完整安装日志：`{INSTALL_LOG}`")
        with gr.Accordion("ComfyUI 运行环境", open=False):
            gr.Markdown(
                "使用独立的 `.venv-comfy`，不会修改 WebUI 或训练环境。NVIDIA 会安装与驱动匹配的 "
                "PyTorch CUDA wheel；AMD ROCm 复用基础容器中的 HIP PyTorch，并在独立环境中安装 ComfyUI 依赖。"
            )
            comfy_install = gr.Button("安装 / 修复 ComfyUI 运行环境", variant="primary")
            comfy_status = gr.Textbox(
                value="尚未检查。安装器会拉取 ComfyUI、安装依赖并执行 PyTorch/GPU 冒烟测试。",
                label="ComfyUI 安装状态与日志",
                lines=14,
                max_lines=24,
                interactive=False,
            )
            comfy_timer = gr.Timer(value=3.0, active=True)
        environment_refresh.click(
            refresh_environment_dashboard,
            outputs=[
                environment_overview,
                environment_compatibility,
                environment_install,
                environment_storage,
            ],
            show_progress="hidden",
        )
        storage_save.click(
            save_environment_storage,
            [storage_total, storage_model, storage_warn, storage_stop, storage_reserve],
            environment_storage,
        )
        demo.load(
            refresh_environment_dashboard,
            outputs=[
                environment_overview,
                environment_compatibility,
                environment_install,
                environment_storage,
            ],
            show_progress="hidden",
        )
        health_timer = gr.Timer(value=3.0, active=True)
        health_run.click(start_training_environment_setup, outputs=health_result, show_progress="hidden")
        health_timer.tick(poll_training_environment_setup, outputs=health_result, show_progress="hidden")
        install.click(install_backends, outputs=[install_status, install_log])
        install_log_refresh.click(read_install_log, outputs=[install_status, install_log])
        comfy_install.click(start_comfy_environment_setup, outputs=comfy_status, show_progress="hidden")
        comfy_timer.tick(poll_comfy_environment_setup, outputs=comfy_status, show_progress="hidden")

    task_state_timer = gr.Timer(value=1.0, active=True)
    task_state_timer.tick(
        poll_task_controls,
        image_model_state,
        outputs=[
            auto_status, h3_status, status,
            auto_start, auto_stop, auto_pause,
            h3_start, h3_stop, h3_pause,
            start, stop, pause,
        ],
    )
    demo.load(
        restore_task_ui,
        outputs=[
            auto_status, h3_status, status,
            auto_job_dir, auto_job_state, auto_summary, auto_command,
            h3_dataset, h3_dataset_state, h3_lora_path, h3_command, command,
            auto_start, auto_stop, auto_pause,
            h3_start, h3_stop, h3_pause,
            start, stop, pause,
            artifact_jobs, artifact_gallery, artifact_status, artifact_outputs,
        ],
    ).then(
        refresh_generated_targets,
        [auto_job_state, auto_exclusions_state],
        [auto_gallery, auto_selected_target, auto_review_status, auto_exclusions, auto_exclusions_state],
    ).then(
        generation_results_signature,
        auto_job_state,
        auto_results_signature,
    )


if __name__ == "__main__":
    share_enabled = os.environ.get("GRADIO_SHARE", "0").lower() in {"1", "true", "yes", "on"}
    root_path = os.environ.get("GRADIO_ROOT_PATH") or None
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": server_port,
        "share": share_enabled,
        "root_path": root_path,
        "allowed_paths": [str(OUTPUTS), str(DATASETS), str(LOGS), str(EXPORTS)],
        "show_error": True,
    }
    if "css" in inspect.signature(demo.launch).parameters:
        launch_kwargs["css"] = IMAGE_WORKBENCH_CSS
    launch_kwargs.update(launch_head_options(demo))
    demo.queue().launch(
        **launch_kwargs,
    )
