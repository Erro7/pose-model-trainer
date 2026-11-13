from __future__ import annotations

import json
import os
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
from pathlib import Path

import gradio as gr
import psutil
import toml


ROOT = Path(__file__).resolve().parent
HOME = Path(os.environ.get("TRAINER_HOME", Path.home() / "modelscope-trainer-data")).expanduser().resolve()
REPOS = HOME / "repos"
MODELS = HOME / "models"
CONFIGS = HOME / "configs"
OUTPUTS = HOME / "outputs"
LOGS = HOME / "logs"
DATASETS = HOME / "datasets"
TOTAL_BUDGET_GB = float(os.environ.get("TRAINER_TOTAL_BUDGET_GB", "100"))
MODEL_BUDGET_GB = float(os.environ.get("TRAINER_MODEL_BUDGET_GB", "80"))
DATA_RESERVE_GB = max(0.0, TOTAL_BUDGET_GB - MODEL_BUDGET_GB)
BASE_IMAGE = os.environ.get(
    "TRAINER_BASE_IMAGE",
    "modelscope:1.40.1-pytorch2.13.0-gpu-py312-cu130-ubuntu22.04",
)
RUNTIME_PROFILE = os.environ.get("TRAINER_RUNTIME_PROFILE", "cu130")
for folder in (REPOS, MODELS, CONFIGS, OUTPUTS, LOGS, DATASETS):
    folder.mkdir(parents=True, exist_ok=True)

STATE = {"process": None, "log": None, "lock": threading.Lock()}
INSTALL_LOG = LOGS / "install-backends.log"
INSTALL_STATE = {"running": False, "step": "尚未开始", "returncode": None, "lock": threading.Lock()}


def q(value: object) -> str:
    return shlex.quote(str(value))


def task_python_path() -> Path:
    configured = os.environ.get("TRAINER_TASK_PYTHON", "").strip()
    candidate = Path(configured) if configured else ROOT / ".venv-h3/bin/python"
    return candidate if candidate.is_file() else Path(sys.executable)


def py() -> str:
    return q(task_python_path())


def accelerate() -> str:
    return f"{py()} -m accelerate.commands.launch"


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
    model_bytes = _tree_bytes([HOME / "modelscope-cache", MODELS])
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
    gpu = subprocess.run(
        ["bash", "-lc", "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true"],
        capture_output=True,
        text=True,
    ).stdout.strip() or "未检测到 NVIDIA GPU"
    disk = psutil.disk_usage(str(HOME))
    venv = "是" if sys.prefix != sys.base_prefix else "否"
    return f"宿主镜像: {BASE_IMAGE}\nH3 运行配置: {RUNTIME_PROFILE}\nWebUI Python: {sys.executable}\nH3 任务 Python: {task_python_path()}\nWebUI 虚拟环境: {venv}\nGPU: {gpu}\nRAM: {psutil.virtual_memory().total / 2**30:.1f} GiB\n可用磁盘: {disk.free / 2**30:.1f} GiB\n工作目录: {HOME}"


def run_visible(command: str) -> str:
    proc = subprocess.run(["bash", "-lc", command], cwd=HOME, text=True, capture_output=True)
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
        ("7/7 修复并验证运行依赖", f"{py()} -m pip install --progress-bar off 'numpy>=1.26,<2' 'opencv-python<4.12' bitsandbytes accelerate && {py()} -c \"import numpy, pandas, pyarrow, modelscope; print('dependency check OK: python=' + __import__('sys').executable + ' numpy=' + numpy.__version__)\""),
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
                    ["bash", "-lc", command],
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
    candidates = [
        Path(configured).expanduser() if configured else None,
        REPOS / "ai-pose-gallery-zh",
        HOME / "ai-pose-gallery-zh",
        ROOT.parent / "ai-pose-gallery-zh",
        Path("/mnt/workspace/ai-pose-gallery-zh"),
        Path("/mnt/workspace/pose-model-trainer-data/repos/ai-pose-gallery-zh"),
        REPOS / "ai-pose-gallery-zh" / "ai-pose-gallery-zh",
    ]
    checked = []
    for candidate in candidates:
        if candidate is None:
            continue
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.append(candidate)
        if (candidate / "data/pose-zh.json").is_file():
            return candidate
    if require_data:
        locations = "\n".join(f"- {path / 'data/pose-zh.json'}" for path in checked)
        raise gr.Error(
            "未找到 Pose 数据文件。程序已检查：\n"
            f"{locations}\n"
            "如果仓库位于其他目录，请在启动前设置："
            "export POSE_GALLERY_DIR=/你的/ai-pose-gallery-zh"
        )
    return (REPOS / "ai-pose-gallery-zh").resolve()

def _has_cached_file(filename: str) -> bool:
    cache = HOME / "modelscope-cache"
    if not cache.is_dir():
        return False
    try:
        return next(cache.rglob(filename), None) is not None
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
        return "正在补齐 H3、EntroPack 与加速依赖…"
    if "installing cu" in joined:
        return "正在安装匹配 GPU 驱动的 PyTorch…"
    profiles = [line for line in lines if "[runtime] applying profile:" in line]
    if profiles:
        return f"正在应用运行配置：{profiles[-1].rsplit(':', 1)[-1].strip()}…"
    return "正在检查并准备 H3 运行环境…"


def start_h3_environment_setup() -> str:
    interpreter = task_python_path()
    script = ROOT / "ensure_h3_runtime.py"
    if not interpreter.is_file():
        return "[失败] 找不到 H3 任务环境。请重新运行一键启动脚本。"
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
        f"后台环境任务已启动（PID {process.pid}）。\n"
        "页面每 3 秒自动更新；DSW 重连或刷新页面不会中断安装。"
    )


def poll_h3_environment_setup():
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
    result = f"[失败] H3 环境准备未完成。\n\n{details}"
    ENV_SETUP_RESULT.write_text(result, encoding="utf-8")
    ENV_SETUP_PID.unlink(missing_ok=True)
    return result


def health_check() -> str:
    passed = "[通过]"
    missing = "[缺失]"
    warning = "[待完成]"
    blocked = False
    lines = ["环境完整性"]

    venv_ok = sys.prefix != sys.base_prefix and Path(sys.executable).parent == ROOT / ".venv/bin"
    task_venv_ok = task_python_path().parent == ROOT / ".venv-h3/bin"
    lines.append(f"{passed if venv_ok else missing} WebUI 环境：{sys.executable}")
    lines.append(f"{passed if task_venv_ok else missing} H3 任务环境：{task_python_path()}")
    blocked = blocked or not venv_ok or not task_venv_ok

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
    model_gb = _tree_bytes([HOME / "modelscope-cache", MODELS]) / gib
    total_gb = _tree_bytes([HOME]) / gib
    storage_ok = model_gb < MODEL_BUDGET_GB and total_gb < TOTAL_BUDGET_GB
    lines.append(
        f"{passed if storage_ok else missing} 存储：模型 {model_gb:.1f}/{MODEL_BUDGET_GB:.0f}GB，"
        f"总计 {total_gb:.1f}/{TOTAL_BUDGET_GB:.0f}GB"
    )
    blocked = blocked or not storage_ok

    candidates = len(list(DATASETS.glob("*/targets/*.png")))
    finalized = len(list(DATASETS.glob("*/metadata.json")))
    loras = len(list(OUTPUTS.rglob("*.safetensors")))
    lines.extend([
        "",
        "流程进度",
        f"{passed if candidates else warning} 目标图：{candidates} 张",
        f"{passed if finalized else warning} 已确认训练集：{finalized} 个",
        f"{passed if loras else warning} LoRA 输出：{loras} 个",
    ])

    if blocked:
        next_step = "先修复上方标记为“缺失”的环境项目。"
    elif not candidates:
        next_step = "环境可用。下一步：H3 工作流 → 1 · 生成目标图。"
    elif not finalized:
        next_step = "下一步：H3 工作流 → 2 · 筛选整理。"
    elif not loras:
        next_step = "下一步：H3 工作流 → 3 · 训练 LoRA。"
    else:
        next_step = "环境和完整流程均已就绪，可继续生成或训练新批次。"
    lines.extend(["", f"下一步：{next_step}"])
    return "\n".join(lines)


def sync_pose_gallery() -> str:
    gallery = pose_gallery_root()
    if (gallery / ".git").is_dir():
        result = run_visible(f"git -C {q(gallery)} pull --ff-only")
    else:
        result = run_visible(f"git clone --depth 1 https://github.com/anhao/ai-pose-gallery-zh.git {q(gallery)}")
    return f"资源目录：{gallery}\n{result}"


def prepare_pose_dataset(keyword: str, max_items: int, safe_only: bool) -> tuple[str, str]:
    gallery = pose_gallery_root(require_data=True)
    index = gallery / "data/pose-zh.json"
    records = json.loads(index.read_text(encoding="utf-8"))
    needle = keyword.strip().lower()
    selected = []
    for item in records:
        searchable = " ".join(str(item.get(k, "")) for k in ("id", "体系", "分类", "姿势名称", "角度", "检索"))
        if needle and needle not in searchable.lower():
            continue
        if safe_only and any(word in searchable for word in ("性感", "双人缠绵")):
            continue
        source = gallery / item.get("原图", "")
        if source.is_file():
            selected.append((item, source))
        if len(selected) >= int(max_items):
            break
    if not selected:
        raise gr.Error("没有找到符合条件且已下载的姿势图片")
    target = DATASETS / "pose_gallery"
    target.mkdir(parents=True, exist_ok=True)
    for old in target.iterdir():
        if old.is_symlink() or old.is_file():
            old.unlink()
    for item, source in selected:
        stem = item["id"]
        image_target = target / f"{stem}{source.suffix.lower()}"
        image_target.symlink_to(source.resolve())
        caption = f"{item.get('姿势名称', '')}, {item.get('分类', '')}, {item.get('角度', '')}"
        (target / f"{stem}.txt").write_text(caption.strip(", "), encoding="utf-8")
    sample = "\n".join(f"{x[0]['id']} · {x[0].get('姿势名称', '')} · {x[0].get('角度', '')}" for x in selected[:20])
    return str(target), f"已生成 {len(selected)} 组图片/描述文件。\n\n前 20 条：\n{sample}"


def _upload_paths(items) -> list[Path]:
    paths = []
    for item in items or []:
        path = Path(str(getattr(item, "name", item))).expanduser().resolve()
        if path.is_file():
            paths.append(path)
    return paths


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
) -> tuple[str, str]:
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
    if not people_files:
        raise gr.Error("至少需要一张人物参考图；人物参考支持多张")
    if not outfit_files:
        raise gr.Error("至少需要一张服装参考图")
    if not pose_files:
        raise gr.Error("至少需要一张 Pose 参考图或有效的本地 Pose 路径")
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", job_name.strip()).strip("-.") or "h3-character"
    root = _new_dataset_root(safe_name)
    image_dir, ref_dir = root / "targets", root / "references"
    image_dir.mkdir()
    ref_dir.mkdir()
    reference_names = []
    for prefix, files in (("person", people_files), ("outfit", outfit_files), ("pose", pose_files)):
        for index, source in enumerate(files, 1):
            reference_names.append(f"references/{_copy_named(source, ref_dir, prefix, index)}")
    person_end = len(people_files)
    outfit_end = person_end + len(outfit_files)
    role_hint = (
        f"Pictures 1-{person_end} are identity references; "
        f"Pictures {person_end + 1}-{outfit_end} are outfit references; "
        f"Pictures {outfit_end + 1}-{len(reference_names)} are pose references."
    )
    prompt = " ".join(x for x in (trigger_word.strip(), role_hint, instruction.strip()) if x)
    records = []
    for index, source in enumerate(target_files, 1):
        target_name = f"targets/{_copy_named(source, image_dir, 'target', index)}"
        records.append({"image": target_name, "prompt": prompt, "edit_image": reference_names})
    metadata = root / "metadata.json"
    metadata.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = (
        f"任务目录：{root}\n目标图：{len(target_files)}\n人物参考：{len(people_files)}\n"
        f"服装参考：{len(outfit_files)}\nPose 参考：{len(pose_files)}\n"
        f"每个样本参考图总数：{len(reference_names)}\nMetadata：{metadata}"
    )
    return str(root), summary


def prepare_h3_generation_job(people, outfits, portrait_keyword, keyword, costume_keyword, max_items, safe_only, job_name, instruction):
    people_files, outfit_files = _upload_paths(people), _upload_paths(outfits)
    gallery, needle = pose_gallery_root(require_data=True), (keyword or "").strip().lower()
    portrait_item = None
    if not people_files:
        portrait_index = gallery / "data/portrait-zh.json"
        if not portrait_index.is_file():
            raise gr.Error("Pose 仓库缺少 data/portrait-zh.json，请重新同步完整图库")
        portrait_needle = (portrait_keyword or "").strip().lower()
        portraits = []
        for item in json.loads(portrait_index.read_text(encoding="utf-8")):
            searchable = " ".join(str(item.get(k, "")) for k in ("id", "分区", "发长", "发型名称", "标题", "检索"))
            if portrait_needle and portrait_needle not in searchable.lower():
                continue
            source = gallery / item.get("原图", "")
            if source.is_file():
                portraits.append((item, source))
        if not portraits:
            raise gr.Error("人像库中没有符合条件且已下载的人物；可清空关键词或重新同步图库")
        portrait_item, portrait_path = random.SystemRandom().choice(portraits)
        people_files = [portrait_path]
    index = gallery / "data/pose-zh.json"
    if not index.is_file():
        raise gr.Error("请先在“姿势图库”页同步 Pose 仓库")
    selected = []
    for item in json.loads(index.read_text(encoding="utf-8")):
        searchable = " ".join(str(item.get(k, "")) for k in ("体系", "分类", "姿势名称", "角度", "检索"))
        if needle and needle not in searchable.lower():
            continue
        if safe_only and any(word in searchable for word in ("性感", "双人缠绵")):
            continue
        pose = gallery / item.get("原图", "")
        if pose.is_file():
            selected.append((item, pose))
        if len(selected) >= int(max_items):
            break
    if not selected:
        raise gr.Error("Pose 仓库中没有符合条件的姿势")
    costumes = []
    if not outfit_files:
        costume_index = gallery / "data/costume-zh.json"
        if not costume_index.is_file():
            raise gr.Error("Pose 仓库缺少 data/costume-zh.json，请重新同步图库")
        costume_needle = (costume_keyword or "").strip().lower()
        for item in json.loads(costume_index.read_text(encoding="utf-8")):
            searchable = " ".join(str(item.get(k, "")) for k in ("分类", "服装名称", "标签", "关键词", "检索"))
            if costume_needle and costume_needle not in searchable.lower():
                continue
            if safe_only and any(word in searchable for word in ("内衣", "性感", "泳装")):
                continue
            source = gallery / item.get("原图", "")
            if source.is_file():
                costumes.append((item, source))
        if not costumes:
            raise gr.Error("服装库中没有符合条件且已下载的服装；可换关键词或重新同步图库")
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", job_name.strip()).strip("-.") or "h3-auto-targets"
    root = _new_dataset_root(safe_name)
    refs = root / "references"
    refs.mkdir()
    person_names = [f"references/{_copy_named(path, refs, 'person', i)}" for i, path in enumerate(people_files, 1)]
    uploaded_outfit_names = [f"references/{_copy_named(path, refs, 'outfit', i)}" for i, path in enumerate(outfit_files, 1)]
    manifest = []
    for i, (pose_item, pose_path) in enumerate(selected, 1):
        if uploaded_outfit_names:
            outfit_names = uploaded_outfit_names
            costume_item = {"id": "uploaded", "服装名称": "the uploaded outfit", "关键词": ""}
        else:
            costume_item, costume_path = costumes[(i - 1) % len(costumes)]
            outfit_names = [f"references/{_copy_named(costume_path, refs, 'costume', i)}"]
        pose_name = f"references/{_copy_named(pose_path, refs, 'pose', i)}"
        all_refs = person_names + outfit_names + [pose_name]
        person_range = f"<Picture 1>" if len(person_names) == 1 else f"<Picture 1>-<Picture {len(person_names)}>"
        outfit_start = len(person_names) + 1
        outfit_range = f"<Picture {outfit_start}>" if len(outfit_names) == 1 else f"<Picture {outfit_start}>-<Picture {outfit_start + len(outfit_names) - 1}>"
        pose_number = len(all_refs)
        pose_text = f"{pose_item.get('体系', '')}, {pose_item.get('分类', '')}, {pose_item.get('姿势名称', '')}, camera angle: {pose_item.get('角度', '')}"
        costume_text = f"{costume_item.get('服装名称', '')}; {costume_item.get('关键词', '')}"
        prompt = (
            f"Preserve the identity, face, hairstyle and body proportions from {person_range}. "
            f"Dress the same person in the clothing from {outfit_range}. "
            f"Match the body pose, limb placement and camera viewpoint in <Picture {pose_number}> exactly. "
            f"Outfit description: {costume_text}. Pose description: {pose_text}. {instruction.strip()}"
        )
        manifest.append({
            "output": f"target_{i:04d}.png", "pose_id": pose_item.get("id", ""),
            "pose_name": pose_item.get("姿势名称", ""), "costume_id": costume_item.get("id", ""),
            "costume_name": costume_item.get("服装名称", ""),
            "portrait_id": portrait_item.get("id", "") if portrait_item else "uploaded",
            "portrait_title": portrait_item.get("标题", "") if portrait_item else "uploaded",
            "prompt": prompt, "edit_image": all_refs,
        })
    (root / "generation.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    person_source = "上传人物" if portrait_item is None else f"仓库随机人物：{portrait_item.get('id', '')} · {portrait_item.get('标题', '')}"
    outfit_source = "上传的服装参考" if outfit_files else "仓库服装库"
    return str(root), f"已构造 {len(manifest)} 个生成任务。\n人物：{person_source}\n服装：{outfit_source}\n同一批次固定同一人物，提示词由人物、服装与 Pose 字段自动组合。"


def build_h3_generation_command(job_dir, width, height, seed):
    job = Path(job_dir).expanduser().resolve()
    if not (job / "generation.json").is_file():
        raise gr.Error("请先构造批量生成任务")
    model_cache = HOME / "modelscope-cache"
    ensure = f"{py()} {q(ROOT / 'ensure_h3_runtime.py')}"
    download = f"{py()} {q(ROOT / 'download_h3.py')} --profile quantized --cache-dir {q(model_cache)}"
    generate = f"{py()} {q(ROOT / 'generate_h3_targets.py')} --job {q(job)} --width {int(width)} --height {int(height)} --seed {int(seed)}"
    return f"export MODELSCOPE_CACHE={q(model_cache)} && {ensure} && {download} && {generate}"


def start_h3_generation(
    people, outfits, portrait_keyword, pose_keyword, costume_keyword,
    max_items, safe_only, job_name, instruction, width, height, seed,
):
    job_dir, summary = prepare_h3_generation_job(
        people, outfits, portrait_keyword, pose_keyword, costume_keyword,
        max_items, safe_only, job_name, instruction,
    )
    command = build_h3_generation_command(job_dir, width, height, seed)
    status = start_training(command)
    return job_dir, summary, command, status


def start_h3_lora(
    dataset_dir, output_name, model_profile, repeats, epochs,
    rank, lr, max_pixels, adapter_path, low_vram,
):
    command, output_dir = build_h3_character_command(
        dataset_dir, output_name, model_profile, repeats, epochs,
        rank, lr, max_pixels, adapter_path, low_vram,
    )
    status = start_training(command)
    return command, output_dir, status


def refresh_generated_targets(job_dir):
    job = Path(job_dir).expanduser().resolve()
    manifest_file = job / "generation.json"
    if not manifest_file.is_file():
        return []
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    return [(str(job / "targets" / item["output"]), f"{item.get('costume_id', '')} · {item['pose_id']} · {item['pose_name']}") for item in manifest if (job / "targets" / item["output"]).is_file()]


def _number_set(raw: str) -> set[int]:
    result = set()
    for part in re.split(r"[,，\s]+", raw or ""):
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            if start.isdigit() and end.isdigit():
                result.update(range(int(start), int(end) + 1))
        elif part.isdigit():
            result.add(int(part))
    return result


def finalize_generated_dataset(job_dir, trigger_word, exclusions, refined_targets):
    job = Path(job_dir).expanduser().resolve()
    manifest_file = job / "generation.json"
    if not manifest_file.is_file():
        raise gr.Error("生成任务不存在")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    refined = _upload_paths(refined_targets)
    for index, source in enumerate(refined, 1):
        if index <= len(manifest):
            target = job / "targets" / manifest[index - 1]["output"]
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(source, target)
    excluded, records = _number_set(exclusions), []
    for index, item in enumerate(manifest, 1):
        target = job / "targets" / item["output"]
        if target.is_file() and index not in excluded:
            records.append({"image": f"targets/{item['output']}", "prompt": " ".join((trigger_word.strip(), item["prompt"])), "edit_image": item["edit_image"]})
    if not records:
        raise gr.Error("还没有可用的生成目标图")
    metadata = job / "metadata.json"
    metadata.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(job), f"已应用 {len(refined)} 张顺序精修图，排除 {len(excluded)} 个序号，保留 {len(records)} 张并重建 metadata.json。"


def build_h3_character_command(dataset_dir, output_name, model_profile, repeats, epochs, rank, lr, max_pixels, adapter_path, low_vram):
    dataset = Path(dataset_dir).expanduser().resolve()
    metadata = dataset / "metadata.json"
    if not metadata.is_file():
        raise gr.Error("找不到 metadata.json，请先生成三合一训练数据集")
    script = REPOS / "DiffSynth-Studio/examples/minimax_h3_image/model_training/train.py"
    if not script.is_file():
        raise gr.Error("尚未安装 DiffSynth-Studio，请先在“环境”页安装训练后端")
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", output_name.strip()).strip("-.") or "h3-character-lora"
    cache_dir = OUTPUTS / f"{name}-split-cache"
    lora_dir = OUTPUTS / name
    model_cache = HOME / "modelscope-cache"
    bf16_models = (
        "MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,"
        "MiniMax/MiniMax-H3-Image:transformer/model*.safetensors,"
        "MiniMax/MiniMax-H3-Image:pdd_fused_heads.safetensors,"
        "MiniMax/MiniMax-H3-Image:vae/diffusion_pytorch_model.safetensors"
    )
    targets = "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2"
    common_tail = (
        f"--data_file_keys image,edit_image --extra_inputs edit_image --max_pixels {int(max_pixels)} "
        f"--learning_rate {float(lr)} --remove_prefix_in_ckpt pipe.dit. --lora_base_model dit "
        f"--lora_target_modules {q(targets)} --lora_rank {int(rank)} --use_gradient_checkpointing"
    )
    if low_vram:
        common_tail += " --use_gradient_checkpointing_offload --cpu_offload --optimize_on_cpu --initialize_model_on_cpu"
    adapter = Path(adapter_path).expanduser() if adapter_path.strip() else None
    if adapter and not adapter.is_file():
        raise gr.Error("训练适配器路径不存在")
    if model_profile == "量化版 <80GB（推荐）":
        quant_models = (
            "DiffSynth-Studio/MiniMax-H3-Image-EntroPack:models/dit_7bit.safetensors,"
            "DiffSynth-Studio/EntroPackPreQuants:models/MiniMax-H3/text_encoder_4bpp.safetensors,"
            "MiniMax/MiniMax-H3-Image:pdd_fused_heads.safetensors,"
            "DiffSynth-Studio/MiniMax-H3-Image-EntroPack:models/vae_int8.safetensors"
        )
        train = (
            f"{accelerate()} {q(script)} --dataset_base_path {q(dataset)} "
            f"--dataset_metadata_path {q(metadata)} --dataset_repeat {int(repeats)} "
            f"--num_epochs {int(epochs)} --model_id_with_origin_paths {q(quant_models)} "
            f"{common_tail} --output_path {q(lora_dir)} --dataset_num_workers 4 --find_unused_parameters"
        )
        if adapter:
            train += f" --preset_lora_path {q(adapter)} --preset_lora_model dit"
        ensure = f"{py()} {q(ROOT / 'ensure_h3_runtime.py')}"
        download = f"{py()} {q(ROOT / 'download_h3.py')} --profile quantized --cache-dir {q(model_cache)}"
        command = f"export MODELSCOPE_CACHE={q(model_cache)} && {ensure} && {download} && {train}"
        return command, str(lora_dir)
    common = f"{common_tail} --model_id_with_origin_paths {q(bf16_models)}"
    stage1 = (
        f"{accelerate()} {q(script)} --dataset_base_path {q(dataset)} "
        f"--dataset_metadata_path {q(metadata)} --dataset_repeat 1 --num_epochs 1 {common} "
        f"--output_path {q(cache_dir)} --task sft:data_process "
        f"--offload_models {q('MiniMax/MiniMax-H3-Image:transformer/model*.safetensors,MiniMax/MiniMax-H3-Image:pdd_fused_heads.safetensors')}"
    )
    stage2 = (
        f"{accelerate()} {q(script)} --dataset_base_path {q(cache_dir)} "
        f"--dataset_repeat {int(repeats)} --num_epochs {int(epochs)} {common} "
        f"--output_path {q(lora_dir)} --task sft:train "
        f"--offload_models {q('MiniMax/MiniMax-H3:FL2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3-Image:vae/diffusion_pytorch_model.safetensors')}"
    )
    if adapter:
        stage2 += f" --preset_lora_path {q(adapter)} --preset_lora_model dit"
    ensure = f"{py()} {q(ROOT / 'ensure_h3_runtime.py')}"
    download = f"{py()} {q(ROOT / 'download_h3.py')} --profile bf16 --cache-dir {q(model_cache)}"
    return f"export MODELSCOPE_CACHE={q(model_cache)} && {ensure} && {download} && {stage1} && {stage2}", str(lora_dir)


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
    dataset = write_dataset_config("image" if image else "video", data_dir, width, height, frames, repeats)
    output = OUTPUTS / output_name.strip()
    output.mkdir(parents=True, exist_ok=True)
    common = (
        f"--dataset_config {q(dataset)} --output_dir {q(output)} --output_name {q(output_name.strip())} "
        f"--mixed_precision bf16 --save_precision bf16 --gradient_checkpointing "
        f"--optimizer_type adamw8bit --learning_rate {float(lr)} --network_dim {int(rank)} "
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
        cache_text = f"{py()} {q(base / 'wan_cache_text_encoder_outputs.py')} --dataset_config {q(dataset)} --t5 {q(t5)} --batch_size 1 --fp8_t5"
        train = f"{accelerate()} --num_cpu_threads_per_process 1 --mixed_precision bf16 {q(base / 'wan_train_network.py')} --task {wan_task} --dit {q(model)} --network_module networks.lora_wan {common} --sdpa --fp8_base --blocks_to_swap 20"
        cmd = f"{cache_latents} && {cache_text} && {train}"
    if extra.strip():
        cmd += " " + extra.strip()
    return cmd


def start_training(command: str) -> str:
    if not command.strip():
        raise gr.Error("请先生成命令")
    with STATE["lock"]:
        proc = STATE["process"]
        if proc and proc.poll() is None:
            raise gr.Error("已有训练任务正在运行")
        log_path = LOGS / "latest.log"
        log_file = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(["bash", "-lc", command], cwd=HOME, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
        STATE.update(process=proc, log=log_file)
    return f"已启动，PID {proc.pid}；日志：{log_path}"


def stop_training() -> str:
    with STATE["lock"]:
        proc = STATE["process"]
        if not proc or proc.poll() is not None:
            return "当前没有运行中的任务"
        os.killpg(proc.pid, signal.SIGTERM)
        return f"已向 PID {proc.pid} 发送停止信号"


def read_log() -> str:
    path = LOGS / "latest.log"
    status = "未启动"
    proc = STATE["process"]
    if proc:
        status = "运行中" if proc.poll() is None else f"已结束（退出码 {proc.returncode}）"
    if not path.exists():
        return status
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-250:]
    return status + "\n\n" + "\n".join(lines)


def list_lora_outputs(output_dir: str):
    if not output_dir.strip():
        return []
    root = Path(output_dir).expanduser().resolve()
    output_root = OUTPUTS.resolve()
    if not root.is_dir() or (root != output_root and output_root not in root.parents):
        return []
    return [str(path) for path in sorted(root.rglob("*.safetensors"), key=lambda item: item.stat().st_mtime, reverse=True)]


with gr.Blocks(title="ModelScope LoRA Trainer") as demo:
    gr.Markdown("# 魔搭 Notebook · 图片 / 视频 LoRA 快速训练器\n24GB 单卡默认配置；先安装环境，再选择数据与权重。")
    with gr.Tab("首页"):
        gr.Markdown("## 环境与持久存储")
        health_run = gr.Button("检查并准备完整环境", variant="primary")
        health_result = gr.Textbox(
            value="点击上方按钮；缺少的 H3 依赖会自动安装，完成后显示环境与流程状态。",
            label="检查结果与下一步",
            lines=16,
            interactive=False,
        )
        storage = gr.Textbox(
            value=storage_info(),
            label="100GB 存储预算",
            lines=6,
            interactive=False,
        )
        with gr.Row():
            storage_refresh = gr.Button("刷新存储用量", variant="primary")
            refresh = gr.Button("刷新运行环境")
        info = gr.Textbox(value=system_info(), label="运行环境", lines=7, interactive=False)
        gr.Markdown(
            f"**宿主基线：** `{BASE_IMAGE}` ；H3 使用独立持久环境，不修改镜像自带 torch 2.0/cu117。\\n\\n"
            "**固定策略：模型最多 80GB，生成数据与 LoRA 输出至少预留 20GB。** "
            "H3 会自动复用缓存；达到模型上限后停止新增下载。"
        )
        with gr.Accordion("安装 / 更新运行环境", open=False):
            gr.Markdown("H3 会在首次运行时自动补齐依赖；只有需要修复或安装其他训练后端时才使用这里。")
            with gr.Row():
                install = gr.Button("安装 / 更新全部训练后端", variant="primary")
                install_log_refresh = gr.Button("刷新安装日志")
            install_status = gr.Textbox(value="尚未开始", label="安装状态", interactive=False)
            install_log = gr.Textbox(label="安装实时日志（最近 500 行）", lines=16, max_lines=26)
            gr.Markdown(f"完整安装日志：`{INSTALL_LOG}`")
        health_timer = gr.Timer(value=3.0, active=True)
        health_run.click(start_h3_environment_setup, outputs=health_result, show_progress="hidden")
        health_timer.tick(poll_h3_environment_setup, outputs=health_result, show_progress="hidden")
        storage_refresh.click(storage_info, outputs=storage)
        refresh.click(system_info, outputs=info)
        install.click(install_backends, outputs=[install_status, install_log])
        install_log_refresh.click(read_install_log, outputs=[install_status, install_log])
    with gr.Tab("姿势图库"):
        gr.Markdown("自动检测 `$TRAINER_HOME/repos/ai-pose-gallery-zh`、`/mnt/workspace/ai-pose-gallery-zh` 或 `$POSE_GALLERY_DIR` 指定的完整仓库。默认安全模式会排除标记为性感和双人缠绵的分类。第三方图片仅用于获授权的个人学习与创作；其 MIT 代码许可不覆盖图片资源。")
        sync_pose = gr.Button("同步 / 更新姿势图库", variant="primary")
        sync_result = gr.Textbox(label="同步结果", lines=5)
        sync_pose.click(sync_pose_gallery, outputs=sync_result)
        with gr.Row():
            pose_keyword = gr.Textbox(label="关键词（可留空）", placeholder="例如：站姿、侧面、挥手")
            pose_limit = gr.Number(value=100, minimum=1, maximum=1439, precision=0, label="最多导入")
            pose_safe = gr.Checkbox(value=True, label="安全模式")
        make_pose_data = gr.Button("生成可训练数据集")
        pose_data_path = gr.Textbox(label="数据集路径")
        pose_preview = gr.Textbox(label="生成结果", lines=12)
        make_pose_data.click(prepare_pose_dataset, [pose_keyword, pose_limit, pose_safe], [pose_data_path, pose_preview])
    with gr.Tab("H3 工作流"):
        gr.Markdown(
            "## MiniMax H3 人物 LoRA\n"
            "按 **1 生成目标图 → 2 筛选整理 → 3 训练 LoRA** 操作。"
            "模型按需自动下载，无需单独填写模型名称。量化权重接近 80GB，完整流程建议预留 100–110GB。"
        )
        with gr.Tabs():
            with gr.Tab("1 · 生成目标图"):
                gr.Markdown("默认从完整图库随机选择一位人物，并自动搭配服装与 Pose；同一批次保持人物一致。")
                with gr.Row():
                    auto_portrait_keyword = gr.Textbox(
                        label="人物范围（可选）",
                        placeholder="短发、长发、正脸参考；留空全库随机",
                    )
                    auto_pose_keyword = gr.Textbox(
                        label="Pose 范围（可选）",
                        placeholder="站姿、坐姿、侧面；留空按图库顺序",
                    )
                    auto_costume_keyword = gr.Textbox(
                        label="服装范围（可选）",
                        placeholder="春季、礼服；留空按图库顺序",
                    )
                    auto_pose_count = gr.Number(
                        value=4, minimum=1, maximum=200, precision=0,
                        label="生成数量",
                    )
                with gr.Accordion("自定义参考（可选）", open=False):
                    gr.Markdown("上传后会覆盖对应的图库随机参考。")
                    with gr.Row():
                        auto_people = gr.Files(label="自定义人物", file_types=["image"])
                        auto_outfits = gr.Files(label="自定义服装", file_types=["image"])
                with gr.Accordion("画面与尺寸（高级）", open=False):
                    auto_instruction = gr.Textbox(
                        value="Create a high-quality realistic full-body fashion photograph with coherent anatomy, natural lighting and a clean background.",
                        label="附加画面要求",
                        lines=2,
                    )
                    with gr.Row():
                        auto_width = gr.Number(value=1024, precision=0, label="宽度")
                        auto_height = gr.Number(value=1024, precision=0, label="高度")
                        auto_seed = gr.Number(value=42, precision=0, label="Seed")
                        auto_pose_safe = gr.Checkbox(value=True, label="安全模式")
                        auto_job_name = gr.Textbox(value="h3-auto-targets", label="批次名称")
                    auto_command = gr.Textbox(label="自动命令", lines=4)
                auto_start = gr.Button("生成目标图", variant="primary")
                with gr.Row():
                    auto_stop = gr.Button("停止")
                    auto_log_refresh = gr.Button("刷新日志")
                auto_status = gr.Textbox(label="生成状态", interactive=False)
                auto_job_dir = gr.Textbox(label="当前批次目录", interactive=False)
                auto_summary = gr.Textbox(label="任务摘要", lines=3, interactive=False)
                auto_log = gr.Textbox(label="生成日志", lines=14, max_lines=24)

            with gr.Tab("2 · 筛选整理"):
                gr.Markdown("刷新候选图，排除失败结果；也可以上传精修图，按顺序替换前 N 张。")
                auto_refresh = gr.Button("刷新候选图片", variant="primary")
                auto_gallery = gr.Gallery(
                    label="候选目标图",
                    columns=4,
                    height="auto",
                )
                with gr.Row():
                    auto_exclusions = gr.Textbox(
                        label="排除序号",
                        placeholder="例如 2,5-7",
                    )
                    auto_refined = gr.Files(
                        label="精修替换图（可选）",
                        file_types=["image"],
                    )
                with gr.Row():
                    auto_trigger = gr.Textbox(value="ohwx person", label="人物触发词")
                    auto_finalize = gr.Button("确认并创建训练集", variant="primary")
                h3_dataset = gr.Textbox(label="训练数据集目录", interactive=False)
                auto_finalize_summary = gr.Textbox(label="整理结果", lines=3, interactive=False)

            with gr.Tab("3 · 训练 LoRA"):
                gr.Markdown(
                    "使用第 2 步确认的数据集训练。默认采用量化配置、Rank 16、重复 5、Epoch 1 和低显存卸载。"
                )
                with gr.Row():
                    h3_output = gr.Textbox(value="h3-character-lora", label="输出名称")
                    h3_profile = gr.Dropdown(
                        ["量化版 <80GB（推荐）", "原始 BF16 两阶段（约 130GB）"],
                        value="量化版 <80GB（推荐）",
                        label="模型配置",
                    )
                with gr.Accordion("训练参数（高级）", open=False):
                    with gr.Row():
                        h3_rank = gr.Number(value=16, precision=0, label="LoRA Rank")
                        h3_lr = gr.Number(value=0.0001, label="学习率")
                        h3_repeats = gr.Number(value=5, precision=0, label="数据重复")
                        h3_epochs = gr.Number(value=1, precision=0, label="Epoch")
                    with gr.Row():
                        h3_pixels = gr.Number(value=1048576, precision=0, label="最大像素")
                        h3_low_vram = gr.Checkbox(value=True, label="低显存 CPU Offload")
                    h3_adapter = gr.Textbox(
                        label="训练适配器（可选）",
                        placeholder=".safetensors 路径",
                    )
                    h3_command = gr.Textbox(label="自动命令", lines=7)
                    h3_lora_path = gr.Textbox(label="输出目录")
                h3_start = gr.Button("开始 LoRA 训练", variant="primary")
                with gr.Row():
                    h3_stop = gr.Button("停止")
                    h3_log_refresh = gr.Button("刷新日志")
                    h3_outputs_refresh = gr.Button("刷新输出")
                h3_status = gr.Textbox(label="训练状态", interactive=False)
                h3_log = gr.Textbox(label="训练日志", lines=16, max_lines=26)
                h3_outputs = gr.Files(label="LoRA 输出文件", interactive=False)

            with gr.Tab("4 · 手动数据（高级）"):
                gr.Markdown("仅在已有目标成品图时使用。常规流程无需操作本页。")
                with gr.Row():
                    h3_targets = gr.Files(label="目标成品图", file_types=["image"])
                    h3_people = gr.Files(label="人物参考图", file_types=["image"])
                with gr.Row():
                    h3_outfits = gr.Files(label="服装参考图", file_types=["image"])
                    h3_poses = gr.Files(label="Pose 参考图", file_types=["image"])
                h3_pose_paths = gr.Textbox(
                    label="Pose 本地路径（可选）",
                    placeholder="每行或逗号分隔",
                )
                with gr.Accordion("数据集描述", open=False):
                    with gr.Row():
                        h3_job = gr.Textbox(value="character-3in1", label="任务名称")
                        h3_trigger = gr.Textbox(value="ohwx person", label="人物触发词")
                    h3_instruction = gr.Textbox(
                        value="Create the same person wearing the referenced outfit and following the referenced pose. Preserve identity, facial features and body proportions.",
                        label="训练指令",
                        lines=3,
                    )
                h3_prepare = gr.Button("创建手动训练集", variant="primary")
                h3_dataset_summary = gr.Textbox(label="数据集摘要", lines=6)

        auto_start.click(
            start_h3_generation,
            [
                auto_people, auto_outfits, auto_portrait_keyword,
                auto_pose_keyword, auto_costume_keyword, auto_pose_count,
                auto_pose_safe, auto_job_name, auto_instruction,
                auto_width, auto_height, auto_seed,
            ],
            [auto_job_dir, auto_summary, auto_command, auto_status],
        )
        auto_stop.click(stop_training, outputs=auto_status)
        auto_log_refresh.click(read_log, outputs=auto_log)
        auto_refresh.click(refresh_generated_targets, auto_job_dir, auto_gallery)
        auto_finalize.click(
            finalize_generated_dataset,
            [auto_job_dir, auto_trigger, auto_exclusions, auto_refined],
            [h3_dataset, auto_finalize_summary],
        )

        h3_start.click(
            start_h3_lora,
            [
                h3_dataset, h3_output, h3_profile, h3_repeats, h3_epochs,
                h3_rank, h3_lr, h3_pixels, h3_adapter, h3_low_vram,
            ],
            [h3_command, h3_lora_path, h3_status],
        )
        h3_stop.click(stop_training, outputs=h3_status)
        h3_log_refresh.click(read_log, outputs=h3_log)
        h3_outputs_refresh.click(list_lora_outputs, h3_lora_path, h3_outputs)

        h3_prepare.click(
            prepare_h3_character_dataset,
            [
                h3_targets, h3_people, h3_outfits, h3_poses,
                h3_pose_paths, h3_job, h3_trigger, h3_instruction,
            ],
            [h3_dataset, h3_dataset_summary],
        )
    with gr.Tab("训练"):
        task = gr.Dropdown(["SDXL LoRA", "FLUX.1 LoRA", "Wan 2.1 T2V 1.3B LoRA", "Wan 2.1 T2V 14B LoRA", "Wan 2.2 T2V A14B LoRA"], value="Wan 2.1 T2V 1.3B LoRA", label="任务")
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
            stop = gr.Button("停止")
            update_log = gr.Button("刷新日志")
        status = gr.Textbox(label="状态")
        log = gr.Textbox(label="最新日志", lines=20, max_lines=30)
        start.click(start_training, command, status)
        stop.click(stop_training, outputs=status)
        update_log.click(read_log, outputs=log)


if __name__ == "__main__":
    share_enabled = os.environ.get("GRADIO_SHARE", "0").lower() in {"1", "true", "yes", "on"}
    root_path = os.environ.get("GRADIO_ROOT_PATH") or None
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=share_enabled,
        root_path=root_path,
        allowed_paths=[str(OUTPUTS), str(DATASETS)],
        show_error=True,
    )
