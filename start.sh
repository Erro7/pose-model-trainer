#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UI_ENV_DIR="${ROOT_DIR}/.venv"
TASK_ENV_DIR="${ROOT_DIR}/.venv-h3"
export TRAINER_HOME="${TRAINER_HOME:-/mnt/workspace/pose-model-trainer-data}"
LOG_MODULE=start
source "${ROOT_DIR}/shell_log.sh"

FORCE=0
CLEAN=0
usage() {
  cat <<EOF
Usage: bash start.sh [--force] [--clean]

  --force  Stop this trainer's running WebUI, install and training processes.
  --clean  --force plus migrate the legacy modelscope-cache into models/ and remove it.
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force|-f) FORCE=1 ;;
    --clean) CLEAN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) log_error "unknown option: $1"; usage >&2; exit 2 ;;
  esac
  shift
done

if (( CLEAN )); then
  bash "${ROOT_DIR}/cleanup.sh" --yes --stop-processes --legacy-cache
elif (( FORCE )); then
  bash "${ROOT_DIR}/cleanup.sh" --yes --stop-processes
fi

export TRAINER_TASK_PYTHON="${TASK_ENV_DIR}/bin/python"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${TRAINER_HOME}/models/.modelscope-cache}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${TRAINER_HOME}/models}"
export GRADIO_SHARE="${GRADIO_SHARE:-0}"
export GRADIO_ANALYTICS_ENABLED="False"
export HF_HUB_DISABLE_TELEMETRY="1"
export DO_NOT_TRACK="1"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.aliyun.com}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
unset GRADIO_ROOT_PATH

log_info "PyPI source: ${PIP_INDEX_URL}"

RECOMMENDED_IMAGE="modelscope:1.40.1-pytorch2.13.0-gpu-py312-cu130-ubuntu22.04"

SYSTEM_TORCH_INFO="$(python3 -c 'import torch; print(f"{torch.__version__}|{torch.version.cuda or ""}|{getattr(torch.version, "hip", "") or ""}")' 2>/dev/null || true)"
if [[ -n "${SYSTEM_TORCH_INFO}" ]]; then
  IFS='|' read -r SYSTEM_TORCH SYSTEM_CUDA SYSTEM_HIP <<< "${SYSTEM_TORCH_INFO}"
else
  SYSTEM_TORCH="not installed"
  SYSTEM_CUDA=""
  SYSTEM_HIP=""
fi

ROCM_INFO=""
ROCM_IMAGE=0
if [[ -n "${SYSTEM_HIP}" ]]; then
  ROCM_INFO="torch HIP ${SYSTEM_HIP}"
elif command -v rocm-smi >/dev/null 2>&1 || [[ -d /opt/rocm ]] || [[ -n "${ROCM_PATH:-}" || -n "${HIP_PATH:-}" ]]; then
  ROCM_ROOT="${ROCM_PATH:-${HIP_PATH:-/opt/rocm}}"
  ROCM_VERSION="$(cat "${ROCM_ROOT}/.info/version" 2>/dev/null || true)"
  ROCM_INFO="${ROCM_VERSION:-ROCm installation detected}"
fi
if [[ -n "${TRAINER_BASE_IMAGE:-}" && "${TRAINER_BASE_IMAGE,,}" == *rocm* ]]; then
  ROCM_IMAGE=1
fi

ROCM_ONLY=0
if ! command -v nvidia-smi >/dev/null 2>&1; then
  if [[ -n "${ROCM_INFO}" || "${ROCM_IMAGE}" == "1" ]]; then
    ROCM_ONLY=1
    export TRAINER_GPU_VENDOR="rocm"
    export TRAINER_ROCM_INFO="${ROCM_INFO:-ROCm image detected}"
    export TRAINER_RUNTIME_PROFILE="rocm"
    log_warn "preflight: AMD ROCm runtime detected: ${TRAINER_ROCM_INFO}"
    log_warn "preflight: H3 Image EntroPack and H3 video are disabled on AMD ROCm"
    log_warn "preflight: Wan 2.1 / 2.2 video training is available as an experimental ROCm path"
  else
    log_error "preflight: NVIDIA GPU is not visible and AMD ROCm was not detected"
    log_error "preflight: create an NVIDIA CUDA GPU Notebook with: ${RECOMMENDED_IMAGE}"
    exit 1
  fi
else
  export TRAINER_GPU_VENDOR="nvidia"
  if [[ "${TRAINER_RUNTIME_PROFILE:-}" == "rocm" ]]; then
    unset TRAINER_RUNTIME_PROFILE
  fi
  if [[ -n "${ROCM_INFO}" ]]; then
    log_warn "preflight: AMD ROCm also detected (${ROCM_INFO}); NVIDIA CUDA will be used"
  fi
fi

if [[ "${ROCM_ONLY}" == "0" && -z "${TRAINER_RUNTIME_PROFILE:-}" ]]; then
  CUDA_MAX="$(nvidia-smi 2>&1 | sed -n 's/.*CUDA Version: \([0-9][0-9.]*\).*/\1/p' | head -n1)"
  if [[ -z "${CUDA_MAX}" ]]; then
    log_error "preflight: cannot determine the driver CUDA capability"
    log_error "preflight: recommended image: ${RECOMMENDED_IMAGE}"
    exit 1
  fi
  if awk -v version="${CUDA_MAX}" 'BEGIN { exit !(version + 0 >= 13.0) }'; then
    export TRAINER_RUNTIME_PROFILE="cu130"
  elif awk -v version="${CUDA_MAX}" 'BEGIN { exit !(version + 0 >= 12.8) }'; then
    export TRAINER_RUNTIME_PROFILE="cu128"
  else
    log_error "preflight: driver supports CUDA ${CUDA_MAX}; H3 EntroPack needs CUDA 12.8+"
    log_error "preflight: recommended image: ${RECOMMENDED_IMAGE}"
    exit 1
  fi
fi
if [[ -z "${TRAINER_BASE_IMAGE:-}" ]]; then
  if [[ "${ROCM_ONLY}" == "1" ]]; then
    export TRAINER_BASE_IMAGE="detected:rocm"
  elif [[ "${TRAINER_RUNTIME_PROFILE}" == "cu130" ]]; then
    export TRAINER_BASE_IMAGE="${RECOMMENDED_IMAGE}"
  else
    export TRAINER_BASE_IMAGE="detected:${TRAINER_RUNTIME_PROFILE}"
  fi
fi
log_info "preflight: runtime profile: ${TRAINER_RUNTIME_PROFILE}"
log_info "preflight: GPU vendor: ${TRAINER_GPU_VENDOR}"
log_info "preflight: system torch: ${SYSTEM_TORCH} (cuda=${SYSTEM_CUDA:-none}, hip=${SYSTEM_HIP:-none})"
if [[ "${ROCM_ONLY}" == "1" ]]; then
  log_info "preflight: video backend: Wan 2.1 / 2.2 (ROCm experimental); H3 disabled"
else
  log_info "preflight: video backend: Wan 2.1 / 2.2 + H3 (NVIDIA CUDA)"
fi
if [[ "${TRAINER_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  log_info "preflight: check complete"
  exit 0
fi

HOST_PYTHON_TAG="$(python3 -c 'import sys; print(f"py{sys.version_info.major}{sys.version_info.minor}")')"
REQUIREMENTS_HASH="$(sha256sum "${ROOT_DIR}/requirements.txt" | awk '{print $1}')"

UI_MARKER="${UI_ENV_DIR}/.pose-trainer-ui-v5-${HOST_PYTHON_TAG}"
VENV_PYTHON_TAG="$("${UI_ENV_DIR}/bin/python" -c 'import sys; print(f"py{sys.version_info.major}{sys.version_info.minor}")' 2>/dev/null || true)"
if [[ ! -x "${UI_ENV_DIR}/bin/python" ]] || [[ "${VENV_PYTHON_TAG}" != "${HOST_PYTHON_TAG}" ]] || grep -Eq '^include-system-site-packages[[:space:]]*=[[:space:]]*true' "${UI_ENV_DIR}/pyvenv.cfg" 2>/dev/null; then
  log_info "rebuilding isolated WebUI environment for ${HOST_PYTHON_TAG}"
  rm -rf -- "${UI_ENV_DIR}"
fi
if [[ ! -x "${UI_ENV_DIR}/bin/python" ]]; then
  python3 -m venv "${UI_ENV_DIR}"
fi

ui_deps_ok() {
  "${UI_ENV_DIR}/bin/python" -c 'import gradio, numpy, psutil, toml, loguru' >/dev/null 2>&1
}

UI_INSTALLED_HASH="$(cat "${UI_MARKER}" 2>/dev/null || true)"
if [[ "${UI_INSTALLED_HASH}" != "${REQUIREMENTS_HASH}" ]]; then
  if [[ -z "${UI_INSTALLED_HASH}" ]] && ui_deps_ok; then
    log_info "reusing existing WebUI dependencies"
  else
    run_logged "升级 WebUI pip 工具链" "${UI_ENV_DIR}/bin/python" -m pip install -U pip wheel setuptools
    run_logged "安装 WebUI 依赖" "${UI_ENV_DIR}/bin/python" -m pip install --upgrade-strategy only-if-needed -r "${ROOT_DIR}/requirements.txt"
  fi
  printf '%s\n' "${REQUIREMENTS_HASH}" > "${UI_MARKER}"
elif ! ui_deps_ok; then
  run_logged "升级 WebUI pip 工具链" "${UI_ENV_DIR}/bin/python" -m pip install -U pip wheel setuptools
  run_logged "安装 WebUI 依赖" "${UI_ENV_DIR}/bin/python" -m pip install --upgrade-strategy only-if-needed -r "${ROOT_DIR}/requirements.txt"
fi

TASK_MARKER="${TASK_ENV_DIR}/.pose-trainer-task-v1-${HOST_PYTHON_TAG}"
if [[ ! -x "${TASK_ENV_DIR}/bin/python" ]] || [[ ! -f "${TASK_MARKER}" ]] || grep -Eq '^include-system-site-packages[[:space:]]*=[[:space:]]*true' "${TASK_ENV_DIR}/pyvenv.cfg" 2>/dev/null; then
  log_info "rebuilding isolated H3 task environment for ${HOST_PYTHON_TAG}"
  rm -rf -- "${TASK_ENV_DIR}"
  python3 -m venv "${TASK_ENV_DIR}"
  run_logged "初始化 H3 任务环境" "${TASK_ENV_DIR}/bin/python" -m pip install -U pip wheel setuptools
  touch "${TASK_MARKER}"
fi

export PATH="${UI_ENV_DIR}/bin:${PATH}"
log_info "starting WebUI on port ${GRADIO_SERVER_PORT:-7860}"
exec "${UI_ENV_DIR}/bin/python" "${ROOT_DIR}/app.py"
