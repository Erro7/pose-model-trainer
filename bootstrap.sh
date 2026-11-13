#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: bash bootstrap.sh [--force] [--clean]"
  exit 0
fi

WORKSPACE="${MODELSCOPE_WORKSPACE:-/mnt/workspace}"
REPO_DIR="${WORKSPACE}/pose-model-trainer"
RECOMMENDED_IMAGE="modelscope:1.40.1-pytorch2.13.0-gpu-py312-cu130-ubuntu22.04"
GITHUB_PROXY="${GITHUB_PROXY:-https://ghfast.top/}"
TRAINER_HOME="${TRAINER_HOME:-${WORKSPACE}/pose-model-trainer-data}"
export TRAINER_HOME
LOG_MODULE=bootstrap
LAUNCHER_LOG="${TRAINER_HOME}/logs/launcher.log"
mkdir -p -- "$(dirname -- "${LAUNCHER_LOG}")"

rotate_launcher_log() {
  local max_bytes=$((20 * 1024 * 1024))
  [[ -f "${LAUNCHER_LOG}" ]] || return 0
  local size
  size="$(stat -c '%s' "${LAUNCHER_LOG}" 2>/dev/null || echo 0)"
  (( size < max_bytes )) && return 0
  rm -f -- "${LAUNCHER_LOG}.3"
  [[ -f "${LAUNCHER_LOG}.2" ]] && mv -f -- "${LAUNCHER_LOG}.2" "${LAUNCHER_LOG}.3"
  [[ -f "${LAUNCHER_LOG}.1" ]] && mv -f -- "${LAUNCHER_LOG}.1" "${LAUNCHER_LOG}.2"
  mv -f -- "${LAUNCHER_LOG}" "${LAUNCHER_LOG}.1"
}

_log() {
  local level="$1"
  shift
  local line
  line="$(date '+%Y-%m-%d %H:%M:%S') | ${level} | [${LOG_MODULE}] $*"
  if [[ "${level}" == "ERROR" ]]; then
    printf '%s\n' "${line}" >&2
  else
    printf '%s\n' "${line}"
  fi
  printf '%s\n' "${line}" >> "${LAUNCHER_LOG}" 2>/dev/null || true
}

log_info() { _log INFO "$@"; }
log_warn() { _log WARNING "$@"; }
log_error() { _log ERROR "$@"; }

run_logged() {
  local label="$1"
  shift
  local temp_file rc=0
  temp_file="$(mktemp)"
  log_info "${label}：开始"
  "$@" >"${temp_file}" 2>&1 || rc=$?
  if (( rc == 0 )); then
    rm -f -- "${temp_file}"
    log_info "${label}：完成"
    return 0
  fi
  while IFS= read -r line; do
    log_warn "${line}"
  done < <(tail -n 80 "${temp_file}" 2>/dev/null || true)
  rm -f -- "${temp_file}"
  log_error "${label}：失败（exit=${rc}）"
  return "${rc}"
}

rotate_launcher_log

if [[ -n "${GITHUB_PROXY}" ]]; then
  GITHUB_PROXY="${GITHUB_PROXY%/}/"
  export GITHUB_PROXY
  export GIT_CONFIG_COUNT=1
  export GIT_CONFIG_KEY_0="url.${GITHUB_PROXY}https://github.com/.insteadOf"
  export GIT_CONFIG_VALUE_0="https://github.com/"
  log_info "GitHub proxy: ${GITHUB_PROXY}"
else
  log_info "GitHub proxy: direct"
fi

log_info "preflight: checking Python and GPU runtime"
if ! command -v python3 >/dev/null 2>&1; then
  log_error "preflight: python3 is missing"
  log_error "preflight: recommended image: ${RECOMMENDED_IMAGE}"
  exit 1
fi

PYTHON_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PYTHON_OK="$(python3 -c 'import sys; print(1 if sys.version_info >= (3, 10) else 0)')"
if [[ "${PYTHON_OK}" != "1" ]]; then
  log_error "preflight: Python ${PYTHON_VERSION}; H3 requires Python 3.10+"
  log_error "preflight: recommended image: ${RECOMMENDED_IMAGE}"
  exit 1
fi

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

if ! command -v nvidia-smi >/dev/null 2>&1; then
  if [[ -n "${ROCM_INFO}" || "${ROCM_IMAGE}" == "1" ]]; then
    log_error "preflight: AMD ROCm runtime detected: ${ROCM_INFO:-ROCm image detected}"
    log_error "preflight: H3 EntroPack requires NVIDIA CUDA 12.8+; AMD ROCm is not supported"
    log_error "preflight: ROCm image ubuntu22.04-rocm7.2.3-py312-torch2.12.0-1.40.1 cannot run H3 EntroPack"
    log_error "preflight: recommended image: ${RECOMMENDED_IMAGE}"
    exit 1
  fi
  log_error "preflight: NVIDIA GPU is not visible and AMD ROCm was not detected"
  log_error "preflight: create an NVIDIA CUDA GPU Notebook with: ${RECOMMENDED_IMAGE}"
  exit 1
fi
if [[ -n "${ROCM_INFO}" ]]; then
  log_warn "preflight: AMD ROCm also detected (${ROCM_INFO}); NVIDIA CUDA will be used"
fi

NVIDIA_OUTPUT="$(nvidia-smi 2>&1 || true)"
CUDA_MAX="$(printf '%s\n' "${NVIDIA_OUTPUT}" | sed -n 's/.*CUDA Version: \([0-9][0-9.]*\).*/\1/p' | head -n1)"
GPU_INFO="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -n1 || true)"
if [[ -z "${CUDA_MAX}" ]]; then
  log_error "preflight: cannot determine the driver CUDA capability"
  log_error "preflight: recommended image: ${RECOMMENDED_IMAGE}"
  exit 1
fi

if awk -v version="${CUDA_MAX}" 'BEGIN { exit !(version + 0 >= 13.0) }'; then
  export TRAINER_RUNTIME_PROFILE="cu130"
  PREFLIGHT_RESULT="recommended"
elif awk -v version="${CUDA_MAX}" 'BEGIN { exit !(version + 0 >= 12.8) }'; then
  export TRAINER_RUNTIME_PROFILE="cu128"
  PREFLIGHT_RESULT="compatible fallback"
else
  log_error "preflight: driver supports CUDA ${CUDA_MAX}; H3 EntroPack needs CUDA 12.8+"
  log_error "preflight: recommended image: ${RECOMMENDED_IMAGE}"
  exit 1
fi

if [[ -z "${TRAINER_BASE_IMAGE:-}" ]]; then
  if [[ "${TRAINER_RUNTIME_PROFILE}" == "cu130" && "${PYTHON_VERSION}" == "3.12" ]]; then
    export TRAINER_BASE_IMAGE="${RECOMMENDED_IMAGE}"
  else
    export TRAINER_BASE_IMAGE="detected:py${PYTHON_VERSION}-driver-cuda${CUDA_MAX}"
  fi
fi

log_info "preflight: result: ${PREFLIGHT_RESULT}"
log_info "preflight: Python: ${PYTHON_VERSION}"
log_info "preflight: GPU: ${GPU_INFO:-unknown}"
log_info "preflight: driver CUDA: ${CUDA_MAX}"
log_info "preflight: system torch: ${SYSTEM_TORCH} (cuda=${SYSTEM_CUDA:-none}, hip=${SYSTEM_HIP:-none})"
log_info "preflight: selected H3 profile: ${TRAINER_RUNTIME_PROFILE}"
if [[ "${PREFLIGHT_RESULT}" != "recommended" || "${PYTHON_VERSION}" != "3.12" ]]; then
  log_warn "preflight: continuing with an isolated compatible runtime"
  log_warn "preflight: recommended for future instances: ${RECOMMENDED_IMAGE}"
fi

mkdir -p "${WORKSPACE}"
if [[ -d "${REPO_DIR}/.git" ]]; then
  run_logged "更新项目仓库" git -C "${REPO_DIR}" pull --ff-only
else
  run_logged "克隆项目仓库" git clone --depth 1 https://github.com/Erro7/pose-model-trainer.git "${REPO_DIR}"
fi

export TRAINER_TOTAL_BUDGET_GB="${TRAINER_TOTAL_BUDGET_GB:-100}"
export TRAINER_MODEL_BUDGET_GB="${TRAINER_MODEL_BUDGET_GB:-80}"
export GRADIO_SHARE="${GRADIO_SHARE:-0}"
unset GRADIO_ROOT_PATH

if pgrep -f "${REPO_DIR}/app.py" >/dev/null 2>&1; then
  log_info "stopping previous trainer process"
  pkill -TERM -f "${REPO_DIR}/app.py" || true
  for _ in {1..20}; do
    pgrep -f "${REPO_DIR}/app.py" >/dev/null 2>&1 || break
    sleep 0.25
  done
fi

log_info "delegating to start.sh"
exec bash "${REPO_DIR}/start.sh" "$@"
