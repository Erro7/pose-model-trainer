#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${MODELSCOPE_WORKSPACE:-/mnt/workspace}"
REPO_DIR="${WORKSPACE}/pose-model-trainer"
RECOMMENDED_IMAGE="modelscope:1.40.1-pytorch2.13.0-gpu-py312-cu130-ubuntu22.04"
GITHUB_PROXY="${GITHUB_PROXY:-https://ghfast.top/}"

if [[ -n "${GITHUB_PROXY}" ]]; then
  GITHUB_PROXY="${GITHUB_PROXY%/}/"
  export GITHUB_PROXY
  export GIT_CONFIG_COUNT=1
  export GIT_CONFIG_KEY_0="url.${GITHUB_PROXY}https://github.com/.insteadOf"
  export GIT_CONFIG_VALUE_0="https://github.com/"
  echo "[bootstrap] GitHub proxy: ${GITHUB_PROXY}"
else
  echo "[bootstrap] GitHub proxy: direct"
fi

echo "[preflight] checking Python and NVIDIA runtime"
if ! command -v python3 >/dev/null 2>&1; then
  echo "[preflight] incompatible: python3 is missing"
  echo "[preflight] recommended image: ${RECOMMENDED_IMAGE}"
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[preflight] incompatible: NVIDIA GPU is not visible"
  echo "[preflight] create a GPU Notebook with: ${RECOMMENDED_IMAGE}"
  exit 1
fi

PYTHON_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PYTHON_OK="$(python3 -c 'import sys; print(1 if sys.version_info >= (3, 10) else 0)')"
NVIDIA_OUTPUT="$(nvidia-smi 2>&1 || true)"
CUDA_MAX="$(printf '%s\n' "${NVIDIA_OUTPUT}" | sed -n 's/.*CUDA Version: \([0-9][0-9.]*\).*/\1/p' | head -n1)"
GPU_INFO="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -n1 || true)"
SYSTEM_TORCH="$(python3 -c 'import torch; print(f"{torch.__version__} cuda={torch.version.cuda}")' 2>/dev/null || echo 'not installed')"

if [[ "${PYTHON_OK}" != "1" ]]; then
  echo "[preflight] incompatible: Python ${PYTHON_VERSION}; H3 requires Python 3.10+"
  echo "[preflight] recommended image: ${RECOMMENDED_IMAGE}"
  exit 1
fi
if [[ -z "${CUDA_MAX}" ]]; then
  echo "[preflight] incompatible: cannot determine the driver CUDA capability"
  echo "[preflight] recommended image: ${RECOMMENDED_IMAGE}"
  exit 1
fi

if awk -v version="${CUDA_MAX}" 'BEGIN { exit !(version + 0 >= 13.0) }'; then
  export TRAINER_RUNTIME_PROFILE="cu130"
  PREFLIGHT_RESULT="recommended"
elif awk -v version="${CUDA_MAX}" 'BEGIN { exit !(version + 0 >= 12.8) }'; then
  export TRAINER_RUNTIME_PROFILE="cu128"
  PREFLIGHT_RESULT="compatible fallback"
else
  echo "[preflight] incompatible: driver supports CUDA ${CUDA_MAX}; H3 EntroPack needs CUDA 12.8+"
  echo "[preflight] recommended image: ${RECOMMENDED_IMAGE}"
  exit 1
fi

if [[ -z "${TRAINER_BASE_IMAGE:-}" ]]; then
  if [[ "${TRAINER_RUNTIME_PROFILE}" == "cu130" && "${PYTHON_VERSION}" == "3.12" ]]; then
    export TRAINER_BASE_IMAGE="${RECOMMENDED_IMAGE}"
  else
    export TRAINER_BASE_IMAGE="detected:py${PYTHON_VERSION}-driver-cuda${CUDA_MAX}"
  fi
fi

echo "[preflight] result: ${PREFLIGHT_RESULT}"
echo "[preflight] Python: ${PYTHON_VERSION}"
echo "[preflight] GPU: ${GPU_INFO:-unknown}"
echo "[preflight] driver CUDA: ${CUDA_MAX}"
echo "[preflight] system torch: ${SYSTEM_TORCH}"
echo "[preflight] selected H3 profile: ${TRAINER_RUNTIME_PROFILE}"
if [[ "${PREFLIGHT_RESULT}" != "recommended" || "${PYTHON_VERSION}" != "3.12" ]]; then
  echo "[preflight] continuing with an isolated compatible runtime"
  echo "[preflight] recommended for future instances: ${RECOMMENDED_IMAGE}"
fi

mkdir -p "${WORKSPACE}"
if [[ -d "${REPO_DIR}/.git" ]]; then
  git -C "${REPO_DIR}" pull --ff-only
else
  git clone --depth 1 https://github.com/Erro7/pose-model-trainer.git "${REPO_DIR}"
fi

export TRAINER_HOME="${TRAINER_HOME:-${WORKSPACE}/pose-model-trainer-data}"
export TRAINER_TOTAL_BUDGET_GB="${TRAINER_TOTAL_BUDGET_GB:-100}"
export TRAINER_MODEL_BUDGET_GB="${TRAINER_MODEL_BUDGET_GB:-80}"
export GRADIO_SHARE="${GRADIO_SHARE:-0}"
unset GRADIO_ROOT_PATH

if pgrep -f "${REPO_DIR}/app.py" >/dev/null 2>&1; then
  echo "[bootstrap] stopping previous trainer process"
  pkill -TERM -f "${REPO_DIR}/app.py" || true
  for _ in {1..20}; do
    pgrep -f "${REPO_DIR}/app.py" >/dev/null 2>&1 || break
    sleep 0.25
  done
fi

exec bash "${REPO_DIR}/start.sh"
