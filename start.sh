#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UI_ENV_DIR="${ROOT_DIR}/.venv"
TASK_ENV_DIR="${ROOT_DIR}/.venv-h3"

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
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if (( CLEAN )); then
  bash "${ROOT_DIR}/cleanup.sh" --yes --stop-processes --legacy-cache
elif (( FORCE )); then
  bash "${ROOT_DIR}/cleanup.sh" --yes --stop-processes
fi

export TRAINER_HOME="${TRAINER_HOME:-/mnt/workspace/pose-model-trainer-data}"
export TRAINER_TASK_PYTHON="${TASK_ENV_DIR}/bin/python"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${TRAINER_HOME}/models/.modelscope-cache}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${TRAINER_HOME}/models}"
export GRADIO_SHARE="${GRADIO_SHARE:-0}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.aliyun.com}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
unset GRADIO_ROOT_PATH

echo "[bootstrap] PyPI source: ${PIP_INDEX_URL}"

RECOMMENDED_IMAGE="modelscope:1.40.1-pytorch2.13.0-gpu-py312-cu130-ubuntu22.04"
if [[ -z "${TRAINER_RUNTIME_PROFILE:-}" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[preflight] incompatible: NVIDIA GPU is not visible"
    echo "[preflight] recommended image: ${RECOMMENDED_IMAGE}"
    exit 1
  fi
  CUDA_MAX="$(nvidia-smi 2>&1 | sed -n 's/.*CUDA Version: \([0-9][0-9.]*\).*/\1/p' | head -n1)"
  if [[ -z "${CUDA_MAX}" ]]; then
    echo "[preflight] incompatible: cannot determine the driver CUDA capability"
    echo "[preflight] recommended image: ${RECOMMENDED_IMAGE}"
    exit 1
  fi
  if awk -v version="${CUDA_MAX}" 'BEGIN { exit !(version + 0 >= 13.0) }'; then
    export TRAINER_RUNTIME_PROFILE="cu130"
  elif awk -v version="${CUDA_MAX}" 'BEGIN { exit !(version + 0 >= 12.8) }'; then
    export TRAINER_RUNTIME_PROFILE="cu128"
  else
    echo "[preflight] incompatible: driver supports CUDA ${CUDA_MAX}; H3 EntroPack needs CUDA 12.8+"
    echo "[preflight] recommended image: ${RECOMMENDED_IMAGE}"
    exit 1
  fi
fi
if [[ -z "${TRAINER_BASE_IMAGE:-}" ]]; then
  if [[ "${TRAINER_RUNTIME_PROFILE}" == "cu130" ]]; then
    export TRAINER_BASE_IMAGE="${RECOMMENDED_IMAGE}"
  else
    export TRAINER_BASE_IMAGE="detected:${TRAINER_RUNTIME_PROFILE}"
  fi
fi
echo "[preflight] selected H3 profile: ${TRAINER_RUNTIME_PROFILE}"

HOST_PYTHON_TAG="$(python3 -c 'import sys; print(f"py{sys.version_info.major}{sys.version_info.minor}")')"
REQUIREMENTS_HASH="$(sha256sum "${ROOT_DIR}/requirements.txt" | awk '{print $1}')"

UI_MARKER="${UI_ENV_DIR}/.pose-trainer-ui-v5-${HOST_PYTHON_TAG}"
VENV_PYTHON_TAG="$("${UI_ENV_DIR}/bin/python" -c 'import sys; print(f"py{sys.version_info.major}{sys.version_info.minor}")' 2>/dev/null || true)"
if [[ ! -x "${UI_ENV_DIR}/bin/python" ]] || [[ "${VENV_PYTHON_TAG}" != "${HOST_PYTHON_TAG}" ]] || grep -Eq '^include-system-site-packages[[:space:]]*=[[:space:]]*true' "${UI_ENV_DIR}/pyvenv.cfg" 2>/dev/null; then
  echo "[bootstrap] rebuilding isolated WebUI environment for ${HOST_PYTHON_TAG}"
  rm -rf -- "${UI_ENV_DIR}"
fi
if [[ ! -x "${UI_ENV_DIR}/bin/python" ]]; then
  python3 -m venv "${UI_ENV_DIR}"
fi

ui_deps_ok() {
  "${UI_ENV_DIR}/bin/python" -c 'import gradio, numpy, psutil, toml' >/dev/null 2>&1
}

UI_INSTALLED_HASH="$(cat "${UI_MARKER}" 2>/dev/null || true)"
if [[ "${UI_INSTALLED_HASH}" != "${REQUIREMENTS_HASH}" ]]; then
  if [[ -z "${UI_INSTALLED_HASH}" ]] && ui_deps_ok; then
    echo "[bootstrap] reusing existing WebUI dependencies"
  else
    "${UI_ENV_DIR}/bin/python" -m pip install -U pip wheel setuptools
    "${UI_ENV_DIR}/bin/python" -m pip install --upgrade-strategy only-if-needed -r "${ROOT_DIR}/requirements.txt"
  fi
  printf '%s\n' "${REQUIREMENTS_HASH}" > "${UI_MARKER}"
elif ! ui_deps_ok; then
  "${UI_ENV_DIR}/bin/python" -m pip install -U pip wheel setuptools
  "${UI_ENV_DIR}/bin/python" -m pip install --upgrade-strategy only-if-needed -r "${ROOT_DIR}/requirements.txt"
fi

TASK_MARKER="${TASK_ENV_DIR}/.pose-trainer-task-v1-${HOST_PYTHON_TAG}"
if [[ ! -x "${TASK_ENV_DIR}/bin/python" ]] || [[ ! -f "${TASK_MARKER}" ]] || grep -Eq '^include-system-site-packages[[:space:]]*=[[:space:]]*true' "${TASK_ENV_DIR}/pyvenv.cfg" 2>/dev/null; then
  echo "[bootstrap] rebuilding isolated H3 task environment for ${HOST_PYTHON_TAG}"
  rm -rf -- "${TASK_ENV_DIR}"
  python3 -m venv "${TASK_ENV_DIR}"
  "${TASK_ENV_DIR}/bin/python" -m pip install -U pip wheel setuptools
  touch "${TASK_MARKER}"
fi

export PATH="${UI_ENV_DIR}/bin:${PATH}"
exec "${UI_ENV_DIR}/bin/python" "${ROOT_DIR}/app.py"
