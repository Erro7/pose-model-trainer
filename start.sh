#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UI_ENV_DIR="${ROOT_DIR}/.venv"
TASK_ENV_DIR="${ROOT_DIR}/.venv-h3"

export TRAINER_HOME="${TRAINER_HOME:-/mnt/workspace/pose-model-trainer-data}"
export TRAINER_TASK_PYTHON="${TASK_ENV_DIR}/bin/python"
export GRADIO_SHARE="${GRADIO_SHARE:-0}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.cloud.aliyuncs.com/pypi/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.cloud.aliyuncs.com}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
unset GRADIO_ROOT_PATH

echo "[bootstrap] PyPI source: ${PIP_INDEX_URL}"

HOST_PYTHON_TAG="$(python3 -c 'import sys; print(f"py{sys.version_info.major}{sys.version_info.minor}")')"

UI_MARKER="${UI_ENV_DIR}/.pose-trainer-ui-v4-${HOST_PYTHON_TAG}"
if [[ ! -x "${UI_ENV_DIR}/bin/python" ]] || [[ ! -f "${UI_MARKER}" ]] || grep -Eq '^include-system-site-packages[[:space:]]*=[[:space:]]*true' "${UI_ENV_DIR}/pyvenv.cfg" 2>/dev/null; then
  echo "[bootstrap] rebuilding isolated WebUI environment for ${HOST_PYTHON_TAG}"
  rm -rf -- "${UI_ENV_DIR}"
fi
python3 -m venv "${UI_ENV_DIR}"
"${UI_ENV_DIR}/bin/python" -m pip install -U pip wheel setuptools
"${UI_ENV_DIR}/bin/python" -m pip install --upgrade-strategy only-if-needed -r "${ROOT_DIR}/requirements.txt"
touch "${UI_MARKER}"

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
