#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINER_HOME="${TRAINER_HOME:-/mnt/workspace/pose-model-trainer-data}"
MODELS_DIR="${DIFFSYNTH_MODEL_BASE_PATH:-${TRAINER_HOME}/models}"
LEGACY_CACHE="${TRAINER_HOME}/modelscope-cache"
LOGS_DIR="${TRAINER_HOME}/logs"

APPLY=0
STOP_PROCESSES=0
CLEAN_LEGACY=0

usage() {
  cat <<EOF
Usage: bash cleanup.sh [--yes] [--stop-processes] [--legacy-cache]

Without --yes this is a dry-run.

  --stop-processes  Stop this trainer's WebUI, install and training processes.
  --legacy-cache    Migrate missing H3 files into ${MODELS_DIR}, then remove the old ${LEGACY_CACHE}.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes) APPLY=1 ;;
    --stop-processes) STOP_PROCESSES=1 ;;
    --legacy-cache) CLEAN_LEGACY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

patterns=(
  "${ROOT_DIR}/app.py"
  "${ROOT_DIR}/ensure_h3_runtime.py"
  "${ROOT_DIR}/download_h3.py"
  "${ROOT_DIR}/generate_h3_targets.py"
  "${ROOT_DIR}/.venv/bin/python -m pip"
  "${ROOT_DIR}/.venv-h3/bin/python -m pip"
  "${TRAINER_HOME}/repos/DiffSynth-Studio/examples/minimax_h3_image/model_training/train.py"
  "${TRAINER_HOME}/repos/sd-scripts/sdxl_train_network.py"
  "${TRAINER_HOME}/repos/sd-scripts/flux_train_network.py"
  "${TRAINER_HOME}/repos/musubi-tuner/src/musubi_tuner/wan_train_network.py"
  "python.*${TRAINER_HOME}/repos/DiffSynth-Studio"
  "python.*${TRAINER_HOME}/repos/sd-scripts"
  "python.*${TRAINER_HOME}/repos/musubi-tuner"
  "git.*${TRAINER_HOME}/repos/DiffSynth-Studio"
  "git.*${TRAINER_HOME}/repos/sd-scripts"
  "git.*${TRAINER_HOME}/repos/musubi-tuner"
)

processes_running() {
  local pattern
  for pattern in "${patterns[@]}"; do
    if pgrep -f -- "$pattern" >/dev/null 2>&1; then
      return 0
    fi
  done
  return 1
}

stop_processes() {
  local pattern
  for pattern in "${patterns[@]}"; do
    if pgrep -f -- "$pattern" >/dev/null 2>&1; then
      if (( APPLY )); then
        echo "[cleanup] stopping: ${pattern}"
        pkill -TERM -f -- "$pattern" || true
      else
        echo "[cleanup] would stop: ${pattern}"
        pgrep -af -- "$pattern" || true
      fi
    fi
  done
  if (( APPLY )); then
    local _
    for _ in {1..40}; do
      processes_running || break
      sleep 0.25
    done
    for pattern in "${patterns[@]}"; do
      pkill -KILL -f -- "$pattern" 2>/dev/null || true
    done
  fi
}

required_models_present() {
  local relative_path
  for relative_path in \
    "DiffSynth-Studio/MiniMax-H3-Image-EntroPack/models/dit_7bit.safetensors" \
    "DiffSynth-Studio/MiniMax-H3-Image-EntroPack/models/vae_int8.safetensors" \
    "DiffSynth-Studio/EntroPackPreQuants/models/MiniMax-H3/text_encoder_4bpp.safetensors" \
    "MiniMax/MiniMax-H3-Image/pdd_fused_heads.safetensors" \
    "MiniMax/MiniMax-H3-Image/processor/tokenizer.json"; do
    if [[ ! -f "${MODELS_DIR}/${relative_path}" ]]; then
      return 1
    fi
  done
  return 0
}

migrate_legacy_file() {
  local src="$1" dst="$2"
  [[ -f "${src}" ]] || return 0
  [[ -e "${dst}" ]] && return 0
  if (( APPLY )); then
    mkdir -p -- "$(dirname -- "${dst}")"
    if ln -- "${src}" "${dst}" 2>/dev/null; then
      echo "[cleanup] linked ${dst}"
    else
      echo "[cleanup] copied ${dst}"
      cp --reflink=auto --sparse=always -- "${src}" "${dst}"
    fi
  else
    echo "[cleanup] would migrate ${dst}"
  fi
}

migrate_legacy_models() {
  local entry target_model legacy_model relative_path src dst
  local -a entries=(
    "DiffSynth-Studio/MiniMax-H3-Image-EntroPack|DiffSynth-Studio--MiniMax-H3-Image-EntroPack|models/dit_7bit.safetensors"
    "DiffSynth-Studio/MiniMax-H3-Image-EntroPack|DiffSynth-Studio--MiniMax-H3-Image-EntroPack|models/dit_7bit.json"
    "DiffSynth-Studio/MiniMax-H3-Image-EntroPack|DiffSynth-Studio--MiniMax-H3-Image-EntroPack|models/vae_int8.safetensors"
    "DiffSynth-Studio/EntroPackPreQuants|DiffSynth-Studio--EntroPackPreQuants|models/MiniMax-H3/text_encoder_4bpp.safetensors"
    "DiffSynth-Studio/EntroPackPreQuants|DiffSynth-Studio--EntroPackPreQuants|models/MiniMax-H3/text_encoder_4bpp.json"
    "MiniMax/MiniMax-H3-Image|MiniMax--MiniMax-H3-Image|pdd_fused_heads.safetensors"
  )
  for entry in "${entries[@]}"; do
    IFS='|' read -r target_model legacy_model relative_path <<< "${entry}"
    src="${LEGACY_CACHE}/models/${legacy_model}/snapshots/master/${relative_path}"
    dst="${MODELS_DIR}/${target_model}/${relative_path}"
    migrate_legacy_file "${src}" "${dst}"
  done

  local src_dir="${LEGACY_CACHE}/models/MiniMax--MiniMax-H3-Image/snapshots/master/processor"
  local dst_dir="${MODELS_DIR}/MiniMax/MiniMax-H3-Image/processor"
  if [[ -d "${src_dir}" ]]; then
    while IFS= read -r -d '' src; do
      relative_path="${src#${src_dir}/}"
      migrate_legacy_file "${src}" "${dst_dir}/${relative_path}"
    done < <(find "${src_dir}" -type f -print0)
  fi
}

if (( STOP_PROCESSES )); then
  stop_processes
fi

if (( CLEAN_LEGACY )); then
  if [[ ! -d "${LEGACY_CACHE}" ]]; then
    echo "[cleanup] legacy cache does not exist: ${LEGACY_CACHE}"
  elif processes_running && (( ! STOP_PROCESSES )); then
    echo "[cleanup] skip legacy cache while trainer processes are running; add --stop-processes"
  else
    if ! required_models_present; then
      if (( APPLY )); then
        echo "[cleanup] migrating required files from ${LEGACY_CACHE}"
        migrate_legacy_models
      else
        echo "[cleanup] would migrate missing H3 files from ${LEGACY_CACHE}"
      fi
    fi
    if required_models_present; then
      legacy_size="$(du -sh "${LEGACY_CACHE}" 2>/dev/null | awk '{print $1}' || true)"
      if (( APPLY )); then
        echo "[cleanup] removing legacy cache ${LEGACY_CACHE} (${legacy_size:-unknown})"
        rm -rf -- "${LEGACY_CACHE}"
      else
        echo "[cleanup] would remove legacy cache ${LEGACY_CACHE} (${legacy_size:-unknown})"
      fi
    elif (( ! APPLY )); then
      echo "[cleanup] dry-run: legacy cache would be removed after migration"
    else
      echo "[cleanup] skip legacy cache: required models are not complete in ${MODELS_DIR}"
    fi
  fi
fi

state_files=(
  "${LOGS_DIR}/h3-environment.pid"
  "${LOGS_DIR}/h3-environment-result.txt"
  "${LOGS_DIR}/h3-runtime.lock"
)

if processes_running && (( ! STOP_PROCESSES )); then
  echo "[cleanup] skip runtime state cleanup while trainer processes are running; add --stop-processes"
else
  for state_file in "${state_files[@]}"; do
    [[ -e "${state_file}" ]] || continue
    if (( APPLY )); then
      echo "[cleanup] removing ${state_file}"
      rm -f -- "${state_file}"
    else
      echo "[cleanup] would remove ${state_file}"
    fi
  done
fi

if (( APPLY )); then
  echo "[cleanup] disk after cleanup:"
  if [[ -d "${TRAINER_HOME}" ]]; then
    df -h "${TRAINER_HOME}" | tail -n1
  else
    echo "${TRAINER_HOME} does not exist"
  fi
else
  echo "[cleanup] dry-run only; add --yes to apply"
fi
