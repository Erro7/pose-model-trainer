#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINER_HOME="${TRAINER_HOME:-/mnt/workspace/pose-model-trainer-data}"
MODELS_DIR="${DIFFSYNTH_MODEL_BASE_PATH:-${TRAINER_HOME}/models}"
LEGACY_CACHE="${TRAINER_HOME}/modelscope-cache"
LOGS_DIR="${TRAINER_HOME}/logs"
DATASETS_DIR="${TRAINER_HOME}/datasets"
LOG_MODULE=cleanup
source "${ROOT_DIR}/shell_log.sh"

APPLY=0
STOP_PROCESSES=0
CLEAN_LEGACY=0
CLEAN_EMPTY_BATCHES=0

usage() {
  cat <<EOF
Usage: bash cleanup.sh [--yes] [--stop-processes] [--legacy-cache] [--empty-batches]

Without --yes this is a dry-run.

  --stop-processes  Stop this trainer's WebUI, install and training processes.
  --legacy-cache    Migrate missing H3 files into ${MODELS_DIR}, then remove the old ${LEGACY_CACHE}.
  --empty-batches   Remove h3-auto-targets-* jobs that contain no target images and no metadata.json.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes) APPLY=1 ;;
    --stop-processes) STOP_PROCESSES=1 ;;
    --legacy-cache) CLEAN_LEGACY=1 ;;
    --empty-batches) CLEAN_EMPTY_BATCHES=1 ;;
    -h|--help) usage; exit 0 ;;
    *) log_error "unknown option: $1"; usage >&2; exit 2 ;;
  esac
  shift
done

workload_patterns=(
  "${ROOT_DIR}/ensure_h3_runtime.py"
  "${ROOT_DIR}/download_h3.py"
  "${ROOT_DIR}/generate_h3_targets.py"
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
patterns=(
  "${ROOT_DIR}/app.py"
  "${ROOT_DIR}/.venv/bin/python -m pip"
  "${ROOT_DIR}/.venv-h3/bin/python -m pip"
  "${workload_patterns[@]}"
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

workload_running() {
  local pattern
  for pattern in "${workload_patterns[@]}"; do
    if pgrep -f -- "$pattern" >/dev/null 2>&1; then
      return 0
    fi
  done
  return 1
}

active_h3_job() {
  local state_file="${LOGS_DIR}/task-state.json"
  [[ -f "${state_file}" ]] || return 0
  python3 - "${state_file}" <<'PY' 2>/dev/null || true
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        task = json.load(handle)
except (OSError, ValueError, TypeError):
    raise SystemExit(0)

if task.get("status") in {"running", "paused"} and task.get("kind") in {"h3-generate", "h3-train"}:
    print(task.get("job_dir") or task.get("dataset_dir") or "")
PY
}

clean_empty_batches() {
  local active_job batch target_count size removed=0
  active_job="$(active_h3_job)"
  for batch in "${DATASETS_DIR}"/h3-auto-targets-*; do
    [[ -d "${batch}" ]] || continue
    if [[ -n "${active_job}" && "$(readlink -f -- "${batch}")" == "$(readlink -f -- "${active_job}")" ]]; then
      log_info "keep active generation batch ${batch}"
      continue
    fi
    [[ -f "${batch}/metadata.json" ]] && continue
    target_count=0
    if [[ -d "${batch}/targets" ]]; then
      target_count="$(find "${batch}/targets" -type f | wc -l)"
    fi
    (( target_count == 0 )) || continue
    size="$(du -sh "${batch}" 2>/dev/null | awk '{print $1}' || true)"
    if (( APPLY )); then
      log_info "removing empty batch ${batch} (${size:-unknown})"
      rm -rf -- "${batch}"
    else
      log_info "would remove empty batch ${batch} (${size:-unknown})"
    fi
    ((removed += 1))
  done
  if (( removed == 0 )); then
    log_info "no empty H3 generation batches found in ${DATASETS_DIR}"
  elif (( APPLY )); then
    log_info "removed ${removed} empty H3 generation batches"
  else
    log_info "dry-run: ${removed} empty H3 generation batches would be removed"
  fi
}

stop_processes() {
  local pattern
  for pattern in "${patterns[@]}"; do
    if pgrep -f -- "$pattern" >/dev/null 2>&1; then
      if (( APPLY )); then
        log_info "stopping: ${pattern}"
        pkill -TERM -f -- "$pattern" || true
      else
        log_info "would stop: ${pattern}"
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
      log_info "linked ${dst}"
    else
      log_info "copied ${dst}"
      cp --reflink=auto --sparse=always -- "${src}" "${dst}"
    fi
  else
    log_info "would migrate ${dst}"
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
    log_info "legacy cache does not exist: ${LEGACY_CACHE}"
  elif processes_running && (( ! STOP_PROCESSES )); then
    log_warn "skip legacy cache while trainer processes are running; add --stop-processes"
  else
    if ! required_models_present; then
      if (( APPLY )); then
        log_info "migrating required files from ${LEGACY_CACHE}"
        migrate_legacy_models
      else
        log_info "would migrate missing H3 files from ${LEGACY_CACHE}"
      fi
    fi
    if required_models_present; then
      legacy_size="$(du -sh "${LEGACY_CACHE}" 2>/dev/null | awk '{print $1}' || true)"
      if (( APPLY )); then
        log_info "removing legacy cache ${LEGACY_CACHE} (${legacy_size:-unknown})"
        rm -rf -- "${LEGACY_CACHE}"
      else
        log_info "would remove legacy cache ${LEGACY_CACHE} (${legacy_size:-unknown})"
      fi
    elif (( ! APPLY )); then
      log_info "dry-run: legacy cache would be removed after migration"
    else
      log_warn "skip legacy cache: required models are not complete in ${MODELS_DIR}"
    fi
  fi
fi

if (( CLEAN_EMPTY_BATCHES )); then
  active_job="$(active_h3_job)"
  if workload_running && [[ -z "${active_job}" ]] && (( ! STOP_PROCESSES )); then
    log_warn "skip empty batch cleanup while generation or training is running; add --stop-processes"
  else
    clean_empty_batches
  fi
fi

state_files=(
  "${LOGS_DIR}/h3-environment.pid"
  "${LOGS_DIR}/h3-environment-result.txt"
  "${LOGS_DIR}/h3-runtime.lock"
)

if processes_running && (( ! STOP_PROCESSES )); then
  log_warn "skip runtime state cleanup while trainer processes are running; add --stop-processes"
else
  for state_file in "${state_files[@]}"; do
    [[ -e "${state_file}" ]] || continue
    if (( APPLY )); then
      log_info "removing ${state_file}"
      rm -f -- "${state_file}"
    else
      log_info "would remove ${state_file}"
    fi
  done
fi

if (( APPLY )); then
  log_info "disk after cleanup:"
  if [[ -d "${TRAINER_HOME}" ]]; then
    log_info "$(df -h "${TRAINER_HOME}" | tail -n1)"
  else
    log_warn "${TRAINER_HOME} does not exist"
  fi
else
  log_info "dry-run only; add --yes to apply"
fi
