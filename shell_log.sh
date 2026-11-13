#!/usr/bin/env bash

LOG_MODULE="${LOG_MODULE:-launcher}"
TRAINER_HOME="${TRAINER_HOME:-/mnt/workspace/pose-model-trainer-data}"
LAUNCHER_LOG="${LAUNCHER_LOG:-${TRAINER_HOME}/logs/launcher.log}"
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
