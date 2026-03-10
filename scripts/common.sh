#!/usr/bin/env bash
set -euo pipefail

OSS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="${OSS_DIR}"

ensure_repo_root() {
  cd "${REPO_ROOT}"
}

read_extra_args() {
  local raw="${1:-}"
  local -n out_ref="$2"
  out_ref=()
  if [[ -n "${raw}" ]]; then
    # Shell-style splitting is sufficient for simple override arguments.
    read -r -a out_ref <<< "${raw}"
  fi
}

run_training_job() {
  local task="$1"
  local default_config="$2"
  local default_log="$3"
  local default_pid="$4"

  ensure_repo_root

  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export TRANSFORMERS_NO_TORCHVISION="${TRANSFORMERS_NO_TORCHVISION:-1}"

  local config="${CONFIG:-${default_config}}"
  local log_file="${LOG:-${default_log}}"
  local pid_file="${PIDFILE:-${default_pid}}"
  local num_processes="${NUM_PROCESSES:-1}"
  local mixed_precision="${MIXED_PRECISION:-bf16}"

  local cmd=(
    accelerate launch
    --num_processes="${num_processes}"
    --mixed_precision="${mixed_precision}"
    -m flux2_litekit.train
    --task "${task}"
    --config "${config}"
  )

  if [[ -n "${RESUME_FROM:-}" ]]; then
    cmd+=(--resume_from "${RESUME_FROM}")
  fi

  local extra_args=()
  read_extra_args "${TRAIN_EXTRA_ARGS:-}" extra_args
  cmd+=("${extra_args[@]}")

  echo "[INFO] Working dir: ${REPO_ROOT}"
  echo "[INFO] Task: ${task}"
  echo "[INFO] Config: ${config}"
  echo "[INFO] Log: ${log_file}"
  echo "[INFO] PID file: ${pid_file}"
  echo "[INFO] Command: ${cmd[*]}"

  if [[ -f "${pid_file}" ]]; then
    local old_pid
    old_pid="$(cat "${pid_file}" || true)"
    if [[ -n "${old_pid}" ]] && ps -p "${old_pid}" >/dev/null 2>&1; then
      echo "[ERROR] Found a running process (pid=${old_pid}). Stop it first or remove ${pid_file}."
      exit 1
    fi
  fi

  mkdir -p "$(dirname "${log_file}")"
  mkdir -p "$(dirname "${pid_file}")"

  nohup "${cmd[@]}" > "${log_file}" 2>&1 &
  echo $! > "${pid_file}"

  echo "[OK] Started. PID=$(cat "${pid_file}")"
  echo "[TIP] tail -f ${log_file}"
  echo "[TIP] stop: kill \$(cat ${pid_file})"
}

run_inference_job() {
  local task="$1"
  local default_config="$2"

  ensure_repo_root

  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export TRANSFORMERS_NO_TORCHVISION="${TRANSFORMERS_NO_TORCHVISION:-1}"

  local config="${CONFIG:-${default_config}}"
  local python_bin="${PYTHON_BIN:-python}"
  local cmd=(
    "${python_bin}"
    -m flux2_litekit.infer
    --task "${task}"
    --config "${config}"
  )

  local extra_args=()
  read_extra_args "${INFER_EXTRA_ARGS:-}" extra_args
  cmd+=("${extra_args[@]}")

  echo "[INFO] Working dir: ${REPO_ROOT}"
  echo "[INFO] Task: ${task}"
  echo "[INFO] Config: ${config}"
  echo "[INFO] Command: ${cmd[*]}"

  "${cmd[@]}"
}
