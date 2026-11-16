#!/bin/bash

# Reusable 1-job-per-GPU asynchronous job queue for launcher scripts.
# Usage:
#   source "$(dirname "$0")/gpu_queue.sh"
#   gpu_pool_init "$LOG_DIR"
#   gpu_pool_run "job_name" command arg1 arg2 ...
#   gpu_pool_wait

_GPU_LIST=()
_GPU_PIDS=()
_GPU_LOG_DIR=""
_GPU_FAILED=0
_FREE_GPU_IDX=0

_gpu_pool_detect() {
  if [[ -n "${GPUS:-}" ]]; then
    read -r -a _GPU_LIST <<< "$GPUS"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a _GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    local num_gpus
    num_gpus=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')
    _GPU_LIST=()
    for ((i = 0; i < num_gpus; i++)); do
      _GPU_LIST+=("$i")
    done
  else
    _GPU_LIST=("0")
  fi
}

_gpu_pool_cleanup() {
  echo "Interrupted; terminating running jobs..."
  for pid in "${_GPU_PIDS[@]}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  exit 130
}

_gpu_pool_acquire() {
  while true; do
    for ((i = 0; i < ${#_GPU_LIST[@]}; i++)); do
      local pid="${_GPU_PIDS[$i]}"
      if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        if [[ -n "$pid" ]]; then
          if ! wait "$pid"; then
            echo "Warning: job PID ${pid} exited with non-zero status"
            _GPU_FAILED=1
          fi
          _GPU_PIDS[$i]=""
        fi
        _FREE_GPU_IDX=$i
        return
      fi
    done
    sleep 2
  done
}

gpu_pool_init() {
  _GPU_LOG_DIR="${1:?Log directory is required}"
  mkdir -p "${_GPU_LOG_DIR}"
  _gpu_pool_detect

  _GPU_PIDS=()
  for ((i = 0; i < ${#_GPU_LIST[@]}; i++)); do
    _GPU_PIDS+=("")
  done
  _GPU_FAILED=0

  trap _gpu_pool_cleanup INT TERM
  echo "GPU pool initialized: [${_GPU_LIST[*]}] | Logs: ${_GPU_LOG_DIR}"
}

gpu_pool_run() {
  local job_name="$1"
  shift

  _gpu_pool_acquire
  local gpu_id="${_GPU_LIST[$_FREE_GPU_IDX]}"
  local log_file="${_GPU_LOG_DIR}/${job_name}.log"

  echo "[GPU ${gpu_id}] Launching ${job_name} -> ${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "$@" > "${log_file}" 2>&1 &
  _GPU_PIDS[$_FREE_GPU_IDX]=$!
}

gpu_pool_wait() {
  for pid in "${_GPU_PIDS[@]}"; do
    if [[ -n "$pid" ]]; then
      if ! wait "$pid"; then
        _GPU_FAILED=1
      fi
    fi
  done

  if [[ "$_GPU_FAILED" -ne 0 ]]; then
    echo "One or more jobs failed. Check logs in ${_GPU_LOG_DIR}."
    return 1
  fi
  echo "All jobs completed successfully."
}
