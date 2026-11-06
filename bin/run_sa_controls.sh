#!/bin/bash

# Default GPU device and settings
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
FAST_DEV_RUN=${FAST_DEV_RUN:-false}
MAX_STEPS=${MAX_STEPS:-500000}

models=("fgdc" "fgc")
slot_sizes=(16 32 64)
num_runs=5

run_control_suite() {
  local task="$1"
  local filter_expr="$2"

  echo "========================================================================="
  echo "Running SA Controls: ${task}"
  echo "========================================================================="

  for model in "${models[@]}"; do
    for ls in "${slot_sizes[@]}"; do
      for run in $(seq 1 "$num_runs"); do
        echo "Training ${model} (slot size: ${ls}, run: ${run}) on ${task}..."

        uv run python -m scripts.train \
          experiment="dsprites_${model}" \
          exp_dir=controls \
          condition_name="control_dsprites_${task}" \
          model_name="${model}_ls${ls}_run${run}" \
          model.slot_size="${ls}" \
          "dataset.held_out_filter=\"${filter_expr}\"" \
          dataset.rebalance_dim=shape \
          trainer.max_steps="${MAX_STEPS}" \
          trainer.fast_dev_run="${FAST_DEV_RUN}"
      done
    done
  done
}

# 1. Slot Attention Control: heart-position
run_control_suite "heart_position" "( shape == 3 ) & ( posX < 0.5 )"

# 2. Slot Attention Control: heart-rotation
run_control_suite "heart_rotation" "( shape == 3 ) & ( orientation < np.pi )"
