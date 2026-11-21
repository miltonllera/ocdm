#!/bin/bash

source "$(dirname "$0")/gpu_queue.sh"

# Configuration and defaults
FAST_DEV_RUN=${FAST_DEV_RUN:-false}
MAX_STEPS=${MAX_STEPS:-200000}
LOG_DIR=${LOG_DIR:-"logs/colored/launcher"}

# 1. Colored Pentominos Rotation (shapes 6 and 8, angle >= 180)
models=("sa")
LATENT_SIZE=${LATENT_SIZE:-64}
BATCH_SIZE=${BATCH_SIZE:-256}
SA_N_ITER=${SA_N_ITER:-3}
filter_expr="((shape == 6) | (shape == 8)) & (angle >= 180)"

echo "========================================================================="
echo "Running Colored Pentominos Rotation Suite"
echo "========================================================================="

gpu_pool_init "${LOG_DIR}"

for model in "${models[@]}"; do
  echo "Training colored_pentominos_${model} (latent/slot size: ${LATENT_SIZE})..."

  model_params=()
  if [[ "$model" == "wae" ]]; then
    model_params=("model.latent_size=${LATENT_SIZE}")
  elif [[ "$model" == "sa" ]]; then
    model_params=("model.slot_size=${LATENT_SIZE}" "model.n_iter=${SA_N_ITER}")
  fi

  gpu_pool_run "colored_pentominos_${model}_ls${LATENT_SIZE}" \
    uv run python -m scripts.train \
    experiment="colored_pentominos_${model}" \
    exp_dir=colored \
    condition_name="combgen_colored_pentominos_rotation" \
    model_name="${model}_ls${LATENT_SIZE}" \
    "dataset.held_out_filter=\"${filter_expr}\"" \
    "dataset.batch_size=${BATCH_SIZE}" \
    dataset.rebalance_dim=shape \
    trainer.max_steps="${MAX_STEPS}" \
    trainer.fast_dev_run="${FAST_DEV_RUN}" \
    "${model_params[@]}"
done

gpu_pool_wait

