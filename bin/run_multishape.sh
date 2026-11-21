#!/bin/bash

source "$(dirname "$0")/gpu_queue.sh"

# Configuration and defaults
FAST_DEV_RUN=${FAST_DEV_RUN:-false}
MAX_STEPS=${MAX_STEPS:-200000}
LATENT_SIZE=${LATENT_SIZE:-64}
BATCH_SIZE=${BATCH_SIZE:-256}
SA_N_ITER=${SA_N_ITER:-3}
LOG_DIR=${LOG_DIR:-"logs/multi_shape/launcher"}

multi_models=("wae" "sa")
multi_filter="((shape == 6) | (shape == 8)) & (angle >= 180)"

echo "========================================================================="
echo "Running Colored Multi-Shape Pentominos Suite"
echo "========================================================================="

gpu_pool_init "${LOG_DIR}"

for model in "${multi_models[@]}"; do
  echo "Training colored_multi_pentominos_${model} (latent/slot size: ${LATENT_SIZE})..."

  model_params=()
  if [[ "$model" == "wae" ]]; then
    model_params=("model.latent_size=${LATENT_SIZE}")
  elif [[ "$model" == "sa" ]]; then
    model_params=(
      "model.slot_size=${LATENT_SIZE}"
      "model.n_iter=${SA_N_ITER}"
    )
  fi

  gpu_pool_run "colored_multi_${model}_ls${LATENT_SIZE}" \
    uv run python -m scripts.train \
    experiment="colored_multi_pentominos_${model}" \
    exp_dir=multi_shape \
    condition_name="combgen_colored_multi_pentominos" \
    model_name="${model}_ls${LATENT_SIZE}" \
    "dataset.held_out_filter=\"${multi_filter}\"" \
    "dataset.batch_size=${BATCH_SIZE}" \
    trainer.max_steps="${MAX_STEPS}" \
    trainer.fast_dev_run="${FAST_DEV_RUN}" \
    "${model_params[@]}"
done

gpu_pool_wait

