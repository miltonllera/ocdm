#!/bin/bash

source "$(dirname "$0")/gpu_queue.sh"

# Configuration and defaults
FAST_DEV_RUN=${FAST_DEV_RUN:-false}
MAX_STEPS=${MAX_STEPS:-500000}
LOG_DIR=${LOG_DIR:-"logs/multi_shape/launcher"}

multi_models=("wae" "sa")
multi_latent_sizes=(16 64)
multi_filter="np.isin( shape , [1,2,3,4,5,6,7,8,10] ) & ( angle >= 180 )"

echo "========================================================================="
echo "Running Colored Multi-Shape Pentominos Suite"
echo "========================================================================="

gpu_pool_init "${LOG_DIR}"

for model in "${multi_models[@]}"; do
  for ls in "${multi_latent_sizes[@]}"; do
    echo "Training colored_multi_pentominos_${model} (latent/slot size: ${ls})..."

    model_params=()
    if [[ "$model" == "wae" ]]; then
      model_params=("model.latent_size=$ls")
    elif [[ "$model" == "sa" ]]; then
      model_params=("model.slot_size=$ls")
    fi

    gpu_pool_run "colored_multi_${model}_ls${ls}" \
      uv run python -m scripts.train \
      experiment="colored_multi_pentominos_${model}" \
      exp_dir=multi_shape \
      condition_name="combgen_colored_multi_pentominos" \
      model_name="${model}_ls${ls}" \
      "dataset.held_out_filter=\"${multi_filter}\"" \
      trainer.max_steps="${MAX_STEPS}" \
      trainer.fast_dev_run="${FAST_DEV_RUN}" \
      "${model_params[@]}"
  done
done

gpu_pool_wait

