#!/bin/bash

# Configuration and defaults
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
FAST_DEV_RUN=${FAST_DEV_RUN:-false}
MAX_STEPS=${MAX_STEPS:-500000}

# 1. Single-shape Colored Pentominos Rotation
models=("wae" "sa" "sa_wwr")
latent_sizes=(16 32 64)
filter_expr="( shape == 6 ) & ( angle >= 180 )"

echo "========================================================================="
echo "Running Colored Pentominos Rotation Suite"
echo "========================================================================="

for model in "${models[@]}"; do
  for ls in "${latent_sizes[@]}"; do
    echo "Training colored_pentominos_${model} (latent/slot size: ${ls})..."

    model_params=()
    if [[ "$model" == "vae" || "$model" == "wae" ]]; then
      model_params=("model.latent_size=$ls")
    elif [[ "$model" == "sa" ]]; then
      model_params=("model.slot_size=$ls")
    elif [[ "$model" == "sa_wwr" ]]; then
      model_params=("model.slot_size=$ls" "use_wasserstein_reg=true")
    fi

    uv run python -m scripts.train \
      experiment="colored_pentominos_${model}" \
      exp_dir=colored \
      condition_name="combgen_colored_pentominos_rotation" \
      model_name="${model}_ls${ls}" \
      "dataset.held_out_filter=\"${filter_expr}\"" \
      dataset.rebalance_dim=shape \
      trainer.max_steps="${MAX_STEPS}" \
      trainer.fast_dev_run="${FAST_DEV_RUN}" \
      "${model_params[@]}"
  done
done

