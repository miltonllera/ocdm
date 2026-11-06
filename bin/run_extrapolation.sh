#!/bin/bash

# Default GPU device and settings
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
FAST_DEV_RUN=${FAST_DEV_RUN:-false}
MAX_STEPS=${MAX_STEPS:-500000}

# Define the shape configurations (nested sets)
# 1 shape: [8] (W)
# 2 shapes: [6, 8] (U, W)
# 4 shapes: [4, 5, 6, 8] (N, T, U, W)
# 6 shapes: [2, 3, 4, 5, 6, 8] (L, P, N, T, U, W)
declare -A shape_filters
shape_filters["1_shape"]="8"
shape_filters["4_shapes"]="4,5,6,8"
shape_filters["9_shapes"]="1,2,3,4,5,6,7,8,10"

# List of shape names in the order we want to run them
shape_names=("1_shape" "4_shapes" "9_shapes")

models=("sa_wwr" "wae")

for i in {1..5}; do
  for model in "${models[@]}"; do
    for shape_name in "${shape_names[@]}"; do
      shape_list="${shape_filters[$shape_name]}"
      echo "========================================================================="
      echo "Training ${model} extrapolation with ${shape_name} (shapes: [${shape_list}]), seed ${seed}"
      echo "========================================================================="

      base_model=""
      model_params=()
      if [ "$model" = "wae" ]; then
        base_model="wae"
        model_params=("model.latent_size=64")
      else
        base_model="sa"
        model_params=("model.slot_size=64" "model.use_wasserstein_reg=true")
      fi

      filter_expr="np.isin(shape, [${shape_list}])"

      uv run python -m scripts.train \
        experiment="pentominos_${base_model}" \
        exp_dir=shape_extrapolation \
        condition_name="extrap_pentominos_${shape_name}" \
        model_name="${model}" \
        "dataset.held_out_filter=\"${filter_expr}\"" \
        dataset.rebalance_dim=shape \
        trainer.max_steps="${MAX_STEPS}" \
        trainer.fast_dev_run="${FAST_DEV_RUN}" \
        "${model_params[@]}"
    done
  done
done
