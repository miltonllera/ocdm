#!/bin/bash

source "$(dirname "$0")/gpu_queue.sh"

# Default GPU device and settings
FAST_DEV_RUN=${FAST_DEV_RUN:-false}
MAX_STEPS=${MAX_STEPS:-200000}
LOG_DIR=${LOG_DIR:-"logs/pentominos_rotation/launcher"}

# Define the shape configurations (nested sets)
declare -A shape_filters
shape_filters["2_shape"]="2,8"
shape_filters["5_shapes"]="2,4,5,6,8"
shape_filters["8_shapes"]="1,2,3,4,5,6,7,8"

# List of shape names in the order we want to run them
# shape_names=("2_shape" "5_shapes" "8_shapes")
shape_names=("2_shape" )

models=("wae" "sa")

gpu_pool_init "${LOG_DIR}"

for s in {1..1}; do
  for model in "${models[@]}"; do
    for shape_name in "${shape_names[@]}"; do
    shape_list="${shape_filters[$shape_name]}"
      echo "========================================================================="
      echo "Training ${model} with ${shape_name} (shapes: [${shape_list}]), run ${s}"
      echo "========================================================================="

      base_model=""
      model_params=()
      if [ "$model" = "wae" ]; then
        base_model="wae"
        model_params=("model.latent_size=64")
      else
        base_model="sa"
        model_params=("model.slot_size=64" )
      fi

      filter_expr="np.isin(shape, [${shape_list}]) & (angle >= 180)"

      gpu_pool_run "${model}_${shape_name}_run${s}" \
        uv run python -m scripts.train \
        experiment="pentominos_${base_model}" \
        exp_dir=pentominos_rotation \
        condition_name="combgen_pentominos_rotation_${shape_name}" \
        model_name="${model}" \
        "dataset.held_out_filter=\"${filter_expr}\"" \
        dataset.rebalance_dim=shape \
        trainer.max_steps="${MAX_STEPS}" \
        trainer.fast_dev_run="${FAST_DEV_RUN}" \
        "${model_params[@]}"
    done
  done
done

gpu_pool_wait
