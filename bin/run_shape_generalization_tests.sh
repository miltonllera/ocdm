#!/bin/bash

# Default GPU device
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

# Define the shape configurations (nested sets)
# 1 shape: [8] (W)
# 2 shapes: [6, 8] (U, W)
# 4 shapes: [4, 5, 6, 8] (N, T, U, W)
# 6 shapes: [2, 3, 4, 5, 6, 8] (L, P, N, T, U, W)

declare -A shape_filters
shape_filters["1_shape"]="8"
shape_filters["2_shapes"]="6,8"
shape_filters["4_shapes"]="4,5,6,8"
shape_filters["6_shapes"]="2,3,4,5,6,8"
shape_filters["7_shapes"]="2,3,4,5,6,8,10"
shape_filters["9_shapes"]="1,2,3,4,5,6,7,8,10"

# List of shape names in the order we want to run them
# shape_names=("1_shape" "2_shapes" "4_shapes" "6_shapes" "8_shapes" "10_shapes")
shape_names=("9_shapes" )

seeds=(101 102 )
models=("sa" )

for model in "${models[@]}"; do
  for shape_name in "${shape_names[@]}"; do
    shape_list="${shape_filters[$shape_name]}"
    for seed in "${seeds[@]}"; do
      echo "========================================================================="
      echo "Training ${model} with ${shape_name} (shapes: [${shape_list}]), seed ${seed}"
      echo "========================================================================="

      # Determine latent parameter based on model type
      if [ "$model" = "wae" ]; then
        model_param="model.latent_size=64"
      else
        model_param="model.slot_size=64 model.use_wasserstein_reg=true"
      fi

      # Build the held-out filter expression
      filter_expr="np.isin(shape, [${shape_list}]) & (angle >= 180)"

      # Run the training command
      # Note: we pass +dataset.num_workers=0 to avoid AttributeError on unique_values
      uv run python -m scripts.train \
        experiment=combgen/pentominos_rotation_${model} \
        ${model_param} \
        seed=${seed} \
        condition_name=combgen_pentominos_rotation_${shape_name}_seed_${seed} \
        model_name='sa_wwr' \
        "dataset.held_out_filter=\"${filter_expr}\""
    done
  done
done
