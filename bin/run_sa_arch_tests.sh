#!/bin/bash

# Configuration and defaults
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
FAST_DEV_RUN=${FAST_DEV_RUN:-false}
MAX_STEPS=${MAX_STEPS:-500000}


models=("wae" )

run_benchmark() {
  local dataset="$1"
  local condition_name="$2"
  local filter_expr="$3"

  echo "========================================================================="
  echo "Running CombGen Benchmark: ${condition_name} on 3DShapes"
  echo "========================================================================="

  for model in "${models[@]}"; do
    echo "Training ${model} (latent/slot size: 64) on ${condition_name}..."

    if [[ "$model" == "vae" || "$model" == "wae" ]]; then
      base_model="$model"
      model_params=("model.latent_size=64")
    elif [[ "$model" == "sa" ]]; then
      base_model="sa"
      model_params=("model.slot_size=64")
    fi

    uv run python -m scripts.train \
      experiment="shapes3d_${model}_diff_arch" \
      exp_dir=combgen_arch_tests \
      condition_name="${condition_name}" \
      model_name="${model}_braodcast_arch" \
      "${model_params[@]}" \
      "dataset.held_out_filter=\"${filter_expr}\"" \
      dataset.rebalance_dim=shape \
      trainer.max_steps="${MAX_STEPS}" \
      trainer.fast_dev_run="${FAST_DEV_RUN}"
  done
}

# 1. LiLS Replication: dSprites heart position
# run_benchmark "dsprites" "combgen_dsprites_heart_position" "( shape == 3 ) & ( posX > 0.5 )"

# 2. LiLS Replication: 3DShapes shape & hue
run_benchmark "shapes_3d" "combgen_shapes3d_shape_hue" "( shape == 3 ) & ( object_hue < 0.5 )"

# 3. Thesis condition from pentominos
# run_benchmark "pentominos" "combgen_pentominos_shape_rotation" "( shape == 8 ) & (angle >= 180)"

# 4. Test with different condition on Shapes 3D
# run_benchmark "shapes_3d" "combgen_shapes3d_shape_scale" "( shape == 3 ) & ( scale > 1.0 )"

