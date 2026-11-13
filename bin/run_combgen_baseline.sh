#!/bin/bash

# Configuration and defaults
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
FAST_DEV_RUN=${FAST_DEV_RUN:-false}
MAX_STEPS=${MAX_STEPS:-500000}

models=("wae" "sa")
latent_sizes=(16 32 64)

run_benchmark() {
  local dataset="$1"
  local condition_name="$2"
  local filter_expr="$3"

  echo "========================================================================="
  echo "Running CombGen Benchmark: ${condition_name} on ${dataset}"
  echo "========================================================================="

  for model in "${models[@]}"; do
    for ls in "${latent_sizes[@]}"; do
      local base_model=""
      local model_params=()

      if [[ "$model" == "vae" || "$model" == "wae" ]]; then
        base_model="$model"
        model_params=("model.latent_size=$ls")
      elif [[ "$model" == "sa" ]]; then
        base_model="sa"
        model_params=("model.slot_size=$ls")
      elif [[ "$model" == "sa_wwr" || "$model" == "sa+wwr" ]]; then
        base_model="sa"
        model="sa_wwr"
        model_params=("model.slot_size=$ls" "model.use_wasserstein_reg=true")
      fi

      echo "Training ${model} (latent/slot size: ${ls}) on ${condition_name}..."

      uv run python -m scripts.train \
        experiment="${dataset}_${base_model}" \
        exp_dir=combgen \
        condition_name="${condition_name}" \
        model_name="${model}_ls${ls}" \
        "dataset.held_out_filter=\"${filter_expr}\"" \
        dataset.rebalance_dim=shape \
        trainer.max_steps="${MAX_STEPS}" \
        trainer.fast_dev_run="${FAST_DEV_RUN}" \
        "${model_params[@]}"
    done
  done
}

# 1. LiLS Replication: dSprites heart position
run_benchmark "dsprites" "combgen_dsprites_heart_position" "( shape == 3 ) & ( posX > 0.5 )"

# 2. LiLS Replication: 3DShapes shape & hue
# run_benchmark "shapes3d" "combgen_shapes3d_shape_hue" "( shape == 3 ) & ( object_hue < 0.5 )"

# 3. Failure Mode Replication: dSprites heart rotation
run_benchmark "dsprites" "combgen_dsprites_heart_rotation" "( shape == 3 ) & ( orientation < np.pi )"

# 4. LiLS Replication: 3DShapes shape & hue
# run_benchmark "shapes3d" "combgen_shapes3d_shape_scale" "( shape == 3 ) & ( scale > 1.0 )"
