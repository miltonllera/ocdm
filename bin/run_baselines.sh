#!/bin/bash

# Configuration and defaults
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
FAST_DEV_RUN=${FAST_DEV_RUN:-false}
MAX_STEPS=${MAX_STEPS:-100000}

datasets=("dsprites" "shapes3d" "pentominos")
models=("vae" "wae" "sa")
latent_sizes=(16 32 64)

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for ls in "${latent_sizes[@]}"; do
      echo "========================================================================="
      echo "Baseline training: ${dataset} - ${model} (latent/slot size: ${ls})"
      echo "========================================================================="

      model_params=()
      if [[ "$model" == "vae" || "$model" == "wae" ]]; then
        model_params=("model.latent_size=$ls")
      elif [[ "$model" == "sa" ]]; then
        model_params=("model.slot_size=$ls")
      fi

      uv run python -m scripts.train \
        experiment="${dataset}_${model}" \
        exp_dir=baseline \
        condition_name="baseline_${dataset}" \
        model_name="${model}_ls${ls}" \
        trainer.max_steps="${MAX_STEPS}" \
        trainer.fast_dev_run="${FAST_DEV_RUN}" \
        "${model_params[@]}"
    done
  done
done
