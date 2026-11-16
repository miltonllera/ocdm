#!/bin/bash

source "$(dirname "$0")/gpu_queue.sh"

# Configuration and defaults
FAST_DEV_RUN=${FAST_DEV_RUN:-false}
MAX_STEPS=${MAX_STEPS:-50000}
LOG_DIR=${LOG_DIR:-"logs/classifiers/launcher"}

datasets=("dsprites" "shapes3d" "pentominos")

gpu_pool_init "${LOG_DIR}"

for dataset in "${datasets[@]}"; do
  echo "========================================================================="
  echo "Training Shape Classifier on ${dataset}"
  echo "========================================================================="

  gpu_pool_run "classifier_${dataset}" \
    uv run python -m scripts.train \
    experiment="${dataset}_shape_classifier" \
    exp_dir=classifiers \
    condition_name="classifier_${dataset}_shape" \
    model_name="classifier" \
    trainer.max_steps="${MAX_STEPS}" \
    trainer.fast_dev_run="${FAST_DEV_RUN}"
done

gpu_pool_wait
