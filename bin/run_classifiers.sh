#!/bin/bash

# Configuration and defaults
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
FAST_DEV_RUN=${FAST_DEV_RUN:-false}
MAX_STEPS=${MAX_STEPS:-50000}

datasets=("dsprites" "shapes3d" "pentominos")

for dataset in "${datasets[@]}"; do
  echo "========================================================================="
  echo "Training Shape Classifier on ${dataset}"
  echo "========================================================================="

  uv run python -m scripts.train \
    experiment="${dataset}_shape_classifier" \
    exp_dir=classifiers \
    condition_name="classifier_${dataset}_shape" \
    model_name="classifier" \
    trainer.max_steps="${MAX_STEPS}" \
    trainer.fast_dev_run="${FAST_DEV_RUN}"
done
