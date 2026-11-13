#!/bin/bash

# Configuration and defaults
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
FAST_DEV_RUN=${FAST_DEV_RUN:-false}
MAX_STEPS=${MAX_STEPS:-500000}

run_benchmark() {
  local condition_name="$1"
  local filter_expr="$2"

  echo "========================================================================="
  echo "Running CombGen Benchmark: ${condition_name} on 3DShapes"
  echo "========================================================================="

  echo "Training SA (latent/slot size: 64) on ${condition_name}..."

  uv run python -m scripts.train \
    experiment="shapes3d_sa_diff_arch" \
    exp_dir=combgen_arch_tests \
    condition_name="${condition_name}" \
    model_name="sa_arch1" \
    "dataset.held_out_filter=\"${filter_expr}\"" \
    dataset.rebalance_dim=shape \
    trainer.max_steps="${MAX_STEPS}" \
    trainer.fast_dev_run="${FAST_DEV_RUN}"
}

# 1. LiLS Replication: 3DShapes shape & hue
run_benchmark "combgen_shapes3d_shape_hue" "( shape == 3 ) & ( object_hue < 0.5 )"

# 2. LiLS Replication: 3DShapes shape & hue
# run_benchmark "combgen_shapes3d_shape_scale" "( shape == 3 ) & ( scale > 1.0 )"

