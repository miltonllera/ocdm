#!/bin/bash

set -e

DATA_DIR=data/datasets/non_pentominos

# Generate white non pentominos
echo "Generating white non-pentominos..."
uv run python -m scripts.generate_non_pentominos \
  --height 60 \
  --width 60 \
  --pad 2 2 \
  --aa 10 \
  --lim_angles 0 360 \
  --num_angles 40 \
  --lim_scales 1.5 3.0 \
  --num_scales 5 \
  --num_colors 1 \
  --lim_xs -10 10 \
  --num_xs 20 \
  --lim_ys -10 10 \
  --num_ys 20 \
  --folder "${DATA_DIR}"

