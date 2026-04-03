#!/bin/bash

set -e

DATA_DIR=data/datasets/pentominos

# # # Generate white pentominos
# echo "Generating white pentominos..."
# uv run python -m scripts.generate_pentominos \
#   --height 60 \
#   --width 60 \
#   --pad 2 2 \
#   --aa 10 \
#   --lim_angles 0 360 \
#   --num_angles 40 \
#   --lim_scales 1.5 3.0 \
#   --num_scales 5 \
#   --num_colors 1 \
#   --lim_xs -10 10 \
#   --num_xs 20 \
#   --lim_ys -10 10 \
#   --num_ys 20 \
#   --folder "${DATA_DIR}"

# # Generate colored pentominos
# echo "Generating colored pentominos (10 colors)..."
# uv run python -m scripts.generate_pentominos \
#   --height 60 \
#   --width 60 \
#   --pad 2 2 \
#   --aa 10 \
#   --lim_angles 0 360 \
#   --num_angles 16 \
#   --lim_scales 1.5 3.0 \
#   --num_scales 1 \
#   --num_colors 10 \
#   --num_bg_colors 1 \
#   --lim_xs -10 10 \
#   --num_xs 16 \
#   --lim_ys -10 10 \
#   --num_ys 16 \
#   --folder "${DATA_DIR}_colored_black_background"

# # Generate dsprite-like pentomino pentominos
# echo "Generating dsprite-matched pentominos..."
# uv run python -m scripts.generate_pentominos \
#   --shapes 1 6 8 \
#   --height 60 \
#   --width 60 \
#   --pad 2 2 \
#   --aa 10 \
#   --lim_angles 0 360 \
#   --num_angles 40 \
#   --lim_scales 1.5 3.0 \
#   --num_scales 6 \
#   --num_colors 1 \
#   --lim_xs -10 10 \
#   --num_xs 32 \
#   --lim_ys -10 10 \
#   --num_ys 32 \
#   --folder "${DATA_DIR}_dsprites-matched"

