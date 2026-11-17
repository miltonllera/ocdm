#!/bin/bash


uv run python -m scripts.generate_pentominos \
  --height 60 --width 60 --pad 2 2 --aa 10 \
  --lim_angles 0 360 --num_angles 16 \
  --lim_scales 1.5 3.0 --num_scales 3 \
  --num_colors 10 --num_bg_colors 1 \
  --lim_xs -10 10 --num_xs 16 \
  --lim_ys -10 10 --num_ys 16 \
  --folder data/datasets/pentominos_colored_black_background


uv run python -m scripts.analyze_hf_latents \
  --dataset_name pentominos_colored_black_background \
  --model_id facebook/dinov2-small \
  --batch_size 256 \
  --sample_size 10000
