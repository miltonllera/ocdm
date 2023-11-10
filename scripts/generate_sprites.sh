#!/bin/bash

# total size 60 + 2 * 2 = 64
export HEIGHT=60
export PAD=2


# black and white sprites

# python -m bin.generate_pentominos \
#   --height 60 --width 60 --pad 2 2  --aa 10 \
#   --num_angles 40 \
#   --num_scales 5 \
#   --num_colors 1 \
#   --num_xs 20 \
#   --num_ys 20 \
#   --folder data/datasets/pentominos/dsprites_like


python -m bin.generate_non_pentominos \
  --height 60 --width 60 --pad 2 2  --aa 10 \
  --num_angles 40 \
  --num_scales 5 \
  --num_colors 1 \
  --num_xs 20 \
  --num_ys 20 \
  --folder data/datasets/non_pentominos/
