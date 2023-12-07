#!/bin/bash

# total size 60 + 2 * 2 = 64
export HEIGHT=60
export WIDTH=60
export PAD=2


# black and white sprites

# python -m bin.generate_pentominos \
#  --height $HEIGHT --width $WIDTH --pad $PAD $PAD --aa 10 \
#   --num_angles 40 \
#   --num_scales 5 \
#   --num_colors 1 \
#   --num_xs 20 \
#   --num_ys 20 \
#   --folder data/datasets/pentominos/dsprites_like


# Test larger scale sizes
# python -m bin.generate_pentominos \
#  --height $HEIGHT --width $WIDTH --pad $PAD $PAD --aa 10 \
#   --num_angles 40 \
#   --num_scales 3 \
#   --lim_scales 3.3 4.6 \
#   --num_colors 1 \
#   --num_xs 20 \
#   --num_ys 20 \
#   --folder data/datasets/pentominos/larger_scales


# NOTE: non-pentominos base size is 16 (as opposed to 10 like in pentominos)
python -m bin.generate_non_pentominos \
  --height $HEIGHT --width $WIDTH --pad $PAD $PAD --aa 10 \
  --num_angles 40 \
  --num_scales 5 \
  --lim_scales 1.0 2.0 \
  --num_colors 1 \
  --num_xs 20 \
  --num_ys 20 \
  --folder data/datasets/non_pentominos/corrected_scales
