#!/bin/bash

# Attention control
# python -m bin.train experiment=slot_attn_control_3dshapes_baseline
# python -m bin.train experiment=slot_attn_control_3dshapes_shape2ohue

# python -m bin.train experiment=slot_attn_control_mpi3d_baseline
# python -m bin.train experiment=slot_attn_control_mpi3d_cyl2vx

# python -m bin.train experiment=slot_attn_control_dsprites_baseline
python -m bin.train experiment=slot_attn_control_dsprites_shape2px

# Decoder control
# python -m bin.train experiment=slot_dec_control_3dshapes_baseline
# python -m bin.train experiment=slot_dec_control_3dshapes_shape2ohue

# python -m bin.train experiment=slot_dec_control_mpi4d_baseline
python -m bin.train experiment=slot_dec_control_mpi3d_cyl2vx

# python -m bin.train experiment=slot_dec_control_dsprites_baseline
python -m bin.train experiment=slot_dec_control_dsprites_shape2px
