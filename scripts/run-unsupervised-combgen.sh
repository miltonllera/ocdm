#!/bin/bash


# SlotAttention
# python -m bin.train experiment=slotae_3dshapes_shape2ohue
python -m bin.train experiment=slotae_mpi3d_cyl2vx

# SLATE
python -m bin.train experiment=slate_3dshapes_shape2ohue
python -m bin.train experiment=slate_mpi3d_cyl2vx
