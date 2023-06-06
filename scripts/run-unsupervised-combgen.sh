#!/bin/bash


# SlotAttention

# 3DShapes
python -m bin.train experiment=slotae_3dshapes_shape2ohue

# MPI
python -m bin.train experiment=slotae_mpi3d_cyl2vx

# dSprites
python -m bin.train experiment=slotae_dsprites_shape2px

# SLATE

# 3DShapes
python -m bin.train experiment=slate_3dshapes_shape2ohue

# MPI3D
python -m bin.train experiment=slate_mpi3d_cyl2vx
