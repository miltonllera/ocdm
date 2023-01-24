#!/bin/bash

# VAE
python -m bin.train experiment=vae_dsprites_baseline
python -m bin.train experiment=vae_3dshapes_baseline

# SlotAttention
python -m bin.train experiment=slotae_3dshapes_baseline
python -m bin.train experiment=slotae_mpi3d_baseline

# SLATE
python -m bin.train experiment=slate_3dshapes_baseline
python -m bin.train experiment=slate_mpi3d_baseline
