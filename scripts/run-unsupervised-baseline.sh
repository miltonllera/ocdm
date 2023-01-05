#!/bin/bash


# VAE
python -m bin.train experiment=unsupervised_vae_dsprites_baseline
python -m bin.train experiment=unsupervised_vae_3dshapes_baseline

# SlotAttention
python -m bin.train experiment=unsupervised_slotae_3dshapes_baseline

# SLATE
python -m bin.train experiment=unsupervised_slate_3dshapes_baseline
