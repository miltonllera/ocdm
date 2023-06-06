#!/bin/bash

# # Slot Attention
python -m bin.train experiment=slotae_dsprites_heart2rot

# # WAE
python -m bin.train experiment=wae_dsprites_heart2rot

# # WAE + SBD
python -m bin.train experiment=sbd_dsprites_heart2rot

# # WAE + SBD (wide)
python -m bin.train experiment=sbd_dsprites_heart2rot model.latent.latent_size=64 model.optimizer.lr=0.0001

# Compnet
python -m bin.train experimetn=compnet_dpsrites_heart2rot
