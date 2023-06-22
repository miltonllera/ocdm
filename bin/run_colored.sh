#!/bin/bash

export CUDA_VISIBLE_DEVICES=0


for i in {1..1}; do
#   uv run python -m scripts.train experiment=combgen/colored_pentominos_rotation_vae model.latent_size=8
#   uv run python -m scripts.train experiment=combgen/colored_pentominos_rotation_wae model.latent_size=8
#   uv run python -m scripts.train experiment=combgen/colored_pentominos_rotation_sa model.slot_size=8

#   uv run python -m scripts.train experiment=combgen/colored_pentominos_rotation_vae model.latent_size=16
#   uv run python -m scripts.train experiment=combgen/colored_pentominos_rotation_wae model.latent_size=16
  # uv run python -m scripts.train experiment=combgen/colored_pentominos_rotation_sa model.slot_size=16
#
#   uv run python -m scripts.train experiment=combgen/colored_pentominos_rotation_vae model.latent_size=32
  # uv run python -m scripts.train experiment=combgen/colored_pentominos_rotation_wae model.latent_size=32
  uv run python -m scripts.train experiment=combgen/colored_pentominos_rotation_sa model.slot_size=32

#   uv run python -m scripts.train experiment=combgen/colored_pentominos_rotation_vae model.latent_size=64
  # uv run python -m scripts.train experiment=combgen/colored_pentominos_rotation_wae model.latent_size=64
#   uv run python -m scripts.train experiment=combgen/colored_pentominos_rotation_sa model.slot_size=64
done
