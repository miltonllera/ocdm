#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

#------------------------------- Replication from LiLS -----------------------------------

# for i in {1..1}; do
  # uv run python -m scripts.train experiment=combgen/dsprites_heart_position_vae model.latent_size=8
  # uv run python -m scripts.train experiment=combgen/dsprites_heart_position_wae model.latent_size=8
  # uv run python -m scripts.train experiment=combgen/dsprites_heart_position_sa model.slot_size=8

  # uv run python -m scripts.train experiment=combgen/dsprites_heart_position_vae model.latent_size=16
  # uv run python -m scripts.train experiment=combgen/dsprites_heart_position_wae model.latent_size=16
  # uv run python -m scripts.train experiment=combgen/dsprites_heart_position_sa model.slot_size=16

  # uv run python -m scripts.train experiment=combgen/dsprites_heart_position_vae model.latent_size=32
  # uv run python -m scripts.train experiment=combgen/dsprites_heart_position_wae model.latent_size=32
  # uv run python -m scripts.train experiment=combgen/dsprites_heart_position_sa model.slot_size=32

  # uv run python -m scripts.train experiment=combgen/dsprites_heart_position_vae model.latent_size=64
  # uv run python -m scripts.train experiment=combgen/dsprites_heart_position_wae model.latent_size=64
  # uv run python -m scripts.train experiment=combgen/dsprites_heart_position_sa model.slot_size=64
# done


# for i in {1..1}; do
#   uv run python -m scripts.train experiment=combgen/shapes3d_shape_hue_vae model.latent_size=8
#   uv run python -m scripts.train experiment=combgen/shapes3d_shape_hue_wae model.latent_size=8
#   uv run python -m scripts.train experiment=combgen/shapes3d_shape_hue_sa model.slot_size=8

#   uv run python -m scripts.train experiment=combgen/shapes3d_shape_hue_vae model.latent_size=16
#   uv run python -m scripts.train experiment=combgen/shapes3d_shape_hue_wae model.latent_size=16
#   uv run python -m scripts.train experiment=combgen/shapes3d_shape_hue_sa model.slot_size=16
#
  # uv run python -m scripts.train experiment=combgen/shapes3d_shape_hue_vae model.latent_size=32
  # uv run python -m scripts.train experiment=combgen/shapes3d_shape_hue_wae model.latent_size=32
  # uv run python -m scripts.train experiment=combgen/shapes3d_shape_hue_sa model.slot_size=32

  # uv run python -m scripts.train experiment=combgen/shapes3d_shape_hue_vae model.latent_size=64
  # uv run python -m scripts.train experiment=combgen/shapes3d_shape_hue_wae model.latent_size=64
  # uv run python -m scripts.train experiment=combgen/shapes3d_shape_hue_sa model.slot_size=64
# done

#------------------------------- Replicating failures ------------------------------------

# for i in {1..1}; do
#   uv run python -m scripts.train experiment=combgen/dsprites_heart_rotation_vae model.latent_size=8
#   uv run python -m scripts.train experiment=combgen/dsprites_heart_rotation_wae model.latent_size=8
#   uv run python -m scripts.train experiment=combgen/dsprites_heart_rotation_sa model.slot_size=8

#   uv run python -m scripts.train experiment=combgen/dsprites_heart_rotation_vae model.latent_size=16
#   uv run python -m scripts.train experiment=combgen/dsprites_heart_rotation_wae model.latent_size=16
#   uv run python -m scripts.train experiment=combgen/dsprites_heart_rotation_sa model.slot_size=16
#
#   uv run python -m scripts.train experiment=combgen/dsprites_heart_rotation_vae model.latent_size=32
#   uv run python -m scripts.train experiment=combgen/dsprites_heart_rotation_wae model.latent_size=32
#   uv run python -m scripts.train experiment=combgen/dsprites_heart_rotation_sa model.slot_size=32

  # uv run python -m scripts.train experiment=combgen/dsprites_heart_rotation_vae model.latent_size=64
  # uv run python -m scripts.train experiment=combgen/dsprites_heart_rotation_wae model.latent_size=64
  # uv run python -m scripts.train experiment=combgen/dsprites_heart_rotation_sa model.slot_size=64
# done

#------------------------------- Results from my thesis ----------------------------------

for i in {1..1}; do
#   uv run python -m scripts.train experiment=combgen/pentominos_rotation_vae model.latent_size=8
#   uv run python -m scripts.train experiment=combgen/pentominos_rotation_wae model.latent_size=8
#   uv run python -m scripts.train experiment=combgen/pentominos_rotation_sa model.slot_size=8

#   uv run python -m scripts.train experiment=combgen/pentominos_rotation_vae model.latent_size=16
#   uv run python -m scripts.train experiment=combgen/pentominos_rotation_wae model.latent_size=16
#   uv run python -m scripts.train experiment=combgen/pentominos_rotation_sa model.slot_size=16

#   uv run python -m scripts.train experiment=combgen/pentominos_rotation_vae model.latent_size=32
#   uv run python -m scripts.train experiment=combgen/pentominos_rotation_wae model.latent_size=32
#   uv run python -m scripts.train experiment=combgen/pentominos_rotation_sa model.slot_size=32

  uv run python -m scripts.train experiment=combgen/pentominos_rotation_vae model.latent_size=64
#   uv run python -m scripts.train experiment=combgen/pentominos_rotation_wae model.latent_size=64
#   uv run python -m scripts.train experiment=combgen/pentominos_rotation_sa model.slot_size=64
done


# for i in {1..1}; do
#   uv run python -m scripts.train experiment=combgen/non_pentominos_rotation_vae model.latent_size=8
#   uv run python -m scripts.train experiment=combgen/non_pentominos_rotation_wae model.latent_size=8
#   uv run python -m scripts.train experiment=combgen/non_pentominos_rotation_sa model.slot_size=8

#   uv run python -m scripts.train experiment=combgen/non_pentominos_rotation_vae model.latent_size=16
#   uv run python -m scripts.train experiment=combgen/non_pentominos_rotation_wae model.latent_size=16
#   uv run python -m scripts.train experiment=combgen/non_pentominos_rotation_sa model.slot_size=16
#
#   uv run python -m scripts.train experiment=combgen/non_pentominos_rotation_vae model.latent_size=32
#   uv run python -m scripts.train experiment=combgen/non_pentominos_rotation_wae model.latent_size=32
#   uv run python -m scripts.train experiment=combgen/non_pentominos_rotation_sa model.slot_size=32

#   uv run python -m scripts.train experiment=combgen/non_pentominos_rotation_vae model.latent_size=64
#   uv run python -m scripts.train experiment=combgen/non_pentominos_rotation_wae model.latent_size=64
#   uv run python -m scripts.train experiment=combgen/non_pentominos_rotation_sa model.slot_size=64
# done
