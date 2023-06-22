#!/bin/bash

export CUDA_VISIBLE_DEVICES=2


#--------------------------------------- One new shape -------------------------------------------

for i in {1..5}; do
  uv run python -m scripts.train experiment=extrap/pentominos_new_shape_vae model.latent_size=8
  uv run python -m scripts.train experiment=extrap/pentominos_new_shape_wae model.latent_size=8
  uv run python -m scripts.train experiment=extrap/pentominos_new_shape_sa model.slot_size=8
done

for i in {1..5}; do
  uv run python -m scripts.train experiment=extrap/pentominos_new_shape_vae model.latent_size=16
  uv run python -m scripts.train experiment=extrap/pentominos_new_shape_wae model.latent_size=16
  uv run python -m scripts.train experiment=extrap/pentominos_new_shape_sa model.slot_size=16
done

for i i
  uv run python -m scripts.train experiment=extrap/pentominos_new_shape_vae model.latent_size=32
  uv run python -m scripts.train experiment=extrap/pentominos_new_shape_wae model.latent_size=32
  uv run python -m scripts.train experiment=extrap/pentominos_new_shape_sa model.slot_size=32
done

for i in {1..5}; do
  uv run python -m scripts.train experiment=extrap/pentominos_new_shape_vae model.latent_size=64
  uv run python -m scripts.train experiment=extrap/pentominos_new_shape_wae model.latent_size=64
  uv run python -m scripts.train experiment=extrap/pentominos_new_shape_sa model.slot_size=64
done


#-------------------------------------- Three new shapes -----------------------------------------

for i in {1..5}; do
  uv run python -m scripts.train experiment=extrap/pentominos_3_new_shapes_vae model.latent_size=8
  uv run python -m scripts.train experiment=extrap/pentominos_3_new_shapes_wae model.latent_size=8
  uv run python -m scripts.train experiment=extrap/pentominos_3_new_shapes_sa model.slot_size=8
done

for i in {1..5}; do
  uv run python -m scripts.train experiment=extrap/pentominos_3_new_shapes_vae model.latent_size=16
  uv run python -m scripts.train experiment=extrap/pentominos_3_new_shapes_wae model.latent_size=16
  uv run python -m scripts.train experiment=extrap/pentominos_3_new_shapes_sa model.slot_size=16
done

for i in {1..5}; do
  uv run python -m scripts.train experiment=extrap/pentominos_3_new_shapes_vae model.latent_size=32
  uv run python -m scripts.train experiment=extrap/pentominos_3_new_shapes_wae model.latent_size=32
  uv run python -m scripts.train experiment=extrap/pentominos_3_new_shapes_sa model.slot_size=32
done

for i in {1..5}; do
  uv run python -m scripts.train experiment=extrap/pentominos_3_new_shapes_vae model.latent_size=64
  uv run python -m scripts.train experiment=extrap/pentominos_3_new_shapes_wae model.latent_size=64
  uv run python -m scripts.train experiment=extrap/pentominos_3_new_shapes_sa model.slot_size=64
done


#--------------------------------------- Six new shapes ------------------------------------------
for i in {1..5}; do
  uv run python -m scripts.train experiment=extrap/pentominos_6_new_shapes_vae model.latent_size=8
  uv run python -m scripts.train experiment=extrap/pentominos_6_new_shapes_wae model.latent_size=8
  uv run python -m scripts.train experiment=extrap/pentominos_6_new_shapes_sa model.slot_size=8
done

for i in {1..5}; do
  uv run python -m scripts.train experiment=extrap/pentominos_6_new_shapes_vae model.latent_size=16
  uv run python -m scripts.train experiment=extrap/pentominos_6_new_shapes_wae model.latent_size=16
  uv run python -m scripts.train experiment=extrap/pentominos_6_new_shapes_sa model.slot_size=16
done


for i in {1..5}; do
  uv run python -m scripts.train experiment=extrap/pentominos_6_new_shapes_vae model.latent_size=32
  uv run python -m scripts.train experiment=extrap/pentominos_6_new_shapes_wae model.latent_size=32
  uv run python -m scripts.train experiment=extrap/pentominos_6_new_shapes_sa model.slot_size=32
done


for i in {1..5}; do
  uv run python -m scripts.train experiment=extrap/pentominos_6_new_shapes_vae model.latent_size=64
  uv run python -m scripts.train experiment=extrap/pentominos_6_new_shapes_wae model.latent_size=64
  uv run python -m scripts.train experiment=extrap/pentominos_6_new_shapes_sa model.slot_size=64
done
