#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

DATASET_PATH="data/datasets/pentominos_dsprites-matched/"

for i in {1..5}; do
    uv run python -m scripts.train experiment=combgen/pentominos_rotation_sa \
      dataset.path=$DATASET_PATH \
      run_name='combgen_pentominos_rotation_sa_dsprites-match'
    uv run python -m scripts.train experiment=combgen/pentominos_rotation_wae \
      dataset.path=$DATASET_PATH \
      run_name='combgen_pentominos_rotation_wae_dsprites-match'
    uv run python -m scripts.train experiment=combgen/pentominos_rotation_vae \
      dataset.path=$DATASET_PATH \
      run_name='combgen_pentominos_rotation_vae-match'
done
