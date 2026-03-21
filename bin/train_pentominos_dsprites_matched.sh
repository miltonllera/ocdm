#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

DATASET_PATH="data/datasets/pentominos_dsprites-matched/"

for i in {1..1}; do
    uv run python -m scripts.train experiment=combgen/pentominos_rotation_sa \
      condition_name='combgen_pentominos_rotation_dsprites-match' \
      dataset.path=$DATASET_PATH \
      dataset.held_out_filter=" ( np.isin( shape, [6] ) & ( angle >= 180 ) " \
      model.slot.size=64
    uv run python -m scripts.train experiment=combgen/pentominos_rotation_wae \
      dataset.path=$DATASET_PATH \
      condition_name='combgen_pentominos_rotation_dsprites-match' \
      dataset.held_out_filter="( np.isin( shape, [6] ) & ( angle >= 180 )" \
      model.latent.size=64
    # uv run python -m scripts.train experiment=combgen/pentominos_rotation_vae \
    #   dataset.path=$DATASET_PATH \
    #   condition_name='combgen_pentominos_rotation_dsprites-match'
    #   dataset.held_out_filter: "( shape == 6 ) & ( angle >= 180 )"
done
