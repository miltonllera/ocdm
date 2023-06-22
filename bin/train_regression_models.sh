#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# classifiers for shape
uv run python -m scripts.train experiment=class/dsprites_shape_classifier
uv run python -m scripts.train experiment=class/pentominos_shape_classifier
uv run python -m scripts.train experiment=class/shapes3d_shape_classifier

# adversarial discriminators
uv run python -m scripts.train experiment=discr/dsprites_vae_shape_position
uv run python -m scripts.train experiment=discr/dsprites_vae_shape_rotation
uv run python -m scripts.train experiment=discr/shapes3d_vae_object_hue
uv run python -m scripts.train experiment=discr/pentominos_vae_shape_rotation

