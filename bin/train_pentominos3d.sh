#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# baseline
uv run python -m scripts.train experiment=baseline/pentominos3d_wae
uv run python -m scripts.train experiment=baseline/pentominos3d_sa

#  combgen
uv run python -m scripts.train experiment=combgen/pentominos3d__rotation_wae
uv run python -m scripts.train experiment=combgen/pentominos3d__rotation_sa
