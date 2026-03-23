#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

uv run python -m scripts.train experiment=baseline/pentominos3d_sa
# uv run python -m scripts.train experiment=baseline/pentominos3d_wae
