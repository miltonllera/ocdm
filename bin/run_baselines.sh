#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

uv run python -m scripts.train experiment=baseline/dsprites_vae
uv run python -m scripts.train experiment=baseline/dsprites_wae
uv run python -m scripts.train experiment=baseline/dsprites_wae_sbd
uv run python -m scripts.train experiment=baseline/dsprites_sa

uv run python -m scripts.train experiment=baseline/dsprites_vae model.latent_size=64
uv run python -m scripts.train experiment=baseline/dsprites_wae model.latent_size=64
uv run python -m scripts.train experiment=baseline/dsprites_wae_sbd model.latent_size=64
uv run python -m scripts.train experiment=baseline/dsprites_sa model.slot_size=10

uv run python -m scripts.train experiment=baseline/dsprites_vae model.latent_size=32
uv run python -m scripts.train experiment=baseline/dsprites_wae model.latent_size=32
uv run python -m scripts.train experiment=baseline/dsprites_wae_sbd model.latent_size=32
uv run python -m scripts.train experiment=baseline/dsprites_sa model.slot_size=32

# uv run python -m scripts.train experiment=baseline/shapes3d_vae
# uv run python -m scripts.train experiment=baseline/shapes3d_wae
# uv run python -m scripts.train experiment=baseline/shapes3d_sa

# uv run python -m scripts.train experiment=baseline/pentominos_vae
# uv run python -m scripts.train experiment=baseline/pentominos_wae
# uv run python -m scripts.train experiment=baseline/pentominos_sa
