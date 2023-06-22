#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

UMAP_METRIC="euclidean"
UMAP_MIN_DIST=0.5
UMAP_N_NEIGHBORS=30


#-------------------------------- dSprites position generalisation -------------------------------
# DATASET="dsprites"
# FILTER_EXPR="( shape == 3 ) & ( posX < 0.5 )"
# RESULTS="data/results/figure1"

# MODEL="vae"
# CKPT="${RESULTS}/combgen_dsprites_heart_position_vae_on:2025-11-12_13-45/epoch=490-step=471360.ckpt"
# uv run python -m scripts.analyze_latents \
#     --dataset_name $DATASET \
#     --filter_expr "$FILTER_EXPR" \
#     --model_name $MODEL \
#     --model_checkpoint $CKPT \
#     --umap_n_neighbors $UMAP_N_NEIGHBORS \
#     --umap_min_dist $UMAP_MIN_DIST \
#     --umap_metric $UMAP_METRIC \
#     --include_test

# MODEL="wae"
# CKPT="${RESULTS}/combgen_dsprites_heart_position_wae_on:2025-11-20_09-21/epoch=357-step=343680.ckpt"
# uv run python -m scripts.analyze_latents \
#     --dataset_name $DATASET \
#     --filter_expr "$FILTER_EXPR" \
#     --model_name $MODEL \
#     --model_checkpoint $CKPT \
#     --umap_n_neighbors $UMAP_N_NEIGHBORS \
#     --umap_min_dist $UMAP_MIN_DIST \
#     --umap_metric $UMAP_METRIC \
#     --include_test

# MODEL="sa1"
# CKPT="${RESULTS}/combgen_dsprites_heart_position_sa_on:2025-11-19_19-16/epoch=517-step=497280.ckpt"
# uv run python -m scripts.analyze_latents \
#     --dataset_name $DATASET \
#     --filter_expr "$FILTER_EXPR" \
#     --model_name $MODEL \
#     --model_checkpoint $CKPT \
#     --umap_n_neighbors $UMAP_N_NEIGHBORS \
#     --umap_min_dist $UMAP_MIN_DIST \
#     --umap_metric $UMAP_METRIC \
#     --include_test


#-------------------------------- dSprites orientation generalisation -------------------------------
DATASET="dsprites"
FILTER_EXPR="( shape == 3 ) & ( orientation < np.pi )"
RESULTS="data/results/figure1"

MODEL="vae"
CKPT=${RESULTS}/combgen_dsprites_heart_rotation_vae_on:2025-11-27_00-02/epoch=500-step=480960.ckpt
uv run python -m scripts.analyze_latents \
    --dataset_name $DATASET \
    --filter_expr "$FILTER_EXPR" \
    --model_name $MODEL \
    --model_checkpoint $CKPT \
    --umap_n_neighbors $UMAP_N_NEIGHBORS \
    --umap_min_dist $UMAP_MIN_DIST \
    --umap_metric $UMAP_METRIC \
    --include_test

MODEL="wae"
CKPT=${RESULTS}/combgen_dsprites_heart_rotation_wae_on:2025-11-27_02-12/epoch=505-step=485760.ckpt
uv run python -m scripts.analyze_latents \
    --dataset_name $DATASET \
    --filter_expr "$FILTER_EXPR" \
    --model_name $MODEL \
    --model_checkpoint $CKPT \
    --umap_n_neighbors $UMAP_N_NEIGHBORS \
    --umap_min_dist $UMAP_MIN_DIST \
    --umap_metric $UMAP_METRIC \
    --include_test

MODEL="sa1"
CKPT=${RESULTS}/combgen_dsprites_heart_rotation_sa_on:2025-11-12_19-42/epoch=510-step=490560.ckpt
uv run python -m scripts.analyze_latents \
    --dataset_name $DATASET \
    --filter_expr "$FILTER_EXPR" \
    --model_name $MODEL \
    --model_checkpoint $CKPT \
    --umap_n_neighbors $UMAP_N_NEIGHBORS \
    --umap_min_dist $UMAP_MIN_DIST \
    --umap_metric $UMAP_METRIC \
    --include_test

#------------------------------- Pentominos rotation generalisation ------------------------------

# DATASET="pentominos"
# FILTER_EXPR="np.isin( shape , [1, 3, 5, 8] ) & ( angle >= 180 )"
# RESULTS="data/results/figure2"

# MODEL="vae"
# CKPT="${RESULTS}/combgen_pentominos_w_rotation_vae_on:2025-11-28_20-36/epoch=77-step=97500.ckpt"
# uv run python -m scripts.analyze_latents \
#     --dataset_name $DATASET \
#     --filter_expr "$FILTER_EXPR" \
#     --model_name $MODEL \
#     --model_checkpoint $CKPT \
#     --umap_n_neighbors $UMAP_N_NEIGHBORS \
#     --umap_min_dist $UMAP_MIN_DIST \
#     --umap_metric $UMAP_METRIC \
#     --include_test

# MODEL="wae"
# CKPT="${RESULTS}/combgen_pentominos_w_rotation_wae_on:2025-12-09_18-53/epoch=68-step=86250.ckpt"
# uv run python -m scripts.analyze_latents \
#     --dataset_name $DATASET \
#     --filter_expr "$FILTER_EXPR" \
#     --model_name $MODEL \
#     --model_checkpoint $CKPT \
#     --umap_n_neighbors $UMAP_N_NEIGHBORS \
#     --umap_min_dist $UMAP_MIN_DIST \
#     --umap_metric $UMAP_METRIC \
#     --include_test

# MODEL="sa1"
# CKPT="${RESULTS}/combgen_pentominos_w_rotation_sa_on:2025-11-27_05-15/epoch=397-step=497500.ckpt"
# uv run python -m scripts.analyze_latents \
#     --dataset_name $DATASET \
#     --filter_expr "$FILTER_EXPR" \
#     --model_name $MODEL \
#     --model_checkpoint $CKPT \
#     --umap_n_neighbors $UMAP_N_NEIGHBORS \
#     --umap_min_dist $UMAP_MIN_DIST \
#     --umap_metric $UMAP_METRIC \
#     --include_test
