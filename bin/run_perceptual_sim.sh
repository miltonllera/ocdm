#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
RESULTS_FLDR=data/results

#-------------------------------- dSprites position generalisation -------------------------------

# FILTER_EXPR="( shape == 3 ) & ( posX < 0.5 )"

# CKPT=figure1/combgen_dsprites_heart_position_vae_on:2025-11-12_13-45/epoch=490-step=471360.ckpt
# uv run python -m scripts.perceptual_sim \
#   --dataset_name dsprites \
#   --filter_expr "${FILTER_EXPR}" \
#   --model_name vae \
#   --model_checkpoint "${RESULTS_FLDR}/${CKPT}"


# CKPT=figure1/combgen_dsprites_heart_position_wae_on:2025-11-20_09-21/epoch=357-step=343680.ckpt
# uv run python -m scripts.perceptual_sim \
#   --dataset_name dsprites \
#   --filter_expr "${FILTER_EXPR}" \
#   --model_name wae \
#   --model_checkpoint "${RESULTS_FLDR}/${CKPT}"


# CKPT=figure1/combgen_dsprites_heart_position_sa_on:2026-03-13_13-12/epoch=503-step=483840.ckpt
# uv run python -m scripts.perceptual_sim \
#   --dataset_name dsprites \
#   --filter_expr "${FILTER_EXPR}" \
#   --model_name sa1 \
#   --model_checkpoint "${RESULTS_FLDR}/${CKPT}"


#------------------------------ dSprites orientation generalisation ------------------------------

# FILTER_EXPR="( shape == 3 ) & ( orientation < np.pi )"

# CKPT=figure1/combgen_dsprites_heart_rotation_vae_on:2025-11-27_00-02/epoch=500-step=480960.ckpt
# uv run python -m scripts.perceptual_sim \
#   --dataset_name dsprites \
#   --filter_expr "${FILTER_EXPR}" \
#   --model_name vae \
#   --model_checkpoint "${RESULTS_FLDR}/${CKPT}"


# CKPT=figure1/combgen_dsprites_heart_rotation_wae_on:2025-11-27_02-12/epoch=505-step=485760.ckpt
# CKPT=figure1/combgen_dsprites_heart_rotation_wae_on:2025-11-26_17-35/epoch=472-step=454080.ckpt
# uv run python -m scripts.perceptual_sim \
#   --dataset_name dsprites \
#   --filter_expr "${FILTER_EXPR}" \
#   --model_name wae \
#   --model_checkpoint "${RESULTS_FLDR}/${CKPT}"


# CKPT=figure1/combgen_dsprites_heart_rotation_sa_on:2026-03-03_05-57/epoch=509-step=489600.ckpt
# CKPT=figure1/combgen_dsprites_heart_rotation_sa_on\:2026-03-03_16-45/epoch=501-step=481920.ckpt
# CKPT=figure1/combgen_dsprites_heart_rotation_sa_on:2026-03-03_13-21/epoch=517-step=497280.ckpt
# uv run python -m scripts.perceptual_sim \
#   --dataset_name dsprites \
#   --filter_expr "${FILTER_EXPR}" \
#   --model_name sa1 \
#   --model_checkpoint "${RESULTS_FLDR}/${CKPT}"


#------------------------------- 3DShapes object hue generalisation ------------------------------

FILTER_EXPR="( shape == 3 ) & ( object_hue > 0.5 )"


CKPT=figure1/combgen_shapes3d_shape_hue_vae_on:2025-11-19_14-21/epoch=144-step=97875.ckpt
uv run python -m scripts.perceptual_sim \
  --dataset_name shapes3d \
  --filter_expr "${FILTER_EXPR}" \
  --model_name vae \
  --model_checkpoint "${RESULTS_FLDR}/${CKPT}"


CKPT=figure1/combgen_shapes3d_shape_hue_wae_on:2025-11-19_16-58/epoch=147-step=99900.ckpt
uv run python -m scripts.perceptual_sim \
  --dataset_name shapes3d \
  --filter_expr "${FILTER_EXPR}" \
  --model_name wae \
  --model_checkpoint "${RESULTS_FLDR}/${CKPT}"


CKPT=figure1/combgen_shapes3d_shape_hue_sa_on:2025-11-19_18-02/epoch=147-step=99900.ckpt
uv run python -m scripts.perceptual_sim \
  --dataset_name shapes3d \
  --filter_expr "${FILTER_EXPR}" \
  --model_name sa \
  --model_checkpoint "${RESULTS_FLDR}/${CKPT}"


#------------------------------- Pentominos rotation generalisation ------------------------------

# FILTER_EXPR="np.isin( shape , [1, 3, 5, 8] ) & ( angle >= 180 )"

# CKPT=figure2/combgen_pentominos_w_rotation_vae_on:2025-12-11_14-09/epoch=394-step=493750.ckpt
# uv run python -m scripts.perceptual_sim \
#   --dataset_name pentominos \
#   --filter_expr "${FILTER_EXPR}" \
#   --model_name vae \
#   --model_checkpoint "${RESULTS_FLDR}/${CKPT}"


# CKPT=figure2/combgen_pentominos_w_rotation_wae_on:2025-12-11_16-16/epoch=390-step=488750.ckpt
# uv run python -m scripts.perceptual_sim \
#   --dataset_name pentominos \
#   --filter_expr "${FILTER_EXPR}" \
#   --model_name wae \
#   --model_checkpoint "${RESULTS_FLDR}/${CKPT}"


# CKPT=figure2/combgen_pentominos_rotation_sa_on:2026-03-07_22-58/epoch=399-step=500000.ckpt
# uv run python -m scripts.perceptual_sim \
#   --dataset_name pentominos \
#   --filter_expr "${FILTER_EXPR}" \
#   --model_name sa1 \
#   --model_checkpoint "${RESULTS_FLDR}/${CKPT}"
