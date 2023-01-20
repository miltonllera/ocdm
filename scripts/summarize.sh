#!/bin/bash


# Slot Attention

# SLOT_AE_3DSHAPES=data/logs/unsupervised/3dshapes/slot_ae/shape_to_ohue/2023-01-27_13-56
# # SLOT_AE_MPI3D=data/logs/unsupervised/mpi3d/slot_ae/cylinder_to_vertical_axis/2023-03-02_14_49
# SLOT_AE_DSPRITES=data/logs/unsupervised/dsprites/slot_ae/square_to_posX/2023-02-12_18-32

# python -m bin.summarize \
#   --model_folders $SLOT_AE_3DSHAPES $SLOT_AE_DSPRITES \
#   --to_aggregate "slot_ae_examples" \
#   --plot_sizes "3,2" "3,2" \
#   --save "plots" \
#   --name "slot_ae_all.png"

# # Slot Attention (attention control)

# SLOT_AE_3DSHAPES=data/logs/unsupervised/3dshapes/slot_attn_control/shape_to_ohue/2023-02-15_18-36/
# # SLOT_AE_MPI3D=data/logs/unsupervised/mpi3d/slot_attn_control/cylinder_to_vertical_axis/2023-02-21_19-47
# SLOT_AE_DSPRITES=data/logs/unsupervised/dsprites/slot_attn_control/square_to_posX/2023-02-21_22-23/

# python -m bin.summarize \
#   --model_folders $SLOT_AE_3DSHAPES $SLOT_AE_DSPRITES \
#   --to_aggregate "slot_ae_examples" \
#   --plot_sizes "3,2" "3,2" \
#   --save "plots" \
#   --name "slot_attn_control_all.png"


# # Slot Attention (decoder control)

# SLOT_AE_3DSHAPES=data/logs/unsupervised/3dshapes/slot_dec_control/shape_to_ohue/2023-02-16_14-12
# # SLOT_AE_MPI3D=data/logs/unsupervised/mpi3d/slot_dec_control/cylinder_to_vertical_axis/2023-02-21_23-10
# SLOT_AE_DSPRITES=data/logs/unsupervised/dsprites/slot_dec_control/square_to_posX/2023-02-22_00-05

# python -m bin.summarize \
#   --model_folders $SLOT_AE_3DSHAPES $SLOT_AE_DSPRITES \
#   --to_aggregate "slot_ae_examples" \
#   --plot_sizes "3,2" "3,2" \
#   --save "plots" \
#   --name "slot_dec_control_all.png"


# dSprites rotation comparisson

# SLOT_AE=data/logs/unsupervised/dsprites/slot_ae/heart_to_rot/2023-02-12_22-26
# # SLOT_AE_MPI3D=data/logs/unsupervised/mpi3d/slot_dec_control/cylinder_to_vertical_axis/2023-02-21_23-10
# WAE=data/logs/unsupervised/dsprites/wae/heart_to_rot/2023-02-25_19-17

# python -m bin.summarize \
#   --model_folders $SLOT_AE $WAE \
#   --titles "Slot Attention" "WAE" \
#   --to_aggregate "traversal_recons" \
#   --plot_sizes "5,2" "5,2" \
#   --save "plots" \
#   --name "slot_vs_wae_heart_to_rot.png"


# Pentominos (shape and rotation)

# SLOT_AE=data/logs/unsupervised/pentominos/slot_ae/shape_and_rotation/2023-04-22_18-23
# WAE=data/logs/unsupervised/pentominos/wae/shape_and_rotation/2023-04-23_13-22

# python -m bin.summarize \
#   --model_folders $SLOT_AE $WAE \
#   --titles "Slot Attention" "WAE" \
#   --to_aggregate "traversal_recons" \
#   --plot_sizes "5,2" "5,2" \
#   --save "plots" \
#   --name "slot_vs_wae_pentomino_shape_and_rotation.png"


# Pentominos (novel shape)

# SLOT_AE=data/logs/unsupervised/pentominos/slot_ae/new_shape/2023-04-23_21-21
# WAE=data/logs/unsupervised/pentominos/wae/new_shape/2023-04-24_09-51

# python -m bin.summarize \
#   --model_folders $SLOT_AE $WAE \
#   --titles "Slot Attention" "WAE" \
#   --to_aggregate "traversal_recons" \
#   --plot_sizes "2,2" "2,2" \
#   --save "plots" \
#   --name "slot_vs_wae_new_shape.png"


# Pentominos (three new shapes)

# SLOT_AE=data/logs/unsupervised/pentominos/slot_ae/three_new_shapes/
# WAE=data/logs/unsupervised/pentominos/wae/new_shape/

# python -m bin.summarize \
#   --model_folders $SLOT_AE $WAE \
#   --titles "Slot Attention" "WAE" \
#   --to_aggregate "traversal_recons" \
#   --plot_sizes "3,2" "3,2" \
#   --save "plots" \
#   --name "slot_vs_wae_new_shape.png"


# # Pentominos (novel shape)

# SLOT_AE=data/logs/unsupervised/pentominos/slot_ae/half_new_shapes/2023-04-27_09-03
# WAE=data/logs/unsupervised/pentominos/wae/new_shape/

# python -m bin.summarize \
#   --model_folders $SLOT_AE $WAE \
#   --titles "Slot Attention" "WAE" \
#   --to_aggregate "traversal_recons" \
#   --plot_sizes "3,2" "3,2" \
#   --save "plots" \
#   --name "slot_vs_wae_new_shape.png"


# Pentominos more than one shape, SA only

THREE_SHAPES=data/logs/unsupervised/pentominos/slot_ae/three_new_shapes/2023-04-26_16-08
SIX_SHAPES=data/logs/unsupervised/pentominos/slot_ae/half_new_shapes/2023-04-27_09-03/

python -m bin.summarize \
  --model_folders  $THREE_SHAPES $SIX_SHAPES \
  --titles "Three Novel Shapes" "Half Novel Shapes" \
  --to_aggregate "traversal_recons" \
  --plot_sizes "4,2" "2,2" \
  --save "plots" \
  --name "slotae-more-than-one-novel-shape.png"
