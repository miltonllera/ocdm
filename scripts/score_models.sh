##!/bin/bash

###------------------- Score Slot Attention reconstrucions -------------------------------------

#SA_3DSHAPES=data/logs/unsupervised/3dshapes/slot_ae/shape_to_ohue/2023-01-27_13-56
#SA_DSPRITES=data/logs/unsupervised/dsprites/slot_ae/square_to_posX/2023-02-12_18-32
#SLOT_AE_MPI3D=data/logs/unsupervised/mpi3d/slot_ae/cylinder_to_vertical_axis/2023-03-02_14_49

#python -m bin.analyze analysis=score_slot_recons run_path=$SA_DSPRITES
#python -m bin.analyze analysis=score_slot_recons run_path=$SA_3DSHAPES


###------------------- Score Slot Attention property prediction --------------------------------

#SA_DSPRITES_SHAPE=data/logs/supervised/dsprites/slot_pred/shape_from_shape2px/2023-05-20_21-13
#SA_DSPRITES_POSX=data/logs/supervised/dsprites/slot_pred/posx_from_shape2px/2023-05-21_18-30

#SA_3DSHAPES_SHAPE=data/logs/supervised/3dshapes/slot_pred/shape_from_shape_to_ohue/2023-04-18_12-30
#SA_3DSHAPES_OHUE=data/logs/supervised/3dshapes/slot_pred/ohue_from_shape_to_ohue/2023-05-19_19-29

#python -m bin.analyze analysis=property_pred_acc run_path=$SA_DSPRITES_SHAPE
#python -m bin.analyze analysis=property_pred_mse run_path=$SA_DSPRITES_POSX

#python -m bin.analyze analysis=property_pred_acc run_path=$SA_3DSHAPES_SHAPE
#python -m bin.analyze analysis=property_pred_mse run_path=$SA_3DSHAPES_OHUE


###---------------------- Score Slot Attention Controls prediction -----------------------------

#GMC_DSPRITES_SHAPE=data/logs/supervised/dsprites/slot_pred/shape_from_shape2px/2023-05-25_13-07
#GMC_DSPRITES_POSX=data/logs/supervised/dsprites/slot_pred/posx_from_shape2px/2023-05-25_14-16

#GMC_3DSHAPES_SHAPE=data/logs/supervised/3dshapes/slot_pred/shape_from_shape_to_ohue/2023-05-25_10-06
#GMC_3DSHAPES_OHUE=data/logs/supervised/3dshapes/slot_pred/ohue_from_shape_to_ohue/2023-05-25_11-09

#python -m bin.analyze analysis=property_pred_acc run_path=$GMC_DSPRITES_SHAPE
#python -m bin.analyze analysis=property_pred_mse run_path=$GMC_DSPRITES_POSX

#python -m bin.analyze analysis=property_pred_acc run_path=$GMC_3DSHAPES_SHAPE
#python -m bin.analyze analysis=property_pred_acc run_path=$GMC_3DSHAPES_OHUE

## Other control

#SDC_DSPRITES_SHAPE=data/logs/supervised/dsprites/slot_pred/shape_from_shape2px/2023-05-26_13-30
#SDC_DSPRITES_POSX=data/logs/supervised/dsprites/slot_pred/shape_from_shape2px/2023-05-25_17-39

#SDC_3DSHAPES_SHAPE=data/logs/supervised/3dshapes/slot_pred/shape_from_shape_to_ohue/2023-05-26_18-02
#SDC_3DSHAPES_OHUE=data/logs/supervised/3dshapes/slot_pred/ohue_from_shape_to_ohue/2023-05-26_18-49

#python -m bin.analyze analysis=property_pred_acc run_path=$SDC_DSPRITES_SHAPE
#python -m bin.analyze analysis=property_pred_acc run_path=$SDC_DSPRITES_POSX


#python -m bin.analyze analysis=property_pred_acc run_path=$SDC_3DSHAPES_SHAPE
#python -m bin.analyze analysis=property_pred_acc run_path=$SDC_3DSHAPES_OHUE


##------------------ Pentomino dataset combinatorial generalization ----------------------------

#SA_PENTOMINO_SHAPE_ROT=data/logs/unsupervised/pentominos/slot_ae/shape_and_rotation/2023-04-22_18-23
#WAE_PENTOMINO_SHAPE_ROT=data/logs/unsupervised/pentominos/wae/shape_and_rotation/2023-04-23_13-22

#python -m bin.analyze analysis=score_slot_recons run_path=$SA_PENTOMINO_SHAPE_ROT
#python -m bin.analyze analysis=score_recons run_path=$WAE_PENTOMINO_SHAPE_ROT


#SA_PENTOMINO_SHAPE=data/logs/supervised/pentominos/slot_pred/shape_from_shape_and_rotation/2023-05-21_23-40
#SA_PENTOMINO_ROT=data/logs/supervised/pentominos/slot_pred/rotation_from_shape_and_rotation/2023-06-05_10-30

#python -m bin.analyze analysis=property_pred_acc run_path=$SA_PENTOMINO_SHAPE
#HYDRA_FULL_ERROR=1 python -m bin.analyze analysis=property_pred_mse run_path=$SA_PENTOMINO_ROT


#WAE_PENTOMINO_SHAPE=data/logs/supervised/pentominos/latent_pred/shape_from_shape_and_rotation/2023-06-05_14-07
#WAE_PENTOMINO_ROT=data/logs/supervised/pentominos/latent_pred/rotation_from_shape_and_rotation/2023-06-05_17-20/

#python -m bin.analyze analysis=property_pred_acc run_path=$WAE_PENTOMINO_SHAPE
#python -m bin.analyze analysis=property_pred_mse run_path=$WAE_PENTOMINO_ROT

##------------------ Pentomino dataset extrapolation -------------------------------------------

#SA_PENTOMINO_1SHAPE=data/logs/unsupervised/pentominos/slot_ae/new_shape/2023-04-23_21-21
#SA_PENTOMINO_3SHAPE=data/logs/unsupervised/pentominos/slot_ae/three_new_shapes/2023-04-26_16-08
#SA_PENTOMINO_6SHAPE=data/logs/unsupervised/pentominos/slot_ae/half_new_shapes/2023-04-27_09-03

#python -m bin.analyze analysis=score_slot_recons run_path=$SA_PENTOMINO_1SHAPE
#python -m bin.analyze analysis=score_slot_recons run_path=$SA_PENTOMINO_3SHAPE
#python -m bin.analyze analysis=score_slot_recons run_path=$SA_PENTOMINO_6SHAPE


#------------------------ Pretrained SA model on ood pentominos ------------------------------------

PENTOMINO_PRETRAINED=data/logs/unsupervised/pentominos/slot_ae/new_shape/2023-04-23_21-21

python -m bin.analyze analysis=slot_recons \
  run_path=$PENTOMINO_PRETRAINED \
  +dataset@overrides.dataset=large_pentominos \
  +overrides.dataset.split_sizes="[0.00, 0.00, 1.00]" \
  visualizations.slot_reconstruction.data_split=test \
  logger.figure_logger.name="large-pentominos"

python -m bin.analyze analysis=slot_recons \
  run_path=$PENTOMINO_PRETRAINED \
  +dataset@overrides.dataset=non_pentominos \
  +overrides.dataset.split_sizes="[0.00, 0.00, 1.00]" \
  visualizations.slot_reconstruction.data_split=test \
  logger.figure_logger.name="non-pentominos"

python -m bin.analyze analysis=slot_recons \
  run_path=$PENTOMINO_PRETRAINED \
  +dataset@overrides.dataset=large_non_pentominos \
  +overrides.dataset.split_sizes="[0.00, 0.00, 1.00]" \
  visualizations.slot_reconstruction.data_split=test \
  logger.figure_logger.name="larger-non-pentominos"

#------------------------ Pretrained FG-Seg model on ood pentominos --------------------------------

PENTOMINO_PRETRAINED=data/logs/unsupervised/pentominos/fgseg/baseline/2023-11-23_10-17

python -m bin.analyze analysis=slot_recons \
  run_path=$PENTOMINO_PRETRAINED \
  +dataset@overrides.dataset=large_pentominos \
  +overrides.dataset.split_sizes="[0.00, 0.00, 1.00]" \
  visualizations.slot_reconstruction.data_split=test \
  logger.figure_logger.name="large_pentominos"

python -m bin.analyze analysis=slot_recons \
  run_path=$PENTOMINO_PRETRAINED \
  +dataset@overrides.dataset=non_pentominos \
  +overrides.dataset.split_sizes="[0.00, 0.00, 1.00]" \
  visualizations.slot_reconstruction.data_split=test \
  logger.figure_logger.name="non-pentominos"

python -m bin.analyze analysis=slot_recons \
  run_path=$PENTOMINO_PRETRAINED \
  +dataset@overrides.dataset=large_non_pentominos \
  +overrides.dataset.split_sizes="[0.00, 0.00, 1.00]" \
  visualizations.slot_reconstruction.data_split=test \
  logger.figure_logger.name="larger-non-pentominos"
