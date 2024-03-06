#!/bin/bash


#------------------------ Pretrained SA model on ood pentominos ------------------------------------

# PENTOMINO_PRETRAINED=data/logs/unsupervised/pentominos/slot_ae/new_shape/2023-04-23_21-21

# python -m bin.analyze analysis=slot_recons \
#   run_path=$PENTOMINO_PRETRAINED \
#   +dataset@overrides.dataset=large_pentominos \
#   +overrides.dataset.split_sizes="[0.00, 0.00, 1.00]" \
#   visualizations.slot_reconstruction.data_split=test \
#   logger.figure_logger.name="large-pentominos"

# python -m bin.analyze analysis=slot_recons \
#   run_path=$PENTOMINO_PRETRAINED \
#   +dataset@overrides.dataset=non_pentominos \
#   +overrides.dataset.split_sizes="[0.00, 0.00, 1.00]" \
#   visualizations.slot_reconstruction.data_split=test \
#   logger.figure_logger.name="non-pentominos"

# python -m bin.analyze analysis=slot_recons \
#   run_path=$PENTOMINO_PRETRAINED \
#   +dataset@overrides.dataset=large_non_pentominos \
#   +overrides.dataset.split_sizes="[0.00, 0.00, 1.00]" \
#   visualizations.slot_reconstruction.data_split=test \
#   logger.figure_logger.name="larger-non-pentominos"

#------------------------ Pretrained FG-Seg model on ood pentominos --------------------------------

# PENTOMINO_PRETRAINED=data/logs/unsupervised/pentominos/fgseg/baseline/2023-11-23_10-17

# python -m bin.analyze analysis=slot_recons \
#   run_path=$PENTOMINO_PRETRAINED \
#   +dataset@overrides.dataset=large_pentominos \
#   +overrides.dataset.split_sizes="[0.00, 0.00, 1.00]" \
#   visualizations.slot_reconstruction.data_split=test \
#   logger.figure_logger.name="large_pentominos"

# python -m bin.analyze analysis=slot_recons \
#   run_path=$PENTOMINO_PRETRAINED \
#   +dataset@overrides.dataset=non_pentominos \
#   +overrides.dataset.split_sizes="[0.00, 0.00, 1.00]" \
#   visualizations.slot_reconstruction.data_split=test \
#   logger.figure_logger.name="non-pentominos"

# python -m bin.analyze analysis=slot_recons \
#   run_path=$PENTOMINO_PRETRAINED \
#   +dataset@overrides.dataset=large_non_pentominos \
#   +overrides.dataset.split_sizes="[0.00, 0.00, 1.00]" \
#   visualizations.slot_reconstruction.data_split=test \
#   logger.figure_logger.name="larger-non-pentominos"


#------------------- Extrapolation on novel scales for all but one shape -------------------------


# FG_SEG_MODEL=data/logs/unsupervised/pentominos/fgseg/scale_combgen_from_loo/2023-12-07_16-16/

# python -m bin.analyze analysis=slot_recons \
#   run_path=$FG_SEG_MODEL \
#   visualizations.slot_reconstruction.data_split=test \
#   logger.figure_logger.name="test_set"


#--------------- Model trained with one excluded shapa, teseted on cannonical rotation -----------


FG_SEG_MODEL=data/logs/unsupervised/pentominos/fgseg/new_shape/2023-09-24_13-17

python -m bin.analyze analysis=slot_recons \
  run_path=$FG_SEG_MODEL \
  visualizations.slot_representation \
  +overrides.dataset.{split_condition=extrap,split_variant=rotated_cannonical_shape}
