#!/bin/bash

# # Shape and rotation

# python -m bin.train experiment=slotae_pentomino_shape_and_rotation
# python -m bin.train experiment=wae_pentomino_shape_and_rotation
# python -m bin.train experiment=vae_pentomino_shape_and_rotation


# # Rotation transformatoin

# python -m bin.train experiment=slotae_pentomino_fixed_rot_transform


# # New shape

# python -m bin.train experiment=slotae_pentomino_new_shape
# python -m bin.train experiment=wae_pentomino_new_shape


# Three new shapes

# python -m bin.train experiment=slotae_pentomino_three_new_shapes
# python -m bin.train experiment=wae_pentomino_three_new_shapes


# # Half novel shapes

# # python -m bin.train experiment=slotae_pentomino_half_new_shapes
# python -m bin.train experiment=wae_pentomino_half_new_shapes


# Novel scales for all but one shape

# python -m bin.train experiment=slotae_pentominios_scale_combgen_from_loo
python -m bin.train experiment=fgseg_pentominos_scale_combgen_from_loo
