#!/bin/bash

#------------------- Fixed rotation transform to and from novel rotation angles  -----------------

# python -m bin.train experiment=slotae_pentomino_fixed_rot_transform

python -m bin.train experiment=fgseg_pentomino_fixed_rot_transform

# python -m bin.train experiment=wae_pentomino_fixed_rot_transform
