#!/bin/bash

# shape and position
python -m bin.train experiment=supervised_ime_dsprites_shape2px

python -m bin.train experiment=supervised_ime_control_dsprites_shape2px

# shape and rotation
python -m bin.train experiment=supervised_ime_dsprites_heart2rot
