#!/bin/bash

export CUDA_VISIBLE_DEVICES=3


#------------------------------ Slot Attention Control heart-position ----------------------------

for i in {1..5}; do
  uv run python -m scripts.train experiment=combgen/dsprites_heart_position_fgdc
  uv run python -m scripts.train experiment=combgen/dsprites_heart_position_fgc
done

for i in {1..5}; do
  uv run python -m scripts.train experiment=combgen/dsprites_heart_position_fgdc model.slot_size=32
  uv run python -m scripts.train experiment=combgen/dsprites_heart_position_fgc model.slot_size=32
done

for i in {1..2}; do
  uv run python -m scripts.train experiment=combgen/dsprites_heart_position_fgdc model.slot_size=64
  uv run python -m scripts.train experiment=combgen/dsprites_heart_position_fgc model.slot_size=64
done


#-------------------------------- Slot Attention Control heart-rotation --------------------------

for i in {1..5}; do
  uv run python -m scripts.train experiment=combgen/dsprites_heart_rotation_fgdc
  uv run python -m scripts.train experiment=combgen/dsprites_heart_rotation_fgc
done

for i in {1..5}; do
  uv run python -m scripts.train experiment=combgen/dsprites_heart_rotation_fgdc model.slot_size=32
  uv run python -m scripts.train experiment=combgen/dsprites_heart_rotation_fgc model.slot_size=32
done

for i in {1..2}; do
  uv run python -m scripts.train experiment=combgen/dsprites_heart_rotation_fgdc model.slot_size=64
  uv run python -m scripts.train experiment=combgen/dsprites_heart_rotation_fgc model.slot_size=64
done
