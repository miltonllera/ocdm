#!/bin/bash

# Systematic architecture ablation on Pentominos (shape rotation held out):
#   - Models: wae, sa
#   - Broadcast widths: 8x8, 16x16
#   - Encoder/decoder tapering:
#       tapered:   16 -> 32 -> 64 (encoder) and reverse (decoder)
#       untapered: 64 -> 64 -> 64 (encoder and decoder)

source "$(dirname "$0")/gpu_queue.sh"

FAST_DEV_RUN=${FAST_DEV_RUN:-false}
MAX_STEPS=${MAX_STEPS:-500000}
LATENT_SIZE=${LATENT_SIZE:-64}

EXP_DIR=${EXP_DIR:-pentominos_arch_tests}
LOG_DIR=${LOG_DIR:-"logs/${EXP_DIR}/launcher"}
CONDITION_NAME="combgen_pentominos_shape_rotation"
FILTER_EXPR="( shape == 8 ) & (angle >= 180)"

models=("wae" "sa")
broadcasts=("8x8" "16x16")
tapers=("tapered" "untapered")

build_encoder_config() {
  local model="$1"
  local taper="$2"

  local convs
  if [[ "$taper" == "tapered" ]]; then
    convs="[conv,[16,4,2,1]],[relu],[conv,[32,4,2,1]],[relu],[conv,[64,4,2,1]],[relu]"
  else
    convs="[conv,[64,4,2,1]],[relu],[conv,[64,4,2,1]],[relu],[conv,[64,4,2,1]],[relu]"
  fi

  if [[ "$model" == "sa" ]]; then
    echo "[${convs},[permute,[0,2,3,1]],[posemb2d,{embed:cardinal}],[flatten,[1,2]],[layer_norm,[-1]],[linear,[64]],[relu]]"
  else
    echo "[${convs},[permute,[0,2,3,1]],[flatten],[layer_norm,[-1]],[linear,[64]],[relu]]"
  fi
}

build_decoder_config() {
  local model="$1"
  local broadcast="$2"
  local taper="$3"

  local out_ch=1
  if [[ "$model" == "sa" ]]; then
    out_ch=2
  fi

  local c1=64
  local c2=64
  if [[ "$taper" == "tapered" ]]; then
    c1=32
    c2=16
  fi

  if [[ "$broadcast" == "8x8" ]]; then
    echo "[[spatbroad,[8,8],{input_last:true}],[posemb2d,{embed:cardinal}],[permute,[0,3,1,2]],[tconv,[${c1},4,2,1]],[relu],[tconv,[${c2},4,2,1]],[relu],[tconv,[${out_ch},4,2,1]]]"
  else
    echo "[[spatbroad,[16,16],{input_last:true}],[posemb2d,{embed:cardinal}],[permute,[0,3,1,2]],[tconv,[${c1},4,2,1]],[relu],[tconv,[${c2},4,2,1]],[relu],[conv,[${out_ch},1,1,0]]]"
  fi
}

echo "========================================================================="
echo "Running Pentominos Architecture Tests: ${CONDITION_NAME}"
echo "Output folder: ${EXP_DIR}"
echo "========================================================================="

gpu_pool_init "${LOG_DIR}"

for model in "${models[@]}"; do
  for broadcast in "${broadcasts[@]}"; do
    for taper in "${tapers[@]}"; do
      encoder_cfg=$(build_encoder_config "$model" "$taper")
      decoder_cfg=$(build_decoder_config "$model" "$broadcast" "$taper")

      if [[ "$model" == "wae" ]]; then
        size_param="model.latent_size=${LATENT_SIZE}"
      else
        size_param="model.slot_size=${LATENT_SIZE}"
      fi

      run_name="${model}_b${broadcast}_${taper}"

      gpu_pool_run "${run_name}" \
        uv run python -m scripts.train \
        experiment="pentominos_${model}" \
        exp_dir="${EXP_DIR}" \
        condition_name="${CONDITION_NAME}" \
        model_name="${run_name}" \
        "${size_param}" \
        "model.encoder_config=${encoder_cfg}" \
        "model.decoder_config=${decoder_cfg}" \
        "dataset.held_out_filter=\"${FILTER_EXPR}\"" \
        "dataset.batch_size=256" \
        dataset.rebalance_dim=shape \
        trainer.max_steps="${MAX_STEPS}" \
        trainer.fast_dev_run="${FAST_DEV_RUN}"
    done
  done
done

gpu_pool_wait
