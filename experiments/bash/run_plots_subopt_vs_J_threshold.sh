#!/bin/bash

TASK=gsm8k
USER_ID=marcellobullo
#SAMPLE_IDS=(2 7 8)
SAMPLE_IDS=(2)
SEED=14
NUM_J_POINTS=10
NUM_EPISODES=5000

export HF_HUB_CACHE="/hdd/mb1921/.cache/huggingface/hub"

MODELS=(google/gemma-3-4b-it meta-llama/Llama-3.1-8B-Instruct)

declare -A THRESHOLDS_BY_MODEL=(
  ["google/gemma-3-4b-it"]=0.57
  ["meta-llama/Llama-3.1-8B-Instruct"]=0.71
)

for SAMPLE_ID in "${SAMPLE_IDS[@]}"; do
  for MODEL in "${MODELS[@]}"; do

    echo -e "\n########## $MODEL - SAMPLE: $SAMPLE_ID - THRESHOLDING ##########"

    # Qwen
    python suboptimality_vs_J_threshold.py \
      --user_id $USER_ID \
      --task $TASK \
      --models "${MODEL}" \
      --i $SAMPLE_ID \
      --seed $SEED \
      --num_j_points $NUM_J_POINTS \
      --num_episodes $NUM_EPISODES \
      --threshold ${THRESHOLDS_BY_MODEL[$MODEL]}
  done
done 


