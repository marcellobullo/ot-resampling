#!/bin/bash

TASK=gsm8k
USER_ID=marcellobullo
SAMPLE_IDS=(7 8)
SEED=14

export HF_HUB_CACHE="/hdd/mb1921/.cache/huggingface/hub"

#MODELS=(Qwen/Qwen3-1.7B Qwen/Qwen3-8B Qwen/Qwen3-14B meta-llama/Llama-3.1-8B-Instruct meta-llama/Llama-3.2-3B-Instruct meta-llama/Llama-3.2-1B-Instruct google/gemma-3-12b-it google/gemma-3-4b-it google/gemma-3-1b-it)
MODELS=(google/gemma-3-4b-it Qwen/Qwen3-14B meta-llama/Llama-3.1-8B-Instruct)


for SAMPLE_ID in "${SAMPLE_IDS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    echo -e "\n\n########## Scoring model: $MODEL ##########"

    accelerate launch \
    --config_file /home/mb1921/ot-resampling/accelerate_config.yaml \
    skywork_scoring.py \
    --user_id $USER_ID \
    --task $TASK \
    --model_id $MODEL \
    --i $SAMPLE_ID 
  done
done
