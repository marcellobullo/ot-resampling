#!/bin/bash

TASK=gsm8k
USER_ID=marcellobullo
#SAMPLE_IDS=(2 7 8)
SAMPLE_IDS=(2 7 8)
SEED=14
NUM_J_POINTS=10
NUM_EPISODES=5000

export HF_HUB_CACHE="/hdd/mb1921/.cache/huggingface/hub"

# QWEN_MODELS=(
#   Qwen/Qwen3-1.7B
#   Qwen/Qwen3-8B
#   Qwen/Qwen3-14B
# )
QWEN_MODELS=(Qwen/Qwen3-1.7B Qwen/Qwen3-8B Qwen/Qwen3-14B)
# LLAMA_MODELS=(
#   meta-llama/Llama-3.2-1B-Instruct
#   meta-llama/Llama-3.2-3B-Instruct
#   meta-llama/Llama-3.1-8B-Instruct
# )
# GEMMA_MODELS=(
#   google/gemma-3-1b-it
#   google/gemma-3-4b-it
#   #google/gemma-3-12b-it 
# )


#--models "${QWEN_MODELS[@]}" \
for SAMPLE_ID in "${SAMPLE_IDS[@]}"; do

  for MODEL in "${QWEN_MODELS[@]}"; do

    echo -e "\n########## $MODEL - SAMPLE: $SAMPLE_ID ##########"

    # Qwen
    python suboptimality_vs_J.py \
      --user_id $USER_ID \
      --task $TASK \
      --models "${MODEL}" \
      --i $SAMPLE_ID \
      --seed $SEED \
      --num_j_points $NUM_J_POINTS \
      --num_episodes $NUM_EPISODES 

    # # Llama
    # python suboptimality_vs_beta.py \
    #   --user_id $USER_ID \
    #   --task $TASK \
    #   --models "${LLAMA_MODELS[@]}" \
    #   --i $SAMPLE_ID \
    #   --seed $SEED \
    #   --num_beta_points $NUM_BETAS_POINTS \
    #   --num_episodes $NUM_EPISODES 

    # # Gemma
    # python suboptimality_vs_beta.py \
    #   --user_id $USER_ID \
    #   --task $TASK \
    #   --models "${GEMMA_MODELS[@]}" \
    #   --i $SAMPLE_ID \
    #   --seed $SEED \
    #   --num_beta_points $NUM_BETAS_POINTS \
    #   --num_episodes $NUM_EPISODES 
  done
done 


