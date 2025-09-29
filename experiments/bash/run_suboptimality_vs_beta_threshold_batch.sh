#!/bin/bash

TASK=gsm8k
USER_ID=marcellobullo
SAMPLE_IDS=(2 7 8)
SEED=14
NUM_BETAS_POINTS=20
NUM_EPISODES=5000

declare -A THRESHOLDS_BY_MODEL=(
  ["google/gemma-3-4b-it"]=0.6
  ["meta-llama/Llama-3.1-8B-Instruct"]=0.68
)

export HF_HUB_CACHE="/hdd/mb1921/.cache/huggingface/hub"

MODELS=(google/gemma-3-4b-it meta-llama/Llama-3.1-8B-Instruct)


# Batched
for SAMPLE_ID in "${SAMPLE_IDS[@]}"; do

  for MODEL in "${MODELS[@]}"; do

    echo -e "\n########## $MODEL - SAMPLE: $SAMPLE_ID - BATCHED - THRESHOLD: ${THRESHOLDS_BY_MODEL[$MODEL]} ##########"

    # Qwen
    python suboptimality_vs_beta_batch_threshold.py \
      --user_id $USER_ID \
      --task $TASK \
      --models "${MODEL}" \
      --i $SAMPLE_ID \
      --seed $SEED \
      --num_beta_points $NUM_BETAS_POINTS \
      --num_episodes $NUM_EPISODES \
      --threshold ${THRESHOLDS_BY_MODEL[$MODEL]}

    # # Gemma
    # python suboptimality_vs_beta.py \
    #   --user_id $USER_ID \
    #   --task $TASK \
    #   --models "${GEMMA_MODELS[@]}" \
    #   --i $SAMPLE_ID \
    #   --seed $SEED \
    #   --num_beta_points $NUM_BETAS_POINTS \
    #   --num_episodes $NUM_EPISODES 

    # # Llama
    # python suboptimality_vs_beta.py \
    #   --user_id $USER_ID \
    #   --task $TASK \
    #   --models "${LLAMA_MODELS[@]}" \
    #   --i $SAMPLE_ID \
    #   --seed $SEED \
    #   --num_beta_points $NUM_BETAS_POINTS \
    #   --num_episodes $NUM_EPISODES 


  done
done 

# # Threshold
# # Batched
# for SAMPLE_ID in "${SAMPLE_IDS[@]}"; do

#   for MODEL in "${QWEN_MODELS[@]}"; do

#     echo -e "\n########## $MODEL - SAMPLE: $SAMPLE_ID - THRESHOLDING ##########"

#     # Qwen
#     python suboptimality_vs_beta_threshold.py \
#       --user_id $USER_ID \
#       --task $TASK \
#       --models "${MODEL}" \
#       --i $SAMPLE_ID \
#       --seed $SEED \
#       --num_beta_points $NUM_BETAS_POINTS \
#       --num_episodes $NUM_EPISODES \
#       --threhsold 0.6

#     # # Llama
#     # python suboptimality_vs_beta.py \
#     #   --user_id $USER_ID \
#     #   --task $TASK \
#     #   --models "${LLAMA_MODELS[@]}" \
#     #   --i $SAMPLE_ID \
#     #   --seed $SEED \
#     #   --num_beta_points $NUM_BETAS_POINTS \
#     #   --num_episodes $NUM_EPISODES 

#     # # Gemma
#     # python suboptimality_vs_beta.py \
#     #   --user_id $USER_ID \
#     #   --task $TASK \
#     #   --models "${GEMMA_MODELS[@]}" \
#     #   --i $SAMPLE_ID \
#     #   --seed $SEED \
#     #   --num_beta_points $NUM_BETAS_POINTS \
#     #   --num_episodes $NUM_EPISODES 
#   done
# done 


