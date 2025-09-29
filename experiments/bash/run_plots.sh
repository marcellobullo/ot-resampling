#!/bin/bash

TASK=gsm8k
USER_ID=marcellobullo
#SAMPLE_IDS=(2 7 8)
SAMPLE_IDS=(2)
SEED=14
NUM_BETAS_POINTS=20
NUM_EPISODES=5000
NUM_BATCH_SIZES=5

export HF_HUB_CACHE="/hdd/mb1921/.cache/huggingface/hub"

# QWEN_MODELS=(
#   Qwen/Qwen3-1.7B
#   Qwen/Qwen3-8B
#   Qwen/Qwen3-14B
# )
QWEN_MODELS=(Qwen/Qwen3-1.7B Qwen/Qwen3-8B Qwen/Qwen3-14B)
#QWEN_MODELS=(Qwen/Qwen3-14B)
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


# #--models "${QWEN_MODELS[@]}" \
# for SAMPLE_ID in "${SAMPLE_IDS[@]}"; do

#   for MODEL in "${QWEN_MODELS[@]}"; do

#     echo -e "\n########## $MODEL - SAMPLE: $SAMPLE_ID ##########"

#     # Qwen
#     python suboptimality_vs_beta.py \
#       --user_id $USER_ID \
#       --task $TASK \
#       --models "${MODEL}" \
#       --i $SAMPLE_ID \
#       --seed $SEED \
#       --num_beta_points $NUM_BETAS_POINTS \
#       --num_episodes $NUM_EPISODES 

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

# # Batched
# for SAMPLE_ID in "${SAMPLE_IDS[@]}"; do

#   for MODEL in "${QWEN_MODELS[@]}"; do

#     echo -e "\n########## $MODEL - SAMPLE: $SAMPLE_ID - BATCHED ##########"

#     # Qwen
#     python suboptimality_vs_beta_batch.py \
#       --user_id $USER_ID \
#       --task $TASK \
#       --models "${MODEL}" \
#       --i $SAMPLE_ID \
#       --seed $SEED \
#       --num_beta_points $NUM_BETAS_POINTS \
#       --num_episodes $NUM_EPISODES \
#       --num_batch_sizes $NUM_BATCH_SIZES

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

# Threshold
for SAMPLE_ID in "${SAMPLE_IDS[@]}"; do

  for MODEL in "${QWEN_MODELS[@]}"; do

    echo -e "\n########## $MODEL - SAMPLE: $SAMPLE_ID - THRESHOLDING ##########"

    # Qwen
    python suboptimality_vs_beta_threshold.py \
      --user_id $USER_ID \
      --task $TASK \
      --models "${MODEL}" \
      --i $SAMPLE_ID \
      --seed $SEED \
      --num_beta_points $NUM_BETAS_POINTS \
      --num_episodes $NUM_EPISODES \
      --threhsold 0.6

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


