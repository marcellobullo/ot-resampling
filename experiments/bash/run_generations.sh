#!/bin/bash

TASK=gsm8k
SYSTEM_INSTRUCTION="You are a helpful assistant that solves math problems. Think step by step. After reasoning, provide your answer in a separate line using the format: #### <answer>"
USER_ID=marcellobullo
DATASET_NAME=openai/gsm8k
DATA_DIR=main
SAMPLE_ID=8
NUM_SAMPLES=10000
PROMPT_KEY=question
LABEL_KEY=answer
SEED=14
INCLUDE_PATH="/home/mb1921/ot-resampling/tasks/$TASK"
OUTPUT_PATH="/home/mb1921/ot-resampling/tasks/$TASK/results-${SAMPLE_ID}"


export HF_HUB_CACHE="/hdd/mb1921/.cache/huggingface/hub"

#MODELS=(Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B Qwen/Qwen3-14B)
# declare -A BATCH_SIZE_BY_MODEL=(
#   ["Qwen/Qwen3-0.6B"]=128
#   ["Qwen/Qwen3-1.7B"]=128
#   ["Qwen/Qwen3-4B"]=64
#   ["Qwen/Qwen3-8B"]=32
#   ["Qwen/Qwen3-14B"]=32
# )

# MODELS=(meta-llama/Llama-3.1-8B-Instruct meta-llama/Llama-3.2-3B-Instruct meta-llama/Llama-3.2-1B-Instruct)
# declare -A BATCH_SIZE_BY_MODEL=(
#   ["meta-llama/Llama-3.2-1B-Instruct"]=128
#   ["meta-llama/Llama-3.2-3B-Instruct"]=64
#   ["meta-llama/Llama-3.1-8B-Instruct"]=32
# )

MODELS=(google/gemma-3-4b-it)
declare -A BATCH_SIZE_BY_MODEL=(
  ["google/gemma-3-1b-it"]=128
  ["google/gemma-3-4b-it"]=64
  ["google/gemma-3-12b-it"]=32
)


# This creates {USER_ID}/{dataset_name}-i{input_index} dataset on the hub
python create_dataset.py \
--user_id $USER_ID \
--dataset_name $DATASET_NAME \
--data_dir $DATA_DIR \
--i $SAMPLE_ID \
--num_samples $NUM_SAMPLES \
--prompt_key $PROMPT_KEY \
--label_key $LABEL_KEY \
--task $TASK

for MODEL in "${MODELS[@]}"; do
    echo -e "\n\n\n Running model: $MODEL with batch size ${BATCH_SIZE_BY_MODEL[$MODEL]}"
    
    # This creates samples_gsm8k_*.jsonl with generations
    accelerate launch \
    --multi_gpu \
    --num_processes 8 \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    -m lm_eval \
    --model hf --model_args parallelize=False,pretrained=$MODEL \
    --tasks $TASK \
    --include_path $INCLUDE_PATH \
    --output_path $OUTPUT_PATH \
    --log_samples \
    --seed $SEED \
    --system_instruction "$SYSTEM_INSTRUCTION" \
    --batch_size ${BATCH_SIZE_BY_MODEL[$MODEL]} \
    --apply_chat_template

    # Retrieve the path of the most recent generation file samples_gsm8k_*.jsonl (not the ll one)
    JSON_PATH1=$(ls -t /home/mb1921/ot-resampling/tasks/$TASK/results-${SAMPLE_ID}/${MODEL//\//__}/samples_gsm8k_*.jsonl \
    | grep -v "gsm8k_ll" \
    | head -n1)
    echo -e "\n\n\n JSON PATH 1: $JSON_PATH1"

    # Process generations to create dataset for ll evaluation
    python process_generations.py \
    --json_file $JSON_PATH1 \
    --user_id $USER_ID

    # Compute loglikelihoods
    accelerate launch \
    --multi_gpu \
    --num_processes 8 \
    --gpu_ids 0,1,2,3,4,5,6,7 \
    -m lm_eval \
    --model hf \
    --model_args parallelize=False,pretrained=$MODEL \
    --tasks "${TASK}_ll" \
    --include_path $INCLUDE_PATH \
    --output_path $OUTPUT_PATH \
    --log_samples \
    --seed $SEED,$SEED,$SEED \
    --batch_size 8 

    # Retrieve the path of the most recent generation file samples_gsm8k_ll*.jsonl
    JSON_PATH2=$(ls -t /home/mb1921/ot-resampling/tasks/$TASK/results-${SAMPLE_ID}/${MODEL//\//__}/samples_${TASK}_ll*.jsonl | head -n1)
    echo -e "\n\n\n JSON PATH 2: $JSON_PATH2"

    # Combine generations and loglikelihoods into a single file and 
    # push to the hub at {USER_ID}/{MODEL}-i{input_index}
    python add_loglikelihoods.py \
    --json_file1 $JSON_PATH1 \
    --json_file2 $JSON_PATH2 \
    --user_id "$USER_ID" \
    --sample_idx $SAMPLE_ID
done
