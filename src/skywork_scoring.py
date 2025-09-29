import sys, os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(FILE_DIR)
sys.path.insert(0, ROOT)


import torch
import argparse
from tqdm import tqdm
from peft import PeftConfig
from datasets import load_dataset
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def parse_args():
    parser = argparse.ArgumentParser(description="Train Pluralistic Reward Model with configurable parameters.")
    parser.add_argument("--user_id", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--i", type=int, required=True)
    return parser.parse_args()

def prepare_responses(example, tokenizer):
    
    conv = [{"role": "user", "content": example["prompt"]}, {"role": "assistant", "content": example["responses"]}]
    conv_formatted = tokenizer.apply_chat_template(conv, tokenize=False)
    
    # Remove duplicate bos if present
    if tokenizer.bos_token and conv_formatted.startswith(tokenizer.bos_token):
        conv_formatted = conv_formatted[len(tokenizer.bos_token):]
    
    return {"reward_input": conv_formatted}

if __name__ == "__main__":

    accelerator = Accelerator()

    # Arguments
    args = parse_args()
    user_id = args.user_id
    model_id = args.model_id
    task = args.task
    i = args.i
    # user_id = "marcellobullo"
    # model_id = "google/gemma-3-1b-it"
    # task = "gsm8k"
    # i = 2

    # Dataset
    batch_size = 32
    model_name = model_id.replace("/", "__")
    dataset_name = f"{user_id}/ot-resampling-{task}-{model_name}-i{i}"
    print(dataset_name)
    dataset = load_dataset(dataset_name, split="train")

    # Reward Model
    model_name = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        #device_map=device,
        #attn_implementation="flash_attention_2",
        #attn_implementation="eager",
        num_labels=1,
    )

    # Reward tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(example):
        return tokenizer(
            example["reward_input"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=1000,
            #return_tesnsors="pt"
        )

    with accelerator.main_process_first():

        # Tokenization
        accelerator.print("Tokenizing dataset...")
        dataset = dataset.map(lambda x: prepare_responses(x, tokenizer=tokenizer), desc="Preparing responses", batched=False)
        dataset = dataset.map(tokenize, desc="Preparing responses", num_proc=32)
        dataset.set_format(type="torch")

    # Dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Prepare model and dataloader with accelerator
    reward_model, dataloader = accelerator.prepare(reward_model, dataloader)

    scores = []
    for batch in tqdm(dataloader, desc="Scoring"):
        with torch.no_grad():

            # Scoring
            outputs = accelerator.unwrap_model(reward_model)(
                input_ids=batch[f"input_ids"].squeeze(),
                attention_mask=batch[f"attention_mask"].squeeze(),
            )
            score = outputs.logits.squeeze().tolist()
            #score = outputs.logits[0][0].item()
            
            all_scores = accelerator.gather_for_metrics(score)
            scores.extend(all_scores)

    if accelerator.is_main_process:
        print("Pushing scores to hub...")
        
        # Add scores to dataset
        dataset = dataset.remove_columns(["input_ids", "attention_mask", "reward_input"])
        dataset = dataset.add_column("skywork_score", scores)

        #dataset_name += "TEST"

        dataset.push_to_hub(
            dataset_name,
            private=True,
        )