import sys, os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(FILE_DIR)
sys.path.insert(0, ROOT)

import yaml
import argparse
from datasets import load_dataset, Dataset, DatasetDict

def parse_args():
    parser = argparse.ArgumentParser(description="Create a dataset with repeated prompts")
    parser.add_argument("--user_id", type=str, required=True, help="Hugging Face user ID.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset path")
    parser.add_argument("--data_dir", type=str, help="Data directory")
    parser.add_argument("--i", type=int, required=True, help="Input sample index")
    parser.add_argument("--num_samples", type=int, required=True, help="Number of samples")
    parser.add_argument("--prompt_key", type=str, required=True, help="Name of the prompt column")
    parser.add_argument("--label_key", type=str, required=True, help="Name of the label column")
    parser.add_argument("--task", type=str, required=True, help="Name of the task")
    return parser.parse_args()

if __name__=="__main__":

    # Arguments
    args = parse_args()
    user_id = args.user_id
    data_dir = args.data_dir
    dataset_name = args.dataset_name
    input_index = args.i
    num_samples = args.num_samples
    prompt_key = args.prompt_key
    label_key = args.label_key
    task = args.task
    
    dataset = load_dataset(dataset_name, data_dir)
    test_input = dataset["test"][input_index]

    test_data = {
        "test": {
            prompt_key: [test_input[prompt_key]] * num_samples,
            label_key: [test_input[label_key]] * num_samples,
        },
        "fewshot": {
            prompt_key: dataset["train"][prompt_key],
            label_key: dataset["train"][label_key]
        }
    }
    
    # Wrap into a DatasetDict
    dataset = DatasetDict({
        "test": Dataset.from_dict(test_data["test"]),
        "fewshot": Dataset.from_dict(test_data["fewshot"])
    })

    dataset_name = dataset_name.split("/")[-1]
    dataset_id = f"{user_id}/{dataset_name}-i{input_index}"
    dataset.push_to_hub(dataset_id, private=True)

    # Write the new dataset path to the YAML file
    yaml_path = os.path.join(ROOT, f"tasks/{task}/{task}.yaml")
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    data["dataset_path"] = dataset_id
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)