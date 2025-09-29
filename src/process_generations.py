import os
import yaml
import argparse
import pandas as pd
from pathlib import Path
from datasets import Dataset

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--user_id", type=str, required=True, help="User ID for the dataset")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    json_file = args.json_file
    user_id = args.user_id

    label_key = "flexible-extract"#"strict-match"

    # Load the JSONL file into a pandas DataFrame
    df = pd.read_json(json_file, lines=True)

    # Filter the DataFrame to only include rows where the "filter" column is "strict-match"
    df = df[df["filter"]==label_key].reset_index()

    # Extract relevant columns and rename them
    df["prompt"] = df["doc"].apply(lambda x: x["question"])
    df["chat_prompt"] = df["arguments"].apply(lambda x: x["gen_args_0"]["arg_0"])
    df["responses"] = df["resps"].apply(lambda r: r[0][0])
    df["filtered_responses"] = df["filtered_resps"].apply(lambda r: r[0])
    cols_to_remove = [c for c in df.columns if c not in ["prompt", "chat_prompt", "target", "exact_match", "responses", "filtered_responses"]]
    df = df.drop(cols_to_remove, axis=1)

    # Create a new DataFrame
    ll_df = pd.DataFrame({
        "question": df["chat_prompt"],
        "completion": df["responses"],
    })

    ds = Dataset.from_pandas(ll_df)
    save_path = (Path(json_file).parent/"result_dataset").as_posix()
    #print(save_path)
    ds.push_to_hub(f"{user_id}/ot-resampling-temp", private=True)

    # # Write the new dataset path to the YAML file
    # yaml_path = (Path(json_file).parent.parent.parent/"gsm8k_ll.yaml").as_posix()
    # with open(yaml_path) as f:
    #     data = yaml.safe_load(f)

    # data["dataset_path"] = save_path
    # with open(yaml_path, "w") as f:
    #     yaml.safe_dump(data, f, sort_keys=False)