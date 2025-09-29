import os
import argparse
import pandas as pd
from pathlib import Path
from datasets import Dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file1", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--json_file2", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--user_id", type=str, required=True, help="User ID for the dataset")
    parser.add_argument("--sample_idx", type=int, required=True, help="User ID for the dataset")
    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_args()
    json_file1 = args.json_file1
    json_file2 = args.json_file2
    user_id = args.user_id
    sample_idx = args.sample_idx

    label_key = "flexible-extract"#"strict-match"

    # Load the JSONL file into a pandas DataFrame
    df = pd.read_json(json_file1, lines=True)

    # Filter the DataFrame to only include rows where the "filter" column is "strict-match"
    df = df[df["filter"]==label_key].reset_index()

    # Extract relevant columns and rename them
    df["prompt"] = df["doc"].apply(lambda x: x["question"])
    df["chat_prompt"] = df["arguments"].apply(lambda x: x["gen_args_0"]["arg_0"])
    df["responses"] = df["resps"].apply(lambda r: r[0][0])
    df["filtered_responses"] = df["filtered_resps"].apply(lambda r: r[0])
    cols_to_remove = [c for c in df.columns if c not in ["prompt", "chat_prompt", "target", "exact_match", "responses", "filtered_responses"]]
    df = df.drop(cols_to_remove, axis=1)

    # Load the second JSONL file into a pandas DataFrame
    df_ll = pd.read_json(json_file2, lines=True)
    df_ll["loglikelihoods"] = df_ll["resps"].apply(lambda x: x[0][0][0])
    df_ll = df_ll.rename(columns={"target": "responses"})
    cols_to_remove = [c for c in df_ll.columns if c not in ["responses", "loglikelihoods"]]
    df_ll = df_ll.drop(cols_to_remove, axis=1)
    
    # Merge the two DataFrames on the "responses" column
    merged_df = pd.merge(df, df_ll.drop_duplicates(subset=['responses']), on="responses")
    hf_dataset = Dataset.from_pandas(merged_df)
    
    # Save the new dataset to disk
    model_id = (Path(json_file1).parent).name
    task = (Path(json_file1).parent.parent.parent).name 
    print("TASK:", task)
    print("MODEL ID:", model_id)
    hf_dataset.push_to_hub(f"{user_id}/ot-resampling-{task}-{model_id}-i{sample_idx}", private=True)
