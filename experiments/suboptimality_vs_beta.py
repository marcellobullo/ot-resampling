import sys, os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(FILE_DIR)
sys.path.insert(0, ROOT)

import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm, trange
from sampling_utils import get_df, get_estimated_df, get_youden_bounds, compute_s, get_membership_fn, rejection_sampling, maximal_coupling, AiC, normalize_column

def parse_args():
    parser = argparse.ArgumentParser(description="Train Pluralistic Reward Model with configurable parameters.")
    parser.add_argument("--user_id", type=str, required=True)
    parser.add_argument("--models", type=str, nargs="+", required=True, help="List of model names to show")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--i", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_beta_points", type=int, default=20)
    parser.add_argument("--num_episodes", type=int, default=1e2)
    return parser.parse_args()

def compute_shat_heuristic(s, fraction):
    return fraction*s

def compute_J_heuristic(s_list, shat_list, fraction):
    upJs = []
    for s, shat in zip(s_list, shat_list):
        _, upJ = get_youden_bounds(s, shat)
        upJs.append(upJ)  
    J = min(upJs)
    return fraction*J

def get_compute_reward(true_df):
    check_true_memebership = get_membership_fn(true_df, "exact_match")
    def compute_reward(sample_id):
        return check_true_memebership(sample_id).astype(int)
    return compute_reward



if __name__ == "__main__":

    # Arguments
    args = parse_args()
    user_id = args.user_id
    model_ids = args.models
    task = args.task
    sample_id = args.i
    seed = args.seed
    num_beta_points = args.num_beta_points
    num_episodes = args.num_episodes

    fraction = 0.8

    # Compute a J feasible for all the models
    s_list, shat_list, true_dfs = [], [], []
    for model_id in model_ids:
        
        # Get the full dataset
        true_df = get_df(user_id, task, model_id.replace("/", "__"), sample_id)
        true_dfs.append(true_df)

        # Compute s and shat
        s = compute_s(true_df, "exact_match")
        print("S:", s)
        s_list.append(s)

        shat = compute_shat_heuristic(s, fraction)
        shat_list.append(shat)
    J = compute_J_heuristic(s_list, shat_list, fraction)

    # Compute shared betas
    mins = min(s_list)
    minshat = min(shat_list)
    betas = np.linspace(1.0, max(1/minshat, 1/mins)*1.3, num_beta_points)

    # Paramters
    rng = np.random.default_rng(seed)
    result_list = []

    for i, (model_id, s, shat, true_df) in enumerate(zip(model_ids, s_list, shat_list, true_dfs)):

        # True df
        check_true_memebership = get_membership_fn(true_df, "exact_match")
        compute_reward = get_compute_reward(true_df)

        # Construct Shat set (estimated_df)
        estimated_shat, estimated_J, estimated_df = get_estimated_df(J, shat, user_id, task, model_id.replace("/", "__"), sample_id)

        # Dataframes
        dfs = {
            "ground_truth": true_df,
            "estimate": estimated_df
        }

        ss = {
            "ground_truth": s,
            "estimate": estimated_shat    
        }

        # Compute
        for beta in tqdm(betas, desc=f"{model_id} - J: {J} - shat {shat}"):

            for mode, df in dfs.items():
                
                # Rejection sampling
                rewards, iters = [], []
                for t in trange(num_episodes, desc="SRS", leave=False):
                    res = rejection_sampling(df, beta, ss[mode], rng=rng, max_iter=1e6)
                    reward = compute_reward(res["sample_id"])
                    rewards.append(reward)
                    iters.append(res["iter"])
                results = {
                    "beta": beta,
                    "model_id": model_id,
                    "s": ss[mode],
                    "reward": np.mean(rewards),
                    "reward_std": np.std(rewards),
                    "iter": np.mean(iters),
                    "iter_std": np.std(iters),
                    "method": "SRS",
                    "mode": mode,
                    "J": estimated_J,
                }
                result_list.append(results)

                # Maximal Coupling
                rewards, iters = [], []
                for t in trange(num_episodes, desc="SMC", leave=False):
                    res = maximal_coupling(df, beta, ss[mode], rng=rng, max_iter=1e6)
                    reward = compute_reward(res["sample_id"])
                    rewards.append(reward)
                    iters.append(res["iter"])
                results = {
                    "beta": beta,
                    "model_id": model_id,
                    "s": ss[mode],
                    "reward": np.mean(rewards),
                    "reward_std": np.std(rewards),
                    "iter": np.mean(iters),
                    "iter_std": np.std(iters),
                    "method": "SMC",
                    "mode": mode,
                    "J": estimated_J,
                }
                result_list.append(results)


                # Accept if Correct (AiC)
                rewards, iters = [], []
                for t in trange(num_episodes, desc="AiC", leave=False):
                    res = AiC(df, rng=rng, max_iter=1e6)
                    reward = compute_reward(res["sample_id"])
                    rewards.append(reward)
                    iters.append(res["iter"])
                results = {
                    "beta": beta,
                    "model_id": model_id,
                    "s": ss[mode],
                    "reward": np.mean(rewards),
                    "reward_std": np.std(rewards),
                    "iter": np.mean(iters),
                    "iter_std": np.std(iters),
                    "method": "AiC",
                    "mode": mode,
                    "J": estimated_J,
                }
                result_list.append(results)

    # fig, axes = plt.subplots(1,2, figsize=(4,8))
    result_df = pd.DataFrame(result_list)

    model_names_flat = ""
    for model_id in model_ids: model_names_flat += model_id.replace("/", "__")

    new_dir = os.path.join(ROOT, f"tasks/{task}/results-{sample_id}/suboptimality_vs_beta")
    os.makedirs(new_dir, exist_ok=True)
    result_df_path = os.path.join(new_dir, f"{task}-{model_names_flat}-episodes{num_episodes}-i{sample_id}.csv")
    result_df.to_csv(result_df_path, index=False)