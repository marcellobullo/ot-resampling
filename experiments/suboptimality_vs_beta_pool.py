import sys, os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(FILE_DIR)
sys.path.insert(0, ROOT)

import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm, trange
from sampling_utils import get_df, get_estimated_df, get_youden_bounds, compute_s, get_membership_fn, rejection_sampling, maximal_coupling, AiC

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


from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm  # or trange

# ---- Globals filled in by Pool initializer ----
_DF = None
_BETA = None
_S = None

def _init_worker(df, beta, s):
    global _DF, _BETA, _S
    _DF, _BETA, _S = df, beta, s

# ---- Worker functions (must be top-level) ----
def _episode_srs(seed):
    rng = np.random.default_rng(seed)
    res = rejection_sampling(_DF, _BETA, _S, rng=rng, max_iter=1e6)
    reward = compute_reward(res["sample_id"])
    return reward, res["iter"]

def _episode_smc(seed):
    rng = np.random.default_rng(seed)
    res = maximal_coupling(_DF, _BETA, _S, rng=rng, max_iter=1e6)
    reward = compute_reward(res["sample_id"])
    return reward, res["iter"]

def _episode_aic(seed):
    rng = np.random.default_rng(seed)
    res = AiC(_DF, rng=rng, max_iter=1e6)
    reward = compute_reward(res["sample_id"])
    return reward, res["iter"]

def _run_method(pool, worker_fn, num_episodes, desc, base_seed=12345, chunksize=64):
    # Deterministic, independent seeds per episode
    seeds = [base_seed for i in range(num_episodes)]
    rewards, iters = [], []
    with tqdm(total=num_episodes, desc=desc, leave=False) as pbar:
        for r, it in pool.imap_unordered(worker_fn, seeds, chunksize=chunksize):
            rewards.append(r)
            iters.append(it)
            pbar.update(1)
    return {
        "reward": float(np.mean(rewards)) if rewards else np.nan,
        "reward_std": float(np.std(rewards)) if rewards else np.nan,
        "iter": float(np.mean(iters)) if iters else np.nan,
        "iter_std": float(np.std(iters)) if iters else np.nan,
    }

# ---- Main loop ----
def run_all(dfs, ss, beta, model_id, estimated_J, num_episodes, base_seed=12345):

    result_list = []

    for mode, df in dfs.items():
        # (Re)use a single pool per mode, so df/beta/s are shipped once
        with Pool( initializer=_init_worker, initargs=(df, beta, ss[mode])) as pool:
            # SRS
            srs = _run_method(pool, _episode_srs, num_episodes, desc="SRS", base_seed=base_seed)
            result_list.append({
                "beta": beta, "model_id": model_id, "s": ss[mode],
                **srs, "method": "SRS", "mode": mode, "J": estimated_J,
            })

            # SMC
            smc = _run_method(pool, _episode_smc, num_episodes, desc="SMC", base_seed=base_seed+10_000)
            result_list.append({
                "beta": beta, "model_id": model_id, "s": ss[mode],
                **smc, "method": "SMC", "mode": mode, "J": estimated_J,
            })

            # AiC
            aic = _run_method(pool, _episode_aic, num_episodes, desc="AiC", base_seed=base_seed+20_000)
            result_list.append({
                "beta": beta, "model_id": model_id, "s": ss[mode],
                **aic, "method": "AiC", "mode": mode, "J": estimated_J,
            })

    return result_list



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

            results = run_all(dfs, ss, beta, model_id, estimated_J, num_episodes)
            result_list.extend(results)

    # fig, axes = plt.subplots(1,2, figsize=(4,8))
    result_df = pd.DataFrame(result_list)

    model_names_flat = ""
    for model_id in model_ids: model_names_flat += model_id.replace("/", "__")

    new_dir = os.path.join(ROOT, f"tasks/{task}/results-{sample_id}/suboptimality_vs_beta")
    os.makedirs(new_dir, exist_ok=True)
    result_df_path = os.path.join(new_dir, f"{task}-{model_names_flat}-episodes{num_episodes}-i{sample_id}.csv")
    result_df.to_csv(result_df_path, index=False)