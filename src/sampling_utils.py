import hashlib
import numpy as np
from datasets import load_dataset


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    w = np.exp(x - np.max(x))
    if not np.isfinite(w.sum()) or w.sum() <= 0:
        raise ValueError("Softmax normalization failed (sum not finite/positive).")
    return w / w.sum()

def get_df(user_id, task, model_id, sample_id):
    dataset_name = f"{user_id}/ot-resampling-{task}-{model_id}-i{sample_id}"
    dataset = load_dataset(dataset_name)
    return dataset["train"].to_pandas()

def get_estimated_df(J, shat, user_id, task, model_id, sample_id):
    df = get_df(user_id, task, model_id, sample_id)
    shat, J, Shat, _, _= construct_imperfect_verifier(df, shat, J)
    return shat, J, Shat

def get_estimated_df_threshold_fn(user_id, task, model_id, sample_id):
    
    df = get_df(user_id, task, model_id, sample_id)
    df = normalize_column(df, "skywork_score", 0, 1)
    df = df.rename(columns={"exact_match": "ground_truth"})
    
    def get_estimated_df_threshold(threshold):
        df["exact_match"] = (df["skywork_score_norm"] > threshold).astype(int)
        TPR, FPR = compute_tpr_fpr(df, "ground_truth", "exact_match")
        J = TPR-FPR
        shat = compute_s(df, "exact_match")
        return shat, J, df
    
    return get_estimated_df_threshold

def get_estimated_df_threshold(threshold, user_id, task, model_id, sample_id):
    df = get_df(user_id, task, model_id, sample_id)
    df = normalize_column(df, "skywork_score", 0, 1)
    df = df.rename(columns={"exact_match": "ground_truth"})
    df["exact_match"] = (df["skywork_score_norm"] > threshold).astype(int)
    TPR, FPR = compute_tpr_fpr(df, "ground_truth", "exact_match")
    J = TPR-FPR
    shat = compute_s(df, "exact_match")
    return shat, J, df

def compute_s(df, col):
    #print("ACCURACY:", df["exact_match"].mean())
    q_weights = softmax(df["loglikelihoods"].values)
    return float(q_weights[df[col].fillna(False).astype(bool)].sum())

def normalize_column(df, col, a=0, b=1):
    xmin, xmax = df[col].min(), df[col].max()
    if xmax == xmin:
        # avoid divide-by-zero: fill with midpoint (a+b)/2 or just 'a'
        df[col + "_norm"] = (a + b) / 2
    else:
        df[col + "_norm"] = a + (df[col] - xmin) * (b - a) / (xmax - xmin)
    return df

def compute_tpr_fpr(df, gt_col, pred_col):
    # Booleans
    # tp = ((df[gt_col] == 1) & (df[pred_col] == 1)).sum()
    # tn = ((df[gt_col] == 0) & (df[pred_col] == 0)).sum()
    # fp = ((df[gt_col] == 0) & (df[pred_col] == 1)).sum()
    # fn = ((df[gt_col] == 1) & (df[pred_col] == 0)).sum()

    tp = ((df[gt_col] == 1) & (df[pred_col] == 1))
    tn = ((df[gt_col] == 0) & (df[pred_col] == 0))
    fp = ((df[gt_col] == 0) & (df[pred_col] == 1))
    fn = ((df[gt_col] == 1) & (df[pred_col] == 0))

    q_weights = softmax(df["loglikelihoods"].values)
    q_weights_tp = q_weights[tp.fillna(False).astype(bool)].sum()
    q_weights_tn = q_weights[tn.fillna(False).astype(bool)].sum()
    q_weights_fp = q_weights[fp.fillna(False).astype(bool)].sum()
    q_weights_fn = q_weights[fn.fillna(False).astype(bool)].sum()

    tpr = q_weights_tp / (q_weights_tp + q_weights_fn) if (q_weights_tp + q_weights_fn) > 0 else 0   # True Positive Rate
    fpr = q_weights_fp / (q_weights_tn + q_weights_fp) if (q_weights_tn + q_weights_fp) > 0 else 0   # False Positive Rate

    # # Rates
    # tpr = tp / (tp + fn) if (tp + fn) > 0 else 0   # True Positive Rate
    # fpr = fp / (tn + fp) if (tn + fp) > 0 else 0   # True Negative Rate

    return tpr, fpr

def otc(beta, s):
    return np.minimum(np.sqrt(np.maximum(0, s*(1 - s)*(beta - 1))), 1 - s)

def get_membership_fn(df, col):
    """Return a callable is_in_S(i) that checks boolean membership from df[col]."""
    def check_membership(i):
        return df[col][i].astype(bool)
    return check_membership

# Likelihood Ratio 
def LR_calculator(beta, s):
    assert (beta >= 1) and (0 <= s <= 1)
    if s == 0:  
        return 0, 1  
    elif s == 1:
        return 1, 0
    else:
        m_star = min(1, s + np.sqrt(max(0, s*(1 - s)*(beta - 1))))
        p = m_star / s
        q = max(0, (1 - m_star) / (1 - s))
        return p, q
    
def compute_max_batch_size(beta, s):
    if beta >= (1-s)/s:
        return np.infty
    elif s*(1-s) <= beta <= (1-s)/s:
        return np.log(1-np.sqrt((beta*s)/(1-s)))/np.log(1-s)
    else:
        raise ValueError("Undetermined")
    

def ci(t, est):
    return (0.016/1)*np.sqrt((est*(1-est))/t) if t>0 else 1


def get_youden_bounds(s, shat):
    # J must ensure 0<=TPR<=1 and 0<=FPR<=1 with
    # TPR = shat + (1-s)J, FPR = shat - sJ.
    L = max(
        (shat-1)/s if s>0 else -np.inf,
        -shat/(1-s) if s<1 else -np.inf
    )
    U = min(
        shat/s if s>0 else np.inf,
        (1-shat)/(1 - s) if s<1 else np.inf
    )
    return L, U

def keys_uniform_01(df, seed=12345):
    """Deterministic U[0,1) keys per row. Uses SHA256(index||seed) to be order-stable."""
    idx = df.index.to_numpy()
    out = np.empty(len(idx), dtype=np.float64)
    salt = str(seed).encode("utf-8")
    for k, i in enumerate(idx):
        h = hashlib.sha256(str(i).encode("utf-8") + b"|" + salt).digest()
        # take first 8 bytes as uint64, map to [0,1)
        val = int.from_bytes(h[:8], "big") / 2**64
        out[k] = val
    return out

def select_by_qmass(
    indices,
    qw,
    keys,
    target_mass,
    order="qw_then_key",      
    mode="min_overshoot",     
    ensure_nonempty=False,  
):
    """
    Select a subset of `indices` whose total Q-mass approximates `target_mass`.

    Ordering:
      - "qw_then_key": primary sort by ascending qw, tie-break by key (recommended: finest mass steps).
      - "key":        sort by key only (your previous behavior).
      - "key_then_qw":stable sort by key, then by qw (keeps more randomness, reduces jumps).

    Mode:
      - "undershoot":    pick largest prefix with mass <= target (except possibly empty).
      - "min_overshoot": pick smallest prefix with mass >= target.
      - "fractional":    pick undershoot prefix, then include the next item with prob p to hit target in expectation
                         (coin is derived from that item’s key → deterministic).

    Returns:
      chosen_indices (np.int64 array), realized_mass (float)
    """
    indices = np.asarray(indices, dtype=int)
    if len(indices) == 0 or target_mass <= 0.0:
        return np.array([], dtype=int), 0.0

    total_mass = float(qw[indices].sum())
    if target_mass >= total_mass:
        # Exact/equal prints were causing confusion; treat equality as "take all".
        return indices.copy(), total_mass

    #  Build ordering 
    if order == "qw_then_key":
        # lexsort sorts by last key as primary
        # primary: qw ascending, secondary: key
        order_idx = np.lexsort((keys[indices], qw[indices]))
        # order_idx = np.argsort(qw[indices])
    elif order == "key":
        order_idx = np.argsort(keys[indices], kind="mergesort")
    elif order == "key_then_qw":
        # stable sort by key, then stable sort by qw (ascending)
        order1 = np.argsort(keys[indices], kind="mergesort")
        idx1 = indices[order1]
        order2 = np.argsort(qw[idx1], kind="mergesort")
        order_idx = order1[order2]
    else:
        raise ValueError(f"Unknown order='{order}'")

    idx_sorted = indices[order_idx]
    q_sorted = qw[idx_sorted]
    cum = np.cumsum(q_sorted)

    # Choose k according to mode 
    if mode == "undershoot":
        k = int(np.searchsorted(cum, target_mass, side="right"))  # first cum > target
        # use k-1 items so mass <= target (unless k==0)
        k = max(0, k - 1)
        if ensure_nonempty and k == 0:
            k = 1  # force one item if requested
        chosen = idx_sorted[:k]
        mass = float(cum[k-1]) if k > 0 else 0.0

    elif mode == "min_overshoot":
        k = int(np.searchsorted(cum, target_mass, side="left"))   # first cum >= target
        k = min(k + 1, len(cum))  # take up to k items so realized >= target
        chosen = idx_sorted[:k]
        mass = float(cum[k-1])

    elif mode == "fractional":
        # Start with the undershoot prefix
        k = int(np.searchsorted(cum, target_mass, side="right"))
        m_prev = float(cum[k-1]) if k > 0 else 0.0
        chosen = idx_sorted[:k] if k > 0 else np.array([], dtype=int)
        mass = m_prev

        # Consider the pivot item k (if any) to probabilistically hit target in expectation
        if k < len(idx_sorted):
            pivot = idx_sorted[k]
            w_piv = float(qw[pivot])
            gap = target_mass - m_prev
            p = 0.0 if w_piv <= 0.0 else np.clip(gap / w_piv, 0.0, 1.0)

            # Deterministic "coin": compare the item's key to p
            if float(keys[pivot]) < p:
                chosen = np.append(chosen, pivot)
                mass = m_prev + w_piv
        else:
            # If there is no pivot (target extremely close to total), keep the undershoot set
            pass
    else:
        raise ValueError(f"Unknown mode='{mode}'")

    # locate the next cumulative after the chosen prefix for diagnostics
    if len(cum) > 0:
        next_idx = min(len(cum)-1, len(chosen))
        prev_idx = max(0, len(chosen)-1)
        mass_before = float(cum[prev_idx]) if len(chosen) > 0 else 0.0
        mass_after  = float(cum[next_idx])

    return chosen.astype(int, copy=False), float(mass)

def construct_imperfect_verifier(df, shat, J, seed=1234):
    s = compute_s(df, "exact_match")

    # Sanity check
    lowJ, upJ = get_youden_bounds(s, shat)
    assert lowJ <= J <= upJ 

    # TPR, FPR
    TPR = shat + (1 - s) * J
    FPR = shat - s * J

    # Targets
    target_mass_positive = TPR*s
    target_mass_negative = FPR*(1-s)

    keys = keys_uniform_01(df, seed=seed)
    idx_pos = df[df["exact_match"]==1].index.values
    idx_neg = df[df["exact_match"]==0].index.values
    q_weights = softmax(df["loglikelihoods"].values)
    sel_pos, mass_pos = select_by_qmass(idx_pos, q_weights, keys, target_mass_positive)
    sel_neg, mass_neg = select_by_qmass(idx_neg, q_weights, keys, target_mass_negative)

    Shat_indexes = np.concatenate([sel_pos, sel_neg])
    df["exact_match"] = 0
    df.loc[Shat_indexes, "exact_match"] = 1

    shat = mass_pos + mass_neg
    TPR = mass_pos / s
    FPR = mass_neg / (1 - s)
    J = TPR - FPR

    return shat, J, df, TPR, FPR
