import sys, os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(FILE_DIR)
sys.path.insert(0, ROOT)

import numpy as np
from sampling_utils import (
    ci,
    softmax,
    get_membership_fn,
    LR_calculator,
)

def rejection_sampling(df, beta, s, rng=None, max_iter=1e6):
    if rng is None: rng=np.random.default_rng()

    # Compute likelihood ratio
    p, q = LR_calculator(beta, s)

    # Compute scaling factor
    M = max(p, q)
    assert M > 0

    # Compute weights and membership function
    check_membership = get_membership_fn(df, 'exact_match')
    q_weights = softmax(df["loglikelihoods"].values)

    # Rejection sampling
    iter = 0
    done = False
    while not done:
        
        # Check max iterations
        if iter > max_iter:
            done = True

        # Sample from reference policy
        sample_id = rng.choice(df.index.values, p=q_weights)

        # Check membership through verifier
        is_member = check_membership(sample_id)
        
        # Rejection sampling acceptance step
        eta = (p if is_member else q)
        if rng.random() <= (1/M)*eta:
            done=True
        
        # Increase iteration count
        iter += 1
    
    return {
        "sample_id": sample_id,
        "is_member": is_member,
        "iter": iter,
        "p": p,
        "q": q,
    }

def batched_rejection_sampling(df, beta, s, batch_size, rng=None):
    if rng is None: rng=np.random.default_rng()

    # Compute likelihood ratio
    p, q = LR_calculator(beta, s)

    # Compute scaling factor
    M = max(p, q)
    assert M > 0

    # Compute weights and membership function
    check_membership = get_membership_fn(df, 'exact_match')
    q_weights = softmax(df["loglikelihoods"].values)

    # Sample bathc_size+1 samples
    sample_ids = rng.choice(df.index.values, size=batch_size+1, p=q_weights)

    # Rejection sampling
    for iter, sample_id in enumerate(sample_ids):

        # Check membership through verifier
        is_member = check_membership(sample_id)
    
        # Rejection sampling acceptance step
        eta = (p if is_member else q)
        if rng.random() <= (1/M)*eta:
            break
    
    #iter += 1
    
    return {
        "sample_id": sample_id,
        "is_member": is_member,
        "iter": batch_size+1,
        "p": p,
        "q": q,
    }

def BoN(df, beta, s, batch_size, rng=None):
    if rng is None: rng=np.random.default_rng()

    # Compute likelihood ratio
    p, q = LR_calculator(beta, s)

    # Compute scaling factor
    M = max(p, q)
    assert M > 0

    # Compute weights and membership function
    check_membership = get_membership_fn(df, 'exact_match')
    q_weights = softmax(df["loglikelihoods"].values)

    # Sample bathc_size+1 samples
    sample_ids = rng.choice(df.index.values, size=batch_size+1, p=q_weights)
    rng.shuffle(sample_ids)
    
    # BoN
    for sample_id in sample_ids:

        # Check membership through verifier
        is_member = check_membership(sample_id)
    
        # Accept if correct acceptance step
        if is_member:
            break
 
    return {
        "sample_id": sample_id,
        "is_member": is_member,
        "iter": batch_size+1,
        "p": p,
        "q": q,
    }


def adaptive_rejection_sampling(df, beta, rng=np.random.default_rng(), max_iter=1e6):

    # Compute weights and membership function
    N = len(df)
    check_membership = get_membership_fn(df, 'exact_match')
    q_weights = softmax(df["loglikelihoods"].values)

    # Rejection sampling
    iter = 0
    done = False
    s_running_estimate = 0
    while not done:

        # Compute likelihood ratio
        p, q = LR_calculator(beta, np.clip(s_running_estimate+ci(iter, s_running_estimate), 0, 1))

        # Compute scaling factor
        M = max(p, q)
        assert M > 0
        
        # Check max iterations
        if iter > max_iter:
            done = True

        # Sample from reference policy
        sample_id = rng.choice(df.index.values, p=q_weights)

        # Check membership through verifier
        is_member = check_membership(sample_id)
        
        # Rejection sampling acceptance step
        eta = (p if is_member else q)
        if rng.random() <= (1/M)*eta:
            done=True
        
        # Increase iteration count
        iter += 1

        s_running_estimate += (int(is_member)-s_running_estimate)/(iter)
    
    return {
        "sample_id": sample_id,
        "is_member": is_member,
        "iter": iter,
        "p": p,
        "q": q,
    }

def maximal_coupling(df, beta, s, rng=None, max_iter=1e6):
    if rng is None: rng = np.random.default_rng()
    
    # Compute likelihood ratio
    p, q = LR_calculator(beta, s)

    # Compute weights and membership function
    check_membership = get_membership_fn(df, 'exact_match')
    q_weights = softmax(df["loglikelihoods"].values)

    # Sample from reference policy
    sample_id = rng.choice(df.index.values, p=q_weights)
    is_member = check_membership(sample_id)
    eta = (p if is_member else q)
    n_iter = 1
    
    if eta < rng.random():
        res = AiC(df, rng=rng, max_iter=max_iter-n_iter)
        sample_id = res["sample_id"]
        is_member = check_membership(sample_id)
    
        #Increase iteration count
        n_iter += res["iter"]
    
    return {
        "sample_id": sample_id,
        "is_member": is_member,
        "iter": n_iter,
        "p": p,
        "q": q,
    }

def maximal_coupling_new(df, beta, s, rng=None, max_iter=1e6):
    if rng is None: rng = np.random.default_rng()

    # Likelihood ratio terms
    p, q = LR_calculator(beta, s)  # assumes these are scalars for this test

    check_membership = get_membership_fn(df, 'exact_match')
    q_weights = softmax(df["loglikelihoods"].values)

    # If no iterations allowed, return immediately
    if max_iter <= 0:
        return {
            "sample_id": None,
            "is_member": False,
            "iter": 0,
            "p": p,
            "q": q,
        }

    # First draw
    sample_id = rng.choice(df.index.values, p=q_weights)
    is_member = check_membership(sample_id)
    eta = p if is_member else q

    n_iter = 1  # we performed one draw

    # Accept with probability eta
    if rng.random() < eta:  # or <= if you prefer
        return {
            "sample_id": sample_id,
            "is_member": is_member,
            "iter": n_iter,
            "p": p,
            "q": q,
        }

    # ---- Continue with AiC for remaining budget ----
    remaining = max_iter - 1
    if remaining <= 0:
        # exhausted budget after first (rejected) draw
        return {
            "sample_id": sample_id,   # last tried
            "is_member": is_member,
            "iter": n_iter,           # exactly 1
            "p": p,
            "q": q,
        }

    res = AiC(df, rng=rng, max_iter=remaining)
    # Total iterations = 1 (first draw) + AiC draws actually taken
    n_iter += res["iter"]

    # Prefer AiC's is_member and sample_id directly
    sample_id = res["sample_id"]
    is_member = res["is_member"]

    return {
        "sample_id": sample_id,
        "is_member": is_member,
        "iter": n_iter,
        "p": p,
        "q": q,
    }

def random(df, beta=None, s=None, rng=None, max_iter=1e6):
    
    check_membership = get_membership_fn(df, 'exact_match')
    q_weights = softmax(df["loglikelihoods"].values)

    iter = 0
    done = False

    while not done:
        
        # Check max iterations
        if iter > max_iter:
            done = True

        # Sample from reference policy
        sample_id = rng.choice(df.index.values, p=q_weights)
        is_member = check_membership(sample_id)

        # Accept if correct
        if rng.random() >= 0.5:
            done=True
        
        # Increase iteration count
        iter += 1
    
    return {
        "sample_id": sample_id,
        "is_member": is_member,
        "iter": iter,
        "p": -1,
        "q": -1,
    }

def AiC(df, rng=None, max_iter=1e6):
    check_membership = get_membership_fn(df, 'exact_match')
    q_weights = softmax(df["loglikelihoods"].values)

    n_iter = 0
    sample_id = -1
    is_member = False

    while n_iter < max_iter:
        sample_id = rng.choice(df.index.values, p=q_weights)
        is_member = check_membership(sample_id)
        n_iter += 1
        if is_member:
            break

    return {
        "sample_id": sample_id,
        "is_member": is_member,
        "iter": n_iter,  # exact number of draws performed
        "p": -1,
        "q": -1,
    }