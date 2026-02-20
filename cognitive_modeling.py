# # python cognitive_modeling.py > ./models/test_fitting/RW/Example_data_output.log 2>&1
# ### Modeling Demo

# - **File:** `Fig1.ipynb/cognitive_modeling.ipynb`
# - **Data:** Uses sample data for 3 human participants in `/example_data/`.
# - **Expected Output:** Parameter fits for alpha and beta, along with learning curve visualizations.
# - **Expected Run Time:** For parameter fitting with 3 participants, the run time is minimal but scales with the number of subjects and free parameters.

import numpy as np
import os
import tempfile, getpass, uuid
import time
from datetime import datetime

from bhv_analysis import *
from preprocessing import *

import pymc as pm
import pytensor
import pytensor.tensor as pt
import arviz as az
from pytensor import config as pt_config

import pandas as pd
import scipy.io as sio

from fitting_code import pad_to_matrix, aggregate_ll_by_subject, extract_individual_and_summary_params
import os, pytensor

# ====== Runtime environment (Windows/Linux compatible), specify writable compilation cache to reduce permission/lock conflicts ======
tmp_path = '/share1/home/sangtian/Temp'
os.makedirs(tmp_path, exist_ok=True)
run_cache = os.path.join(
    tmp_path, f"pytensor_cache_{getpass.getuser()}_{os.getpid()}_{uuid.uuid4().hex[:6]}"
)
os.environ["PYTENSOR_FLAGS"] = ",".join([
    f"base_compiledir={run_cache}",
    f"compiledir={run_cache}",
    "openmp=False"
])

pytensor.config.compiledir = f"{tmp_path}/{os.getenv('USER','user')}/pytensor_{os.getpid()}"

# If available, enable jax float64 (optional)
try:
    import jax
    from jax import config as jax_config
    jax_config.update("jax_enable_x64", True)
except Exception:
    pass

# ------------ A utility: Load data by type ------------
def load_data_by_type(data_type: str, root_dir: str, folder_name: str = None):
    """
    Returns:
        actions_list: List[np.ndarray[int8]]
        rewards_list: List[np.ndarray[int8]]
        masks_list:   List[np.ndarray[int8]]
        tag:          Short tag for file naming (e.g., 'human', 'monkey', 'agentH')
        mapping_used: Mapping actually used (for recording/debugging only)
    """
    if data_type == "Human":
        if folder_name is None:
            data_dir = os.path.join(root_dir, "LYL_37_subj_csv")
        else:
            data_dir = os.path.join(root_dir, folder_name)
        # Mapping consistent with previous: A/S: 1->0, 2->1; R: -1->0
        mapping = {'A': {1: 0, 2: 1}, 'S': {1: 0, 2: 1}, 'R': {-1: 0}}

        paths, csv_names = get_csv_files(data_dir)
        actions, rewards, _ = read_csv_raw_data(paths, delete_miss=True, mapping=mapping, pad=False)

        actions = [np.asarray(a, dtype=np.int8) for a in actions]
        rewards = [np.asarray(r, dtype=np.int8) for r in rewards]
        masks   = [np.ones(len(a), dtype=np.int8) for a in actions]
        return actions, rewards, masks, "human", mapping, csv_names

    elif data_type == "Monkey":
        if folder_name is None:
            data_dir = os.path.join(root_dir, "bhv_3m")
        else:
            data_dir = os.path.join(root_dir, folder_name)
        mapping = {'A': {1: 0, 2: 1}, 'S': {1: 0, 2: 1}, 'R': {-1: 0}}

        paths, csv_names = get_csv_files(data_dir)
        actions, rewards, _ = read_csv_raw_data(paths, delete_miss=True, mapping=mapping, pad=False)

        actions = [np.asarray(a, dtype=np.int8) for a in actions]
        rewards = [np.asarray(r, dtype=np.int8) for r in rewards]
        masks   = [np.ones(len(a), dtype=np.int8) for a in actions]
        return actions, rewards, masks, "monkey", mapping, csv_names

    elif data_type == "Agent":
        # best-agent (Default uses human agent; to switch to monkey agent, uncomment the line below and comment out the human line)
        agent_human_dir  = os.path.join(root_dir, folder_name)

        # Human agent (Default)
        data_dir = agent_human_dir
        tag = "sim"
        # If using Monkey agent:
        # data_dir = agent_monkey_dir; tag = "agentM"

        # Mapping for best agent (as per your example)
        mapping = {'A': {1: 1, 0: 0}, 'S': {1: 1, 0: 0}, 'R': {1: 1}}

        paths, csv_names = get_csv_files(data_dir)
        actions, rewards, _ = read_csv_raw_data(paths, delete_miss=True, mapping=mapping, pad=False)

        actions = [np.asarray(a, dtype=np.int8) for a in actions]
        rewards = [np.asarray(r, dtype=np.int8) for r in rewards]
        masks   = [np.ones(len(a), dtype=np.int8) for a in actions]
        return actions, rewards, masks, tag, mapping, csv_names

    else:
        raise ValueError("data_type must be 'Human' | 'Monkey' | 'Agent'")


def build_model_rw(
    human_actions, human_rewards, human_masks=None,
    draws=1000, tune=1000, chains=4, seed=42, target_accept=0.9,
    hierarchical: bool = True,
    beta_scale: float = 10.0,
    include_kappa: bool = True,          # New: Whether to include stickiness parameter kappa (Default True)
    save_zarr_path: str = None, save_summary_csv: str = None,
    save_ll_csv: str = None, save_ll_mat: str = None
):
    """
    Single function support:
      - hierarchical=True: Population hyperpriors + non-centered logistic-normal (Hierarchical)
      - hierarchical=False: Independent priors per subject (Non-hierarchical)

    Common features:
      - ragged -> padding
      - Single scan vectorized update for Q of all subjects (synchronously tracking last_a)
      - Single vectorized Bernoulli likelihood
      - JAX/Numpyro sampling (when available)
      - Online calculation of pointwise log-likelihood, aggregated to total loglik per subject

    include_kappa:
      - True: Q update includes a stickiness bias kappa when a_t = last_a; logits = beta*Q
      - False : 
    """
    import pandas as pd
    import scipy.io as sio

    S = len(human_actions)
    if human_masks is None:
        human_masks = [np.ones_like(a, dtype=np.int8) for a in human_actions]

    # ---------- 1) padding ----------
    actions_pad, _, _ = pad_to_matrix(human_actions, pad_value=0, dtype=np.int64)
    rewards_pad, _, _ = pad_to_matrix(human_rewards, pad_value=0, dtype=np.int64)
    masks_pad,   _, _ = pad_to_matrix(human_masks,  pad_value=0, dtype=np.int64)

    assert actions_pad.shape == rewards_pad.shape == masks_pad.shape
    T_max, S2 = actions_pad.shape
    assert S2 == S

    # Observed vector (numpy side): take actions for valid positions where t>=1
    valid_mask_np = masks_pad[1:, :] > 0
    observed_vec = actions_pad[1:, :][valid_mask_np].astype(np.int8)  # (K,)

    with pm.Model() as m:
        # ---------- 2) Priors ----------
        if hierarchical:
            # Population hyperpriors
            mu_alpha, sigma_alpha = pm.Normal("mu_alpha", 0.0, 1.0), pm.HalfNormal("sigma_alpha", 1.0)
            mu_beta,      sigma_beta      = pm.Normal("mu_beta",      0.0, 1.0), pm.HalfNormal("sigma_beta",      1.0)

            # Non-centered logistic-normal -> (0,1)
            def logistic01(mu, sigma, name, shape):
                z = pm.Normal(name + "_z", 0.0, 1.0, shape=shape)
                return pm.Deterministic(name, pm.math.sigmoid(mu + sigma * z))

            alpha_s = logistic01(mu_alpha, sigma_alpha, "alpha", shape=S)
            beta_raw_s  = logistic01(mu_beta,      sigma_beta,      "beta_raw",  shape=S)

            # Hierarchical prior for kappa in [-0.5, 0.5] (logistic->(0,1) then -0.5)
            if include_kappa:
                mu_kappa    = pm.Normal("mu_kappa", 0.0, 0.2)
                sigma_kappa = pm.HalfNormal("sigma_kappa", 0.2)
                z_kappa     = pm.Normal("z_kappa", 0.0, 1.0, shape=S)
                kappa_s     = pm.Deterministic("kappa", pm.math.sigmoid(mu_kappa + sigma_kappa * z_kappa) - 0.5)
            else:
                kappa_s = pt.zeros((S,), dtype="float32")  # or Q0.dtype

        else:
            # Non-hierarchical: Independent priors per subject
            alpha_s = pm.Beta("alpha", alpha=2.0, beta=2.0, shape=S)
            beta_raw_s  = pm.Beta("beta_raw",  alpha=2.0, beta=2.0, shape=S)

            if include_kappa:
                # kappa_raw in (0,1) -> kappa in (-0.5, 0.5)
                kappa_raw_s = pm.Beta("kappa_raw", alpha=2.0, beta=2.0, shape=S)
                kappa_s     = pm.Deterministic("kappa", kappa_raw_s - 0.5)
            else:
                kappa_s = pt.zeros((S,), dtype="float32")  # or Q0.dtype
                # pm.Deterministic("kappa", kappa_s)         # Optional: Ensure output with same name exists

        beta_s = pm.Deterministic("beta", beta_scale * beta_raw_s)

        # ---------- 3) Constant Tensors ----------
        actions_pt = pt.constant(actions_pad.astype("int64"))   # (T,S)
        rewards_pt = pt.constant(rewards_pad.astype("int64"))   # (T,S)
        masks_pt   = pt.constant(masks_pad.astype("int64"))     # (T,S)

        # Initial Q and initial last_a (set to -1, making first step stickiness=0)
        Q0      = pt.ones((S, 2), dtype=pt_config.floatX) * 0.5
        last_a0 = pt.full((S,), -1, dtype="int64")

        # ---------- 4) Vectorized step (do not update Q/last_a when mask==0) ----------
        def step(a_t, r_t, m_t, Q_prev, last_a_prev, alpha, p):
            """
            Pure RW: Only update chosen action Q[a], no counterfactual update for unchosen options, and no use of stickiness parameter p.
            Parameter p is kept only for API compatibility.

            a_t, r_t, m_t: (S,) int64
            Q_prev: (S,2) float
            last_a_prev: (S,)
            alpha: (S,) or scalar
            p: kept for API compatibility (unused here)
            """
            idx = pt.arange(a_t.shape[0], dtype="int64")
            r_f = pt.cast(r_t, Q_prev.dtype)

            # --- Only update chosen action (Pure RW) ---
            Qa  = Q_prev[idx, a_t]                 # (S,)
            QaN = Qa + alpha * (r_f - Qa)          # Q[a] <- Q[a] + alpha (r - Q[a])

            # --- stickiness: add p if repeated choice (and valid trial) ---
            repeated   = pt.eq(a_t, last_a_prev)                          # (S,)
            valid_trial = pt.neq(m_t, 0)                                  # (S,)
            add_bias   = pt.cast(repeated & valid_trial, Q_prev.dtype)    # (S,)
            QaN        = QaN + p * add_bias

            # Write back: Unchosen actions remain unchanged
            Q_cand = pt.set_subtensor(Q_prev[idx, a_t], QaN)

            # Apply update only on valid trials
            m_f   = pt.cast(m_t, Q_prev.dtype)
            Q_new = Q_prev * (1.0 - m_f)[:, None] + Q_cand * m_f[:, None]

            # last_a updates only on valid trials (although this function does not rely on last_a)
            valid_trial = pt.neq(m_t, 0)
            last_a_new  = pt.switch(valid_trial, a_t, last_a_prev)

            return Q_new, last_a_new


        (Qs, last_as), _ = pytensor.scan(
            fn=step,
            sequences=[actions_pt, rewards_pt, masks_pt],
            outputs_info=[Q0, last_a0],
            non_sequences=[alpha_s, kappa_s],
            strict=True,
        )
        # Qs: (T,S,2); last_as: (T,S)

        # ---------- 5) Probabilities, Vectorized Likelihood ----------
        Q_pred = Qs[:-1, :, :]          # (T-1,S,2)
        last_a = last_as[:-1, :]        # (T-1,S)

        # beta*Q
        logits = Q_pred * beta_s[None, :, None]
        log_p1 = logits[:, :, 1] - pt.logsumexp(logits, axis=2)
        p1     = pt.clip(pt.exp(log_p1), 1e-6, 1-1e-6)  # (T-1,S)

        Tm1, S_ = p1.shape
        p1_flat = p1.reshape((Tm1 * S_,))
        idx_flat = np.flatnonzero(valid_mask_np.ravel(order="C")).astype("int64")
        p_obs = pt.take(p1_flat, pt.constant(idx_flat))
        pm.Bernoulli("obs", p=p_obs, observed=observed_vec)

        # ---------- 6) Sampling ----------
        try:
            from pymc.sampling_jax import sample_numpyro_nuts
            idata = sample_numpyro_nuts(
                draws=draws, tune=tune, chains=chains,
                target_accept=target_accept, random_seed=seed,
                chain_method="parallel" if chains > 1 else "sequential"
            )
        except Exception:
            idata = pm.sample(
                draws=draws, tune=tune, chains=chains,
                target_accept=target_accept, random_seed=seed
            )

        # ---------- 7) Online calculation of pointwise log-likelihood ----------
        idata = pm.compute_log_likelihood(idata, model=m, var_names=["obs"])

    # ---------- 8) Aggregate to total loglik per subject ----------
    ll_per_subject = aggregate_ll_by_subject(idata, masks_pad=masks_pad, how="mean")

    # ---------- 9) Optional Save ----------
    if save_zarr_path is not None:
        az.InferenceData.to_zarr(idata, save_zarr_path)
    if save_summary_csv is not None:
        summary_df = az.summary(idata, hdi_prob=0.95)
        summary_df.to_csv(save_summary_csv, index=True, encoding="utf-8-sig")
    if save_ll_csv is not None:
        os.makedirs(os.path.dirname(save_ll_csv), exist_ok=True)
        pd.DataFrame({
            "subject_index": np.arange(S, dtype=int),
            "log_likelihood": ll_per_subject
        }).to_csv(save_ll_csv, index=False, encoding="utf-8-sig")
    if save_ll_mat is not None:
        os.makedirs(os.path.dirname(save_ll_mat), exist_ok=True)
        sio.savemat(save_ll_mat, {"log_likelihoods": ll_per_subject})

    return idata, ll_per_subject

if __name__ == "__main__":

    # for hierarchical in [True, False]:   # True=Hierarchical; False=Non-hierarchical
    hierarchical = True   # True=Hierarchical; False=Non-hierarchical
    # hierarchical = False   # True=Hierarchical; False=Non-hierarchical
    # include_kappa = False
    include_kappa = True
    out_prefix = 'RW'
    if include_kappa:
        param_names = ['alpha', 'beta', 'kappa']
    else:
        param_names = ['alpha', 'beta']
    print("Start processing")
    start_time = time.time()
    print("Start time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if hierarchical:
        print("Currently performing hierarchical sampling")
    else:
        print("Currently performing non-hierarchical sampling")

    # ------------ Config: Data root path & Select data type ------------
    real_csv_path = "./data"
    fitting_save_path = f'./models/test_fitting/{out_prefix}'

    data_type = "Human"   # Options: "Human" | "Monkey" | "Agent" | "Example_data"
    folder_name = "Example_data"

    # data_type = "Monkey"   
    # folder_name = "bhv_3m"

    # real_csv_path = f"./models/sim_csv_and_fitting/{out_prefix}"
    # fitting_save_path = real_csv_path
    # data_type = 'Agent'
    # # folder_name = "Monkey_nohier"
    # folder_name = "Human_nohier"
    # ------------ Load Data ------------
    # actions_list, rewards_list, masks_list, tag, mapping_used = load_data_by_type(data_type, real_csv_path)
    actions_list, rewards_list, masks_list, tag, mapping_used, subj_list = load_data_by_type(data_type, real_csv_path, folder_name=folder_name)
    S_total = len(actions_list)
    print(f"[{data_type}] Loading complete: #subjects = {S_total}; First 5 lengths = {[len(x) for x in actions_list[:5]]}")

    # If downsampling/debugging is needed, slice here; default is full amount
    acts = actions_list  # [:N]
    rews = rewards_list  # [:N]
    msks = masks_list    # [:N]
    S_use = len(acts)

    out_dir = f'{fitting_save_path}/{data_type}_{out_prefix}_{"hier" if hierarchical else "nohier"}_{"add_p" if include_kappa else "no_p" }_subj_{folder_name}_zarr'
    os.makedirs(out_dir, exist_ok=True)

    idata, ll_per_subject = build_model_rw(
        acts, rews, msks,
        draws=1000, tune=1000, chains=4, seed=42, target_accept=0.95,
        hierarchical=hierarchical,
        beta_scale=10.0,
        include_kappa=include_kappa,
        save_zarr_path=out_dir,
        save_summary_csv=os.path.join(out_dir, f'summary.csv'),
        save_ll_csv=os.path.join(out_dir, f'loglik_by_subject.csv'),
        save_ll_mat=os.path.join(out_dir, f'loglik_by_subject.mat'),
    )

    # Extract fitting results
    summary_csv_path = os.path.join(out_dir, 'summary.csv')
    summary_df = pd.read_csv(summary_csv_path)

    # Extract individual and population parameters
    new_df, summary_df = extract_individual_and_summary_params(summary_df, subj_list, param_names, save_path=out_dir)

    end_time = time.time()
    print("End time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Total time: {:.2f} seconds ({:.2f} minutes)".format(
        end_time - start_time, (end_time - start_time) / 60
    ))