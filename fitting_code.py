import numpy as np
import os
import arviz as az
import pandas as pd

# -----------------------------
# Helper: ragged -> padding (T_max, S) + mask
# -----------------------------
def pad_to_matrix(list_of_1d_arrays, pad_value=0, dtype=np.int64):
    """
    Input: list of 1D np.ndarray (can be of different lengths)
    Output:
      mat: Shape (T_max, S), filled with pad_value
      mask: Shape (T_max, S), valid positions are 1, otherwise 0
      lengths: Original length for each subject
    """
    S = len(list_of_1d_arrays)
    lengths = [len(a) for a in list_of_1d_arrays]
    T_max = max(lengths) if lengths else 0
    mat = np.full((T_max, S), pad_value, dtype=dtype)
    mask = np.zeros((T_max, S), dtype=np.int8)
    for s, arr in enumerate(list_of_1d_arrays):
        L = len(arr)
        if L == 0:
            continue
        mat[:L, s] = np.asarray(arr, dtype=dtype)
        mask[:L, s] = 1
    return mat, mask, np.asarray(lengths, dtype=np.int64)


# -----------------------------
# Aggregate pointwise loglik into "total loglik per subject"
# -----------------------------
def aggregate_ll_by_subject(idata: az.InferenceData,
                            masks_pad: np.ndarray,
                            how: str = "mean") -> np.ndarray:
    """
    idata.log_likelihood['obs']: (chain, draw, K) or (draw, chain, K)
    The order of observation points is consistent with modeling: flatten (t>=1, s) in C-order, then subset with valid_mask.
    Returns: Total loglik per subject (S,)
    """
    if not hasattr(idata, "log_likelihood") or ("obs" not in idata.log_likelihood):
        raise RuntimeError("log_likelihood['obs'] not found. "
                           "Call pm.compute_log_likelihood(idata, model=m, var_names=['obs']) first.")
    ll = idata.log_likelihood["obs"].values  # (chain, draw, K) or (draw, chain, K)
    dims = idata.log_likelihood["obs"].dims
    
    # Unify dimensions to (chain, draw, K)
    if dims[0] == "draw":
        ll = np.swapaxes(ll, 0, 1)

    # Aggregate by sample to each observation point
    if how == "mean":
        ll_point = ll.mean(axis=(0, 1))  # (K,)
    elif how == "median":
        ll_point = np.median(ll, axis=(0, 1))
    else:
        raise ValueError("how must be 'mean' or 'median'")

    # Generate the subject index corresponding to each observation point
    valid_mask = masks_pad[1:, :] > 0  # (T-1, S)
    Tm1, S = valid_mask.shape
    subj_grid = np.tile(np.arange(S, dtype=np.int64), (Tm1, 1))  # (Tm1, S)
    subj_idx_for_point = subj_grid.ravel(order="C")[valid_mask.ravel(order="C")]  # (K,)

    ll_per_subject = np.zeros(S, dtype=np.float64)
    for s in range(S):
        ll_per_subject[s] = ll_point[subj_idx_for_point == s].sum()
    return ll_per_subject


# -----------------------------
# Extract fitting parameters
# -----------------------------
def extract_individual_and_summary_params(df, subj_list, param_names, save_path=None, human_subj_prefix='H_'):
    """
    Extract individual-level and population-level parameters from MCMC model output.

    Parameters:
        df: pandas.DataFrame, containing original parameter results (e.g., 'alpha[0]' and 'mu_alpha')
        subj_list: list, individual names (e.g., ['subj1', 'subj2', ...])
        param_names: list of str, prefixes of parameters of interest (e.g., ['alpha', 'beta'])
        save_path: Path to save the results

    Returns:
        new_df: Individual parameter table (columns are param and param_r_hat)
        summary_df: Population parameters (mu_xx, sigma_xx)
    """
    new_data = {'subj_name': subj_list}
    
    # Extract individual parameters
    for param in param_names:
        # Extract rows for this parameter, e.g., alpha[0], alpha[1], ...
        param_df = df[df['Unnamed: 0'].str.contains(rf'{param}\[\d+\]')].copy()
        param_df['index'] = param_df['Unnamed: 0'].str.extract(rf'{param}\[(\d+)\]').astype(int)
        
        # Extract mean and r_hat
        new_data[param] = [param_df.loc[param_df['index'] == i, 'mean'].values[0] for i in range(len(subj_list))]
        new_data[f'{param}_r_hat'] = [param_df.loc[param_df['index'] == i, 'r_hat'].values[0] for i in range(len(subj_list))]

    # Create a DataFrame containing individual parameters
    new_df = pd.DataFrame(new_data)
    new_df['Species'] = new_df['subj_name'].apply(lambda x: 'Human' if x.startswith(human_subj_prefix) else 'non-human')

    # Extract population parameters: mu_xx and sigma_xx
    target_rows = []
    for param in param_names:
        target_rows.extend([f'mu_{param}', f'sigma_{param}'])
    
    summary_df = df[df['Unnamed: 0'].isin(target_rows)].copy()

    # If a save path is provided, save to the specified path
    if save_path is not None:
        new_df.to_csv(os.path.join(save_path, 'fitting_individual_params.csv'), index=False)
        summary_df.to_csv(os.path.join(save_path, 'fitting_summary_params.csv'), index=False)
    
    return new_df, summary_df