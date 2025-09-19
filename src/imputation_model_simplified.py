import numpy as np
from tqdm import tqdm
import pandas as pd
from joblib import Parallel, delayed
from numpy.linalg import LinAlgError
from imputation_utils import *
from sklearn.cluster import KMeans
import statsmodels.api as sm
import tensorly as tl
from pykalman import KalmanFilter
from tensorly.decomposition import parafac
from tensorly.cp_tensor import cp_to_tensor
from tqdm import trange

############### Core Factor Imputation Model ####################

def conditional_mean(Sigma, mu, i_mask, i_data):
    '''
    return the conditional mean of missing data give the covariance matrix and mean of the characteristcs
    '''
    Sigma_11 = Sigma[~i_mask, :][:, ~i_mask]
    Sigma_12 = Sigma[~i_mask, :][:, i_mask]
    Sigma_22 = Sigma[i_mask, :][:, i_mask]
    mu1, mu2 = mu[~i_mask], mu[i_mask]
    
    conditional_mean = mu1 + Sigma_12 @ np.linalg.solve(Sigma_22, i_data[i_mask] - mu2)

    assert np.all(~np.isnan(conditional_mean))
    
    return conditional_mean


def get_optimal_A(B, A, present, cl, idxs=[], reg=0, 
                 mu=None,
                 resid_regs=None):
    """
    Get optimal A for cl = AB given that X is (potentially) missing data
    Parameters
    ----------
        B : matrix B
        A : matrix A, will be overwritten
        present: boolean mask of present data
        cl: matrix cl
        idxs: indexes which to impute
        reg: optinal regularization penalty
        min_chars: minimum number of entries to require present
        infer_lr_entries: optionally require fewer entries present, regress on first
            i columns of B where i is the number of observed entries
    """
    A[:,:] = np.nan
    for i in idxs:
        present_i = present[i,:]
        Xi = cl[i,:]
        Xi = Xi[present_i]
        Bi = B[:,present_i]
        assert np.all(~np.isnan(Bi)) and np.all(~np.isinf(Bi))
        effective_reg = reg 
        if resid_regs is None:
            lmbda = effective_reg * np.eye(Bi.shape[1])
        else:
            lmbda = np.diag(resid_regs[present_i])

        if mu is not None:
            Xi = Xi - mu[present_i]
        try:
            A[i,:] = Bi @ np.linalg.lstsq(Bi.T @ Bi + lmbda, Xi, rcond=0)[0]
        except LinAlgError as e:
            lmbda = np.eye(Bi.shape[1])
            A[i,:] = Bi @ np.linalg.lstsq(Bi.T @ Bi + lmbda, Xi, rcond=0)[0]
    return A


def _symmetrize(a):
    return 0.5 * (a + a.T)

def _eig_lambda_from_cov(cov, K, reg=0.0, eval_weight=True, shrink=False):
    """Return (L,K) lambda from an (L,L) covariance. Always finite."""
    if cov is None:
        return None
    cov = np.array(cov, dtype=float)
    L = cov.shape[0]
    if not np.all(np.isfinite(cov)):
        cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov = _symmetrize(cov)
    # numerical ridge to avoid tiny negative eigenvalues
    jitter = 1e-10 if reg == 0 else 0.0
    cov = cov + np.eye(L) * jitter
    w, v = np.linalg.eigh(cov)
    idx = np.argsort(np.abs(w))[::-1][:K]
    w = np.maximum(w[idx], 0.0)  # clip
    v = v[:, idx]
    if eval_weight:
        if shrink:
            scale = np.maximum(np.sqrt(np.sqrt(w)) - reg, 0.0)
        else:
            scale = np.sqrt(w)
        Lmb = v * scale.reshape(1, -1)
    else:
        Lmb = v
    Lmb[~np.isfinite(Lmb)] = 0.0
    return Lmb

# --- main: estimate_lambda -------------------------------------------------

def estimate_lambda(char_panel, return_panel, num_months_train, K, min_chars,
                    window_size=1, time_varying_lambdas=False,
                    mu=None, eval_weight_lmbda=True,
                    shrink_lmbda=False, reg=0.0):
    """
    Compute lambda via eigen-decomposition of characteristic covariances.
    Always returns finite lambda (list[T] if time_varying, else (L,K)) and a cov-matrix (or list of them).
    """
    # Build per-month covariances for the first `num_months_train` months
    T0 = int(num_months_train)
    T0 = max(1, min(T0, char_panel.shape[0]))
    cov_mats = []
    L = char_panel.shape[2]

    # robust covariance per month: ignore NaNs, center by mu[t] when given
    for t in range(T0):
        X = char_panel[t]  # (N,L)
        if mu is not None:
            m = np.asarray(mu[t]).reshape(1, L)
        else:
            m = np.nanmean(X, axis=0, keepdims=True)
            m[~np.isfinite(m)] = 0.0
        Xc = X - m
        Xc = np.nan_to_num(Xc, nan=0.0, posinf=0.0, neginf=0.0)
        # weight only rows with ≥ min_chars observed
        present = np.sum(~np.isnan(char_panel[t]), axis=1) >= min_chars
        Xw = Xc[present]
        if Xw.size == 0:
            cov = np.zeros((L, L))
        else:
            cov = (Xw.T @ Xw) / max(len(Xw), 1)
        cov_mats.append(cov)

    if time_varying_lambdas:
        # sliding window over available covs, produce λ for each month in training window
        lmbdas = []
        cov_ret = []
        for t in range(T0):
            lo = max(0, t - window_size + 1)
            win = cov_mats[lo:t+1]
            cov_sum = sum(win) / len(win)
            Lmb = _eig_lambda_from_cov(cov_sum, K, reg=reg,
                                       eval_weight=eval_weight_lmbda,
                                       shrink=shrink_lmbda)
            if Lmb is None:
                Lmb = np.zeros((L, K))
            lmbdas.append(Lmb)
            cov_ret.append(cov_sum)
        return lmbdas, cov_ret
    else:
        cov_avg = sum(cov_mats) / len(cov_mats)
        Lmb = _eig_lambda_from_cov(cov_avg, K, reg=reg,
                                   eval_weight=eval_weight_lmbda,
                                   shrink=shrink_lmbda)
        if Lmb is None:
            Lmb = np.zeros((L, K))
        return Lmb, cov_avg

# --- main: impute ----------------------------------------------------------

def impute_panel_xp_lp(char_panel, return_panel, min_chars, K, num_months_train,
                       reg=0.01,
                       time_varying_lambdas=False,
                       window_size=1,
                       n_iter=1,
                       eval_data=None,
                       allow_mean=False,
                       eval_weight_lmbda=True,
                       resid_reg=False,
                       shrink_lmbda=False,
                       run_in_parallel=True):
    """
    Cross-sectional imputation.
    """

    T, N, L = char_panel.shape

    X = np.copy(char_panel)

    # 1) Drop firm-months with < min_chars observed (proper 3D mask)
    present_counts = np.sum(~np.isnan(X), axis=2)            # (T,N)
    X = np.where(present_counts[..., None] < min_chars, np.nan, X)

    missing_mask_overall = np.isnan(X)
    imputed = np.copy(X)

    # 2) Initialize mean term
    mu = np.zeros((T, L), dtype=float)

    # 3) Iterations
    prev_lmbda = None
    for _ in range(max(1, n_iter)):
        # 3a) Estimate lambda robustly
        lmbda, cov_mat = estimate_lambda(
            imputed, return_panel,
            num_months_train=num_months_train,
            K=K, min_chars=min_chars, window_size=window_size,
            time_varying_lambdas=time_varying_lambdas,
            mu=mu if allow_mean else None,
            eval_weight_lmbda=eval_weight_lmbda,
            shrink_lmbda=shrink_lmbda, reg=reg
        )

        # ensure finite lambda
        if time_varying_lambdas:
            for t in range(len(lmbda)):
                if not np.all(np.isfinite(lmbda[t])):
                    lmbda[t] = prev_lmbda[t] if (prev_lmbda is not None and t < len(prev_lmbda)) else np.zeros((L, K))
        else:
            if not np.all(np.isfinite(lmbda)):
                lmbda = prev_lmbda if prev_lmbda is not None else np.zeros((L, K))
        prev_lmbda = lmbda if time_varying_lambdas else np.copy(lmbda)

        # 3b) Compute gammas for all months
        def _get_gamma_t(t):
            ct = X[t]                               # (N,L)
            present = ~np.isnan(ct)                 # (N,L)
            to_impute = np.where(np.sum(~np.isnan(ct), axis=1) >= min_chars)[0]
            if time_varying_lambdas:
                lt = lmbda[min(t, len(lmbda)-1)]    # (L,K)
                g0 = lt.T.dot(ct.T).T               # (N,K)
                g  = get_optimal_A(lt.T, g0, present, ct, idxs=to_impute, reg=reg,
                                    mu=mu[t] if allow_mean else None, resid_regs=None)
            else:
                g0 = lmbda.T.dot(ct.T).T
                g  = get_optimal_A(lmbda.T, g0, present, ct, idxs=to_impute, reg=reg,
                                    mu=mu[t] if allow_mean else None, resid_regs=None)
            return g

        if run_in_parallel:
            gammas = Parallel(n_jobs=min(30, T), verbose=0)(
                delayed(_get_gamma_t)(t) for t in range(T)
            )
        else:
            gammas = [_get_gamma_t(t) for t in range(T)]

        gamma_ts = np.full((T, N, K), np.nan, dtype=float)
        # place where defined (rows with enough chars)
        ok = present_counts >= min_chars
        for t in range(T):
            gamma_ts[t, ok[t]] = gammas[t][ok[t]]

        # 3c) New imputation from factor scores
        new_imp = np.empty_like(X, dtype=float)
        if time_varying_lambdas:
            for t in range(T):
                lt = lmbda[min(t, len(lmbda)-1)]
                new_imp[t] = gamma_ts[t] @ lt.T + (mu[t] if allow_mean else 0.0)
        else:
            LT = lmbda.T
            for t in range(T):
                new_imp[t] = gamma_ts[t] @ LT + (mu[t] if allow_mean else 0.0)

        # 3d) Optional residual regularization
        if resid_reg:
            resids = X - new_imp
            _ = np.square(np.nanstd(resids, axis=1))  # placeholder if you use it elsewhere

        # 3e) Fill only missing
        imputed[missing_mask_overall] = new_imp[missing_mask_overall]

        # 3f) Update mean if allowed
        if allow_mean:
            mu = np.nanmean(imputed, axis=1)
        else:
            mu[:] = 0.0

    return gamma_ts, lmbda



def get_cov_mat(char_matrix, mu=None):
    '''
    utility method to get the covariance matrix of a partially observed set of chatacteristics
    '''
    ct_int = (~np.isnan(char_matrix)).astype(float)
    ct = np.nan_to_num(char_matrix)
    if mu is None:
        mu = np.nanmean(char_matrix, axis=0).reshape(-1, 1)
    temp = ct.T.dot(ct) 
    temp_counts = ct_int.T.dot(ct_int)
    sigma_t = temp / temp_counts - mu @ mu.T
    return sigma_t # sigma is the covariance matrix of characteristics at time t



def get_sufficient_statistics_xs(gamma_ts, characteristics_panel):
    return gamma_ts, None


# Impute All
def get_sufficient_statistics_last_val(characteristics_panel, max_delta=None,
                                      residuals=None):
    '''
    utility method to get the last observed value of a characteristic if it has been previously observed
    '''
    T, N, L = characteristics_panel.shape

    last_val = np.copy(characteristics_panel[0])
    if residuals is not None:
        last_resid = np.copy(residuals[0])
    lag_amount = np.zeros_like(last_val)
    lag_amount[:] = np.nan
    if residuals is None:
        sufficient_statistics = np.zeros((T,N,L, 1), dtype=float)
    else:
        sufficient_statistics = np.zeros((T,N,L, 2), dtype=float)
    sufficient_statistics[:,:,:,:] = np.nan
    deltas = np.copy(sufficient_statistics[:,:,:,0])
    for t in range(1, T):
        lag_amount += 1
        sufficient_statistics[t, :, :, 0] = np.copy(last_val)
        deltas[t] = np.copy(lag_amount)
        present_t = ~np.isnan(characteristics_panel[t])
        last_val[present_t] = np.copy(characteristics_panel[t, present_t])
        if residuals is not None:
            sufficient_statistics[t, :, :, 1] = np.copy(last_resid)
            last_resid[present_t] = np.copy(residuals[t, present_t])
        lag_amount[present_t] = 0
        if max_delta is not None:
            last_val[lag_amount >= max_delta] = np.nan
    return sufficient_statistics, deltas


def get_sufficient_statistics_next_val(characteristics_panel, max_delta=None, residuals=None):
    '''
    utility method to get the next observed value of a characteristic if it is observed in the future
    '''
    suff_stats, deltas = get_sufficient_statistics_last_val(characteristics_panel[::-1], max_delta=max_delta,
                                      residuals=None if residuals is None else residuals[::-1])
    return suff_stats[::-1], deltas[::-1]


# ------------------------------------------------------------------------------------------------------------------------------------------------

def smoothing(imputed_sub, mask_bool, smooth_method, ts_method=None, ts_kwargs=None):
    T, N, L = imputed_sub.shape
    # Cross-series smoother
    if smooth_method == 'ema':
        a = 0.5
        for t in range(T - 1):
            imputed_sub[t+1] = a * imputed_sub[t] + (1-a) * imputed_sub[t+1]
    elif smooth_method == 'center_ma':
        flat_mat = imputed_sub.reshape(T, -1)
        imputed_sub = pd.DataFrame(flat_mat).rolling(window=5, center=True, min_periods=1).mean().values.reshape(imputed_sub.shape)
    elif smooth_method == 'kalman_filter':
        kf = KalmanFilter(transition_matrices=[[0.9]], transition_covariance=[[0.01]], observation_covariance=[[0.1]])
        for i in range(imputed_sub.shape[1]):
            for j in range(L):
                smooth_vals, _ = kf.smooth(imputed_sub[:,i,j][:,None])
                imputed_sub[:,i,j] = smooth_vals.ravel()

    return imputed_sub

def get_sufficient_statistics_cp(
    characteristics_panel,
    return_panel,
    smooth_method=None,
    rank=40,
    n_iter_max=100,            # (kept for compat; not used by coupled)
    init='svd',
    tol=1e-6,
):
    """
    Run masked CP directly on the entire (T, N, L) tensor and impute missing entries.
    Does not overwrite any originally observed values.
    Returns a fully imputed NumPy array of shape (T, N, L).
    """
    # Dimensions
    T, N, L = characteristics_panel.shape
    print(f"Running global CP imputation on full tensor (T={T}, N={N}, L={L})...")

    # Mask observed entries
    mask = ~np.isnan(characteristics_panel)

    # Initialize missing entries to zero
    filled = np.nan_to_num(characteristics_panel, nan=0.0)

    # Perform masked CP on full tensor
    factors = parafac(
        tensor=tl.tensor(filled, dtype=tl.float32),
        rank=rank,
        mask=tl.tensor(mask.astype(float), dtype=tl.float32),
        n_iter_max=n_iter_max,
        tol=tol,
        init=init,
    )
    recon = tl.to_numpy(tl.cp_to_tensor(factors))

    # X: (T,N,L) chars; Y: (T,N) returns or excess returns; both may have NaNs
    X = characteristics_panel
    Y = return_panel
    Mx = ~np.isnan(X); My = ~np.isnan(return_panel)

    if smooth_method == 'center_ma':
        X_imp = smoothing(X_imp, None, smooth_method)

    X_imp[Mx] = characteristics_panel[Mx]
    return X_imp




def get_sufficient_statistics_kmeans_cp_smooth(
    characteristics_panel,
    return_panel,            
    n_clusters=10,
    rank=40,
    n_iter_max=100,            # (kept for compat; not used by coupled)
    init='svd',
    tol=1e-6,
    skip_char_threshold=0.0,
    smooth_method=None,
    min_chars = 1,
):
    """
    CP-based imputation with clustering + optional smoothing.
    Now supports masking skipped companies and fully masking low-info firms.
    """
    # Copy to avoid modifying the input
    char_panel = np.copy(characteristics_panel)
    final_imputed = np.full_like(char_panel, np.nan, dtype=float)

    # now build mask_global *after* the injection so fixed values count as observed
    mask_global = ~np.isnan(char_panel)

    T, N, L = char_panel.shape
    mask_global = ~np.isnan(char_panel)

    # # fraction of observed cells per firm across all time × characteristics
    # obs_ratio_firm = np.isfinite(char_panel).mean(axis=(0, 2))  # shape (N,)
    # usable_firms = np.flatnonzero(obs_ratio_firm > FIRM_OBS_THR)
    # # 1) Determine usable firms (avoid (~None))
    obs_counts_firm = np.sum(~np.isnan(char_panel), axis=(0, 2))  # (N,)
    has_any_obs = obs_counts_firm > 0
    usable_firms = np.where(has_any_obs)[0]
    if usable_firms.size == 0:
        raise ValueError("No usable firms for KMeans (all skipped or fully missing).")
    
    kept_firms = usable_firms

    # ------------------------------
    # 2) KMeans
    # ------------------------------
    flat_usable = np.nan_to_num(char_panel[:, kept_firms, :], nan=0.0) \
                    .transpose(1, 0, 2).reshape(kept_firms.size, -1)
    k = min(n_clusters, max(1, kept_firms.size))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(flat_usable)
    labels_usable = kmeans.labels_

    # Map labels back to full (N,) array; skipped firms get -1
    labels_full = -1 * np.ones(N, dtype=int)
    labels_full[kept_firms] = labels_usable

    # Build cluster -> global firm indices using only usable firms
    cluster_companies = {c: [] for c in range(k)}
    for uf_idx, firm in enumerate(kept_firms):
        c = labels_usable[uf_idx]
        cluster_companies[c].append(firm)

    # 3) Apply masking for min_chars (keep your skip mask already applied)
    obs_counts_t_firm = np.sum(~np.isnan(char_panel), axis=2)  # (T, N)
    low_info = obs_counts_t_firm < int(min_chars)
    if np.any(low_info):
        char_panel[low_info, :] = np.nan

    mask_global = ~np.isnan(char_panel)


    # --------------------------------------------
    # 4) Dense vs sparse clusters (on masked panel)
    # --------------------------------------------
    obs_frac_per_cluster = {}
    for c in range(k):
        firms = cluster_companies[c]
        if len(firms) == 0:
            obs_frac_per_cluster[c] = 0.0
            continue
        sub = char_panel[:, firms, :]  # (T, |firms|, L)
        obs_frac_per_cluster[c] = float((~np.isnan(sub)).mean())

    dense_clusters  = [c for c, f in obs_frac_per_cluster.items() if f >= skip_char_threshold]
    sparse_clusters = [c for c, f in obs_frac_per_cluster.items() if f <  skip_char_threshold]

    print(f"Usable firms for KMeans: {kept_firms.size} / {N}")
    print(f"Dense clusters: {dense_clusters}")
    print(f"Sparse clusters: {sparse_clusters}")

    def _cp_impute(sub_tensor, mask_bool, l2_reg=0.0):
        filled = np.nan_to_num(sub_tensor, nan=0.0)
        mask_float = mask_bool.astype(float)
        factors = parafac(
            tl.tensor(filled, dtype=tl.float32),
            rank=rank,
            mask=tl.tensor(mask_float, dtype=tl.float32),
            n_iter_max=n_iter_max,
            tol=tol,
            init=init,
            l2_reg=l2_reg
        )
        return tl.to_numpy(tl.cp_to_tensor(factors))

    def _apply_smoothing_and_ts(imputed_sub):
        # cross-series smoother (optional)
        N = imputed_sub.shape[1]

        if smooth_method == 'ema':
            a = 0.5
            for t in range(T-1):
                # apply EMA only to not-skipped firms
                imputed_sub[t+1, :, :] = (
                    a * imputed_sub[t, :, :] +
                    (1-a) * imputed_sub[t+1, :, :]
                )
        elif smooth_method == 'cma':
            for j in range(N):
                flat_mat = imputed_sub[:, j, :].reshape(T, -1)
                smoothed = (pd.DataFrame(flat_mat)
                            .rolling(window=5, center=True, min_periods=1)
                            .mean().values)
                imputed_sub[:, j, :] = smoothed.reshape(imputed_sub[:, j, :].shape)

        elif smooth_method == 'kalman_filter':
            kf = KalmanFilter(transition_matrices=[[0.9]],
                            transition_covariance=[[0.01]],
                            observation_covariance=[[0.1]])
            for j in tqdm(range(N), desc="Firms"):
                for l in range(L):
                    smooth_vals, _ = kf.smooth(imputed_sub[:, j, l][:, None])
                    imputed_sub[:, j, l] = smooth_vals.ravel()

        return imputed_sub

    # 5) Local CP for dense clusters — USE char_panel (masked), not characteristics_panel
    for c in dense_clusters:
        firms = cluster_companies[c]
        if not firms:
            continue
        sub = char_panel[:, firms, :]            # <-- masked panel
        mask_sub = ~np.isnan(sub)
        print(f"Dense cluster {c}: size={len(firms)}, obs_frac={obs_frac_per_cluster[c]:.3f}")
        recon = _cp_impute(sub, mask_sub)

        # keep your “fill only missing” policy
        imputed = np.where(mask_sub, sub, recon)
        for idx, firm in enumerate(firms):
            miss_global_firm = np.isnan(char_panel[:, firm, :])    # <-- masked panel
            final_imputed[:, firm, :][miss_global_firm] = imputed[:, idx, :][miss_global_firm]


    # 6) Sparse clusters — build pool with ORIGINAL indices, dedupe, map local->global
    if sparse_clusters:
        donors_all = [f for c in dense_clusters for f in cluster_companies[c]]
        print(f"Sparse clusters: {sparse_clusters} — using {len(donors_all)} donor firms from dense clusters")
        for cs in sparse_clusters:
            target_firms = cluster_companies[cs]
            if not target_firms:
                continue
            # dedupe while preserving order
            pool_list = list(dict.fromkeys(target_firms + donors_all))
            sub_union  = char_panel[:, pool_list, :]
            mask_union = ~np.isnan(sub_union)
            print(f"  Imputing sparse cluster {cs}: targets={len(target_firms)}, pooled size={len(pool_list)}")
            recon_union = _cp_impute(sub_union, mask_union, l2_reg=1e-4)

            # local->global map
            local_of = {g_idx: l_idx for l_idx, g_idx in enumerate(pool_list)}
            for firm in target_firms:
                lp = local_of[firm]
                miss_global_firm = np.isnan(char_panel[:, firm, :])  # masked panel
                final_imputed[:, firm, :][miss_global_firm] = recon_union[:, lp, :][miss_global_firm]


    # Ensure observed values are present before smoothing
    final_imputed[mask_global] = characteristics_panel[mask_global]
    final_imputed = _apply_smoothing_and_ts(final_imputed)
    

    final_imputed[mask_global] = char_panel[mask_global]

    return final_imputed




def impute_chars(char_data, imputed_chars, residuals,  # imputed_chars is XS info
                 suff_stat_method, constant_beta, tag, smooth_method=None, ts_method=None, beta_weight=True, noise=None, return_panel = None, dates=None, chars=None, char_map=None):
    '''
    run imputation as described in the paper, based on the type of sufficient statistics
    - last-val B-XS
    - next-val F-XS
    - fwbw BF-XS
    '''
    suff_stats = None
       
    if suff_stat_method == 'last_val':
        if (suff_stats is None):
            suff_stats, _ = get_sufficient_statistics_last_val(char_data, max_delta=None,
                                                                            residuals=residuals)
            
        if len(suff_stats.shape) == 3:
            suff_stats = np.expand_dims(suff_stats, axis=3)
        beta_weights = None
        
    elif suff_stat_method == 'next_val':
        if (suff_stats is None):
            suff_stats, deltas = get_sufficient_statistics_next_val(char_data, max_delta=None, residuals=residuals)

        if len(suff_stats.shape) == 3:
            suff_stats = np.expand_dims(suff_stats, axis=3)

        beta_weights = None

    elif suff_stat_method == 'fwbw':
        next_val_suff_stats, fw_deltas = get_sufficient_statistics_next_val(char_data, max_delta=None,
                                                                                            residuals=residuals)
        prev_val_suff_stats, bw_deltas = get_sufficient_statistics_last_val(char_data, max_delta=None,
                                                                                            residuals=residuals)
        suff_stats = np.concatenate([prev_val_suff_stats, next_val_suff_stats], axis=3)
        
        if beta_weight:            
            beta_weight_arr = np.concatenate([np.expand_dims(fw_deltas, axis=3), 
                                                  np.expand_dims(bw_deltas, axis=3)], axis=3)
            beta_weight_arr = 2 * beta_weight_arr / np.sum(beta_weight_arr, axis=3, keepdims=True)
            beta_weights = {}
            one_arr = np.ones((1, 1))
            for t, i, j in np.argwhere(np.logical_and(~np.isnan(fw_deltas), ~np.isnan(bw_deltas))):
                beta_weights[(t,i,j)] = np.concatenate([one_arr, beta_weight_arr[t,i,j].reshape(-1, 1)], axis=0).squeeze()
        else:
            beta_weights = None


    elif suff_stat_method == 'cp_pooling_smooth':
        imputed_chars = get_sufficient_statistics_kmeans_cp_smooth(char_data, smooth_method=smooth_method, return_panel=return_panel)

    elif suff_stat_method == 'cp':
        imputed_chars = get_sufficient_statistics_cp(char_data, smooth_method=smooth_method, return_panel=return_panel)
        beta_weights = None  

    elif suff_stat_method == 'None':
        suff_stats = None
        beta_weights = None
            
    if suff_stats is None:
        return imputed_chars
    
    else:
        tensor_methods = ['cp_pooling_smooth', 'cp']
        if suff_stat_method in tensor_methods:
            return suff_stats
        else:
            return impute_beta_combined_regression(
                char_data, imputed_chars, sufficient_statistics=suff_stats, 
                beta_weights=None, constant_beta=constant_beta,
                noise=noise
            )
            


def impute_beta_combined_regression(characteristics_panel, xs_imps, sufficient_statistics=None, 
                           beta_weights=None, constant_beta=False, get_betas=False, gamma_ts=None,
                                   use_factors=False, noise=None, reg=None, switch_off_on_suff_stats=False):
    '''
    Run the regression to fit the parameters of the regression model combining time series with cross-sectional information
    '''
    T, N, L = characteristics_panel.shape
    # print(sufficient_statistics.shape)
    K = 0
    if xs_imps is not None:
        K += 1
    if (sufficient_statistics is not None) and (len(sufficient_statistics.shape) == 4):
        K += sufficient_statistics.shape[3]
    if use_factors:
        K += gamma_ts.shape[-1]
    betas = np.zeros((T, L, K))
    imputed_data = np.copy(characteristics_panel)
    imputed_data[:,:,:]=np.nan
    
    if reg is not None and not switch_off_on_suff_stats and use_factors:
        gamma_ts = gamma_ts * np.sqrt(45)
    
    for l in range(L):
        fit_suff_stats = []
        fit_tgts = []
        inds = []
        curr_ind = 0
        all_suff_stats = []
        
        for t in range(T):
            inds.append(curr_ind)
            
            if xs_imps is not None:
                if noise is not None:
                    print("There is noise")
                    suff_stat = np.concatenate([xs_imps[t,:,l:l+1] + noise[t,:,l:l+1], sufficient_statistics[t,:,l]], axis=1)
                else:
                    suff_stat = np.concatenate([xs_imps[t,:,l:l+1], sufficient_statistics[t,:,l]], axis=1)  ## HERE !!!!! HOW TO COMBINE XS AND TS
                
            else:
                if use_factors:
                    suff_stat = np.concatenate([gamma_ts[t,:,l,:], sufficient_statistics[t,:,l]], axis=1)
                else:
                    suff_stat = sufficient_statistics[t,:,l]
            
            available_for_imputation = np.all(~np.isnan(suff_stat), axis=1)
            available_for_fit = np.logical_and(~np.isnan(characteristics_panel[t,:,l]),
                                                  available_for_imputation)
            all_suff_stats.append(suff_stat)

            fit_suff_stats.append(suff_stat[available_for_fit, :])
            fit_tgts.append(characteristics_panel[t,available_for_fit,l])
            curr_ind += np.sum(available_for_fit)
        
        
        inds.append(curr_ind)
        fit_suff_stats = np.concatenate(fit_suff_stats, axis=0)
        fit_tgts = np.concatenate(fit_tgts, axis=0)
        
        if constant_beta:
            if reg is None:
                beta = np.linalg.lstsq(fit_suff_stats, fit_tgts, rcond=None)[0]
            else:
                X = fit_suff_stats
                lmbda = np.eye(fit_suff_stats.shape[1]) * reg * fit_suff_stats.shape[0]
                if switch_off_on_suff_stats:
                    skip_reg_num = 0 if sufficient_statistics is None else sufficient_statistics.shape[-1]
                    for i in range(1, skip_reg_num+1):
                        lmbda[-i, -i] = 0

                
                beta = np.linalg.lstsq(X.T@ X + lmbda, X.T@fit_tgts, rcond=None)[0]
                
            betas[:,l,:] = beta.reshape(1, -1)
        else:
            for t in range(T):
                
                
                if reg is None:
                    beta_l_t = np.linalg.lstsq(fit_suff_stats[inds[t]:inds[t+1]],
                                           fit_tgts[inds[t]:inds[t+1]], rcond=None)[0]
                else:
                    X = fit_suff_stats[inds[t]:inds[t+1]]
                    y = fit_tgts[inds[t]:inds[t+1]]
                    lmbda = np.eye(X.shape[1]) * reg * X.shape[0]
                    
                    if switch_off_on_suff_stats:
                        skip_reg_num = 0 if sufficient_statistics is None else sufficient_statistics.shape[-1]
                        for i in range(1, skip_reg_num+1):
                            lmbda[-i, -i] = 0
                    
                    beta_l_t = np.linalg.lstsq(X.T@X + lmbda, 
                                           X.T@y, rcond=None)[0]
                betas[t,l,:] = beta_l_t
                
        for t in range(T):
            beta_l_t = betas[t,l]
            suff_stat = all_suff_stats[t]
            
            available_for_imputation = np.all(~np.isnan(suff_stat), axis=1)
            
            if beta_weights is None:
                imputed_data[t,available_for_imputation,l] = suff_stat[available_for_imputation,:] @ beta_l_t
            else:
                for i in np.argwhere(available_for_imputation).squeeze():
                    if (t,i,l) in beta_weights:
                        assert np.all(~np.isnan(beta_weights[(t,i,l)]))
                        imputed_data[t,i,l] = suff_stat[i,:] @ np.diag(beta_weights[(t,i,l)]) @ betas[l]
                    else:
                        imputed_data[t,i,l] = suff_stat[i,:] @ betas[l]

    # noise_std = 0.02
    # noise_matrix = np.random.normal(loc=0.0, scale=noise_std, size=imputed_data.shape)

    # # add only to entries we actually imputed (not originally observed)
    # mask_imputed = np.isnan(characteristics_panel) & np.isfinite(imputed_data)
    # imputed_data[mask_imputed] += noise_matrix[mask_imputed]

    if get_betas:
        return imputed_data, betas
    else:
        return imputed_data ### ADD HERE

def simple_imputation(gamma_ts, char_data, suff_stat_method, monthly_update_mask, char_groupings,
                                 eval_char_data=None, num_months_train=None, median_imputation=False,
                                 industry_median=False, industries=None):
    '''
    utility method to do either previous value, median or industry median imputation
    '''
    if eval_char_data is None:
        eval_char_data = char_data
    imputed_chars = simple_impute(char_data)
    if median_imputation:
        imputed_chars[:,:,:] = 0
    elif industry_median:
        imputed_chars = xs_industry_median_impute(char_panel=char_data, industry_codes=industries)
        
    return imputed_chars

def simple_impute(char_panel):
    """
    imputes using the last value of the characteristic time series
    """
    imputed_panel = np.copy(char_panel)
    imputed_panel[:,:,:] = np.nan
    imputed_panel[0] = np.copy(char_panel[0])
    for t in range(1, imputed_panel.shape[0]):
        present_t_l = ~np.isnan(char_panel[t-1])
        imputed_t_1 = ~np.isnan(imputed_panel[t-1])
        imputed_panel[t, present_t_l] = char_panel[t-1, present_t_l]
        imputed_panel[t, np.logical_and(~present_t_l, 
                                     imputed_t_1)] = imputed_panel[t-1, 
                                                                   np.logical_and(~present_t_l, imputed_t_1)]
        imputed_panel[t, ~np.logical_or(imputed_t_1, present_t_l)] = np.nan
        
    return imputed_panel

def xs_industry_median_impute(char_panel, industry_codes):
    """
    imputes using the last value of the characteristic time series
    """
    imputed_panel = np.copy(char_panel)
    for t in range(imputed_panel.shape[0]):
        for c in range(imputed_panel.shape[2]):
            for x in np.unique(industry_codes):
                industry_filter = industry_codes==x
                present_t_l_i = np.logical_and(~np.isnan(char_panel[t,:, c]), industry_filter)
                imputed_panel[t, industry_filter, c] = np.median(char_panel[t,present_t_l_i, c])        
    return imputed_panel


## Impute All
def get_all_xs_vals(chars, reg, Lmbda, time_varying_lmbda=False, get_factors=False):
    '''
    utility method to get the "out of sample estimate" of an observed characteristic based on the XP method
    '''
    C = chars.shape[-1]
    def impute_t(t_chars, reg, C, Lmbda, get_factors=False):
        if not get_factors:
            imputation = np.copy(t_chars) * np.nan
        else:
            imputation = np.zeros((t_chars.shape[0], t_chars.shape[1], Lmbda.shape[1])) * np.nan
        mask = ~np.isnan(t_chars)
        net_mask = np.sum(mask, axis=1)
        K = Lmbda.shape[1]
        for n in range(t_chars.shape[0]):
            if net_mask[n] == 1:
                imputation[n,:] = 0
            elif net_mask[n] > 1:
                for i in range(C):
                    tmp = mask[n, i]
                    mask[n,i] = False
                    y = t_chars[n, mask[n]]
                    X = Lmbda[mask[n], :]
                    L = np.eye(K) * reg
                    params = np.linalg.lstsq(X.T @ X + L, X.T @ y, rcond=None)[0]
                    if get_factors:
                        imputation[n,i] = params
                    else:
                        imputation[n,i] = Lmbda[i] @ params
                    
                    mask[n,i] = tmp
        return np.expand_dims(imputation, axis=0)
    chars = [chars_t for chars_t in chars]
    
    if time_varying_lmbda:
        imputation = list(Parallel(n_jobs=60)(delayed(impute_t)(chars_t, reg, C, l, get_factors=get_factors) 
                                              for chars_t, l in zip(chars, Lmbda)))
    else:
        imputation = list(Parallel(n_jobs=60)(delayed(impute_t)(chars_t, reg, C, Lmbda, get_factors=get_factors)
                                              for chars_t in chars))
    return np.concatenate(imputation, axis=0)


