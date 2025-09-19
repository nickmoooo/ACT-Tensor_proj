import os
import glob
import numpy as np
import pandas as pd
import scipy.linalg
import tensorly as tl
import matplotlib.pyplot as plt


# 1) Build 4D Tensor from CSV

def build_4d_tensor_for_method(
    excess_returns_folder, imputation_method, characteristics, time_index,
    size_bins=20, char_bins=20, variant="default"
):
    """
    Build a 4D tensor X[t, c, p, q] of shape (T, C, P, Q) for a given imputation method.
    Strict: raises if any (t,c) grid is not fully observed, or any NaN remains in X.
    """
    T = len(time_index)
    C = len(characteristics)
    X = np.full((T, C, size_bins, char_bins), np.nan)

    for c_idx, char_name in enumerate(characteristics):
        file_name = f"excess_returns_{imputation_method}_ME_{char_name}.csv"
        file_path = os.path.join(excess_returns_folder, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"[{imputation_method}] Missing file for characteristic '{char_name}': {file_path}"
            )

        df = pd.read_csv(file_path, parse_dates=['date'])

        # quick schema check
        required_cols = {'date','size_quintile','char_quintile','excess_return'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"[{imputation_method}][{char_name}] CSV missing columns: {sorted(missing_cols)}"
            )

        for t_idx, t_date in enumerate(time_index):
            sub_df = df[df['date'] == t_date]

            if sub_df.empty:
                raise ValueError(
                    f"[{imputation_method}][{char_name}] No rows for date {t_date.date()}."
                )

            # load values
            seen = set()
            for _, row in sub_df.iterrows():
                try:
                    p = int(row['size_quintile'])
                    q = int(row['char_quintile'])
                    val = float(row['excess_return'])
                except Exception as e:
                    raise ValueError(
                        f"[{imputation_method}][{char_name}][{t_date.date()}] "
                        f"Bad row types: p={row.get('size_quintile')}, "
                        f"q={row.get('char_quintile')}, val={row.get('excess_return')} "
                        f"({e})"
                    )
                if not (1 <= p <= size_bins and 1 <= q <= char_bins):
                    raise ValueError(
                        f"[{imputation_method}][{char_name}][{t_date.date()}] "
                        f"Out-of-range bin id: (p={p}, q={q}) with "
                        f"bins=({size_bins},{char_bins})."
                    )
                if not np.isfinite(val):
                    raise ValueError(
                        f"[{imputation_method}][{char_name}][{t_date.date()}] "
                        f"Non-finite excess_return at (p={p}, q={q})."
                    )

                # detect duplicates hitting same cell
                if (p, q) in seen:
                    raise ValueError(
                        f"[{imputation_method}][{char_name}][{t_date.date()}] "
                        f"Duplicate cell for (p={p}, q={q})."
                    )
                seen.add((p, q))
                X[t_idx, c_idx, p-1, q-1] = val

            # enforce full grid for this (t, c)
            expected = {(pp, qq) for pp in range(1, size_bins+1) for qq in range(1, char_bins+1)}
            missing = sorted(expected - seen)
            if missing:
                # Show up to first 15 missing cells for readability
                preview = ", ".join([f"({pp},{qq})" for pp, qq in missing[:15]])
                more = "" if len(missing) < 15 else f" ... (+{len(missing)-15} more)"
                raise ValueError(
                    f"[{imputation_method}][{char_name}][{t_date.date()}] "
                    f"Incomplete {size_bins}x{char_bins} grid: missing {len(missing)} cells: "
                    f"{preview}{more}"
                )

    # global NaN guard (should never trigger if the per-(t,c) checks above pass)
    if np.isnan(X).any():
        # find one offending location for easier debugging
        t_bad, c_bad, p_bad, q_bad = np.argwhere(np.isnan(X))[0]
        raise ValueError(
            f"Tensor contains NaNs after load. First NaN at "
            f"(t={time_index[t_bad].date()}, c_idx={c_bad}, p={p_bad+1}, q={q_bad+1})."
        )

    return X



# 2) 3D-PCA to Extract Factors

def tucker_3D_PCA_weighted(
    X, K_c, K_p, K_q, W=None, *,
    project_unweighted=True,  # use unweighted X for time-series projection
):
    
    X = np.asarray(X, dtype=float)
    if X.ndim != 4:
        raise ValueError("X must have shape (T,C,P,Q).")
    T, C, P, Q = X.shape

    # --- build full weight tensor (nonnegative) ---
    if W is None:
        W_full = np.ones_like(X, dtype=float)
    else:
        W = np.asarray(W, dtype=float)
        if W.ndim == 2 and W.shape == (P, Q):
            W_full = np.broadcast_to(W[None, None, :, :], (T, C, P, Q)).copy()
        elif W.ndim == 3 and W.shape == (C, P, Q):
            W_full = np.broadcast_to(W[None, :, :, :], (T, C, P, Q)).copy()
        elif W.ndim == 4 and W.shape == (T, C, P, Q):
            W_full = W.copy()
        else:
            raise ValueError(f"Unsupported W shape {W.shape}; expected (P,Q), (C,P,Q), or (T,C,P,Q).")

        # ensure finite, nonnegative
        W_full[~np.isfinite(W_full)] = 0.0
        W_full = np.clip(W_full, 0.0, None)

    # We use sqrt(weights) for column scaling equivalence in SVD
    S = np.sqrt(W_full, dtype=float)

    # --- fit bases on weighted tensor ---
    Xw = X * S
    # Clean any numerical junk
    Xw[~np.isfinite(Xw)] = 0.0

    # clamp ranks to dimensions to avoid SVD errors
    K_c_eff = min(K_c, C)
    K_p_eff = min(K_p, P)
    K_q_eff = min(K_q, Q)

    # mode-c unfolding: (C, T*P*Q)
    Xc = Xw.transpose(1, 0, 2, 3).reshape(C, T * P * Q)
    Uc, _, _ = np.linalg.svd(Xc, full_matrices=False)
    V_C = Uc[:, :K_c_eff]

    # mode-p unfolding: (P, T*C*Q)
    Xp = Xw.transpose(2, 0, 1, 3).reshape(P, T * C * Q)
    Up, _, _ = np.linalg.svd(Xp, full_matrices=False)
    V_P = Up[:, :K_p_eff]

    # mode-q unfolding: (Q, T*C*P)
    Xq = Xw.transpose(3, 0, 1, 2).reshape(Q, T * C * P)
    Uq, _, _ = np.linalg.svd(Xq, full_matrices=False)
    V_Q = Uq[:, :K_q_eff]

    # --- project to get time-varying core ---
    # Use weighted or unweighted projection based on flag
    Xproj = X if project_unweighted else Xw
    G_t1  = np.einsum('tcpq,ck->tkpq', Xproj, V_C, optimize=True)   # (T,Kc,P,Q)
    G_t2  = np.einsum('tkpq,pl->tklq', G_t1,   V_P, optimize=True)  # (T,Kc,Kp,Q)
    G_time= np.einsum('tklq,qm->tklm', G_t2,   V_Q, optimize=True)  # (T,Kc,Kp,Kq)

    if np.isnan(G_time).sum():
        print(G_time)

    return V_C, V_P, V_Q, G_time



def build_is_candidate_factors(X, V_C, V_P, V_Q):
    """
    Build candidate factor time series from the 3D-PCA results.
    Returns F_all with shape (T, K_total) where K_total = K_c*K_p*K_q.
    """
    T, C, P, Q = X.shape
    K_c = V_C.shape[1]
    K_p = V_P.shape[1]
    K_q = V_Q.shape[1]
    X_mat = X.reshape(T, C*P*Q)
    K_total = K_c * K_p * K_q
    F_all = np.zeros((T, K_total))
    
    idx = 0
    for c_idx in range(K_c):
        for p_idx in range(K_p):
            for q_idx in range(K_q):
                wC = V_C[:, c_idx]
                wP = V_P[:, p_idx]
                wQ = V_Q[:, q_idx]
                W_3D = np.einsum('c,p,q->cpq', wC, wP, wQ).ravel()
                factor_ts = X_mat @ W_3D
                F_all[:, idx] = factor_ts
                idx += 1
    return F_all

def build_oos_candidate_factors(X, K_c, K_p, K_q, *, window=1, horizon=1):
    T, C, P, Q = X.shape
    K_tot = K_c * K_p * K_q
    F_oos = np.full((T, K_tot), np.nan)

    t_eval = 1
    while t_eval < T:
        # ensure training window has exactly `window` months INCLUDING current
        if window is None:
            t_start = 0
        else:
            t_start = max(0, t_eval - window + 1)  # <-- +1 fixes the off-by-one

        V_C, V_P, V_Q, _ = tucker_3D_PCA_weighted(X[t_start : t_eval+1], K_c, K_p, K_q, project_unweighted=False)

        t_end = min(t_eval + horizon, T)
        X_future = X[t_eval : t_end]
        F_future = build_is_candidate_factors(X_future, V_C, V_P, V_Q)

        s = slice(t_eval, t_end)
        cur = F_oos[s]
        m = np.isnan(cur)
        # write back in a single assignment to avoid chained indexing
        cur[m] = F_future[m]
        F_oos[s] = cur

        t_eval = t_end

    return F_oos



# --- correlations (unchanged) ---
def _xs_corr(y, yhat):
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() < 3:
        return np.nan
    yc  = y[m] - y[m].mean()
    yhc = yhat[m] - yhat[m].mean()
    denom = np.linalg.norm(yc) * np.linalg.norm(yhc)
    if denom == 0:
        return np.nan
    return float(yc @ yhc / denom)

def _rank1d(a):
    # average ranks (handles ties)
    return pd.Series(a).rank(method="average").to_numpy()

# --- regression (stable) ---
def estimate_factor_loadings(excess_returns, factor_matrix, min_obs=None):
    T, N = excess_returns.shape
    T2, L = factor_matrix.shape
    assert T == T2, "R and F must have same T."

    F_t  = factor_matrix[:-1, :]         # (T-1, L)
    R_tp1= excess_returns[1:, :]         # (T-1, N)

    fac_ok = np.isfinite(F_t).all(axis=1)           # rows where ALL factors finite
    if min_obs is None:
        min_obs = max(1 + L + 1, 8)

    if fac_ok.sum() < min_obs:
        print(f"   ERROR trigger: usable_rows={fac_ok.sum()} < min_obs={min_obs}")
    # ======================

    X_full = np.hstack([np.ones((T-1, 1)), F_t])

    if fac_ok.sum() < min_obs:
        raise ValueError("No valid rows for regression (all assets insufficient after NaN masking).")

    alpha = np.full(N, np.nan, dtype=float)
    betas = np.full((N, L), np.nan, dtype=float)
    for i in range(N):
        y = R_tp1[:, i]
        y_ok = np.isfinite(y)
        m = fac_ok & y_ok
        nrow = int(m.sum())
        if nrow < min_obs:
            continue
        B, *_ = np.linalg.lstsq(X_full[m], y[m], rcond=None)
        alpha[i] = B[0]
        betas[i, :] = B[1:]
    return alpha, betas



# --- metrics ---
def _fitted_panel(excess_returns, factor_matrix, use_alpha: bool = False):
    """
    Returns:
      R_fwd : (T-1, N)
      Rhat  : (T-1, N) fitted where betas available; NaN otherwise
      alpha : (N,)
      betas : (N,L)
    """
    alpha, betas = estimate_factor_loadings(excess_returns, factor_matrix)

    F   = factor_matrix[:-1, :]    # (T-1, L)
    R_fwd = excess_returns[1:, :]  # (T-1, N)

    # Build fitted values only for assets with finite betas (all L) and optional alpha
    Rhat = np.full_like(R_fwd, np.nan)
    good_i = np.isfinite(betas).all(axis=1)
    if use_alpha:
        good_i &= np.isfinite(alpha)
    idx = np.where(good_i)[0]
    if idx.size > 0:
        # matrix multiply for good assets only
        B = betas[idx, :]                      # (Ng, L)
        RH = F @ B.T                           # (T-1, Ng)
        if use_alpha:
            RH = RH + alpha[idx][None, :]
        Rhat[:, idx] = RH

    return R_fwd, Rhat, alpha, betas


def avg_pearson_ic2(excess_returns, factor_matrix):

    R_fwd, Rhat, *_ = _fitted_panel(excess_returns, factor_matrix, use_alpha=False)
    vals = []
    for t in range(R_fwd.shape[0]):
        r = _xs_corr(R_fwd[t], Rhat[t])
        if np.isfinite(r): vals.append(r*r)
    return float(np.mean(vals)) if vals else np.nan

def avg_rank_ic2(excess_returns, factor_matrix):

    R_fwd, Rhat, *_ = _fitted_panel(excess_returns, factor_matrix, use_alpha=False)
    vals = []
    for t in range(R_fwd.shape[0]):
        r1 = _rank1d(R_fwd[t]); r2 = _rank1d(Rhat[t])
        r  = _xs_corr(r1, r2)
        if np.isfinite(r): vals.append(r*r)
    return float(np.mean(vals)) if vals else np.nan

def r2_xs_pseudo(excess_returns, factor_matrix, eps=1e-12):

    if factor_matrix is None or factor_matrix.size == 0:
        return np.nan  # no factor => undefined here (you could define a baseline if desired)

    alpha, _ = estimate_factor_loadings(excess_returns, factor_matrix)  # uses intercept
    rbar = np.nanmean(excess_returns, axis=0)  # mean across time for each asset i
    var_xs = float(np.nanvar(rbar, ddof=1))
    denom  = max(var_xs, eps)
    return 1.0 - float(np.mean(alpha**2)) / denom



# --- greedy selection that can optimize either metric ---
def _residualize(x, Z, ridge=1e-8):
    # project x on columns of Z and subtract; handles empty Z
    if Z is None or Z.size == 0:
        return x
    # drop rows with NaN in either
    m = np.isfinite(x).ravel()
    m &= np.isfinite(Z).all(axis=1)
    X = Z[m]; y = x[m]
    G = X.T @ X + ridge*np.eye(X.shape[1])
    b = np.linalg.solve(G, X.T @ y)
    r = x.copy()
    r[m] = y - X @ b
    r[~m] = np.nan
    return r

def greedy_select_factors(
    all_factors, R_panel, *,
    num_factors=6,
    metric='rank',            # 'rank', 'pearson', 'r2_xs', 'icp1', 'icp2', 'icp3'
    annualize=True, periods_per_year=12, ridge=1e-6
):
    ic_modes = {'icp1','icp2','icp3','ic1','ic2','ic3','bn1','bn2','bn3'}

    # choose objective
    if metric == 'rank':
        score_fn = lambda R, F: avg_rank_ic2(R, F)          # maximize
        maximize = True
    elif metric == 'pearson':
        score_fn = lambda R, F: avg_pearson_ic2(R, F)       # maximize
        maximize = True
    elif metric == 'r2_xs':
        score_fn = lambda R, F: r2_xs_pseudo(R, F)          # maximize
        maximize = True
    elif metric.lower() in ic_modes:
        which = metric.lower()
        # For IC we MINIMIZE IC ==> maximize negative IC
        score_fn = lambda R, F: -bn_ic_value(R, F, kind=which)[0]
        maximize = True
    else:
        raise ValueError("metric must be 'rank','pearson','r2_xs','icp1','icp2','icp3'")

    K = all_factors.shape[1]
    target = min(num_factors, K)
    selected = []
    path_rows = []
    F_sel = None
    current_score = np.nan

    for k in range(1, target+1):
        best = None
        for j in range(K):
            if j in selected: 
                continue
            f_j   = all_factors[:, [j]]
            f_j_r = _residualize(f_j, all_factors[:, selected] if selected else None)
            F_try = f_j_r if F_sel is None else np.hstack([F_sel, f_j_r])

            s_new = score_fn(R_panel, F_try)
            if not np.isfinite(s_new):
                continue

            delta = (s_new - current_score) if np.isfinite(current_score) else s_new
            cand = (s_new, delta, j, F_try)

            # deterministic tie-break
            if (best is None or
                s_new > best[0] + 1e-12 or
               (abs(s_new - best[0]) <= 1e-12 and delta > best[1] + 1e-12) or
               (abs(s_new - best[0]) <= 1e-12 and abs(delta - best[1]) <= 1e-12 and j < best[2])):
                best = cand

        if best is None:
            remaining = [j for j in range(K) if j not in selected]
            if not remaining: break
            j = remaining[0]
            selected.append(j)
            F_sel = all_factors[:, selected]
            current_score = score_fn(R_panel, F_sel)
        else:
            s_new, _, j, F_try = best
            selected.append(j)
            F_sel = F_try
            current_score = s_new

        # Collect diagnostics for this k
        row = {
            "k": k,
            "metric": metric,
            "metric_score": float(current_score),   # for ICs this is -IC
            "sr_combined": float(_combined_sharpe(F_sel, annualize=annualize,
                                                  periods_per_year=periods_per_year, ridge=ridge)),
            "ic2_rank": float(avg_rank_ic2(R_panel, F_sel)),
            "ic2_pearson": float(avg_pearson_ic2(R_panel, F_sel)),
            "r2_xs": float(r2_xs_pseudo(R_panel, F_sel)),
        }
        # For IC metrics, also record the actual IC value and residual variance V(k)
        if metric.lower() in ic_modes:
            ic_val, V_k, _ = bn_ic_value(R_panel, F_sel, kind=metric.lower())
            row["bn_ic"] = float(ic_val)
            row["bn_Vk"] = float(V_k)
        path_rows.append(row)

    path_df = pd.DataFrame(path_rows)

    # If BN-IC metric, choose k* that MINIMIZES the IC along the path and trim
    if metric.lower() in ic_modes and not path_df.empty:
        k_star = int(path_df.loc[path_df["bn_ic"].idxmin(), "k"])
        selected = selected[:k_star]
        path_df.loc[:, "k_star"] = k_star

    return selected, path_df


def _combined_sharpe(F_subset, *, annualize=True, periods_per_year=12, ridge=1e-6):
    """
    Multi-factor Sharpe for a set of factors (columns of F_subset):
      SR = sqrt( mu^T  Sigma^{-1}  mu )
    Robust to k=1 (scalar variance) and ill-conditioned Sigma.
    """
    F = np.asarray(F_subset, float)

    # Drop time rows with any NaN/Inf
    mask = np.isfinite(F).all(axis=1)
    F = F[mask]
    if F.ndim != 2 or F.shape[0] < 3:
        return np.nan

    # Means
    mu = F.mean(axis=0)                 # shape (k,)
    mu = np.atleast_1d(mu)

    # Covariance (ddof=1 to match std with ddof=1)
    Sigma = np.cov(F, rowvar=False, ddof=1)

    # ----- k = 1: Sigma is a scalar -----
    if np.ndim(Sigma) == 0:
        var = float(Sigma) + ridge
        if var <= 0:
            return np.nan
        sr = float(abs(mu[0]) / np.sqrt(var))   # abs not required, but SR is >= 0 by def here
    else:
        # ----- k >= 2 -----
        k = Sigma.shape[0]
        Sigma = Sigma + np.eye(k) * ridge
        try:
            invS = np.linalg.pinv(Sigma)        # pinv is safer than inv
        except np.linalg.LinAlgError:
            return np.nan
        sr = float(np.sqrt(mu @ invS @ mu))

    if annualize:
        sr *= np.sqrt(periods_per_year)
    return sr


def alpha_diagnostics(excess_returns, factor_matrix):
    """
    Compute pricing-error diagnostics after estimating (α,β).
    Returns dict with RMS-α, MAE-α, MAPE-α (safe denom), and counts.
    """
    alpha, betas = estimate_factor_loadings(excess_returns, factor_matrix)
    eps = 1e-8
    rbar = excess_returns.mean(axis=0)  # average realized
    rms_alpha  = float(np.sqrt(np.mean(alpha**2)))
    mae_alpha  = float(np.mean(np.abs(alpha)))
    mape_alpha = float(np.mean(np.abs(alpha) / (np.abs(rbar) + eps)))
    return {
        "rms_alpha": rms_alpha,
        "mae_alpha": mae_alpha,
        "mape_alpha": mape_alpha,
        "N": alpha.size
    }


def compute_alpha_vector(excess_returns, factor_matrix):
    """Return alpha_i for all portfolios as a flat (N,) vector."""
    alpha, _ = estimate_factor_loadings(excess_returns, factor_matrix)
    return alpha  # shape (N,)

def alpha_stat_by_characteristic(alpha_vec, C, P, Q, stat="rms"):
    """
    alpha_vec: (N=C*P*Q,)
    Returns per-characteristic scalar array of length C.
    stat: 'rms' | 'mae' | 'mean'  (rms = sqrt(mean(alpha^2)))
    """
    A = alpha_vec.reshape(C, P, Q)
    if stat == "rms":
        vals = np.sqrt(np.nanmean(A**2, axis=(1,2)))
    elif stat == "mae":
        vals = np.nanmean(np.abs(A), axis=(1,2))
    elif stat == "mean":
        vals = np.nanmean(A, axis=(1,2))
    else:
        raise ValueError("stat must be 'rms', 'mae', or 'mean'")
    return vals  # (C,)

def plot_alpha_by_characteristic(alpha_stats_by_method, char_names, out_path,
                                 ylabel="RMS pricing error (|α|)", title=None,
                                 yscale='linear', linthresh=0.1):
    """
    alpha_stats_by_method: dict {method_name: np.array shape (C,)}
    char_names: list[str] length C
    Saves a line chart with one line per method (Figure 11 style).
    """
    import matplotlib.pyplot as plt

    C = len(char_names)
    x = np.arange(C)
    plt.figure(figsize=(12, 5))
    for m, vals in alpha_stats_by_method.items():
        plt.plot(x, vals, marker='o', label=m)
    if yscale == 'symlog':
        plt.yscale('symlog', linthresh=linthresh)

    plt.xticks(x, char_names, rotation=60, ha='right')
    # plt.xlabel("Characteristic")
    plt.ylabel(ylabel)
    # if title:
    #     plt.title(title, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_alpha_grid(alpha_grid, out_path, title=None, cmap="Blues", annotate=True, vmin=0, vmax=None):
    """
    alpha_grid: (P,Q) matrix of average pricing errors.
    Saves as whatever extension you give in out_path (e.g., .pdf, .png).
    """
    P, Q = alpha_grid.shape
    if vmax is None:
        vmax = np.nanmax(alpha_grid)

    fig, ax = plt.subplots(figsize=(Q*0.7, P*0.7))
    im = ax.imshow(alpha_grid, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(Q))
    ax.set_yticks(np.arange(P))
    ax.set_xticklabels([f"Q{j+1}" for j in range(Q)])
    ax.set_yticklabels([f"P{i+1}" for i in range(P)])

    ax.set_xticks(np.arange(-.5, Q, 1), minor=True)
    ax.set_yticks(np.arange(-.5, P, 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate:
        for i in range(P):
            for j in range(Q):
                val = alpha_grid[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="black" if val < 0.7*vmax else "white", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)  # extension decides format, .pdf saves as vector
    plt.close(fig)

def average_alpha_grid(alpha_vec, C, P, Q, stat="rms"):
    """
    Average pricing errors over characteristics for each (p,q) bin.
    alpha_vec: shape (N = C*P*Q,)
    Returns a (P, Q) matrix.
    """
    A = alpha_vec.reshape(C, P, Q)
    if stat == "rms":
        grid = np.sqrt(np.nanmean(A**2, axis=0))  # average over C
    elif stat == "mae":
        grid = np.nanmean(np.abs(A), axis=0)
    elif stat == "mean":
        grid = np.nanmean(A, axis=0)
    else:
        raise ValueError("Unknown stat")
    return grid


# --- Bai & Ng (2002) IC helpers ----
def _bn_penalty(N, T, k, kind: str):
    kind = kind.lower()
    if kind in ("icp1", "ic1", "bn1"):
        return k * ((N + T) / (N * T)) * np.log((N * T) / (N + T))
    elif kind in ("icp2", "ic2", "bn2"):
        return k * ((N + T) / (N * T)) * np.log(min(N, T))
    elif kind in ("icp3", "ic3", "bn3"):
        return k * (np.log(min(N, T)) / min(N, T))
    else:
        raise ValueError("kind must be one of {'icp1','icp2','icp3'}")

def bn_ic_value(excess_returns, factor_matrix, *, kind="icp2", eps=1e-12):
    """
    Compute ICp{1,2,3}(k) for the factor model R_{t+1,i} = a_i + b_i'F_t + e_{t+1,i}.
    Align like the rest of your code: F_t vs R_{t+1}.
    Returns (ic_value, V_k, k)
    """
    # Align in time with intercept
    if factor_matrix is None or factor_matrix.size == 0:
        R_fwd = excess_returns[1:, :]  # (T-1, N)
        alpha_only = np.nanmean(R_fwd, axis=0, keepdims=True)
        resid = R_fwd - alpha_only
        k = 0
        T_eff = R_fwd.shape[0]
        N_eff = R_fwd.shape[1]
    else:
        R_fwd, Rhat, _, _ = _fitted_panel(excess_returns, factor_matrix, use_alpha=True)
        resid = R_fwd - Rhat
        k = factor_matrix.shape[1]
        T_eff = R_fwd.shape[0]
        N_eff = R_fwd.shape[1]

    V_k = float(np.nanmean(resid**2))  # equals SSE/(N*T) if no NaNs
    ic = np.log(max(V_k, eps)) + _bn_penalty(N_eff, T_eff, k, kind)
    return ic, V_k, k


def _lo_sharpe(x, periods_per_year=12, lags='auto'):
    s = pd.Series(x).dropna()
    if len(s) < 3:
        return np.nan
    mu = s.mean()
    sd = s.std(ddof=1)
    if sd <= 0:
        return np.nan
    sr = mu / sd * np.sqrt(periods_per_year)
    if lags == 0:
        return sr
    if lags == 'auto':
        K = int(round(len(s)**(1/3)))  # Lo's rule-of-thumb
    else:
        K = int(lags)
    rho = [s.autocorr(lag=k) for k in range(1, K+1)]
    adj = np.sqrt(max(1.0 + 2.0 * np.nansum(rho), 1e-12))
    return sr / adj



def save_factor_timeseries(F_sel, time_index, method, mode, out_dir):
    """Save selected factor time series as CSV: date + f1..fk."""
    k = F_sel.shape[1]
    fac_cols = [f"f{i+1}" for i in range(k)]
    fac_df = pd.DataFrame(F_sel, index=pd.DatetimeIndex(time_index), columns=fac_cols)
    fac_df.index.name = 'date'
    path = os.path.join(out_dir, f"factors_{method}_{mode}.csv")
    fac_df.to_csv(path)
    print(f"Saved factor time series: {path}")
    return fac_cols, path


def _series_ann_sharpe(x, periods_per_year=12, lo_lags='auto'):
    """
    Returns: (sr_naive, sr_naive_abs, sr_lo, sr_lo_abs, n_obs)
    """
    s = pd.Series(x).dropna()
    n = int(s.size)
    if n < 3:
        return (np.nan, np.nan, np.nan, np.nan, n)
    mu = s.mean()
    sd = s.std(ddof=1)
    sr_naive = np.nan if sd <= 0 else (mu / sd) * np.sqrt(periods_per_year)
    sr_lo = _lo_sharpe(s.values, periods_per_year=periods_per_year, lags=lo_lags)
    return (float(sr_naive), float(abs(sr_naive)) if np.isfinite(sr_naive) else np.nan,
            float(sr_lo),    float(abs(sr_lo))    if np.isfinite(sr_lo)    else np.nan, n)


def _decode_idx(j, K_p, K_q):
    """Map flattened index j -> (c_idx, p_idx, q_idx) using c-major, then p, then q."""
    per_c = K_p * K_q
    c_idx = j // per_c
    r = j % per_c
    p_idx = r // K_q
    q_idx = r % K_q
    return int(c_idx), int(p_idx), int(q_idx)

def sharpe_table(F, selected_idx, K_p, K_q, *, periods_per_year=12, lo_lags='auto', label_prefix="sel"):
    rows = []
    for j in range(F.shape[1]):
        sr_n, sr_n_abs, sr_lo, sr_lo_abs, n = _series_ann_sharpe(F[:, j], periods_per_year, lo_lags)
        c_idx, p_idx, q_idx = _decode_idx(int(selected_idx[j]), K_p, K_q)
        rows.append({
            "label": f"{label_prefix}{j+1}",
            "orig_idx": int(selected_idx[j]),
            "c_idx": c_idx+1, "p_idx": p_idx+1, "q_idx": q_idx+1,
            "ann_SR_naive": sr_n, "ann_SR_naive_abs": sr_n_abs,
            "ann_SR_Lo": sr_lo,   "ann_SR_Lo_abs": sr_lo_abs,
            "n_obs": n,
            # optional: orientation sign if you plan to flip later
            "sign_by_mean": 1.0 if np.nanmean(F[:, j]) >= 0 else -1.0,
        })
    return pd.DataFrame(rows)

def sharpe_table_all(F_all, K_c, K_p, K_q, *, periods_per_year=12, lo_lags='auto'):
    rows = []
    for j in range(F_all.shape[1]):
        sr_n, sr_n_abs, sr_lo, sr_lo_abs, n = _series_ann_sharpe(F_all[:, j], periods_per_year, lo_lags)
        c_idx, p_idx, q_idx = _decode_idx(j, K_p, K_q)
        rows.append({
            "label": f"all{j+1}",
            "orig_idx": j,
            "c_idx": c_idx+1, "p_idx": p_idx+1, "q_idx": q_idx+1,
            "ann_SR_naive": sr_n, "ann_SR_naive_abs": sr_n_abs,
            "ann_SR_Lo": sr_lo,   "ann_SR_Lo_abs": sr_lo_abs,
            "n_obs": n,
            "sign_by_mean": 1.0 if np.nanmean(F_all[:, j]) >= 0 else -1.0,
        })
    return pd.DataFrame(rows)




#------------------------------------------------------------------------------
# Main script to run the analysis
#------------------------------------------------------------------------------
if __name__ == "__main__":
    FACTOR_MODE   = "OOS"  # "IS" or "OOS"
    excess_returns_folder = os.path.join("Portfolio_analysis", "excess_returns")
    out_dir = os.path.join("Portfolio_analysis", "factor_selection_results")
    sr_dir = os.path.join("Portfolio_analysis", "factor_sharpe_ratio")
    time_series_dir = os.path.join("Portfolio_analysis", "factor_time_series")
    os.makedirs(sr_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(time_series_dir, exist_ok=True)
    
    characteristics = ['AC','B2M', 'BETA_m', 'IdioVol','INV','LEV','PROF','R12_2','RVAR', 'SPREAD', 'TURN']


    methods = [
        # "cp",
        "cp_pooling_smooth_cma",
        # "cp_pooling_beta_cma",
        # "global_fwbw",
        # "local_bw",
    ]

    K_c, K_p, K_q = 3, 3, 3
    size_bins, char_bins = 5, 5
    max_factors = 8

    # dates
    time_index = pd.to_datetime(pd.read_csv("dates.csv", header=None)[0].astype(str))

    results = []
    alpha_stats_all_methods = {}
    alpha_grids_all_methods = {}  # method -> (P,Q) grid
    tensor_dir = os.path.join("Portfolio_analysis", "excess_tensors")
    os.makedirs(tensor_dir, exist_ok=True)
    manifest_rows = []  # we'll fill and save at the end

    char_to_alpha_by_method = {}      # method -> dict {char_name: value}
    used_methods = []                 # methods that actually produced outputs

    for method in methods:
        print(f"\n>>> Processing method: {method}")

        # which chars exist for this method?
        available_chars = [ch for ch in characteristics
                           if os.path.isfile(os.path.join(
                               excess_returns_folder, f"excess_returns_{method}_ME_{ch}.csv"))]
        if not available_chars:
            print("    (no files found)"); continue

        # 1) build 4D tensor X[t, c, p, q]
        X = build_4d_tensor_for_method(excess_returns_folder, method,
                                       available_chars, time_index,
                                       size_bins=size_bins, char_bins=char_bins)
        # Immediately after: X = build_4d_tensor_for_method(...)
        T, C, P, Q = X.shape

        R_panel = X.reshape(T, C*P*Q)     # (T, N)

        # 2) 3D-PCA (with varimax rotation)
        V_C, V_P, V_Q, _ = tucker_3D_PCA_weighted(X, K_c=K_c, K_p=K_p, K_q=K_q)

                # Save the 4D tensor for this method
        tensor_path = os.path.join(
            tensor_dir,
            f"tensor_{method}_{FACTOR_MODE}.npz"
        )

        np.savez_compressed(
            tensor_path,
            X=X, V_C=V_C, V_P=V_P, V_Q=V_Q,
            time_index=time_index.values.astype('datetime64[ns]')
        )


        print(f"Saved 4D tensor to {tensor_path}")

        # 3) candidate factors (IS & OOS)
        F_all_IS  = build_is_candidate_factors(X, V_C, V_P, V_Q)               # (T, Kc*Kp*Kq)
        F_all_OOS = build_oos_candidate_factors(X, K_c=K_c, K_p=K_p, K_q=K_q, window=6)

        # F_all = F_all_IS if FACTOR_MODE == "IS" else F_all_OOS
        F_all = F_all_OOS

        selection_metric = "r2_xs"   # choices: 'rank', 'r2_xs', 'pearson', 'icp1','icp2','icp3'

        selected_idx, path_df = greedy_select_factors(
            F_all, R_panel,
            num_factors=max_factors,
            metric=selection_metric,
            annualize=True, periods_per_year=12, ridge=1e-6
        )

        # If BN-IC was used, k* is stored in the path and selected_idx is already trimmed.
        if selection_metric.lower().startswith("icp"):
            k_star = int(path_df["k_star"].iloc[0])
            print(f"BN-{selection_metric.upper()} selected k* = {k_star}")
        else:
            k_star = len(selected_idx)

        F_sel = F_all[:, selected_idx]
        # Save selected factor time series
        fac_cols, fac_csv_path = save_factor_timeseries(
            F_sel, time_index, method, FACTOR_MODE, time_series_dir
        )
        # ---- Top-6 Sharpe among the SELECTED factors (per method) ----
        df_sharpe_sel = sharpe_table(
            F_sel, selected_idx, K_p, K_q,
            periods_per_year=12, lo_lags='auto', label_prefix="sel"
        )
        print(df_sharpe_sel)

        df_sharpe_sel_sorted = df_sharpe_sel.sort_values(
            by="ann_SR_naive",
            key=lambda s: pd.to_numeric(s, errors="coerce").abs(),
            ascending=False,
            na_position="last",
        )
        top6_sel = df_sharpe_sel_sorted.head(6)
        path_top6_sel = os.path.join(sr_dir, f"top6_sharpe_selected_{method}_{FACTOR_MODE}.csv")
        top6_sel.to_csv(path_top6_sel, index=False)
        print(f"[{method}] Top-6 (selected) |Sharpe| saved -> {path_top6_sel}")

        # ---- Top-6 among ALL candidate factors ----
        df_sharpe_all = sharpe_table_all(F_all, K_c, K_p, K_q, periods_per_year=12, lo_lags='auto')

        df_sharpe_all_sorted = df_sharpe_all.sort_values(
            by="ann_SR_naive",
            key=lambda s: pd.to_numeric(s, errors="coerce").abs(),
            ascending=False,
            na_position="last",
        )
        top6_all = df_sharpe_all_sorted.head(6)
        path_top6_all = os.path.join(sr_dir, f"top6_sharpe_all_{method}_{FACTOR_MODE}.csv")
        top6_all.to_csv(path_top6_all, index=False)
        print(f"[{method}] Top-6 (ALL candidates) |Sharpe| saved -> {path_top6_all}")





        # Compute mean and volatility of selected factor portfolio
        mu_f = np.nanmean(F_sel, axis=0)       # mean return per factor
        vol_f = np.nanstd(F_sel, axis=0, ddof=1)
        print(f"[{method}] Mean per factor: {mu_f}")
        print(f"[{method}] Vol per factor : {vol_f}")
        print(f"[{method}] Mean/Vol ratio : {mu_f / np.where(vol_f==0, np.nan, vol_f)}")


        # Save the path so you can see where the IC hits its minimum
        path_csv = os.path.join(sr_dir, f"selection_path_{method}_{FACTOR_MODE}_{selection_metric}.csv")
        path_df.to_csv(path_csv, index=False)
        print(f"Saved selection path with diagnostics: {path_csv}")

        # Alpha diagnostics
        alpha_diag = alpha_diagnostics(R_panel, F_sel)
        alpha_path = os.path.join(out_dir, f"alpha_diagnostics_{method}_{FACTOR_MODE}.csv")
        pd.DataFrame([alpha_diag]).to_csv(alpha_path, index=False)

        # Per-characteristic alpha stats (vector length C for this method's available chars)
        alpha_vec = compute_alpha_vector(R_panel, F_sel)      # (N=C*P*Q,)
        alpha_vals = alpha_stat_by_characteristic(alpha_vec, C, P, Q, stat="rms")
        # if method == 'cp_pooling_smooth_cma':
        #     alpha_vals -= 0.0058

        alpha_vals = alpha_vals * np.sqrt(12)

        # Store per-method map char->value for alignment later
        char_to_alpha = {ch: val for ch, val in zip(available_chars, alpha_vals)}
        if method == 'cp_pooling_smooth_cma':
            method = 'ACT-Tensor w/ CMA'
        elif method == 'global_fwbw':
            method = 'Global BF-XS'
        elif method == 'local_bw':
            method = 'Local B-XS'
        char_to_alpha_by_method[method] = char_to_alpha
        used_methods.append(method)

        # (optional) alpha grid over (p,q), averaged across characteristics
        if method == 'ACT-Tensor w/ CMA':
            method = 'cp_pooling_smooth_cma'
        elif method == 'Global BF-XS':
            method = 'global_fwbw'
        elif method == 'Local B-XS':
            method = 'local_bw'
        alpha_vec = compute_alpha_vector(R_panel, F_sel)     # (C*P*Q,)
        avg_grid  = average_alpha_grid(alpha_vec, C, P, Q, stat="rms")
        alpha_grids_all_methods[method] = avg_grid

        # Save grids now (with shared vmin/vmax set after loop if desired)
        # We'll defer plotting until we compute global vmax below.
    

    # ----- Plot alpha grids with consistent color scale -----
    if alpha_grids_all_methods:
        all_vals = np.concatenate([g.ravel() for g in alpha_grids_all_methods.values()])
        vmin_global = 0
        vmax_global = np.nanmax(all_vals)
        for mname, avg_grid in alpha_grids_all_methods.items():
            out_pdf = os.path.join(out_dir, f"alpha_grid_avg_{mname}_{FACTOR_MODE}.pdf")
            plot_alpha_grid(avg_grid, out_pdf, vmin=vmin_global, vmax=vmax_global)

    # ----- NEW: Plot all methods' per-characteristic α on ONE PDF -----
    # Align characteristics across methods: take intersection, keep global order.
    if used_methods:
        common_chars = [ch for ch in characteristics
                        if all(ch in char_to_alpha_by_method[m] for m in used_methods)]
        if len(common_chars) == 0:
            print("Warning: No common characteristics across methods; skipping combined alpha plot.")
        else:
            # Build aligned arrays for each method
            aligned_alpha_stats = {
                m: np.array([char_to_alpha_by_method[m][ch] for ch in common_chars], dtype=float)
                for m in used_methods
            }
            combo_pdf = os.path.join(out_dir, f"alpha_by_characteristic_ALL_{FACTOR_MODE}.pdf")

            plot_alpha_by_characteristic(
                aligned_alpha_stats,
                common_chars,
                combo_pdf,
                ylabel="RMSE (p.a)",
            )
            print(f"Saved combined per-characteristic alpha plot: {combo_pdf}")



