import os, numpy as np, pandas as pd

# ---- CONFIG ----
FOLDER = os.path.join("Portfolio_analysis", "excess_tensors")
IMPUTATION_METHODS = [
    "cp_pooling_smooth_cma",
    # add more methods here...
]
FREQ = 12           # 12=monthly, 252=daily
FLIP_SIGN = True    # flip sign so Sharpe >= 0 (handles PCA sign ambiguity)
OUT_CSV = os.path.join("Portfolio_analysis", "factor_weights", "top6_ann_sharpe.csv")

# Try these file patterns and keys to locate factor return matrices
CAND_FILES = [
    "tensor_{method}_OOS_FACTORS.npz",
    "tensor_{method}_factors.npz",
    "tensor_{method}_OOS_factors.npz",
    "3D_PCA_factors_{method}.npz",   # sometimes factors live here too
    "factors_{method}.npz"
    "tensor_cp_pooling_smooth_cma_OOS_LOADING.npz"
]
CAND_KEYS = [
    "F", "factors", "factor_returns", "F_t", "time_factors", "scores", "Scores"
]

def load_factor_matrix(folder, method):
    """Return (T,K) factor matrix, file_path, key. Else (None,None,None)."""
    for pat in CAND_FILES:
        path = os.path.join(folder, pat.format(method=method))
        if not os.path.exists(path):
            continue
        z = np.load(path, allow_pickle=True)
        # preferred keys
        for key in CAND_KEYS:
            if key in z:
                arr = np.asarray(z[key])
                if arr.ndim == 2 and arr.shape[0] >= 24:
                    return arr, path, key
        # fallback: pick the largest 2D array that looks like (T,K)
        best = None; best_key = None
        for k in z.files:
            arr = np.asarray(z[k])
            if arr.ndim == 2 and arr.shape[0] >= 24:
                if best is None or arr.size > best.size:
                    best, best_key = arr, k
        if best is not None:
            return best, path, best_key
    return None, None, None

def ann_sharpe(x, freq=12, flip_sign=True):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan, x.size
    mu = x.mean()
    sd = x.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan, x.size
    if flip_sign and mu < 0:
        mu = -mu
    return (mu / sd) * np.sqrt(freq), x.size

rows = []
for m in IMPUTATION_METHODS:
    F, path, key = load_factor_matrix(FOLDER, m)
    if F is None:
        print(f"[skip] No factor returns found for method: {m}")
        continue
    T, K = F.shape
    for k in range(K):
        s, n = ann_sharpe(F[:, k], freq=FREQ, flip_sign=FLIP_SIGN)
        rows.append({"method": m, "factor_idx": k+1, "ann_sharpe": s, "n_obs": n, "file": os.path.basename(path), "key": key})

df = pd.DataFrame(rows).sort_values("ann_sharpe", ascending=False, na_position="last")
top6 = df.head(6)

# Save & print
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
top6.to_csv(OUT_CSV, index=False)
print("\nTop-6 factors by annualized Sharpe (across all methods):")
print(top6.to_string(index=False))
print(f"\nSaved to: {OUT_CSV}")
