import numpy as np
import imputation_metrics
import matplotlib.pyplot as plt
import tensorly as tl
import pandas as pd
import os
from imputation_metrics import get_imputation_metrics_all, get_sparse_imputation_metrics

line_styles = [
     'solid',      # Same as (0, ()) or '-'
     'dotted',    # Same as (0, (1, 1)) or ':'
     'dashed',    # Same as '--'
     'dashdot',  # Same as '-.'
     (5, (10, 3)),
     (0, (5, 10)),
     (0, (3, 10, 1, 10)),
     (0, (3, 5, 1, 5, 1, 5)),
     (0, (3, 1, 1, 1, 1, 1)),

    # --- new additions ---
    (0, (4, 2)),               # dash, short gap
    (0, (2, 2)),               # dash, dash, gap
    (0, (1, 3)),               # dot, longer gap
    (0, (1, 1, 5, 1)),         # dot, gap, longer dash, gap
    (0, (5, 2, 1, 2)),         # dash, gap, dot, gap
    (0, (2, 1, 1, 1, 1, 1)),   # dash, dot, dot, dot…
    (0, (10, 1, 1, 1)),        # extra-long dash, dot, dot, dot
    (0, (1, 1, 1, 1, 5, 1)),   # dots then a long dash
]


from collections import defaultdict
def get_nyse_permnos_mask(dates_ap, permnos):
    '''
    get a boolean mask for the permno's of companies which are listed on the NYSE at a certain point in time
    '''
    permnos_nyse_data = pd.io.parsers.read_csv('nyse_permnos.csv')
    permnos_nyse_data[['permno', 'date']].to_numpy()
    permnos_to_ind = {}
    for i, p in enumerate(permnos):
        permnos_to_ind[p] = i

    dates_to_ind = defaultdict(list)
    for i, date in enumerate(dates_ap):
        dates_to_ind[date // 10000].append(i)
    T,N,_ = percentile_rank_chars.shape
    permno_mask = np.zeros((T,N), dtype=bool)
    for permno, date in permnos_nyse_data[['permno', 'date']].to_numpy():
        date = int(date.replace('-', ''))
        if date//10000 in dates_to_ind and permno in permnos_to_ind:

            pn_ind = permnos_to_ind[permno]
            for d_ind in dates_to_ind[date//10000]:
                permno_mask[d_ind, pn_ind] = 1
    return permno_mask

def get_deciles_nyse_cutoffs(permno_mask, size_chars):
    '''
    get the cutoffs for the size deciles over time based only on companies listed on NYSE
    '''
    T, N = size_chars.shape
    decile_data = np.zeros((T, N, 10))
    to_decide_deciles = np.logical_and(~np.isnan(size_chars), permno_mask)
    for t in range(T):
        valid_values_sorted = np.sort(size_chars[t, to_decide_deciles[t]])
        interval_size = int(valid_values_sorted.shape[0] / 10)
        cutoffs = [[valid_values_sorted[i * interval_size], 
                    valid_values_sorted[min((i+1) * interval_size, 
                       valid_values_sorted.shape[0] - 1)]] for i in range(10)]
        cutoffs[-1][1] = 2 # don't ignore the biggest stock lol
        cutoffs[0][0] = -1
        for i in range(10):
            in_bucket = np.logical_and(size_chars[t,:] > cutoffs[i][0], 
                                      size_chars[t,:] <= cutoffs[i][1])
            decile_data[t,in_bucket,i] = 1
    return decile_data


def plot_metrics_over_time(metrics, names, dates, save_name=None, extra_line=None, nans_ok=False):
    '''
    utility method to plot the imputation metrics over time
    '''
    save_base = '../images-pdfs/section5/metrics_over_time-'
    

    date_vals = np.array(dates) // 10000 + ((np.array(dates) // 100) % 100) / 12
    
    start_idx = metrics[0][0][0].shape[0] - date_vals.shape[0]

    plot_names = ["aggregate", "quarterly_chars", "monthly_chars"]
    
    for i, plot_name in enumerate(plot_names):
        plt.tight_layout() 
        fig, axs = plt.subplots(1, 1, figsize=(20,10))
        fig.patch.set_facecolor('white')

        for j, (data, label) in enumerate(zip(metrics, names)):

            metrics_i = data[i]
            
            label = f'{label}'
            if nans_ok:
                plt.plot(dates, np.sqrt(np.nanmean(np.array(metrics_i), axis=0))[start_idx:], label=label,
                        linestyle=line_styles[j])
            else:
                plt.plot(dates, np.sqrt(np.mean(np.array(metrics_i), axis=0))[start_idx:], label=label,
                        linestyle=line_styles[j])

        if extra_line is not None:
            ax2 = axs.twinx()
            ax2.plot(dates, extra_line, label="extra_line", c='red')
            ax2.legend(prop={'size': 14})
        if i == 0:
            axs.legend(prop={'size': 20}, loc='upper center', bbox_to_anchor=(0.5, 1.2),
              ncol=4, framealpha=1)
        
        if save_name is not None:
            plt.title(f'RMSE over time for {save_name}')
            fig.savefig(save_base + save_name + f'-{plot_name}.pdf', bbox_inches='tight')
            
        plt.show()
        
        
def plot_metrics_by_mean_vol(mean_vols, input_metrics, names, chars, save_name=None, ylabel=None):
    '''
    utility method to plot the imputation metrics by each characteristic, with the characteristics 
    ordered in increasing volatility
    '''
    char_names = []
    metrics_by_type = [[] for _ in input_metrics] 

    for i in np.argsort(mean_vols):
        metrics = [round(np.sqrt(np.nanmean(y[0][i])), 5) for y in input_metrics]
        char_names.append(chars[i])
        for j, m in enumerate(metrics):
            metrics_by_type[j].append(m)
    plt.tight_layout() 
    fig = plt.figure(figsize=(20,10))
    fig.patch.set_facecolor('white')
    # mycolors = ['#152eff', '#e67300', '#0e374c', '#6d904f', '#8b8b8b', '#30a2da', '#e5ae38', '#fc4f30', '#6d904f', '#8b8b8b', '#0e374c']
    mycolors = [
    '#152eff',  # deep blue
    '#e67300',  # burnt orange
    '#0e374c',  # dark teal
    '#6d904f',  # olive green
    '#8b8b8b',  # medium grey
    '#30a2da',  # bright cyan
    '#e5ae38',  # mustard yellow
    '#fc4f30',  # tomato red
    '#9467bd',  # muted purple
    '#d62728',  # brick red
    '#8c564b',  # chocolate brown
    '#bcbd22',  # olive drab
    '#17becf',  # aquamarine
    '#e377c2',  # soft pink
    '#7f7f7f',  # slate grey
    ]
    for j, (c, line_name, metrics_series) in enumerate(zip(mycolors, names,
                                 metrics_by_type)):
        plt.plot(np.arange(45), metrics_series, label=line_name, c=c, linestyle=line_styles[j])
    plt.plot(np.arange(45), np.array(mean_vols)[np.argsort(mean_vols)], label="mean volatility of char", c='black')
    plt.xticks(np.arange(45), chars[np.argsort(mean_vols)], rotation='vertical')
    if ylabel is None:
        plt.ylabel("RMSE")
    else:
        plt.ylabel("ylabel")
    plt.legend(prop={'size': 14}, loc='center', bbox_to_anchor=(1.25, 0.5), ncol=1, framealpha=1)
    plt.minorticks_off()
    
    if save_name is not None:
        save_base = '../images-pdfs/section5/metrics_by_char_vol_sort-'
        save_path = save_base + save_name + '.pdf'
        plt.title(f'RMSE by characteristics for {save_name}')
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        
    plt.show()

def save_imputation(imputed_data, dates, permnos, chars, name):
    base_path = '/Users/nickmoooo/Desktop/Research/SPUR/missing_data_pub-main/data/imputation_cache/'
    result_file_name = base_path + name + '.npz'
    np.savez(result_file_name, data=imputed_data, dates=dates, permnos=permnos, chars=chars)
    
def load_imputation(name, full=False):
    base_path = '/Users/nickmoooo/Desktop/Research/SPUR/missing_data_pub-main/data/imputation_cache/'
    result_file_name = base_path + name + '.npz'
    res = np.load(result_file_name)
    if not full:
        return res['data']
    else:
        return res['data'], res['dates'], res['permnos'], res['chars']
    

def save_metrics_per_tag(
    imputed_chars,
    eval_char_data,
    monthly_update_mask,
    char_groupings,
    sparse_firms,
    tag=None,
    norm_func=None,
    clip=True,
    output_dir='/Users/nickmoooo/Desktop/Research/SPUR/missing_data_pub-main/src/error_metrics/'
):
    """
    For each tag in `tags`, compute overall and sparse metrics,
    then write to {output_dir}/metrics<TAG>.csv with columns:
      metric, overall, sparse
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Determine file path and next method index
    file_path = f"{output_dir}{tag.replace('_', '', 1)}.csv"
    if os.path.exists(file_path):
        try:
            df_existing = pd.read_csv(file_path)
            if 'method_index' in df_existing.columns and not df_existing.empty:
                # coerce to numeric, drop non‐numeric entries
                idx_ser = pd.to_numeric(df_existing['method_index'], errors='coerce')
                valid = idx_ser.dropna().astype(int)
                if not valid.empty:
                    method_index = valid.max() + 1
                else:
                    method_index = 1
            else:
                method_index = 1
        except Exception:
            method_index = 1
    else:
        method_index = 1

    # 1) overall
    overall = get_imputation_metrics_all(
        imputed_chars,
        eval_char_data,
        char_groupings,
        net_mask=None,
        norm_funcs=norm_func,
        clip=clip
    )
    # overall is {'rmse_overall':..., 'mae_overall':..., 'mape_overall':...}
    
    # # 2) sparse
    # sparse = get_sparse_imputation_metrics(
    #     imputed_chars,
    #     eval_char_data,
    #     sparse_firms,
    #     char_groupings,
    #     net_mask=None,
    #     norm_funcs=norm_func
    # )
    # # sparse is same structure
    
    # 3) build DataFrame
    df = pd.DataFrame({
        'method_index': [method_index] * 4,
        "metric": ["rmse", "mae", "mape", "r2"],
        "overall": [
            overall["rmse_overall"],
            overall["mae_overall"],
            overall["mape_overall"],
            overall["r2_overall"]
        ],
        # "sparse": [
        #     sparse["rmse_overall"],
        #     sparse["mae_overall"],
        #     sparse["mape_overall"],
        #     sparse["r2_overall"]
        # ]
    })
    
    # 4) save
    df.to_csv(file_path, mode='a', index=False)
    # print(f"Wrote metrics to {file_path}")


def get_imputation_metrics(imputed_chars, eval_char_data, monthly_update_mask, char_groupings, norm_func=None,
                          clip=True, tag=None):
    '''
    utility method to calculate RMSE metrics for imputed chars
    '''
    
    by_char_metrics, by_char_m_metrics, by_char_q_metrics  = imputation_metrics.get_aggregate_imputation_metrics(imputed_chars,
                                                          eval_char_data, None, monthly_update_mask, char_groupings,
                                                          norm_func=norm_func, clip=clip)

    sparse_firms = np.loadtxt(f"sparse_firms{tag}.txt", dtype=int).tolist() ################### NEED TO REVISE SPARSE FIRMS HERE BASED ON OUR NEW DATA
    save_metrics_per_tag(imputed_chars, eval_char_data, None,
                            char_groupings, sparse_firms, tag=tag, norm_func=norm_func, clip=clip)

    return by_char_metrics, by_char_q_metrics, by_char_m_metrics


def get_present_flags(raw_char_panel):
    '''
    utility method to get state of a characteristic from 
    - observed
    - missing at the start
    - missing in the middle
    - missing at the end
    - company not observed
    '''
    T, N, C = raw_char_panel.shape
    flag_panel = np.zeros_like(raw_char_panel, dtype=np.int8)
    
    first_occurances = np.argmax(~np.isnan(raw_char_panel), axis=0)
    not_in_sample = np.all(np.isnan(raw_char_panel), axis=0)
    last_occurances = T - 1 - np.argmax(~np.isnan(raw_char_panel[::-1]), axis=0)
    
    for t in range(raw_char_panel.shape[0]):
        
        present_mask = ~np.isnan(raw_char_panel[t])
        previous_entry = t == first_occurances
        next_entry = t == last_occurances
        
        flag_panel[t, np.logical_and(present_mask, previous_entry)] = -1
        
        flag_panel[t, np.logical_and(present_mask, next_entry)] = -3
        
        both = np.logical_and(t > first_occurances, t < last_occurances)
        
        flag_panel[t, np.logical_and(present_mask, both)] = -2
        previous_entry[present_mask] = 1
    flag_panel[:,not_in_sample] = 0
    return flag_panel



char_groupings  = [('A2ME', "Q"),
                   ('AC', 'Q'),
('AT', 'Q'),
('ATO', 'Q'),
('B2M', 'QM'),
('BETA_d', 'M'),
('BETA_m', 'M'),
('C2A', 'Q'),
('CF2B', 'Q'),
('CF2P', 'QM'),
('CTO', 'Q'),
('D2A', 'Q'),
('D2P', 'M'),
('DPI2A', 'Q'),
('E2P', 'QM'),
('FC2Y', 'QY'),
('IdioVol', 'M'),
('INV', 'Q'),
('LEV', 'Q'),
('ME', 'M'),
('TURN', 'M'),
('NI', 'Q'),
('NOA', 'Q'),
('OA', 'Q'),
('OL', 'Q'),
('OP', 'Q'),
('PCM', 'Q'),
('PM', 'Q'),
('PROF', 'QY'),
('Q', 'QM'),
('R2_1', 'M'),
('R12_2', 'M'),
('R12_7', 'M'),
('R36_13', 'M'),
('R60_13', 'M'),
('HIGH52', 'M'),
('RVAR', 'M'),
('RNA', 'Q'),
('ROA', 'Q'),
('ROE', 'Q'),
('S2P', 'QM'),
('SGA2S', 'Q'),
('SPREAD', 'M'),
('SUV', 'M'),
('VAR', 'M')]
char_maps = {x[0]:x[1] for x in char_groupings}
char_map = char_maps



def build_4d_tensor_for_method(
    src,                       # str (folder)  OR ndarray (T,N,L)
    imputation_method=None,    # str – only if reading CSVs
    characteristics=None,      # list[str] – always required
    time_index=None,           # DatetimeIndex – always required
    *,
    size_bins=20,
    char_bins=20,
    variant="default",
    # --- only needed when src is an ndarray -----------------------------
    bin_lookup=None            # pd.DataFrame with cols:
                               # company, size_quintile, char_quintile
):
    """
    Build a tensor X[t,c,p,q]  (T × C × P × Q).

    Two modes
    ---------
    1)  CSV mode (back-compat) ::
            X = build_4d_tensor_for_method(
                    'excess_returns', 'tucker_pooling', char_list, dates)
    2)  In-memory mode ::
            X = build_4d_tensor_for_method(
                    imputed_panel,       # ndarray (T,N,L)
                    characteristics=char_list,
                    time_index=dates,
                    bin_lookup=meta_df   # company→{size_q,char_q}
            )
    """
    # ------------------------------------------------------------------
    #  A. INPUT  VALIDATION
    # ------------------------------------------------------------------
    if isinstance(src, str):                      # ------- CSV mode ----
        folder = src
        if imputation_method is None:
            raise ValueError("Need `imputation_method=` when src is a folder")
    else:                                         # ------- ndarray mode
        imputed = np.asarray(src)
        if imputed.ndim != 3:
            raise ValueError("imputed panel must be 3-D (T,N,L)")
        if bin_lookup is None:
            raise ValueError("Need `bin_lookup=` (company→quintiles) "
                             "when src is a panel ndarray")
    if characteristics is None or time_index is None:
        raise ValueError("`characteristics` and `time_index` are required")

    T   = len(time_index)
    C   = len(characteristics)
    P   = size_bins
    Q   = char_bins
    X4  = np.full((T, C, P, Q), np.nan)

    # ------------------------------------------------------------------
    #  B. CSV  MODE
    # ------------------------------------------------------------------
    if isinstance(src, str):
        folder = src
        for c_idx, char_name in enumerate(characteristics):
            fname = (f"excess_returns_{imputation_method}_ME_{char_name}_20%.csv"
                     if variant == "20%" else
                     f"excess_returns_{imputation_method}_ME_{char_name}.csv")
            fpath = os.path.join(folder, fname)
            if not os.path.exists(fpath):
                print(f"[skip] {fname} not found")
                continue
            df = pd.read_csv(fpath, parse_dates=["date"])

            for t_idx, t_date in enumerate(time_index):
                sub = df.loc[df["date"] == t_date]
                if sub.empty:
                    continue
                for _, row in sub.iterrows():
                    p = int(row["size_quintile"])
                    q = int(row["char_quintile"])
                    if 1 <= p <= P and 1 <= q <= Q:
                        X4[t_idx, c_idx, p-1, q-1] = row["excess_return"]

    # ------------------------------------------------------------------
    #  C. IN-MEMORY  MODE
    # ------------------------------------------------------------------
    else:
        # imputed has shape (T,N,L)  ;  bin_lookup has company→(p,q)
        if imputed.shape[0] != T:
            raise ValueError("time dimension of panel ≠ len(time_index)")

        # bin_lookup must have one row per company (index 0..N-1)
        if not {"size_quintile", "char_quintile"}.issubset(bin_lookup.columns):
            raise ValueError("bin_lookup must have 'size_quintile' "
                             "and 'char_quintile' columns")

        for n in range(imputed.shape[1]):
            p = int(bin_lookup.loc[n, "size_quintile"])
            q = int(bin_lookup.loc[n, "char_quintile"])
            if not (1 <= p <= P and 1 <= q <= Q):
                continue
            for c_idx, char_name in enumerate(characteristics):
                X4[:, c_idx, p-1, q-1] = imputed[:, n, c_idx]

    # ------------------------------------------------------------------
    #  D. ZERO-FILL AND RETURN
    # ------------------------------------------------------------------
    X4 = np.nan_to_num(X4, nan=0.0)
    return X4



def unfold_ignore_time(X, mode, T, C, P, Q):
    """
    Unfold the 4D tensor X[t, c, p, q] into a 2D matrix ignoring the time dimension.
    mode = 0: characteristic (c) dimension
    mode = 1: size (p) dimension
    mode = 2: characteristic-bin (q) dimension
    """
    if mode == 0:
        return X.transpose(1, 0, 2, 3).reshape(C, T*P*Q)
    elif mode == 1:
        return X.transpose(2, 0, 1, 3).reshape(P, T*C*Q)
    else:
        return X.transpose(3, 0, 1, 2).reshape(Q, T*C*P)

def tucker_3D_PCA(X, K_c, K_p, K_q):
    """
    Perform a naive HOSVD-like decomposition of X[t, c, p, q] ignoring the time dimension.
    Returns factor matrices V_C, V_P, V_Q and the core tensor G.
    """
    T, C, P, Q = X.shape
    
    X_c = unfold_ignore_time(X, mode=0, T=T, C=C, P=P, Q=Q)
    U_c, _, _ = np.linalg.svd(X_c, full_matrices=False)
    V_C = U_c[:, :K_c]
    
    X_p = unfold_ignore_time(X, mode=1, T=T, C=C, P=P, Q=Q)
    U_p, _, _ = np.linalg.svd(X_p, full_matrices=False)
    V_P = U_p[:, :K_p]
    
    X_q = unfold_ignore_time(X, mode=2, T=T, C=C, P=P, Q=Q)
    U_q, _, _ = np.linalg.svd(X_q, full_matrices=False)
    V_Q = U_q[:, :K_q]
    
    X_temp = np.einsum('tcpq,ck->tkpq', X, V_C)
    X_temp2 = np.einsum('tkpq,pl->tklq', X_temp, V_P)
    G = np.einsum('tklq,qm->tklm', X_temp2, V_Q)
    
    return V_C, V_P, V_Q, G


def estimate_factor_loadings(excess_returns, factor_matrix):
    """
    Now aligns so that R_{i,t+1} is regressed on F_t.

    excess_returns : array, shape (T, N)
    factor_matrix  : array, shape (T, L)

    Returns
    -------
    alpha : array, shape (N,)
    betas : array, shape (N, L)
    """
    T, N = excess_returns.shape
    _, L = factor_matrix.shape

    # 1) drop the last factor row (no t to predict t+1)
    F = factor_matrix[:-1, :]      # shape (T-1, L)
    # 2) drop the first return row (we have nothing to predict at t=0)
    R = excess_returns[1:, :]      # shape (T-1, N)

    # build X = [1, F_t]
    X_reg = np.hstack([np.ones((T-1,1)), F])

    alpha = np.zeros(N)
    betas = np.zeros((N, L))

    for i in range(N):
        y = R[:, i]
        coef, _, _, _ = np.linalg.lstsq(X_reg, y, rcond=None)
        alpha[i]  = coef[0]
        betas[i,:] = coef[1:]
    return alpha, betas


def build_candidate_factors(X, V_C, V_P, V_Q):
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