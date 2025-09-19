import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# -----------------------------
# Helpers for binning / merging
# -----------------------------
def _assign_bins(series, n_bins, use_quantiles=True):
    """Return integer bin labels in {1..n_bins}; fall back deterministically."""
    s = pd.Series(series)
    if use_quantiles:
        try:
            cats = pd.qcut(s, q=n_bins, labels=False, duplicates='drop')
            if pd.isna(cats).all():
                raise ValueError
            # If duplicates dropped, cats max may be < n_bins-1
            labels = cats.astype(float)
            # Remap to 1..B_used
            uniq = np.sort(pd.unique(labels.dropna()))
            mapper = {old: i+1 for i, old in enumerate(uniq)}
            labels = labels.map(mapper)
            return labels.astype('Int64').astype(int)
        except Exception:
            pass
    # fallback to equal-width
    cats = pd.cut(s, bins=n_bins, labels=False, include_lowest=True)
    labels = cats.astype(float)
    uniq = np.sort(pd.unique(labels.dropna()))
    mapper = {old: i+1 for i, old in enumerate(uniq)}
    labels = labels.map(mapper)
    return labels.astype('Int64').astype(int)

def _merge_sparse_char_bins_within_size(df_bins, min_n, n_bins_char):

    df_bins = df_bins.copy()
    df_bins['char_eff'] = df_bins['char_quintile']

    mapping = {}  # size_bin -> dict eff_bin: [orig_bins]
    for s in sorted(df_bins['size_quintile'].dropna().unique()):
        mask_s = (df_bins['size_quintile'] == s)

        # initialize groups with ALL possible bins (so empties are included)
        groups = {q: [q] for q in range(1, n_bins_char + 1)}

        while True:
            # counts for current effective bins (0 for empty)
            counts = (df_bins.loc[mask_s]
                      .groupby('char_eff').size()
                      .reindex(sorted(groups.keys()), fill_value=0))

            if len(groups) <= 1:
                break
            # check only active groups
            active_keys = list(groups.keys())
            if counts[active_keys].min() >= min_n:
                break

            # pick the smallest bin to merge
            q_min = int(counts[active_keys].idxmin())

            # choose nearest neighbor (tie-breaker: larger count, then right)
            keys_sorted = sorted(groups.keys())
            idx = keys_sorted.index(q_min)
            if idx == 0:
                q_nei = keys_sorted[1]
            elif idx == len(keys_sorted) - 1:
                q_nei = keys_sorted[-2]
            else:
                left, right = keys_sorted[idx - 1], keys_sorted[idx + 1]
                if counts.loc[right] != counts.loc[left]:
                    q_nei = right if counts.loc[right] > counts.loc[left] else left
                else:
                    q_nei = right  # deterministic tie

            # merge q_min into q_nei
            df_bins.loc[mask_s & (df_bins['char_eff'] == q_min), 'char_eff'] = q_nei
            groups[q_nei].extend(groups[q_min])
            del groups[q_min]

        mapping[s] = groups

    return df_bins, mapping


def _aggregate_and_expand(df_bins, mapping, n_bins_char, rf_col='rf'):

    rf_val = float(df_bins[rf_col].iloc[0])

    def agg_func(g):
        w = g['ME']
        tot = w.sum()
        vwret = np.nan if tot == 0 else float((g['ret'] @ w) / tot)
        return pd.Series({'vwret': vwret, 'n_firms': int(len(g))})

    # aggregated returns for effective groups that actually have stocks
    agg = (df_bins
           .groupby(['size_quintile', 'char_eff'], as_index=False)
           .apply(agg_func))

    # build a full map (size, char_quintile) -> char_eff using the provided mapping
    size_levels = sorted(df_bins['size_quintile'].dropna().unique())
    map_rows = []
    for s in size_levels:
        groups = mapping.get(s)
        if not groups:
            # identity mapping if somehow missing
            groups = {q: [q] for q in range(1, n_bins_char + 1)}
        # invert to char_quintile -> char_eff
        inv = {}
        for eff, orig_list in groups.items():
            for q in orig_list:
                inv[q] = eff
        for q in range(1, n_bins_char + 1):
            # nearest fallback if q not found (shouldn't happen)
            eff = inv.get(q)
            if eff is None:
                eff = min(groups.keys(), key=lambda e: abs(e - q))
            map_rows.append((s, q, eff))

    df_map = pd.DataFrame(map_rows, columns=['size_quintile', 'char_quintile', 'char_eff'])

    # join agg back to each original (size,char) via its char_eff
    expanded = (df_map
                .merge(agg, on=['size_quintile', 'char_eff'], how='left'))

    expanded['excess_return'] = expanded['vwret'] - rf_val
    expanded = expanded.drop(columns=['vwret'])

    # we keep which effective group this cell inherited from
    expanded = expanded.rename(columns={'char_eff': 'merged_from'})

    # final schema
    return expanded[['size_quintile', 'char_quintile', 'excess_return', 'n_firms', 'merged_from']].reset_index(drop=True)


# -----------------------------
# Double-sort for one month (patched)
# -----------------------------
def compute_portfolio_returns_for_month(
    df, sort_var,
    n_bins_size=5,
    n_bins_char=5,
    min_n_per_cell=5,
    do_merge=True
):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['ME', sort_var, 'ret', 'rf'])
    if df.empty:
        return None

    df = df.copy()
    df['size_quintile'] = _assign_bins(df['ME'], n_bins_size, use_quantiles=True)
    df['char_quintile'] = _assign_bins(df[sort_var], n_bins_char, use_quantiles=True)

    if do_merge:
        df_m, mapping = _merge_sparse_char_bins_within_size(
            df[['permno', 'ret', 'rf', 'ME', 'size_quintile', 'char_quintile']].copy(),
            min_n=min_n_per_cell,
            n_bins_char=n_bins_char
        )
    else:
        df_m = df[['permno', 'ret', 'rf', 'ME', 'size_quintile', 'char_quintile']].copy()
        df_m['char_eff'] = df_m['char_quintile']
        # identity mapping per size bin
        mapping = {int(s): {q: [q] for q in range(1, n_bins_char + 1)}
                   for s in sorted(df_m['size_quintile'].dropna().unique())}

    expanded = _aggregate_and_expand(df_m, mapping, n_bins_char=n_bins_char)
    return expanded




# ==============================
# Main
# ==============================
if __name__ == '__main__':
    base = np.load('../data/raw_rank_trunk_chars.npz')

    raw_chars       = base['raw_chars']          # (T, N, num_chars[,1])
    rts             = base['rfs']                # (T,)
    return_panel    = base['returns']            # (T, N)
    chars           = base['chars']              # (num_chars,)
    dates           = base['dates']              # (T,)
    permnos         = base['permnos']            # (N,)

    if raw_chars.ndim == 4:
        raw_chars = raw_chars[:, :, :, 0]

    time_index = pd.to_datetime(dates.astype(str))

    chars_list = list(chars)
    size_index = chars_list.index("ME")
    double_sort_chars = [c for c in chars_list if c != "ME"]

    out_root = os.path.join('Portfolio_analysis')
    output_folder = os.path.join(out_root, 'excess_returns')
    coverage_folder = os.path.join(out_root, 'coverage')
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(coverage_folder, exist_ok=True)

    methods = [
        # "raw",
        # "cp",
        "cp_pooling_smooth_cma",
        # "cp_pooling_beta_cma",
        # "global_fwbw",
        # "local_bw",
        # "xs_median"

    ]

    n_bins_size = 5
    n_bins_char = 5
    min_n_per_cell = 2   # << tweak here
    do_merge = True

    for method in methods:
        print(f"\n>>> Processing method: {method}")

        # Load characteristic tensor
        if method == 'raw':
            reg_chars = raw_chars
            complete_mask = ~np.isnan(return_panel).any(axis=0)
            permnos_masked = permnos[complete_mask]
            return_panel_masked = return_panel[:, complete_mask]
            reg_chars_masked = reg_chars[:, complete_mask, :]
        else:
            impt = np.load(
                f'../data/imputation_cache/{method}_out_of_sample_block_LOADING.npz'
            )
            reg_chars = impt['data']
            if reg_chars.ndim == 4:
                reg_chars = reg_chars[:, :, :, 0]

            nan_before = int(np.isnan(reg_chars).sum())
            print(f"NaN count in reg_chars for {method}: {nan_before:,}")

            if method in ('global_fwbw', 'local_bw', 'cp_pooling_smooth_cma') and nan_before > 0:
                try:
                    global_npz = np.load(
                        '../data/imputation_cache/global_xs_out_of_sample_block.npz'
                    )
                    local_npz = np.load(
                        '../data/imputation_cache/local_xs_out_of_sample_block.npz'
                    )
                    cp_npz = np.load(
                        '../data/imputation_cache/global_xs_out_of_sample_block.npz'
                    )
                    global_data = global_npz['data']
                    local_data = local_npz['data']
                    cp_data = cp_npz['data']
                    if global_data.ndim == 4:
                        global_data = global_data[:, :, :, 0]
                    if local_data.ndim == 4:
                        local_data = local_data[:, :, :, 0]

                    if method in ('global_fwbw'):
                        m = np.isnan(reg_chars)
                        reg_chars = np.where(m, global_data, reg_chars)
                        nan_after = int(np.isnan(reg_chars).sum())
                        print(f"    Filled {nan_before - nan_after:,} NaNs in {method} (remaining: {nan_after:,}).")
                    elif method in ('local_bw'):
                        m = np.isnan(reg_chars)
                        reg_chars = np.where(m, local_data, reg_chars)
                        nan_after = int(np.isnan(reg_chars).sum())
                        print(f"    Filled {nan_before - nan_after:,} NaNs in {method} (remaining: {nan_after:,}).")
                    elif method in ('cp_pooling_beta_cma'):
                        m = np.isnan(reg_chars)
                        reg_chars = np.where(m, cp_data, reg_chars)
                        nan_after = int(np.isnan(reg_chars).sum())
                        print(f"    Filled {nan_before - nan_after:,} NaNs in {method} (remaining: {nan_after:,}).")

                except FileNotFoundError:
                    print("    [warn] global_xs fallback file not found; cannot fill NaNs.")

            # Only firms with complete return history
            complete_mask = ~np.isnan(return_panel).any(axis=0)
            permnos_masked = permnos[complete_mask]
            return_panel_masked = return_panel[:, complete_mask]
            reg_chars_masked = reg_chars[:, complete_mask, :]


        # Coverage tracker: per characteristic, a P×Q matrix of hits (post-merge)
        coverage_hits = {ch: np.zeros((n_bins_size, n_bins_char), dtype=int) for ch in double_sort_chars}
        coverage_den  = {ch: 0 for ch in double_sort_chars}

        drop_log = []

        for char in double_sort_chars:
            print(f"  • {char}")
            c_idx = chars_list.index(char)
            monthly_rows = []

            for t_idx, dt in enumerate(time_index):
                df_month = pd.DataFrame({
                    'permno': permnos_masked,
                    'ret'   : return_panel_masked[t_idx, :],
                    'rf'    : rts[t_idx],
                    'ME'    : reg_chars_masked[t_idx, :, size_index],
                    char    : reg_chars_masked[t_idx, :, c_idx],
                })

                if pd.isna(df_month['rf'].iloc[0]):
                    print(f"    [{dt.date()}] skipped: rf is NaN")
                    continue

                total_n = len(df_month)
                n_missing = int(df_month['ret'].isna().sum())
                df_month = df_month.loc[~df_month['ret'].isna()].copy()
                kept_n = len(df_month)

                drop_log.append({
                    'method': method, 'date': dt, 'characteristic': char,
                    'n_total': total_n, 'n_dropped_ret_na': n_missing,
                    'n_kept': kept_n, 'drop_rate': n_missing / total_n if total_n else np.nan
                })

                if df_month.empty:
                    continue

                port_df = compute_portfolio_returns_for_month(
                    df_month, sort_var=char,
                    n_bins_size=n_bins_size, n_bins_char=n_bins_char,
                    min_n_per_cell=min_n_per_cell,
                    do_merge=do_merge
                )
                if port_df is None:
                    continue

                full_index = pd.MultiIndex.from_product(
                    [range(1, n_bins_size+1), range(1, n_bins_char+1)],
                    names=['size_quintile','char_quintile']
                )
                port_df = port_df.set_index(['size_quintile','char_quintile']).reindex(full_index, fill_value=0).reset_index()

                port_df['date'] = dt
                monthly_rows.append(port_df)

            # Save CSV for this characteristic
            if monthly_rows:
                out_df = pd.concat(monthly_rows, ignore_index=True)
                out_df.sort_values(['date', 'size_quintile', 'char_quintile'], inplace=True)
                out_path = os.path.join(output_folder, f"excess_returns_{method}_ME_{char}.csv")
                out_df.to_csv(out_path, index=False)
            else:
                print(f"    (no valid rows for {char})")

            # Save coverage for this characteristic & method
            den = coverage_den[char]
            if den > 0:
                cov_frac = coverage_hits[char] / den
                cov_path = os.path.join(coverage_folder, f"coverage_{method}_{char}.npz")
                np.savez_compressed(cov_path,
                                    coverage_hits=coverage_hits[char],
                                    coverage_frac=cov_frac,
                                    n_months=den,
                                    size_bins=n_bins_size,
                                    char_bins=n_bins_char)
                print(f"    Saved coverage: {cov_path}")

        # Save the per-method drop log
        if drop_log:
            log_df = pd.DataFrame(drop_log)
            log_path = os.path.join(out_root, f"return_drop_log_{method}.csv")
            log_df.sort_values(['date', 'characteristic'], inplace=True)
            log_df.to_csv(log_path, index=False)
            print(f"Saved return drop log: {log_path}")
