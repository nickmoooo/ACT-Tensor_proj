import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# 1. Define your folder containing the factor NPZ files.
factors_folder = os.path.join('Portfolio_analysis', 'excess_tensors')

# 2. List of imputation methods for which you have factor files.
imputation_methods = [
    'cp_pooling_smooth_cma',
]

# 3. Define the full list of characteristics (row‐labels for V_C).
# characteristics = [
#     'A2ME', 'AC', 'AT', 'ATO', 'B2M', 'BETA_d', 'BETA_m', 'C2A', 
#     'CF2B', 'CF2P', 'CTO', 'D2A', 'D2P', 'DPI2A', 'E2P', 'FC2Y', 
#     'IdioVol', 'INV', 'LEV', 'TURN', 'NI', 'NOA', 'OA', 'OL', 
#     'OP', 'PCM', 'PM', 'PROF', 'Q', 'R2_1', 'R12_2', 'R12_7', 
#     'R36_13', 'R60_13', 'HIGH52', 'RVAR', 'RNA', 'ROA', 'ROE', 
#     'S2P', 'SGA2S', 'SPREAD', 'SUV', 'VAR'
# ]
characteristics = ['AC','B2M','BETA_m', 'IdioVol','INV','LEV','PROF','R12_2','RVAR', 'SPREAD', 'TURN']

# 4. Create an output PDF to store all heatmaps.
output_pdf = os.path.join("Portfolio_analysis", "factor_weights", "all_imputation_methods_heatmaps.pdf")

# Pre-filter to avoid empty-PDF warning:
methods_with_file = []
for method in imputation_methods:
    candidate = os.path.join(factors_folder, f"tensor_{method}_OOS_LOADING.npz")
    if os.path.exists(candidate):
        methods_with_file.append(method)

if len(methods_with_file) == 0:
    print("No factor files found. Exiting without creating PDF.")
else:
    with PdfPages(output_pdf) as pdf:
        for method in methods_with_file:
            npz_path = os.path.join(factors_folder, f"tensor_{method}_OOS_LOADING.npz")
            data = np.load(npz_path)
            V_C = data.get("V_C", None)  # shape = (num_characteristics, K_c)
            V_P = data.get("V_P", None)  # shape = (P, K_p)
            V_Q = data.get("V_Q", None)  # shape = (Q, K_q)

            #
            # ────────────────────────────────────────────────────────────────────
            # PART A: Plot V^(C) all K_c columns in one combined heatmap,
            #         with dynamically computed page size, but narrower cells
            #         and extra top margin so the title is fully visible.
            # ────────────────────────────────────────────────────────────────────
            #
            if V_C is not None:
                num_chars, K_c = V_C.shape

                # === 1) Choose per‐cell dimensions in inches ===
                #    - Each characteristic row will be 0.15" tall (was 0.18")
                row_height_in = 1    # ← reduce this if cells are too tall
                #    - Each factor column will be 1.0" wide (was 1.2")
                col_width_in  = 1.0       # ← reduce this if cells are too wide

                #    - Extra margin at top for the title (in inches)
                top_margin_in = 1.5       # ← increase this if title is getting cut off
                #    - Extra margin on each side (left + right) for tick‐labels
                side_margin_in = 1.0

                # === 2) Compute figure height and width ===
                # Make sure it's at least 6" tall, but scale up if many rows:
                fig_height = max(8, num_chars * row_height_in + top_margin_in)
                # Make sure it's at least 6" wide, but scale up if many columns:
                fig_width  = 8

                # === 3) Create figure with exactly that size ===
                fig, ax_c = plt.subplots(
                    nrows=1,
                    ncols=1,
                    figsize=(fig_width, fig_height),
                    constrained_layout=False
                )

                # === 4) Build a DataFrame for all K_c factors side by side ===
                row_labels = characteristics
                col_labels = [r"$\mathbf{v}^{(C)}_{%d}$" % (k+1) for k in range(K_c)]
                df_VC = pd.DataFrame(V_C, index=row_labels, columns=col_labels)

                # === 5) Plot the combined heatmap ===
                sns.heatmap(
                    df_VC,
                    ax=ax_c,
                    cmap="RdBu_r",
                    center=0,
                    annot=True,
                    fmt=".2f",
                    square=True,       # allow rectangular cells
                    linewidths=0,       # no white gridlines
                    cbar=False,          # no colorbar in Part A
                    annot_kws={"weight": "bold", "fontsize": 12}
                )

                # === 6) Title above the heatmap (slightly smaller font) ===
                ax_c.set_title(
                    r"$\widehat{W}$" ,
                    fontsize=12,   # shrink from 16 to 14 so it fits more easily
                    pad=12
                )

                # === 7) Remove axis labels (leave ticklabels only) ===
                ax_c.set_xlabel("")
                ax_c.set_ylabel("")

                # === 8) Format y‐tick labels (characteristic names) ===
                ax_c.tick_params(axis="y", labelsize=8)  # shrink from 7 to 6 if needed

                # === 9) Format x‐tick labels (Factor 1…Factor K_c) ===
                ax_c.tick_params(axis="x", labelsize=8)
                ax_c.set_xticklabels(
                    ax_c.get_xticklabels(),
                    fontsize=8,
                )

                # === 10) Tight layout, leaving top fraction for title ===
                #      Reserve top ~10% instead of 6% so title definitely fits
                plt.tight_layout(rect=[0, 0, 1, 0.90])

                # === 11) Save and close ===
                pdf.savefig(fig)
                plt.close(fig)


            #
            # ────────────────────────────────────────────────────────────────────
            # PART B: Plot V^(P) and V^(Q) side by side, share one colorbar [-1,1].
            # ────────────────────────────────────────────────────────────────────
            #
            if V_P is not None and V_Q is not None:
                P, K_p = V_P.shape    # Expect K_p = 3
                Q, K_q = V_Q.shape    # Expect K_q = 3

                # 1) Build DataFrames with the exact column headers you want:
                size_labels = [f"P{i+1}" for i in range(P)]
                col_names_P = [r"$\mathbf{u}_{%d}$" % (k+1) for k in range(K_p)]
                df_VP = pd.DataFrame(V_P, index=size_labels, columns=col_names_P)

                charbin_labels = [f"Q{j+1}" for j in range(Q)]
                col_names_Q = [r"$\mathbf{v}_{%d}$" % (k+1) for k in range(K_q)]
                df_VQ = pd.DataFrame(V_Q, index=charbin_labels, columns=col_names_Q)

                # 2) Fix the color range to [-1, +1]:
                vmin_PQ, vmax_PQ = -1.0, +1.0

                # 3) Create a Figure + GridSpec so we can place one colorbar to the right
                fig2 = plt.figure(figsize=(12, 6))
                #   - 3 columns: [heatmap_P, heatmap_Q, colorbar]
                #   - width_ratios: heatmaps get “1,1”, colorbar gets “0.12” (tweak as needed)
                gs = fig2.add_gridspec(
                    1, 3,
                    width_ratios=[1, 1, 0.12],
                    wspace=0.3
                )
                ax_p = fig2.add_subplot(gs[0, 0])
                ax_q = fig2.add_subplot(gs[0, 1])

                # 4) Common heatmap kwargs
                hm_kwargs_PQ = {
                    "cmap": "RdBu_r",
                    "center": 0,
                    "annot": True,
                    "fmt": ".2f",
                    "square": True,
                    "linewidths": 0.5,
                    "linecolor": "white",
                    # We do NOT draw a colorbar here (use cbar=False)
                    "cbar": False,
                    # Bold annotation
                    "annot_kws": {"weight": "bold", "fontsize": 12}
                }

                # ─── Plot V^(P) on ax_p ─────────────────────────────────────────
                sns.heatmap(
                    df_VP,
                    ax=ax_p,
                    **hm_kwargs_PQ
                )
                ax_p.set_title(
                    r"$\widehat{U}$",
                    fontsize=12, pad=12
                )
                ax_p.set_xlabel("")  # no x‐label
                ax_p.set_ylabel("")  # no y‐label
                # Row ticks (P1…P5)
                ax_p.tick_params(axis="y", labelsize=12)
                # Column ticks (v^(P)_i) rotated & center‐aligned
                ax_p.tick_params(axis="x", labelsize=12)

                # ─── Plot V^(Q) on ax_q ─────────────────────────────────────────
                sns.heatmap(
                    df_VQ,
                    ax=ax_q,
                    **hm_kwargs_PQ
                )
                ax_q.set_title(
                    r"$\widehat{V}$",
                    fontsize=12, pad=12
                )
                ax_q.set_xlabel("")
                ax_q.set_ylabel("")
                ax_q.tick_params(axis="y", labelsize=12)
                ax_q.tick_params(axis="x", labelsize=12)

                # ─── Reserve top ~7% of the figure for the two titles ──────────
                # The rectangle [left, bottom, right, top] is in figure‐fraction coordinates.
                plt.tight_layout(rect=[0, 0, 0.98, 0.93])

                pdf.savefig(fig2)
                plt.close(fig2)

                print(f"[✓] Saved combined V^(P), V^(Q) page for method = {method}")


            #
            # ────────────────────────────────────────────────────────────────────
            # PART C: Plot the 3×3 grid of W^{(PQ)} – unchanged from your original.
            # ────────────────────────────────────────────────────────────────────
            #
            if (V_P is not None) and (V_Q is not None):
                K_p = V_P.shape[1]
                K_q = V_Q.shape[1]
                size_labels    = [f"P{i+1}" for i in range(V_P.shape[0])]
                charbin_labels = [f"Q{j+1}" for j in range(V_Q.shape[0])]

                fig, axes = plt.subplots(
                    nrows=K_p,
                    ncols=K_q,
                    figsize=(4 * K_q, 4 * K_p),
                    constrained_layout=False
                )

                heatmap_kwargs = {
                    "cmap": "RdBu_r",
                    "center": 0,
                    "annot": True,
                    "fmt": ".2f",
                    "square": True,
                    "linewidths": 0.5,
                    "linecolor": "white",
                    "cbar": False,  
                    # "cbar_kws": {"shrink": 0.6, "pad": 0.02}
                    "annot_kws": {"weight": "bold", "fontsize": 8}
                }

                panel_letters = [chr(ord("A") + i) for i in range(K_p * K_q)]
                idx = 0
                for p_idx in range(K_p):
                    for q_idx in range(K_q):
                        W_pq = np.outer(V_P[:, p_idx], V_Q[:, q_idx])
                        dfW = pd.DataFrame(W_pq,
                                           index=size_labels,
                                           columns=charbin_labels)

                        ax = axes[p_idx, q_idx]
                        sns.heatmap(
                            dfW,
                            ax=ax,
                            **heatmap_kwargs,
                            # cbar=(q_idx == K_q - 1)
                        )
                        letter = panel_letters[idx]
                        ax.set_title(f"{letter}: " + r"$\mathbf{W}^{(PQ)}_{" + f"{p_idx+1}{q_idx+1}" + r"}$",
                                     fontsize=11, pad=8)
                        ax.set_xlabel("")
                        ax.set_ylabel("")
                        idx += 1

                # fig.suptitle(r"$\mathbf{W}^{(PQ)}_{pq}$",
                #              fontsize=16, y=0.98)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                pdf.savefig(fig)
                plt.close(fig)

            print(f"[✓] Finished plotting all pages for method = {method}")

    print(f"All heatmaps saved to {output_pdf}")
