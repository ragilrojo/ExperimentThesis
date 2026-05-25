import json
import os

nb_path = r'g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\guidiciti_reexperiment\RLNetworkMarkowitz_4Metrics_CalmarRatioReward_Fixed_modif.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def to_source(text):
    return [line + "\n" for line in text.split("\n")]

for cell in nb['cells']:
    source_text = "".join(cell.get('source', []))
    
    # Cell 1: Markdown Title & Table
    if cell['cell_type'] == 'markdown' and '# Ablation Study' in source_text:
        cell['source'] = to_source("""# Ablation Study: Feature Selection for SAC-Based Gamma Controller
## Evaluasi dengan 4 Metrik: Sharpe Ratio, Sortino Ratio, Calmar Ratio, CVaR (95%)

**Tujuan:** Mengevaluasi kontribusi masing-masing kelompok fitur terhadap performa portofolio berdasarkan **4 metrik risk-adjusted** yang komprehensif untuk keperluan tesis.

> **Catatan:** Eksperimen ini menggunakan **Calmar Ratio reward** (`ann_ret / |max_dd|`).
> Agent didorong memaksimalkan return per unit risiko kejatuhan terburuk.
> Cocok untuk kripto yang sangat volatile.

### Metrik Evaluasi (4 Metrik Utama Tesis)
| Metrik | Formula | Interpretasi |
|--------|---------|-------------|
| **Sharpe Ratio** | Ann.Return / Ann.Volatility | Return per unit volatilitas total |
| **Sortino Ratio** | Ann.Return / Downside Volatility | Lebih adil untuk kripto — hanya hukum volatilitas negatif |
| **Calmar Ratio** | Ann.Return / \|Max Drawdown\| | Return per unit risiko kejatuhan terburuk |
| **CVaR (95%)** | E[Loss \| Loss > VaR95%] | Rata-rata kerugian di skenario terburuk 5% (standar akademik) |

### Eksperimen Utama & Ablasi (Leave-One-Out)
| ID | Nama Eksperimen | Fitur Network | γ (Gamma) |
|----|----------------|---------------|-----------|
| Comp_Static_Gamma0/1/2 | **Static Baseline** | Lengkap (5) | Tetap (0/1/2) |
| E2_NoMarket | **Proposed** | Lengkap (5) | Dynamic (SAC) |
| E2_drop_CentStd | **Ablasi LOO** | Hapus F1 | Dynamic (SAC) |
| E2_drop_CentMean | **Ablasi LOO** | Hapus F2 | Dynamic (SAC) |
| E2_drop_MSTDist | **Ablasi LOO** | Hapus F3 | Dynamic (SAC) |
| E2_drop_MaxCent | **Ablasi LOO** | Hapus F4 | Dynamic (SAC) |
| E2_drop_NetDens | **Ablasi LOO** | Hapus F5 | Dynamic (SAC) |""")

    # Cell 2: Global Settings
    if cell['cell_type'] == 'code' and 'GLOBAL SETTINGS' in source_text:
        cell['source'] = to_source("""# ================================================================
# GLOBAL SETTINGS
# ================================================================
SEEDS         = [42, 123]          # [42, 123, 77] untuk multi-seed
TRAIN_STEPS   = 100
GAMMA_CENTER  = 0
SET_WINDOW    = 30
SET_REBALANCE = 7
REWARD_WINDOW = 20            # sliding window untuk Calmar Ratio reward
CVAR_LEVEL    = 0.95          # confidence level CVaR

# Definisi eksperimen ablation
ABLATION_CONFIGS = {
    # --- Baseline: Static Gamma (Full Network Features) ---
    'Comp_Static_Gamma0'      : {'use_network': True,  'use_market': False, 'extra_features': [], 'static_gamma': 0.0},
    'Comp_Static_Gamma1'      : {'use_network': True,  'use_market': False, 'extra_features': [], 'static_gamma': 1.0},
    'Comp_Static_Gamma2'      : {'use_network': True,  'use_market': False, 'extra_features': [], 'static_gamma': 2.0},

    # --- Proposed: E2_NoMarket (Full Network Features, SAC Dynamic) ---
    'E2_NoMarket'             : {'use_network': True,  'use_market': False, 'extra_features': []},

    # --- Ablation E2: Leave-One-Out (Drop 1 Network Feature) ---
    'E2_drop_CentStd'         : {'use_network': True,  'use_market': False, 'extra_features': [], 'drop_nw_idx': 0},
    'E2_drop_CentMean'        : {'use_network': True,  'use_market': False, 'extra_features': [], 'drop_nw_idx': 1},
    'E2_drop_MSTDist'         : {'use_network': True,  'use_market': False, 'extra_features': [], 'drop_nw_idx': 2},
    'E2_drop_MaxCent'         : {'use_network': True,  'use_market': False, 'extra_features': [], 'drop_nw_idx': 3},
    'E2_drop_NetDens'         : {'use_network': True,  'use_market': False, 'extra_features': [], 'drop_nw_idx': 4},
}

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import seaborn as sns
import warnings
import os
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from scipy.optimize import minimize

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
os.makedirs('ablation_results_4metrics', exist_ok=True)

print('Libraries loaded.')
print(f'Ablation configs: {list(ABLATION_CONFIGS.keys())}')
print(f'CVaR confidence level: {CVAR_LEVEL*100:.0f}%')""")

    # Cell 4: Core Functions (specifically the colors/exps part)
    if cell['cell_type'] == 'code' and 'ABLATION_COLORS =' in source_text:
        # I'll keep the functions but replace the config part at the end of the cell
        parts = source_text.split('# ────────────────────────────────────────────────────────────────\n# 2D. Shared Visualisation Helpers')
        header = parts[0]
        # Redefine the visualization section
        viz_section = """# ────────────────────────────────────────────────────────────────
# 2D. Shared Visualisation Helpers  (eliminasi duplikasi plot)
# ────────────────────────────────────────────────────────────────

ABLATION_COLORS = {
    'E2_NoMarket'             : '#FF9800',
    'E2_drop_CentStd'         : '#FFB74D',
    'E2_drop_CentMean'        : '#FFA726',
    'E2_drop_MSTDist'         : '#FB8C00',
    'E2_drop_MaxCent'         : '#F57C00',
    'E2_drop_NetDens'         : '#EF6C00',
    'Comp_Static_Gamma0'      : '#9E9E9E',
    'Comp_Static_Gamma1'      : '#795548',
    'Comp_Static_Gamma2'      : '#607D8B',
}

MAIN_EXPS = ['Comp_Static_Gamma0', 'Comp_Static_Gamma1', 'Comp_Static_Gamma2', 'E2_NoMarket']
LOO_EXPS  = ['E2_NoMarket', 'E2_drop_CentStd', 'E2_drop_CentMean', 'E2_drop_MSTDist', 'E2_drop_MaxCent', 'E2_drop_NetDens']

METRIC_KEY = {
    'Sharpe Ratio' : 'SharpeRatio',
    'Sortino Ratio': 'SortinoRatio',
    'Calmar Ratio' : 'CalmarRatio',
    'CVaR (95%)'   : 'CVaR95pct',
}
FOUR_METRICS = list(METRIC_KEY.keys())


def _style_table(tbl, col_labels, n_rows):
    \"\"\"Terapkan gaya tabel: header biru, baris zebra.\"\"\"
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#1565C0')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, n_rows + 1):
        fc = '#E3F2FD' if i % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(fc)


def _plot_metric_bars(ax, summary_df, exp_ids, colors, mean_col, std_col,
                      label, lower_better, short_labels=True):
    \"\"\"Bar chart satu metrik dengan highlight bar terbaik.\"\"\"
    vals = summary_df.loc[exp_ids, mean_col].values
    errs = summary_df.loc[exp_ids, std_col].values
    bars = ax.bar(range(len(exp_ids)), vals, yerr=errs,
                  color=colors, edgecolor='white', capsize=3, alpha=0.85)
    best_idx = np.argmin(vals) if lower_better else np.argmax(vals)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(2.5)
    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_xticks(range(len(exp_ids)))
    xlabels = ([e.split('_')[0] for e in exp_ids] if short_labels
               else [e.replace('_', '\\n') for e in exp_ids])
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.set_title(label, fontsize=9, fontweight='bold')
    ax.set_ylabel(label, fontsize=8)
    return bars, vals, errs


def _mean_cumret(ablation_results, exp_id, period):
    \"\"\"Rata-rata cumulative return lintas seed untuk satu eksperimen.\"\"\"
    return pd.concat(
        [ablation_results[exp_id][period][s] for s in SEEDS], axis=1
    ).mean(axis=1)


def _build_heatmap_data(summary_df, exp_ids, period):
    \"\"\"Buat DataFrame heatmap raw + normalized untuk satu period ('Test'/'Train').\"\"\"
    metric_cols = [(f'{METRIC_KEY[m]}_{period}_Mean', m.split()[0]) for m in FOUR_METRICS]
    # rename CVaR kolom agar pendek
    metric_cols[-1] = (f'CVaR95pct_{period}_Mean', 'CVaR(95%)')
    heatmap_data = pd.DataFrame(
        {lbl: summary_df[col] for col, lbl in metric_cols},
        index=exp_ids
    ).astype(float)

    heatmap_norm = heatmap_data.copy()
    for col in ['Sharpe', 'Sortino', 'Calmar']:
        mn, mx = heatmap_norm[col].min(), heatmap_norm[col].max()
        heatmap_norm[col] = (heatmap_norm[col] - mn) / (mx - mn + 1e-8)
    mn, mx = heatmap_norm['CVaR(95%)'].min(), heatmap_norm['CVaR(95%)'].max()
    heatmap_norm['CVaR(95%)'] = 1 - (heatmap_norm['CVaR(95%)'] - mn) / (mx - mn + 1e-8)
    return heatmap_data, heatmap_norm


print('Core functions + 4 metrik evaluasi tesis + visualisation helpers defined.')
print('Metrik utama: Sharpe Ratio | Sortino Ratio | Calmar Ratio | CVaR (95%)')"""
        cell['source'] = to_source(header + viz_section)

    # Cell 5: Feature Engineering
    if cell['cell_type'] == 'code' and 'def build_observation' in source_text:
        cell['source'] = to_source("""def compute_network_features(returns_window):
    \"\"\"5 network features dari MST (original).\"\"\"
    T, N = returns_window.shape
    corr_f  = apply_rmt_filter(returns_window)
    # 1. Network Density (standard undirected: ignore diagonal)
    upper_idx = np.triu_indices(N, k=1)
    density   = np.sum(np.abs(corr_f[upper_idx]) > 0.1) / (N * (N - 1) / 2) if N > 1 else 0.0
    _, cent_vec = _build_mst_centrality(N, corr_f)
    dist_mat = np.sqrt(np.maximum(0, 2 * (1 - corr_f)))
    mst = nx.minimum_spanning_tree(nx.from_numpy_array(dist_mat))
    mst_dist = sum(d['weight'] for _, _, d in mst.edges(data=True))
    return np.array([
        np.std(cent_vec)  * 10,   # F1: Centrality Std
        np.mean(cent_vec) * 10,   # F2: Centrality Mean
        mst_dist          * 0.1,  # F3: MST Distance
        np.max(cent_vec),          # F4: Max Centrality
        density                    # F5: Network Density
    ], dtype=np.float32), corr_f, cent_vec


def compute_market_features(returns_window, port_val=0.0):
    \"\"\"4 market features (original).\"\"\"
    short_ret  = returns_window.iloc[-5:].mean().mean()
    long_ret   = returns_window.mean().mean()
    momentum   = short_ret - long_ret
    recent_vol = returns_window.iloc[-5:].std().mean()
    return np.array([
        short_ret  * 100,
        momentum   * 100,
        recent_vol * 100,
        port_val
    ], dtype=np.float32)


def compute_extra_features(returns_window, corr_f, extra_list):
    \"\"\"Fitur tambahan untuk ablation study.\"\"\"
    extra = []
    if 'downside_vol' in extra_list:
        ret_flat = returns_window.values.flatten()
        downside = ret_flat[ret_flat < 0]
        dv = np.std(downside) * np.sqrt(252) if len(downside) > 0 else 0.0
        extra.append(float(dv) * 10)
    if 'avg_corr' in extra_list:
        upper_tri = corr_f[np.triu_indices_from(corr_f, k=1)]
        extra.append(float(np.mean(np.abs(upper_tri))))
    return np.array(extra, dtype=np.float32)


def build_observation(returns_window, config, port_val=0.0):
    nw_feat, corr_f, cent_vec = compute_network_features(returns_window)
    
    # Drop network feature if specified (Leave-One-Out Ablation)
    if 'drop_nw_idx' in config:
        nw_feat = np.delete(nw_feat, config['drop_nw_idx'])
        
    mkt_feat   = compute_market_features(returns_window, port_val)
    extra_feat = compute_extra_features(returns_window, corr_f, config['extra_features'])
    parts = []
    if config['use_network']: parts.append(nw_feat)
    if config['use_market']:  parts.append(mkt_feat)
    if len(extra_feat) > 0:   parts.append(extra_feat)
    obs = np.concatenate(parts) if parts else np.array([0.0], dtype=np.float32)
    return np.nan_to_num(obs), corr_f, cent_vec


def get_obs_dim(config):
    dim = 0
    if config['use_network']: 
        dim += (4 if 'drop_nw_idx' in config else 5)
    if config['use_market']:  dim += 4
    dim += len(config['extra_features'])
    return max(dim, 1)


print('Feature engineering functions defined.')
print('Observation dimensions per config:')
for name, cfg in ABLATION_CONFIGS.items():
    print(f'  {name}: {get_obs_dim(cfg)} features')""")

    # --- 3. Fix plot_dashboard Interpretation (KeyError: 'E0_Baseline') ---
    if cell['cell_type'] == 'code' and 'def plot_dashboard' in source_text:
        old_interp_start = "base = {m: summary_df.loc['E0_Baseline', f'{METRIC_KEY[m]}_Test_Mean']"
        new_interp = """        base = {m: summary_df.loc['Comp_Static_Gamma0', f'{METRIC_KEY[m]}_Test_Mean']
                for m in FOUR_METRICS}

        def delta(exp, m):
            return summary_df.loc[exp, f'{METRIC_KEY[m]}_Test_Mean'] - base[m]

        def drop_delta(exp, m):
            # Perbandingan vs E2_NoMarket (Full Network)
            return summary_df.loc[exp, f'{METRIC_KEY[m]}_Test_Mean'] - summary_df.loc['E2_NoMarket', f'{METRIC_KEY[m]}_Test_Mean']

        interpretasi = (
            "INTERPRETASI ABLATION STUDY — 4 METRIK TESIS:\\n\\n"
            f"  Best Sharpe Ratio   : {best_sharpe} = {summary_df.loc[best_sharpe,  'SharpeRatio_Test_Mean']:.4f}\\n"
            f"  Best Sortino Ratio  : {best_sortino} = {summary_df.loc[best_sortino, 'SortinoRatio_Test_Mean']:.4f}\\n"
            f"  Best Calmar Ratio   : {best_calmar} = {summary_df.loc[best_calmar,  'CalmarRatio_Test_Mean']:.4f}\\n"
            f"  Best CVaR (95%)     : {best_cvar} = {summary_df.loc[best_cvar, 'CVaR95pct_Test_Mean']:.5f} (terkecil)\\n\\n"
            "  PERFORMA PROPOSED (E2_NoMarket vs Comp_Static_Gamma0):\\n"
            f"  Sharpe Delta: {delta('E2_NoMarket','Sharpe Ratio'):+.4f} | "
            f"Sortino Delta: {delta('E2_NoMarket','Sortino Ratio'):+.4f}\\n\\n"
            "  KONTRIBUSI FITUR NETWORK (Drop Performance vs E2_NoMarket):\\n"
            f"  Drop CentStd  -> Sharpe: {drop_delta('E2_drop_CentStd', 'Sharpe Ratio'):+.4f}\\n"
            f"  Drop CentMean -> Sharpe: {drop_delta('E2_drop_CentMean','Sharpe Ratio'):+.4f}\\n"
            f"  Drop MSTDist  -> Sharpe: {drop_delta('E2_drop_MSTDist', 'Sharpe Ratio'):+.4f}\\n"
            f"  Drop MaxCent  -> Sharpe: {drop_delta('E2_drop_MaxCent', 'Sharpe Ratio'):+.4f}\\n"
            f"  Drop NetDens  -> Sharpe: {drop_delta('E2_drop_NetDens', 'Sharpe Ratio'):+.4f}\\n"
            "\\n* Nilai negatif pada Drop Performance menunjukkan fitur tersebut berkontribusi positif."
        )"""
        
        if old_interp_start in source_text:
            start_idx = source_text.find(old_interp_start)
            end_idx = source_text.find("ax_interp.text", start_idx)
            if start_idx != -1 and end_idx != -1:
                new_source = source_text[:start_idx] + new_interp + "\n        " + source_text[end_idx:]
                cell['source'] = to_source(new_source)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook reconstructed and modified successfully.")
