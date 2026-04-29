import nbformat as nbf
import os

nb_path = 'RLNetworkMarkowitz_Optimized_2000Seed2.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# 1. Update ABLATION_COLORS and MAIN_EXPS in Phase 2
for cell in nb.cells:
    if cell.cell_type == 'code' and 'ABLATION_COLORS =' in cell.source:
        cell.source = cell.source.replace(
            "'Comp_Static_Gamma2'      : '#607D8B',",
            "'Comp_Static_Gamma2'      : '#607D8B',\n    'EW'                      : '#4CAF50',\n    'B&H'                     : '#8BC34A',\n    'Classic-MV'              : '#9C27B0',"
        )
        cell.source = cell.source.replace(
            "MAIN_EXPS = ['Comp_Static_Gamma0', 'Comp_Static_Gamma1', 'Comp_Static_Gamma2', 'E2_NoMarket']",
            "MAIN_EXPS = ['Comp_Static_Gamma0', 'Comp_Static_Gamma1', 'Comp_Static_Gamma2', 'E2_NoMarket', 'EW', 'B&H', 'Classic-MV']"
        )
        break

# 2. Add Baseline Strategy implementations after Phase 6
baseline_code = """# ────────────────────────────────────────────────────────────────
# 6B. Baseline Strategies (EW, B&H, Classic Markowitz)
# ────────────────────────────────────────────────────────────────

class EqualWeightStrategy:
    def __init__(self, n_assets):
        self.name = 'EW'
        self.weights = np.ones(n_assets) / n_assets
        self.is_static = True
    def compute_weights(self, *args, **kwargs):
        return self.weights

def run_backtest_baselines(data, assets, window=30):
    \"\"\"Backtest khusus untuk baselines yang tidak mengikuti logic rebalancing standard.\"\"\"
    n_assets = len(assets)
    dates = data.index[window:]
    
    # 1. Equal Weight
    ew_rets = data.iloc[window:].mean(axis=1)
    
    # 2. Buy & Hold (Initial EW, then drift)
    bh_rets = []
    w = np.ones(n_assets) / n_assets
    for i in range(window, len(data)):
        ret_row = data.iloc[i].values
        daily_ret = np.dot(w, ret_row)
        bh_rets.append(daily_ret)
        # Update weights by drift
        w = w * (1 + ret_row)
        if np.sum(w) > 0:
            w = w / np.sum(w)
        else:
            w = np.ones(n_assets) / n_assets
        
    # 3. Classic Markowitz (No RMT, No Network, Gamma=0)
    cmv_rets = []
    w_cmv = np.ones(n_assets) / n_assets
    for i in range(window, len(data)):
        if (i - window) % 7 == 0:
            window_df = data.iloc[i-window:i]
            mu = window_df.mean().values
            # Classic: No RMT filter, just sample covariance
            cov = window_df.cov().values + np.eye(n_assets) * 1e-8
            # Gamma=0, Cent_vec=0
            w_cmv = _solve_weights(cov, np.zeros(n_assets), mu, 0.0, n_assets)
        cmv_rets.append(np.dot(w_cmv, data.iloc[i].values))
        
    return {
        'EW': pd.Series(ew_rets, index=dates),
        'B&H': pd.Series(bh_rets, index=dates),
        'Classic-MV': pd.Series(cmv_rets, index=dates)
    }

print('Baseline strategies (EW, B&H, Classic-MV) defined.')"""

# Find Phase 6 cell and insert after it
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code' and 'class AblationStrategy:' in cell.source:
        nb.cells.insert(i + 1, nbf.v4.new_code_cell(baseline_code))
        break

# 3. Update Backtest Loop in Phase 8
for cell in nb.cells:
    if cell.cell_type == 'code' and 'ablation_results = {}' in cell.source:
        cell.source = cell.source.replace(
            "print('\\nSemua backtest selesai.')",
            "print('\\nSemua SAC/Static-Gamma backtest selesai.')\n\n# --- Running Baselines ---\nprint('Running Baselines (EW, B&H, Classic-MV)...')\nbaselines_train = run_backtest_baselines(ret_train, assets, SET_WINDOW)\nbaselines_test  = run_backtest_baselines(ret_test, assets, SET_WINDOW)\n\nfor b_name in ['EW', 'B&H', 'Classic-MV']:\n    ablation_results[b_name] = {'train': {}, 'test': {}}\n    for seed in SEEDS:\n        ablation_results[b_name]['train'][seed] = baselines_train[b_name]\n        ablation_results[b_name]['test'][seed]  = baselines_test[b_name]\n    \n    m_tr = calculate_all_metrics(baselines_train[b_name], CVAR_LEVEL)\n    m_te = calculate_all_metrics(baselines_test[b_name], CVAR_LEVEL)\n    print(f'  [BASELINE: {b_name}]')\n    print(f'    TRAIN: Sharpe={m_tr[\"Sharpe Ratio\"]:.3f} | Calmar={m_tr[\"Calmar Ratio\"]:.3f}')\n    print(f'    TEST : Sharpe={m_te[\"Sharpe Ratio\"]:.3f} | Calmar={m_te[\"Calmar Ratio\"]:.3f}')"
        )
        break

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

# --- Second Pass: Aggregation and Visualization ---
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# 4. Update Phase 9 Aggregation
for cell in nb.cells:
    if cell.cell_type == 'code' and 'for exp_id, config in ABLATION_CONFIGS.items():' in cell.source:
        cell.source = cell.source.replace(
            "for exp_id, config in ABLATION_CONFIGS.items():",
            "ALL_IDS = list(ABLATION_CONFIGS.keys()) + ['EW', 'B&H', 'Classic-MV']\nfor exp_id in ALL_IDS:\n    config = ABLATION_CONFIGS.get(exp_id, {})"
        )
        cell.source = cell.source.replace(
            "    if config.get('static_gamma') is not None:",
            "    if exp_id in ['EW', 'B&H', 'Classic-MV']:\n        feature_desc = ['Standard Baseline']\n    elif config.get('static_gamma') is not None:"
        )
        cell.source = cell.source.replace(
            "'Obs Dim'   : get_obs_dim(config),",
            "'Obs Dim'   : get_obs_dim(config) if exp_id in ABLATION_CONFIGS else 0,"
        )
        break

# 5. Update Phase 10 Visualization
for cell in nb.cells:
    if cell.cell_type == 'code' and 'exp_ids = list(ABLATION_CONFIGS.keys())' in cell.source:
        cell.source = cell.source.replace(
            "exp_ids = list(ABLATION_CONFIGS.keys())",
            "exp_ids = list(summary_df.index)"
        )
        break

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook updated successfully with Aggregation and Visualization fixes.")
