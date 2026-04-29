import nbformat as nbf

nb_path = 'RLNetworkMarkowitz_Optimized.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# 1. Update Phase 2 (Colors and MAIN_EXPS)
for cell in nb.cells:
    if cell.cell_type == 'code' and 'ABLATION_COLORS = {' in cell.source:
        print("Updating ABLATION_COLORS and MAIN_EXPS in Phase 2...")
        cell.source = cell.source.replace(
            "'Comp_Static_Gamma2'      : '#607D8B',",
            "'Comp_Static_Gamma2'      : '#607D8B',\n    'EW'                      : '#4CAF50',\n    'B&H'                     : '#8BC34A',\n    'Classic-MV'              : '#9C27B0',"
        )
        cell.source = cell.source.replace(
            "MAIN_EXPS = ['Comp_Static_Gamma0', 'Comp_Static_Gamma1', 'Comp_Static_Gamma2', 'E2_NoMarket']",
            "MAIN_EXPS = ['Comp_Static_Gamma0', 'Comp_Static_Gamma1', 'Comp_Static_Gamma2', 'E2_NoMarket', 'EW', 'B&H', 'Classic-MV']"
        )

# 2. Add run_backtest_baselines to Phase 6
for cell in nb.cells:
    if cell.cell_type == 'code' and 'def run_backtest(' in cell.source:
        print("Adding run_backtest_baselines to Phase 6...")
        baseline_code = """
def run_backtest_baselines(data, assets, window=30):
    \"\"\"Simulasi strategi benchmark: Equal-Weight, Buy-and-Hold, Classic Markowitz.\"\"\"
    n_assets = len(assets)
    dates = data.index[window:]
    
    # 1. Equal-Weight (EW)
    ew_rets = []
    w_ew = np.ones(n_assets) / n_assets
    for i in range(window, len(data)):
        ew_rets.append(np.dot(w_ew, data.iloc[i].values))
        
    # 2. Buy-and-Hold (B&H)
    bh_rets = []
    # Inisialisasi unit (asumsi harga awal = 1)
    units = np.ones(n_assets) / n_assets 
    port_val = 1.0
    for i in range(window, len(data)):
        daily_asset_rets = data.iloc[i].values
        # Update unit value
        asset_vals = units * (1 + daily_asset_rets)
        new_port_val = np.sum(asset_vals)
        bh_rets.append((new_port_val - port_val) / port_val)
        port_val = new_port_val
        units = asset_vals / port_val # normalize units to weights
        
    # 3. Classic Markowitz (Classic-MV) - No SAC, No Centrality
    cmv_rets = []
    for i in range(window, len(data)):
        if (i - window) % 7 == 0:
            win   = data.iloc[i-window:i]
            mu    = win.mean().values
            sigma = win.std().values
            corr  = apply_rmt_filter(win)
            cov   = np.outer(sigma, sigma) * corr + np.eye(n_assets)*1e-8
            # Classic MV: gamma=0 (no centrality penalty)
            w_mv = _solve_weights(cov, np.zeros(n_assets), mu, 0.0, n_assets)
        cmv_rets.append(np.dot(w_mv, data.iloc[i].values))
        
    return {
        'EW'        : pd.Series(ew_rets,  index=dates),
        'B&H'       : pd.Series(bh_rets,  index=dates),
        'Classic-MV': pd.Series(cmv_rets, index=dates)
    }
"""
        cell.source += baseline_code

# 3. Update Phase 8 (Run Baselines)
for cell in nb.cells:
    if cell.cell_type == 'code' and 'ablation_results = {}' in cell.source:
        print("Updating Phase 8 to include baseline execution...")
        cell.source = cell.source.replace(
            "print('\\nSemua backtest selesai.')",
            """# --- Running Baselines ---
print('Running Baselines (EW, B&H, Classic-MV)...')
baselines_train = run_backtest_baselines(ret_train, assets, SET_WINDOW)
baselines_test  = run_backtest_baselines(ret_test, assets, SET_WINDOW)

for b_name in ['EW', 'B&H', 'Classic-MV']:
    ablation_results[b_name] = {'train': {}, 'test': {}}
    for seed in SEEDS:
        ablation_results[b_name]['train'][seed] = baselines_train[b_name]
        ablation_results[b_name]['test'][seed]  = baselines_test[b_name]
    
    m_tr = calculate_all_metrics(baselines_train[b_name], CVAR_LEVEL)
    m_te = calculate_all_metrics(baselines_test[b_name], CVAR_LEVEL)
    print(f'  [BASELINE: {b_name}]')
    print(f'    TRAIN: Sharpe={m_tr["Sharpe Ratio"]:.3f} | Calmar={m_tr["Calmar Ratio"]:.3f}')
    print(f'    TEST : Sharpe={m_te["Sharpe Ratio"]:.3f} | Calmar={m_te["Calmar Ratio"]:.3f}')

print('\\nSemua backtest selesai.')"""
        )

# 4. Update Phase 9 (Aggregation)
for cell in nb.cells:
    if cell.cell_type == 'code' and 'summary_rows = []' in cell.source and 'for exp_id, config in ABLATION_CONFIGS.items():' in cell.source:
        print("Updating Phase 9 Aggregation...")
        cell.source = cell.source.replace(
            "for exp_id, config in ABLATION_CONFIGS.items():",
            "ALL_IDS = list(ABLATION_CONFIGS.keys()) + ['EW', 'B&H', 'Classic-MV']\nfor exp_id in ALL_IDS:\n    config = ABLATION_CONFIGS.get(exp_id, {})"
        )
        cell.source = cell.source.replace(
            "if config.get('static_gamma') is not None:",
            "if exp_id in ['EW', 'B&H', 'Classic-MV']:\n        feature_desc = ['Standard Baseline']\n    elif config.get('static_gamma') is not None:"
        )
        cell.source = cell.source.replace(
            "'Obs Dim'   : get_obs_dim(config),",
            "'Obs Dim'   : get_obs_dim(config) if exp_id in ABLATION_CONFIGS else 0,"
        )

# 5. Update Phase 10 (Visualization)
for cell in nb.cells:
    if cell.cell_type == 'code' and 'exp_ids = list(ABLATION_CONFIGS.keys())' in cell.source:
        print("Updating Phase 10 Visualization labels...")
        cell.source = cell.source.replace(
            "exp_ids = list(ABLATION_CONFIGS.keys())",
            "exp_ids = list(summary_df.index)"
        )

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook updated successfully.")
