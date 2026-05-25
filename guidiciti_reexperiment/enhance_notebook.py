import nbformat as nbf
import os

notebook_path = r'g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\guidiciti_reexperiment\RLNetworkMarkowitz_ThesisReady25000step3seed5ModelFixedNext.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# New cells to add
new_cells = []

new_cells.append(nbf.v4.new_markdown_cell("## Phase 14: Thesis Enhancement - Portfolio Turnover & Transaction Cost Analysis\n\nAnalisis tingkat pergantian portofolio (Turnover) dan dampak biaya transaksi (0.1%) terhadap performa."))

new_cells.append(nbf.v4.new_code_cell("""
def calculate_turnover(strategy, data, window=30, rebalance_freq=7):
    \"\"\"Menghitung rata-rata turnover per rebalancing event.\"\"\"
    prev_weights = None
    turnovers = []
    port_val = 1.0
    
    for i in range(window, len(data)):
        if (i - window) % rebalance_freq == 0:
            window_df = data.iloc[i - window : i]
            curr_weights = strategy.compute_weights(window_df, port_val=port_val)
            
            if prev_weights is not None:
                # Turnover = sum(|w_new - w_old|)
                to = np.sum(np.abs(curr_weights - prev_weights))
                turnovers.append(to)
            
            prev_weights = curr_weights
            
        if prev_weights is not None:
            daily_ret = np.dot(prev_weights, data.iloc[i].values)
            port_val *= (1 + daily_ret)
            
    return np.mean(turnovers) if turnovers else 0.0

print('=== Portfolio Turnover Analysis (Test Period) ===')
turnover_results = []
# Analyze E2_NoMarket, Static Gamma 0, and Classic-MV
for exp_id in ['E2_NoMarket', 'Comp_Static_Gamma0', 'Classic-MV']:
    tos = []
    for seed in SEEDS:
        model_path = trained_model_paths.get((exp_id, seed), 'static')
        config = ABLATION_CONFIGS.get(exp_id, {})
        strat = AblationStrategy(exp_id, model_path, config, GAMMA_CENTER)
        tos.append(calculate_turnover(strat, ret_test, SET_WINDOW, SET_REBALANCE))
    
    avg_to = np.mean(tos)
    turnover_results.append({'Experiment': exp_id, 'Avg_Turnover': avg_to})
    print(f'{exp_id:<20}: {avg_to:.4f} (per rebalance)')

to_df = pd.DataFrame(turnover_results)
to_df.to_csv('ablation_results_thesis/portfolio_turnover.csv', index=False)

# Plot Turnover
plt.figure(figsize=(10, 5))
sns.barplot(data=to_df, x='Experiment', y='Avg_Turnover', palette='viridis')
plt.title('Average Portfolio Turnover per Rebalance (Lower is Better)', fontweight='bold')
plt.ylabel('Turnover (Sum of Abs Weight Delta)')
plt.savefig('ablation_results_thesis/turnover_comparison.png', dpi=150)
plt.show()
"""))

new_cells.append(nbf.v4.new_markdown_cell("## Phase 15: Thesis Enhancement - Explainable AI (Action Sensitivity)\n\nMenganalisis bagaimana input fitur jaringan (Network Features) mempengaruhi keputusan agen dalam menentukan `gamma`."))

new_cells.append(nbf.v4.new_code_cell("""
def analyze_policy_sensitivity(exp_id, seed):
    \"\"\"Analisis sensitivitas Gamma terhadap fitur Network.\"\"\"
    model_path = trained_model_paths.get((exp_id, seed))
    if not model_path or model_path == 'static': return None
    
    model = SAC.load(model_path)
    config = ABLATION_CONFIGS[exp_id]
    
    # Sample observations dari test set cache
    obs_cache, _ = build_caches_from_global(config, GLOBAL_CACHE)
    sample_indices = list(obs_cache.keys())[:200]
    obs_samples = np.array([obs_cache[i] for i in sample_indices])
    
    # Feature names
    feature_names = ['Net_Std', 'Net_Mean', 'Net_Max', 'Net_Density', 'Net_Dist']
    if config['use_market']:
        feature_names += ['Mkt_Ret', 'Mkt_Vol', 'Mkt_Mom', 'Port_Val']
    
    importances = []
    epsilon = 1e-2
    
    for i in range(min(len(feature_names), obs_samples.shape[1])):
        obs_plus = obs_samples.copy()
        obs_plus[:, i] += epsilon
        
        act_orig, _ = model.predict(obs_samples, deterministic=True)
        act_plus, _ = model.predict(obs_plus, deterministic=True)
        
        # Sensitivity = mean(|delta_action| / delta_feature)
        sensitivity = np.mean(np.abs(act_plus - act_orig) / epsilon)
        importances.append(sensitivity)
        
    return dict(zip(feature_names, importances))

print('=== SAC Policy Sensitivity Analysis (XAI) ===')
# Run for the best seed of E2_NoMarket
xai_results = analyze_policy_sensitivity('E2_NoMarket', SEEDS[0])
if xai_results:
    xai_df = pd.DataFrame(list(xai_results.items()), columns=['Feature', 'Sensitivity'])
    xai_df = xai_df.sort_values('Sensitivity', ascending=False)
    print(xai_df)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=xai_df, x='Sensitivity', y='Feature', palette='magma')
    plt.title(f'Action Sensitivity (dGamma/dFeature) - E2_NoMarket', fontweight='bold')
    plt.xlabel('Mean Absolute Sensitivity')
    plt.tight_layout()
    plt.savefig('ablation_results_thesis/xai_sensitivity.png', dpi=150)
    plt.show()
"""))

new_cells.append(nbf.v4.new_markdown_cell("## Phase 16: Thesis Enhancement - Network Minimum Spanning Tree (MST) Visualization\n\nVisualisasi topologi aset menggunakan MST untuk menunjukkan struktur korelasi pasar pada periode pengujian."))

new_cells.append(nbf.v4.new_code_cell("""
def plot_market_mst(returns_df, date_idx, title_suffix=''):
    \"\"\"Plot MST dari korelasi aset pada tanggal tertentu.\"\"\"
    corr = returns_df.iloc[date_idx-30 : date_idx].corr()
    # Distance matrix: d = sqrt(2 * (1 - rho))
    dist = np.sqrt(2 * (1 - corr))
    
    G = nx.from_pandas_adjacency(dist)
    mst = nx.minimum_spanning_tree(G)
    
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(mst, seed=42)
    
    # Node size based on centrality
    cent = nx.eigenvector_centrality_numpy(mst)
    node_sizes = [v * 5000 for v in cent.values()]
    
    nx.draw_networkx_nodes(mst, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8)
    nx.draw_networkx_edges(mst, pos, width=1.5, edge_color='gray', alpha=0.5)
    nx.draw_networkx_labels(mst, pos, font_size=10, font_weight='bold')
    
    date_str = returns_df.index[date_idx].date()
    plt.title(f'Asset Minimum Spanning Tree (MST) - {date_str} {title_suffix}', fontsize=12, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'ablation_results_thesis/mst_topology_{date_idx}.png', dpi=150)
    plt.show()

print('Generating MST Snapshots for Thesis...')
# Visualisasi MST pada hari terakhir data test
plot_market_mst(ret_test, len(ret_test)-1, '(End of Test Period)')
"""))

new_cells.append(nbf.v4.new_markdown_cell("## Final Conclusion for Thesis\n\nSemua analisis tambahan (Turnover, XAI, MST) telah disimpan di folder `ablation_results_thesis/` untuk disertakan dalam dokumen skripsi/tesis."))

# Append cells to notebook
nb.cells.extend(new_cells)

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook successfully enhanced with Phase 14, 15, and 16.")
