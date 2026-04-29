import nbformat as nbf

nb_path = 'RLNetworkMarkowitz_Optimized_2000Seed2.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Target the cell in Phase 9 (Aggregation)
found = False
for cell in nb.cells:
    if cell.cell_type == 'code' and 'summary_rows = []' in cell.source and 'summary_df = pd.DataFrame(summary_rows)' in cell.source:
        print("Found Aggregation Cell. Updating...")
        
        # Replace the loop
        cell.source = cell.source.replace(
            "for exp_id, config in ABLATION_CONFIGS.items():",
            "ALL_IDS = list(ABLATION_CONFIGS.keys()) + ['EW', 'B&H', 'Classic-MV']\nfor exp_id in ALL_IDS:\n    config = ABLATION_CONFIGS.get(exp_id, {})"
        )
        
        # Replace metadata logic
        cell.source = cell.source.replace(
            "    if config.get('static_gamma') is not None:",
            "    if exp_id in ['EW', 'B&H', 'Classic-MV']:\n        feature_desc = ['Standard Baseline']\n    elif config.get('static_gamma') is not None:"
        )
        
        # Replace Obs Dim logic
        cell.source = cell.source.replace(
            "'Obs Dim'   : get_obs_dim(config),",
            "'Obs Dim'   : get_obs_dim(config) if exp_id in ABLATION_CONFIGS else 0,"
        )
        
        # Also ensure config.get is used safely
        cell.source = cell.source.replace("config['use_network']", "config.get('use_network', False)")
        cell.source = cell.source.replace("config['use_market']", "config.get('use_market', False)")
        cell.source = cell.source.replace("config['extra_features']", "config.get('extra_features', [])")
        
        found = True
        break

if found:
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Aggregation cell updated successfully.")
else:
    print("Could not find the aggregation cell.")
