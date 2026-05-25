import nbformat as nbf

nb_path = 'RLNetworkMarkowitz_Optimized.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# 1. Revert ABLATION_CONFIGS keys and create THESIS_LABELS mapping
thesis_labels = {
    'Comp_Static_Gamma0': 'Static Gamma = 0.0',
    'Comp_Static_Gamma1': 'Static Gamma = 1.0',
    'Comp_Static_Gamma2': 'Static Gamma = 2.0',
    'E2_NoMarket': 'Proposed (SAC + Network)',
    'E2_drop_CentStd': 'Ablation: Drop Cent.Std',
    'E2_drop_CentMean': 'Ablation: Drop Cent.Mean',
    'E2_drop_MSTDist': 'Ablation: Drop MST.Dist',
    'E2_drop_MaxCent': 'Ablation: Drop Max.Cent',
    'E2_drop_NetDens': 'Ablation: Drop Net.Density',
    'EW': 'Equal-Weight',
    'B&H': 'Buy-and-Hold',
    'Classic-MV': 'Classic Markowitz'
}

# Invert the mapping to help with restoration
reverse_labels = {v: k for k, v in thesis_labels.items()}

for cell in nb.cells:
    if cell.cell_type == 'code':
        # Revert ABLATION_CONFIGS keys
        for thesis_name, simple_key in reverse_labels.items():
            if f"'{thesis_name}'" in cell.source:
                cell.source = cell.source.replace(f"'{thesis_name}'", f"'{simple_key}'")
        
        # Also revert lists like MAIN_EXPS and LOO_EXPS
        if 'MAIN_EXPS' in cell.source or 'LOO_EXPS' in cell.source:
            for thesis_name, simple_key in reverse_labels.items():
                cell.source = cell.source.replace(f"'{thesis_name}'", f"'{simple_key}'")

# 2. Add THESIS_LABELS dictionary to Phase 2 (Global Settings)
for cell in nb.cells:
    if cell.cell_type == 'code' and 'ABLATION_CONFIGS = {' in cell.source:
        print("Adding THESIS_LABELS to Global Settings...")
        labels_code = f"\n# Mapping experiment ID to formal thesis names\nTHESIS_LABELS = {thesis_labels}\n"
        # Insert before ABLATION_CONFIGS or after imports
        cell.source = cell.source.replace("import pandas", labels_code + "import pandas")
        break

# 3. Update Visualisation and Aggregation to use THESIS_LABELS
for cell in nb.cells:
    if cell.cell_type == 'code':
        # Replace exp_id in labels/titles with THESIS_LABELS.get(exp_id, exp_id)
        if 'exp_id' in cell.source and ('plt.title' in cell.source or 'label=' in cell.source or 'tbl_data.append' in cell.source):
            print("Updating cell to use THESIS_LABELS for display...")
            cell.source = cell.source.replace("exp_id.replace('_', ' ')", "THESIS_LABELS.get(exp_id, exp_id)")
            cell.source = cell.source.replace("exp_id", "THESIS_LABELS.get(exp_id, exp_id)")

# 4. Specific fix for Phase 11 table (which uses exp_id)
for cell in nb.cells:
    if cell.cell_type == 'code' and 'tbl_data.append' in cell.source:
        cell.source = cell.source.replace("exp_id,", "THESIS_LABELS.get(exp_id, exp_id),")

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook updated: Experiment IDs reverted and THESIS_LABELS introduced.")
