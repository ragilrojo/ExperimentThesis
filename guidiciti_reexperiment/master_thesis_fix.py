import nbformat as nbf

nb_path = 'RLNetworkMarkowitz_Optimized.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# 1. Labels Mapping
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

# 2. Revert/Clean ABLATION_CONFIGS keys to be safe for filenames
for cell in nb.cells:
    if cell.cell_type == 'code':
        # Safely revert any colons or spaces in keys back to simple IDs
        for simple_key, thesis_name in thesis_labels.items():
            if f"'{thesis_name}'" in cell.source:
                cell.source = cell.source.replace(f"'{thesis_name}'", f"'{simple_key}'")
        
        # Robust Hyperparameters
        if 'SEEDS         =' in cell.source:
            cell.source = cell.source.replace("SEEDS         = [42]", "SEEDS         = [42, 123, 77]")
            cell.source = cell.source.replace("TRAIN_STEPS   = 500", "TRAIN_STEPS   = 2000")
            cell.source = cell.source.replace("TRAIN_STEPS   = 1000", "TRAIN_STEPS   = 2000")

# 3. Inject THESIS_LABELS and ensure Phase 10/11 use them
for cell in nb.cells:
    if cell.cell_type == 'code' and 'ABLATION_CONFIGS = {' in cell.source:
        if 'THESIS_LABELS =' not in cell.source:
            labels_def = f"\n# Mapping experiment ID to formal thesis names\nTHESIS_LABELS = {thesis_labels}\n"
            cell.source = cell.source.replace("import pandas", labels_def + "import pandas")

for cell in nb.cells:
    if cell.cell_type == 'code':
        if 'exp_id' in cell.source and ('plt.title' in cell.source or 'label=' in cell.source or 'tbl_data.append' in cell.source):
            cell.source = cell.source.replace("exp_id.replace('_', ' ')", "THESIS_LABELS.get(exp_id, exp_id)")
            # Avoid double replacement if already patched
            if "THESIS_LABELS.get(exp_id, exp_id)" not in cell.source:
                 cell.source = cell.source.replace("exp_id", "THESIS_LABELS.get(exp_id, exp_id)")

# 4. Professional Styling
for cell in nb.cells:
    if cell.cell_type == 'code' and 'def _style_table' in cell.source:
        cell.source = cell.source.replace("tbl.set_fontsize(8)", "tbl.set_fontsize(10)\n    tbl.scale(1.0, 1.5)")
        cell.source = cell.source.replace("#1565C0", "#2C3E50")
    
    if cell.cell_type == 'code' and 'plt.savefig' in cell.source:
        cell.source = cell.source.replace("dpi=150", "dpi=300")

# 5. Intro Cell
if "# Eksperimen Tesis" not in nb.cells[0].source:
    intro_cell = nbf.v4.new_markdown_cell(
        "# Eksperimen Tesis: SAC-Based Gamma Controller untuk Optimasi Portofolio Berbasis Network\n"
        "## Perbandingan Strategi Proposed vs Benchmark Industri\n\n"
        "**Peneliti:** [Nama Anda]\n"
        "**Tujuan:** Mengevaluasi efektivitas reinforcement learning (Soft Actor-Critic) dalam mengontrol parameter "
        "regularisasi network (Gamma) pada model Markowitz yang difilter dengan Random Matrix Theory (RMT) dan Minimum Spanning Tree (MST)."
    )
    nb.cells.insert(0, intro_cell)

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Master Fix applied successfully.")
