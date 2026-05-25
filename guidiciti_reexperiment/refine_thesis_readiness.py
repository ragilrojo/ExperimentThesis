import nbformat as nbf

nb_path = 'RLNetworkMarkowitz_Optimized.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# 1. Update Labels and Descriptions for Thesis Readiness
label_mapping = {
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

for cell in nb.cells:
    if cell.cell_type == 'code':
        # Update ABLATION_COLORS keys
        for old, new in label_mapping.items():
            if f"'{old}'" in cell.source:
                cell.source = cell.source.replace(f"'{old}'", f"'{new}'")
        
        # Update MAIN_EXPS and LOO_EXPS lists if they contain old names
        if 'MAIN_EXPS = [' in cell.source or 'LOO_EXPS = [' in cell.source:
             for old, new in label_mapping.items():
                cell.source = cell.source.replace(f"'{old}'", f"'{new}'")

# 2. Improve Visualization Aesthetics in Phase 10 & 11
for cell in nb.cells:
    if cell.cell_type == 'code' and ('plt.subplots' in cell.source or 'gridspec' in cell.source):
        # Increase font sizes and figure quality
        cell.source = cell.source.replace("fontsize=7", "fontsize=9")
        cell.source = cell.source.replace("fontsize=8", "fontsize=10")
        cell.source = cell.source.replace("fontsize=11", "fontsize=12")
        cell.source = cell.source.replace("fontsize=14", "fontsize=16")
        cell.source = cell.source.replace("dpi=150", "dpi=300") # Higher resolution for thesis
        
        # Add grid customization
        if 'ax.grid(True' in cell.source:
            cell.source = cell.source.replace("ax.grid(True)", "ax.grid(True, linestyle='--', alpha=0.6)")

# 3. Professionalize the Table Rendering in Phase 2 (_style_table)
for cell in nb.cells:
    if cell.cell_type == 'code' and 'def _style_table' in cell.source:
        print("Professionalizing _style_table definition...")
        cell.source = cell.source.replace("tbl.set_fontsize(8)", "tbl.set_fontsize(10)\n    tbl.scale(1.0, 1.5)")
        cell.source = cell.source.replace("#1565C0", "#2C3E50") # Darker, more academic header
        cell.source = cell.source.replace("#E3F2FD", "#F2F4F4") # Subtle grey for rows

# 4. Add a Thesis Title and Abstract/Introduction Markdown cell at the top
if "# Eksperimen Tesis" not in nb.cells[0].source:
    intro_cell = nbf.v4.new_markdown_cell(
        "# Eksperimen Tesis: SAC-Based Gamma Controller untuk Optimasi Portofolio Berbasis Network\n"
        "## Perbandingan Strategi Proposed vs Benchmark Industri\n\n"
        "**Peneliti:** [Nama Anda]\n"
        "**Tujuan:** Mengevaluasi efektivitas reinforcement learning (Soft Actor-Critic) dalam mengontrol parameter "
        "regularisasi network (Gamma) pada model Markowitz yang difilter dengan Random Matrix Theory (RMT) dan Minimum Spanning Tree (MST).\n\n"
        "**Metodologi:**\n"
        "1. **Proposed:** SAC mengontrol Gamma secara dinamis berdasarkan fitur network (centrality, density, MST distance).\n"
        "2. **Baselines:** Equal-Weight, Buy-and-Hold, Classic Markowitz, dan Static Gamma (0.0, 1.0, 2.0).\n"
        "3. **Ablation Study:** Leave-One-Out (LOO) untuk mengevaluasi kontribusi setiap fitur network terhadap performa agen RL."
    )
    nb.cells.insert(0, intro_cell)
else:
    print("Introduction cell already exists. Skipping insertion.")

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook updated for thesis readiness.")
