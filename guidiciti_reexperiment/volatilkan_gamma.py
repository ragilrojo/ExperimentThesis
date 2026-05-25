import json

file_path = r'g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\guidiciti_reexperiment\RLNetworkMarkowitz_SAC_ThesisDataTrainingSajaVolatilkanGamma.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update GAMMA_RANGE and SAC Hyperparameters (Phase 5)
found_params = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'GAMMA_RANGE  =' in source and 'sac_kwargs' in source:
            # Increase range and force higher entropy
            new_source = []
            for line in cell['source']:
                if 'GAMMA_RANGE  =' in line:
                    new_source.append("GAMMA_RANGE  = 5.0   # Diperluas agar lebih volatil (gamma ∈ [-4.0, 6.0])\n")
                elif "ent_coef        = 'auto'" in line:
                    new_source.append("    ent_coef        = 0.2,            # Ditingkatkan agar agen lebih eksploratif/volatil\n")
                else:
                    new_source.append(line)
            cell['source'] = new_source
            found_params = True
            break

# 2. Add 'total_return' to reward_modes (Phase 5 Training Loop)
found_loop = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'reward_modes = {' in source and 'excess_nw' in source:
            new_source = []
            for line in cell['source']:
                if "'sharpe_incremental':" in line:
                    new_source.append(line)
                    new_source.append("    'total_return'    : 'SAC-Net (Total Return)',\n")
                else:
                    new_source.append(line)
            cell['source'] = new_source
            found_loop = True
            break

# 3. Add Color for Total Return (Phase 6)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'COLORS = {' in source:
            new_source = []
            for line in cell['source']:
                if "'SAC-Net (Sharpe Incr)'" in line:
                    new_source.append(line)
                    new_source.append("    'SAC-Net (Total Return)': 'purple',\n")
                else:
                    new_source.append(line)
            cell['source'] = new_source
            break

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Successfully updated notebook for higher Gamma volatility and 'Total Return' reward mode.")
