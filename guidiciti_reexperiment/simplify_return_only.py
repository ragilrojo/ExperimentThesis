import json

file_path = r'g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\guidiciti_reexperiment\RLNetworkMarkowitz_SAC_ThesisDataTrainingSajaVolatilkanGammaReturnSaja.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update SEEDS (Libraries cell)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'SEEDS =' in source and 'import' in source:
            new_source = []
            for line in cell['source']:
                if 'SEEDS =' in line:
                    new_source.append("SEEDS = [42]  # Fokus hanya pada seed 42\n")
                else:
                    new_source.append(line)
            cell['source'] = new_source
            break

# 2. Update Reward Modes (Training Loop cell)
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'reward_modes = {' in source and 'excess_nw' in source:
            new_source = [
                "reward_modes = {\n",
                "    'total_return'    : 'SAC-Net (Total Return)',\n",
                "}\n"
            ]
            cell['source'] = new_source
            break

# 3. Ensure Volatility Settings are applied
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'GAMMA_RANGE  =' in source and 'sac_kwargs' in source:
            new_source = []
            for line in cell['source']:
                if 'GAMMA_RANGE  =' in line:
                    new_source.append("GAMMA_RANGE  = 5.0   # Diperluas agar lebih volatil (gamma ∈ [-4.0, 6.0])\n")
                elif "ent_coef        = 'auto'" in line:
                    new_source.append("    ent_coef        = 0.2,            # Ditingkatkan agar agen lebih eksploratif/volatil\n")
                else:
                    new_source.append(line)
            cell['source'] = new_source

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Successfully simplified notebook to 'Total Return' and seed 42.")
