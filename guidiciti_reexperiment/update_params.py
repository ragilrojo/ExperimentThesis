import json

file_path = r'g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\guidiciti_reexperiment\RLNetworkMarkowitz_Thesis_FinalResettingGamma.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        changed = False
        for i, line in enumerate(source):
            if 'STEPS         =' in line:
                source[i] = 'SET_REBALANCE = 7\n' if 'STEPS' not in source[i] else source[i] # Just checking context
                if 'STEPS         =' in line:
                    source[i] = 'STEPS         = 5000\n'
                    changed = True
            if 'SET_WINDOW    =' in line:
                source[i] = 'SET_WINDOW    = 30\n'
                changed = True
        if changed:
            cell['source'] = source

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Berhasil memperbarui STEPS=5000 dan SET_WINDOW=30.")
