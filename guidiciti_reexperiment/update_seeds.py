import json

file_path = r'g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\guidiciti_reexperiment\RLNetworkMarkowitz_SAC_ThesisDataTrainingSaja.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Search for the cell containing SEEDS = [42]
updated = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if 'SEEDS = [42]' in line:
                source[i] = line.replace('SEEDS = [42]', 'SEEDS = [42, 123, 77]')
                updated = True
                break
        if updated:
            break

if updated:
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Successfully updated SEEDS in the notebook.")
else:
    print("Could not find SEEDS = [42] in the notebook.")
