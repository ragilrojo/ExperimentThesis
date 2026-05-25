import json
nb_path = 'RLNetworkMarkowitz_SAC_Thesis.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

mod_steps = False
mod_seeds = False

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        for line in source:
            # 1. Reduce TRAIN_STEPS
            if 'TRAIN_STEPS  = 100_000' in line:
                new_source.append(line.replace('100_000', '5_000'))
                mod_steps = True
                print("Reduced TRAIN_STEPS to 5,000")
            elif 'TRAIN_STEPS = 100_000' in line:
                new_source.append(line.replace('100_000', '5_000'))
                mod_steps = True
                print("Reduced TRAIN_STEPS to 5,000")
            # 2. Reduce SEEDS
            elif 'SEEDS = [42, 123, 999]' in line:
                new_source.append(line.replace('[42, 123, 999]', '[42]'))
                mod_seeds = True
                print("Reduced SEEDS to [42]")
            else:
                new_source.append(line)
        cell['source'] = new_source

if mod_steps or mod_seeds:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated successfully with reduced steps/seeds.")
else:
    print("Target lines not found in the notebook.")
