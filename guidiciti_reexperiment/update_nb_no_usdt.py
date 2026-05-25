import json
nb_path = 'RLNetworkMarkowitz_Thesis_FinalResettingGammaNoUsdt.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'load_and_preprocess' in ''.join(cell['source']):
        new_source = []
        for line in cell['source']:
            # Normalizing whitespace for matching
            if 'assets = list(set(r_old.columns) & set(r_2024.columns))' in line:
                new_source.append(line)
                new_source.append('    # FILTER OUT USDT (User Request)\n')
                new_source.append('    if "USDT" in assets:\n')
                new_source.append('        assets.remove("USDT")\n')
                new_source.append('        print("USDT matched and removed from assets.")\n')
            elif 'print(f"Data loaded. Assets: {len(assets)}")' in line:
                new_source.append('print(f"Data loaded. Assets: {len(assets)}")\n')
                new_source.append('print(f"Remaining assets: {assets}")\n')
            else:
                new_source.append(line)
        cell['source'] = new_source
        found = True
        break

if found:
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated successfully.")
else:
    print("Target cell not found.")
