import json

notebook_path = 'strategy_comparison_coba_toGrid2stage4Matriks_adv.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        new_source = []
        for line in source:
            if 'AdaptiveNetworkMarkowitz' in line and 'gamma=' in line:
                # Replace gamma=1.0 with gamma_up=0.5, gamma_down=1.0
                # We handle both single and double quotes
                updated_line = line.replace('gamma=1.0', 'gamma_up=0.5, gamma_down=1.0')
                # If there are cases with other values like gamma=1, handle them too
                updated_line = updated_line.replace('gamma=1', 'gamma_up=0.5, gamma_down=1.0')
                new_source.append(updated_line)
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("All AdaptiveNetworkMarkowitz calls updated to use new parameters.")
