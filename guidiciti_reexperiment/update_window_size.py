import json

notebook_path = 'strategy_comparison_coba_toGrid2stage4Matriks_adv.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The cell we want to update is the one near the end with 'strategies_2024'
found = False
for cell in reversed(nb['cells']):
    if cell['cell_type'] == 'code' and 'strategies_2024 =' in "".join(cell['source']):
        source = cell['source']
        new_source = []
        for line in source:
            if 'AdaptiveNetworkMarkowitz' in line:
                # Update window_size to 120 and update label
                processed = line.replace('window_size=30', 'window_size=120')
                if 'W=30' in processed:
                    processed = processed.replace('W=30', 'W=120')
                else:
                    # If W=120 not in there, add it to name
                    processed = processed.replace('Down=1.0)', 'Down=1.0, W=120)')
                new_source.append(processed)
            else:
                new_source.append(line)
        cell['source'] = new_source
        found = True
        break

if found:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Adaptive strategy window_size updated to 120 in the 2024 simulation.")
else:
    print("2024 simulation cell not found.")
