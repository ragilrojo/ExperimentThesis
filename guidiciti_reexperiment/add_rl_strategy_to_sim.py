import json

notebook_path = 'strategy_comparison_coba_toGrid2stage4Matriks_adv.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update the 2024 comparison cell
for cell in reversed(nb['cells']):
    source_str = "".join(cell.get('source', []))
    if 'strategies_2024 =' in source_str:
        source = cell['source']
        new_source = []
        for line in source:
            if 'AdaptiveNetworkMarkowitz' in line and ']' in line:
                # If the closing bracket is on the same line, insert before it
                line = line.replace(']', '    RLNetworkMarkowitz("NW RL Dynamic Gamma", model_path="ppo_gamma_controller", window_size=120)\n    ]')
                new_source.append(line)
            elif 'AdaptiveNetworkMarkowitz' in line:
                new_source.append(line)
                new_source.append('    RLNetworkMarkowitz("NW RL Dynamic Gamma", model_path="ppo_gamma_controller", window_size=120),\n')
            else:
                new_source.append(line)
        cell['source'] = new_source
        break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("RL Strategy added to 2024 comparison successfully.")
