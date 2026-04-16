import json

notebook_path = 'strategy_comparison_coba_toGrid2stage4Matriks_adv.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in reversed(nb['cells']):
    source_str = "".join(cell.get('source', []))
    if 'strategies_2024 =' in source_str:
        source = cell['source']
        new_source = []
        for line in source:
            if 'AdaptiveNetworkMarkowitz' in line and not line.strip().endswith(','):
                # Add the missing comma
                new_source.append(line.rstrip() + ",\n")
            else:
                new_source.append(line)
        cell['source'] = new_source
        break

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Formatting fix: Added missing comma in strategies_2024 list.")
