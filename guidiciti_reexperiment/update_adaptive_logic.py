import json

notebook_path = 'strategy_comparison_coba_toGrid2stage4Matriks_adv.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'class AdaptiveNetworkMarkowitz' in "".join(cell['source']):
        source = cell['source']
        new_source = []
        for line in source:
            if "current_gamma = -1" in line:
                new_source.append(line.replace("-1", "0.5") + "  # Uptrend: Low Gamma (Flexibility)\n")
            elif "current_gamma = self.base_gamma" in line:
                new_source.append(line.replace("self.base_gamma", "1.0") + "  # Downtrend: High Gamma (Defense)\n")
            elif "# Use slope_threshold instead of hardcoded 0" in line:
                new_source.append("        # Improved logic for active risk management:\n")
            else:
                new_source.append(line)
        cell['source'] = new_source
        found = True
        break

if found:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("AdaptiveNetworkMarkowitz logic updated successfully.")
else:
    print("Class definition not found.")
