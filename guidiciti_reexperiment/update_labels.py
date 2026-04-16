import json

notebook_path = 'strategy_comparison_coba_toGrid2stage4Matriks_adv.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The cell we want to update is the last one (index -1)
source = nb['cells'][-1]['source']
new_source = []
for line in source:
    if 'NW Adaptis (G=1, W=30, S=0.013333)' in line:
        new_source.append(line.replace('NW Adaptis (G=1, W=30, S=0.013333)', 'NW Adaptis (Up=0.5, Down=1.0)'))
    elif "plt.title('Performance Comparison 2024: BTC vs Adaptive Network Markowitz (Validation)', fontsize=16)" in line:
        new_source.append(line.replace('(Validation)', '(Validation - Improved Logic)'))
    else:
        new_source.append(line)

nb['cells'][-1]['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook 2024 simulation labels updated.")
