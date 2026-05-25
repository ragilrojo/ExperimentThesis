import json
with open('RLNetworkMarkowitz_RewardAblation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

changed = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for i, line in enumerate(cell['source']):
            if 'SAC strategy untuk reward ablation. Mendukung static benchmark.' in line and 'def __init__' in line:
                cell['source'][i] = line.replace('"""    def', '"""\n    def')
                changed = True

if changed:
    with open('RLNetworkMarkowitz_RewardAblation.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print('Fixed the syntax error')
else:
    print('Line not found')
