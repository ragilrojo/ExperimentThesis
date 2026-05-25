import json
nb = json.load(open('RLNetworkMarkowitz_SAC_XAI_AggressiveOnly.ipynb', 'r', encoding='utf-8'))
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source'])
    if 'run_multiseed_backtest' in src:
        tag = 'DEF' if 'def run_multiseed_backtest' in src else 'USE'
        print(f"Cell {i}: {tag} run_multiseed_backtest")
    if 'results_train' in src:
        tag = 'DEF' if 'results_train = {}' in src or 'results_train =' in src else 'USE'
        print(f"Cell {i}: {tag} results_train")
