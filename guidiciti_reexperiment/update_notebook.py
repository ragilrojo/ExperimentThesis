import json

notebook_path = r"g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\guidiciti_reexperiment\RLNetworkMarkowitz_4Metrics.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        if "| E5 | **+Both** | 9 fitur + downside_vol + avg_corr | Semua fitur tambahan |" in source:
            new_source = source.replace(
                "| E5 | **+Both** | 9 fitur + downside_vol + avg_corr | Semua fitur tambahan |",
                "| E5 | **+Both** | 9 fitur + downside_vol + avg_corr | Semua fitur tambahan |\n| E6 | **-Network +Downside Vol** | 4 market + downside_vol | Tanpa network, dengan risk feature |"
            )
            # Reconstruct list of strings maintaining trailing newlines appropriately
            lines = new_source.split('\n')
            cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "'E5_AddBoth'" in source and "ABLATION_CONFIGS = {" in source:
            new_source = source.replace(
                "    'E5_AddBoth'         : {'use_network': True,  'use_market': True,  'extra_features': ['downside_vol', 'avg_corr']},\n",
                "    'E5_AddBoth'         : {'use_network': True,  'use_market': True,  'extra_features': ['downside_vol', 'avg_corr']},\n    'E6_NoNetwork_AddDownside': {'use_network': False, 'use_market': True,  'extra_features': ['downside_vol']},\n"
            )
            
            lines = new_source.split('\n')
            cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
