import json

notebook_path = r"g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\guidiciti_reexperiment\RLNetworkMarkowitz_4Metrics.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "ABLATION_COLORS = {" in source and "'E5_AddBoth'" in source:
            new_source = source.replace(
                "    'E5_AddBoth'         : '#E91E63',\n",
                "    'E5_AddBoth'         : '#E91E63',\n    'E6_NoNetwork_AddDownside': '#9C27B0',\n"
            )
            
            lines = new_source.split('\n')
            cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook colors updated successfully.")
