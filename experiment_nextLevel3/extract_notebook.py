import json
import os

notebook_path = r"g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\experiment_nextLevel3\enhanced_strategy_v42_lq45_ma_tpfp_hybrid.ipynb"
script_path = r"g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\experiment_nextLevel3\v42_auto_run.py"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_cells = [cell['source'] for cell in nb['cells'] if cell['cell_type'] == 'code']

with open(script_path, 'w', encoding='utf-8') as f:
    for source in code_cells:
        if isinstance(source, list):
            f.writelines(source)
        else:
            f.write(source)
        f.write('\n\n')

print(f"Extracted code to {script_path}")
