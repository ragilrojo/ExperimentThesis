import json
import sys

def ipynb_to_py(ipynb_file, py_file):
    with open(ipynb_file, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    with open(py_file, 'w', encoding='utf-8') as f:
        f.write("import matplotlib\nmatplotlib.use('Agg')\n")
        f.write("import os\nos.environ['PYTHONUNBUFFERED'] = '1'\n")
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                f.write('#' + '-'*40 + '\n')
                f.write(''.join(cell['source']))
                f.write('\n\n')

if __name__ == "__main__":
    ipynb_to_py('RLNetworkMarkowitz_SAC_ThesisDataTrainingSaja.ipynb', 'RLNetworkMarkowitz_SAC_ThesisDataTrainingSaja.py')
    print("Converted notebook to script with non-interactive backend.")
