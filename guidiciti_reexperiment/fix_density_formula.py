import json
import os
import numpy as np

def fix_notebook(file_path):
    print(f"Checking {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            nb = json.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return
    
    modified = False
    old_formula = "np.sum(np.abs(corr_f) > 0.1) / (N * N)"
    new_formula_lines = [
        "# 1. Network Density (standard undirected: ignore diagonal)\n",
        "        upper_idx = np.triu_indices(N, k=1)\n",
        "        density   = np.sum(np.abs(corr_f[upper_idx]) > 0.1) / (N * (N - 1) / 2) if N > 1 else 0.0"
    ]
    
    # We need to be careful with indentation and JSON strings.
    # In the notebooks, it might look like:
    # "density = np.sum(np.abs(corr_f) > 0.1) / (N * N)\n"
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            new_source = []
            cell_modified = False
            for line in source:
                if old_formula in line:
                    # Capture indentation
                    indent = line[:line.find(old_formula)].replace('density', '').replace('=', '').rstrip()
                    # Reconstruct the replacement lines with same indentation
                    # Note: We replace the entire line that contained the old formula.
                    # Usually it's 'density = ...'
                    
                    # Simple replacement first
                    # We'll replace the line with 3 lines
                    new_line_1 = indent + "# 1. Network Density (standard undirected: ignore diagonal)\n"
                    new_line_2 = indent + "upper_idx = np.triu_indices(N, k=1)\n"
                    new_line_3 = indent + "density   = np.sum(np.abs(corr_f[upper_idx]) > 0.1) / (N * (N - 1) / 2) if N > 1 else 0.0\n"
                    
                    new_source.append(new_line_1)
                    new_source.append(new_line_2)
                    new_source.append(new_line_3)
                    cell_modified = True
                    modified = True
                else:
                    new_source.append(line)
            if cell_modified:
                cell['source'] = new_source
                
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Fixed {file_path}")
    else:
        print(f"No changes needed for {file_path}")

if __name__ == "__main__":
    directory = r'g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\guidiciti_reexperiment'
    for file in os.listdir(directory):
        if file.endswith(".ipynb"):
            fix_notebook(os.path.join(directory, file))
