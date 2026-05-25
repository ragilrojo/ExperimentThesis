import nbformat as nbf
import re

nb_path = 'RLNetworkMarkowitz_Optimized.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Pattern to find the illegal assignment in for loops
faulty_pattern = r"for\s+THESIS_LABELS\.get\(exp_id,\s*exp_id\)\s+in"
fixed_replacement = "for exp_id in"

modified_count = 0
for cell in nb.cells:
    if cell.cell_type == 'code':
        if re.search(faulty_pattern, cell.source):
            new_source = re.sub(faulty_pattern, fixed_replacement, cell.source)
            if new_source != cell.source:
                cell.source = new_source
                modified_count += 1

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Fixed {modified_count} faulty for-loops in the notebook.")
