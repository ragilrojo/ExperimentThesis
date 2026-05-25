import nbformat
from nbformat.v4 import new_notebook, new_code_cell

with open('RLNetworkMarkowitz_thesis_CVaR.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Modify content for Jupyter Notebook (inline plotting)
content = content.replace("matplotlib.use('Agg')", "# matplotlib.use('Agg')  # Di-comment agar bisa tampil inline di Jupyter")
content = content.replace("plt.close()", "plt.show()\n    # plt.close()")

# Fix __file__ error in Jupyter Notebook
content = content.replace("os.path.dirname(os.path.abspath(__file__))", "os.getcwd()")

# Split by the section headers to create logical cells
parts = content.split('# ================================================================')

nb = new_notebook()
cells = []

# First part (before first separator)
if parts[0].strip():
    cells.append(new_code_cell(parts[0].strip()))

for part in parts[1:]:
    if part.strip():
        cell_content = '# ================================================================\n' + part.strip()
        cells.append(new_code_cell(cell_content))

nb.cells = cells

with open('RLNetworkMarkowitz_thesis_CVaR.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Berhasil membuat file RLNetworkMarkowitz_thesis_CVaR.ipynb")
