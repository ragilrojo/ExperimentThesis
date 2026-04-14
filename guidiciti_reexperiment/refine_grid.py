import nbformat
import sys

def update_tuning_grid(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    target_found = False
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'COARSE_W_STEP = 30' in cell.source:
            # Update documentation comments to reflect new density
            cell.source = cell.source.replace('(~96 combinations per metric)', '(~300+ combinations per metric)')
            cell.source = cell.source.replace('(~288 trials vs 7308', '(~higher trials vs 7308')
            
            # Update parameters
            cell.source = cell.source.replace('COARSE_W_STEP = 30', 'COARSE_W_STEP = 15')
            cell.source = cell.source.replace('COARSE_G_STEP = 10', 'COARSE_G_STEP = 5')
            cell.source = cell.source.replace('FINE_MARGIN_W = 2', 'FINE_MARGIN_W = 5')
            cell.source = cell.source.replace('FINE_MARGIN_G = 1', 'FINE_MARGIN_G = 3')
            
            # Update comments
            cell.source = cell.source.replace('# every 30 windows', '# every 15 windows (denser)')
            cell.source = cell.source.replace('# every 10 units (0.0, 1.0, 2.0)', '# every 5 units (0.0, 0.5, 1.0, 1.5, 2.0)')
            cell.source = cell.source.replace('± 2 around coarse best window', '± 5 around coarse best window')
            cell.source = cell.source.replace('± 1 index (0.3)', '± 3 index (0.3)')
            
            target_found = True
            break

    if target_found:
        with open(file_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"Successfully updated tuning grid in {file_path}")
    else:
        print("Target tuning parameters not found in any cell.")

if __name__ == "__main__":
    update_tuning_grid('strategy_comparison_coba_toGrid2stage4Matriks_clean.ipynb')
