import nbformat
import sys

def update_hybrid_grid(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    target_found = False
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'hybrid_w_coarse = [20, 40, 60, 80, 100, 120]' in cell.source:
            # Update coarse search space
            cell.source = cell.source.replace('hybrid_w_coarse = [20, 40, 60, 80, 100, 120]', 
                                            'hybrid_w_coarse = list(range(10, 121, 10))')
            cell.source = cell.source.replace('hybrid_g_coarse = [0.0, 0.5, 1.0, 1.5, 2.0]', 
                                            'hybrid_g_coarse = [round(x * 0.2, 1) for x in range(0, 11)]')
            cell.source = cell.source.replace('hybrid_t_coarse = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]', 
                                            'hybrid_t_coarse = [round(x * 0.05, 2) for x in range(1, 13)]')
            
            # Update fine margins
            cell.source = cell.source.replace('best_h_c[\'W\'] - 3', 'best_h_c[\'W\'] - 5')
            cell.source = cell.source.replace('best_h_c[\'W\'] + 4', 'best_h_c[\'W\'] + 6')
            cell.source = cell.source.replace('range(-2, 3)', 'range(-3, 4)') # ±0.3 instead of ±0.2
            cell.source = cell.source.replace('i*0.02', 'i*0.02') # Will change range instead
            cell.source = cell.source.replace('h_t_fine = [round(best_h_c[\'T\'] + i*0.02, 2) for i in range(-2, 3)]',
                                            'h_t_fine = [round(best_h_c[\'T\'] + i*0.02, 2) for i in range(-3, 4)]') # ±0.06 instead of ±0.04
            
            target_found = True
            break

    if target_found:
        with open(file_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"Successfully updated hybrid grid in {file_path}")
    else:
        print("Target hybrid tuning parameters not found in any cell.")

if __name__ == "__main__":
    update_hybrid_grid('strategy_comparison_coba_toGrid2stage4Matriks_clean.ipynb')
