import json
import os

file_path = r'g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\guidiciti_reexperiment\strategy_comparison_coba_toGrid2stage4Matriks_clean.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and any('objective_hybrid' in line for line in cell['source']):
        source = cell['source']
        new_source = []
        for line in source:
            # Fix Stage 1 (Remove indentation)
            if 'hybrid_w_coarse = [20, 40, 60, 80, 100, 120]' in line:
                new_source.append('hybrid_w_coarse = [20, 40, 60, 80, 100, 120]\n')
            elif 'hybrid_g_coarse = [0.0, 0.5, 1.0, 1.5, 2.0]' in line:
                new_source.append('hybrid_g_coarse = [0.0, 0.5, 1.0, 1.5, 2.0]\n')
            elif 'hybrid_t_coarse = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]' in line:
                new_source.append('hybrid_t_coarse = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]\n')
            elif '# Search space lebih luas untuk optimasi lebih mendalam' in line:
                new_source.append('# Search space lebih luas untuk optimasi lebih mendalam\n')
            
            # Fix Stage 2 (Ensure proper indentation for logic but 0 for definitions)
            elif "h_w_fine = list(range(max(5, best_h_c['W'] - 3), min(120, best_h_c['W'] + 4)))" in line:
                new_source.append("h_w_fine = list(range(max(5, best_h_c['W'] - 3), min(120, best_h_c['W'] + 4)))\n")
            elif "h_g_fine = [round(best_h_c['G'] + i*0.1, 1) for i in range(-2, 3)]" in line:
                new_source.append("h_g_fine = [round(best_h_c['G'] + i*0.1, 1) for i in range(-2, 3)]\n")
            elif "h_g_fine = [x for x in h_g_fine if 0 <= x <= 2.0]" in line:
                new_source.append("h_g_fine = [x for x in h_g_fine if 0 <= x <= 2.0]\n")
            elif "h_t_fine = [round(best_h_c['T'] + i*0.02, 2) for i in range(-2, 3)]" in line:
                new_source.append("h_t_fine = [round(best_h_c['T'] + i*0.02, 2) for i in range(-2, 3)]\n")
            elif "h_t_fine = [x for x in h_t_fine if 0.1 <= x <= 0.8]" in line:
                new_source.append("h_t_fine = [x for x in h_t_fine if 0.1 <= x <= 0.8]\n")
            
            # Also check the fine tuning results line
            elif 'tuned_hybrid_params = best_hybrid_tuned' in line:
                new_source.append('tuned_hybrid_params = best_hybrid_tuned\n')
            elif 'print("\\n[Stage 2] Fine tuning for Hybrid MIS-NW...")' in line:
                new_source.append('print("\\n[Stage 2] Fine tuning for Hybrid MIS-NW...")\n')
            elif 'valid_h_fine = []' in line:
                new_source.append('valid_h_fine = []\n')
                
            else:
                new_source.append(line)
        cell['source'] = new_source
        break

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook indentation fixed.")
