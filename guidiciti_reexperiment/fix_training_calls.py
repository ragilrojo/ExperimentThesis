import json

file_path = r'g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\guidiciti_reexperiment\RLNetworkMarkowitz_Thesis_FinalResettingGamma.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'STEPS         = 50000' in source and 'train_optimized_model(' in source:
            # Rebuild the source with correct calls
            new_source = []
            i = 0
            lines = cell['source']
            skip_next_empty_call = 0
            # We need to fix the two empty train_optimized_model() calls
            # and add back the RL strategies to the strategies list
            fixed_lines = []
            j = 0
            while j < len(lines):
                line = lines[j]
                # Fix empty train_optimized_model() calls
                if line.strip() == 'train_optimized_model(' and j + 1 < len(lines) and lines[j+1].strip() == ')':
                    # Determine which call this is (first=mdd, second=raw_return)
                    if skip_next_empty_call == 0:
                        fixed_lines.append("train_optimized_model(\n")
                        fixed_lines.append("    GammaPortfolioEnv(train_data, reward_mode='defensive'),\n")
                        fixed_lines.append("    'final_mdd', val_data, 'RL-Net (Defensive)'\n")
                        fixed_lines.append(")\n")
                        skip_next_empty_call = 1
                    else:
                        fixed_lines.append("train_optimized_model(\n")
                        fixed_lines.append("    GammaPortfolioEnv(train_data, reward_mode='raw_return'),\n")
                        fixed_lines.append("    'final_raw', val_data, 'RL-Net (Total Return)'\n")
                        fixed_lines.append(")\n")
                    j += 2  # Skip the ")\n" line that follows
                    continue
                # Add back RL strategies to the strategies list
                elif "NetworkMarkowitz('NW (Gamma=0.0)', gamma=0.0)," in line and '    NetworkMarkowitz' in line:
                    # Check if the next line is "]" (meaning RL strategies are missing)
                    if j + 1 < len(lines) and lines[j+1].strip() == ']':
                        fixed_lines.append(line)
                        fixed_lines.append("    RLNetworkMarkowitz('RL-Net (Defensive)',    'final_mdd'),\n")
                        fixed_lines.append("    RLNetworkMarkowitz('RL-Net (Total Return)', 'final_raw'),\n")
                        j += 1
                        continue
                fixed_lines.append(line)
                j += 1
            cell['source'] = fixed_lines
            break

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Done!")
