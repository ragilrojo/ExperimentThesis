import json

file_path = r'g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\guidiciti_reexperiment\RLNetworkMarkowitz_Thesis_FinalResettingGamma.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'action, _ = self.model.predict(obs, deterministic=True)' in "".join(cell['source']):
        source = cell['source']
        new_source = []
        for line in source:
            if 'obs = np.array([m, s, c, st_c], dtype=np.float32)' in line:
                new_source.append('        obs = np.array([m, s, c, st_c], dtype=np.float32)\n')
                new_source.append('        # Fix for potential NaN values and ensure shape compatibility\n')
                new_source.append('        obs = np.nan_to_num(obs)\n')
            else:
                new_source.append(line)
        cell['source'] = new_source

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Berhasil menambahkan penanganan NaN pada observasi model.")
