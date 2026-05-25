import json

notebook_path = r"g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\guidiciti_reexperiment\RLNetworkMarkowitz_4Metrics_CalmarRatioReward.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        if "Reward tetap menggunakan **Rolling Sharpe Ratio**" in source:
            new_source = source.replace(
                "Reward tetap menggunakan **Rolling Sharpe Ratio**",
                "Reward menggunakan **Calmar Ratio**"
            )
            
            lines = new_source.split('\n')
            cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "sharpe     = arr.mean() * np.sqrt(252) / (arr.std() + 1e-8)" in source:
            old_code = """        arr = np.array(self._returns_buffer)
        if len(arr) < 2:
            raw_reward = port_ret * 100
        else:
            sharpe     = arr.mean() * np.sqrt(252) / (arr.std() + 1e-8)
            raw_reward = float(np.clip(sharpe, -10.0, 10.0))"""
            
            new_code = """        arr = np.array(self._returns_buffer)
        if len(arr) < 2:
            raw_reward = port_ret * 100
        else:
            cumulative = (1 + arr).cumprod()
            peak = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - peak) / peak
            max_dd = drawdown.min()
            
            ann_ret = arr.mean() * 252
            
            denominator = abs(max_dd) if abs(max_dd) > 1e-8 else 1e-8
            calmar = ann_ret / denominator
            raw_reward = float(np.clip(calmar, -10.0, 10.0))"""
            
            new_source = source.replace(old_code, new_code)
            
            # also replace the class docstring
            new_source = new_source.replace(
                "Reward: Rolling Sharpe Ratio (konsisten dengan metrik evaluasi #1).",
                "Reward: Calmar Ratio (Return/MaxDD)."
            )
            
            lines = new_source.split('\n')
            cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Reward updated to Calmar Ratio successfully.")
