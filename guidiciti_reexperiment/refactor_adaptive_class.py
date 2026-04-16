import json

notebook_path = 'strategy_comparison_coba_toGrid2stage4Matriks_adv.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 1. Update Class Definition
class_found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'class AdaptiveNetworkMarkowitz' in "".join(cell['source']):
        new_class_source = [
            "class AdaptiveNetworkMarkowitz(NetworkMarkowitz):\n",
            "    def __init__(self, name=\"Adaptive NW\", gamma_up=0.5, gamma_down=1.0, window_size=30, slope_threshold=0, p_threshold=0.05):\n",
            "        # Kita gunakan gamma_down sebagai default base_gamma\n",
            "        super().__init__(name, gamma_down, window_size)\n",
            "        self.gamma_up = gamma_up\n",
            "        self.gamma_down = gamma_down\n",
            "        self.W = window_size\n",
            "        self.slope_threshold = slope_threshold\n",
            "        self.p_threshold = p_threshold\n",
            "        self.gamma_history = []\n",
            "\n",
            "    def get_weights(self, returns_data):\n",
            "        mean_rets = returns_data.mean(axis=1)\n",
            "        cum_rets = (1 + mean_rets).cumprod()\n",
            "        x = np.arange(len(cum_rets))\n",
            "        slope, intercept, r_value, p_value, std_err = linregress(x, cum_rets.values)\n",
            "        \n",
            "        # Logika adaptive berdasarkan parameter:\n",
            "        if slope > self.slope_threshold and p_value < self.p_threshold:\n",
            "            current_gamma = self.gamma_up\n",
            "        else:\n",
            "            current_gamma = self.gamma_down\n",
            "            \n",
            "        self.gamma = current_gamma\n",
            "        current_date = returns_data.index[-1]\n",
            "        self.gamma_history.append({'date': current_date, 'gamma': current_gamma})\n",
            "        \n",
            "        return super().get_weights(returns_data)\n",
            "    \n",
            "    def get_params(self):\n",
            "        return {'W': self.W, 'gamma_up': self.gamma_up, 'gamma_down': self.gamma_down, 'slope_threshold': self.slope_threshold}\n"
        ]
        cell['source'] = new_class_source
        class_found = True
        break

# 2. Update Simulation Call in the last cell
sim_found = False
for cell in reversed(nb['cells']):
    if cell['cell_type'] == 'code' and 'strategies_2024 =' in "".join(cell['source']):
        source = cell['source']
        new_source = []
        for line in source:
            if 'AdaptiveNetworkMarkowitz' in line:
                # Update call to use new parameters
                new_line = '    AdaptiveNetworkMarkowitz("NW Adaptis (Up=0.5, Down=1.0, W=120)", gamma_up=0.5, gamma_down=1.0, window_size=120, slope_threshold=0.013333)\n'
                new_source.append(new_line)
            else:
                new_source.append(line)
        cell['source'] = new_source
        sim_found = True
        break

if class_found and sim_found:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("AdaptiveNetworkMarkowitz refactored with gamma_up and gamma_down parameters successfully.")
else:
    print(f"Update failed. Class found: {class_found}, Simulation found: {sim_found}")
