import json
import os

path = r'g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\guidiciti_reexperiment\strategy_comparison_coba_toGrid2stage4Matriks_adv.ipynb'

if not os.path.exists(path):
    print(f"Error: File not found at {path}")
    exit(1)

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_code = [
    "def objective_adaptive(params, target_metric='RETURN'):\n",
    "    s_thresh = params[0]\n",
    "    # Menggunakan W=30, G=1.0 (Basis), dan dalam class sudah kita ubah G_up = -1\n",
    "    temp_strat = AdaptiveNetworkMarkowitz('GS Temp', gamma=1.0, window_size=30, slope_threshold=s_thresh)\n",
    "    try:\n",
    "        res = backtest_strategy(temp_strat, df_returns, global_start=120)\n",
    "        rets = res['returns']\n",
    "        if len(rets) < 10: return 1e9\n",
    "        \n",
    "        if target_metric == 'RETURN': \n",
    "            return -(1 + pd.Series(rets)).cumprod().iloc[-1]\n",
    "        return 1e9\n",
    "    except:\n",
    "        return 1e9\n",
    "\n",
    "# --- 2-STAGE GRID SEARCH FOR ADAPTIVE SLOPE THRESHOLD ---\n",
    "\n",
    "# STAGE 1: COARSE SEARCH\n",
    "print('\\n[Adaptive] Stage 1: Coarse Searching for slope_threshold...')\n",
    "COARSE_SLOPE = [0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01]\n",
    "scores_coarse = []\n",
    "for s in COARSE_SLOPE:\n",
    "    score = objective_adaptive([s], 'RETURN')\n",
    "    scores_coarse.append({'slope_threshold': s, 'score': score})\n",
    "    print(f' > Checked s={s}: Return={-score:.4f}')\n",
    "\n",
    "best_coarse = min(scores_coarse, key=lambda x: x['score'])\n",
    "print(f' > Stage 1 Best: {best_coarse[\"slope_threshold\"]} (Return: {-best_coarse[\"score\"]:.4f})')\n",
    "\n",
    "# STAGE 2: FINE SEARCH\n",
    "print('\\n[Adaptive] Stage 2: Fine Searching around best coarse value...')\n",
    "base_s = best_coarse['slope_threshold']\n",
    "if base_s == 0:\n",
    "    FINE_SLOPE = [0, 0.00005, 0.0001, 0.00015, 0.0002]\n",
    "else:\n",
    "    FINE_SLOPE = np.linspace(base_s * 0.5, base_s * 1.5, 7)\n",
    "\n",
    "scores_fine = []\n",
    "for s in FINE_SLOPE:\n",
    "    s = round(s, 6)\n",
    "    score = objective_adaptive([s], 'RETURN')\n",
    "    scores_fine.append({'slope_threshold': s, 'score': score})\n",
    "    print(f' > Checked s={s}: Return={-score:.4f}')\n",
    "\n",
    "best_adaptive = min(scores_fine, key=lambda x: x['score'])\n",
    "print(f'\\n >>> FINAL BEST Adaptive Slope Threshold: {best_adaptive[\"slope_threshold\"]} <<<')\n",
    "print(f' >>> Optimal Return: {-best_adaptive[\"score\"]:.4f}')\n",
    "\n",
    "tuned_params['ADAPTIVE_SLOPE'] = best_adaptive['slope_threshold']\n"
]

cell_updated = False
for cell in nb['cells']:
    if cell.get('id') == 'adaptive_grid_search':
        cell['source'] = new_code
        cell_updated = True
        break

if cell_updated:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Successfully updated the adaptive_grid_search cell.")
else:
    print("Error: Cell with id 'adaptive_grid_search' not found.")
