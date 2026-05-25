"""
fix_dashboard.py
Surgically corrects the IndentationError in the plot_dashboard cell of the notebook.
The problem: `base = {...}` was injected with 16-space indent instead of 8-space.
"""
import json

NB_PATH = r'g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\guidiciti_reexperiment\RLNetworkMarkowitz_4Metrics_CalmarRatioReward_Fixed_modif.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The correct interpretation block with proper 8-space indentation
# (inside `if period == 'Test':` which is itself inside `def plot_dashboard(...)`)
CORRECT_BLOCK = [
    "        base = {m: summary_df.loc['Comp_Static_Gamma0', f'{METRIC_KEY[m]}_Test_Mean']\n",
    "                for m in FOUR_METRICS}\n",
    "\n",
    "        def delta(exp, m):\n",
    "            return summary_df.loc[exp, f'{METRIC_KEY[m]}_Test_Mean'] - base[m]\n",
    "\n",
    "        def drop_delta(exp, m):\n",
    "            # Perbandingan vs E2_NoMarket (Full Network)\n",
    "            return (summary_df.loc[exp, f'{METRIC_KEY[m]}_Test_Mean']\n",
    "                    - summary_df.loc['E2_NoMarket', f'{METRIC_KEY[m]}_Test_Mean'])\n",
    "\n",
    "        interpretasi = (\n",
    "            \"INTERPRETASI ABLATION STUDY \\u2014 4 METRIK TESIS:\\n\\n\"\n",
    "            f\"  Best Sharpe Ratio   : {best_sharpe} = {summary_df.loc[best_sharpe,  'SharpeRatio_Test_Mean']:.4f}\\n\"\n",
    "            f\"  Best Sortino Ratio  : {best_sortino} = {summary_df.loc[best_sortino, 'SortinoRatio_Test_Mean']:.4f}\\n\"\n",
    "            f\"  Best Calmar Ratio   : {best_calmar} = {summary_df.loc[best_calmar,  'CalmarRatio_Test_Mean']:.4f}\\n\"\n",
    "            f\"  Best CVaR (95%)     : {best_cvar} = {summary_df.loc[best_cvar, 'CVaR95pct_Test_Mean']:.5f} (terkecil)\\n\\n\"\n",
    "            \"  PERFORMA PROPOSED (E2_NoMarket vs Comp_Static_Gamma0):\\n\"\n",
    "            f\"  Sharpe Delta: {delta('E2_NoMarket','Sharpe Ratio'):+.4f} | \"\n",
    "            f\"Sortino Delta: {delta('E2_NoMarket','Sortino Ratio'):+.4f}\\n\\n\"\n",
    "            \"  KONTRIBUSI FITUR NETWORK (Drop Performance vs E2_NoMarket):\\n\"\n",
    "            f\"  Drop CentStd  -> Sharpe: {drop_delta('E2_drop_CentStd', 'Sharpe Ratio'):+.4f}\\n\"\n",
    "            f\"  Drop CentMean -> Sharpe: {drop_delta('E2_drop_CentMean','Sharpe Ratio'):+.4f}\\n\"\n",
    "            f\"  Drop MSTDist  -> Sharpe: {drop_delta('E2_drop_MSTDist', 'Sharpe Ratio'):+.4f}\\n\"\n",
    "            f\"  Drop MaxCent  -> Sharpe: {drop_delta('E2_drop_MaxCent', 'Sharpe Ratio'):+.4f}\\n\"\n",
    "            f\"  Drop NetDens  -> Sharpe: {drop_delta('E2_drop_NetDens', 'Sharpe Ratio'):+.4f}\\n\"\n",
    "            \"\\n* Nilai negatif pada Drop Performance menunjukkan fitur tersebut berkontribusi positif.\"\n",
    "        )\n",
]

fixed = False
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    src = cell.get('source', [])
    # Find the bad line: 16-space indent on `base =`
    bad_idx = None
    end_idx = None
    for i, line in enumerate(src):
        if "                base = {m: summary_df.loc['Comp_Static_Gamma0'" in line:
            bad_idx = i
        if bad_idx is not None and "ax_interp.text" in line:
            end_idx = i
            break

    if bad_idx is not None and end_idx is not None:
        print(f"Found bad block at lines {bad_idx}-{end_idx} in cell")
        cell['source'] = src[:bad_idx] + CORRECT_BLOCK + src[end_idx:]
        fixed = True
        break

if fixed:
    with open(NB_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("✅ Fixed! Indentation corrected in plot_dashboard.")
else:
    print("⚠️  Pattern not found — check notebook manually.")
    # Print first few lines of each code cell that has 'plot_dashboard' to debug
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and 'plot_dashboard' in ''.join(cell.get('source', [])):
            print(f"\n--- Cell {i} (first 10 lines) ---")
            for line in cell['source'][:10]:
                print(repr(line))
