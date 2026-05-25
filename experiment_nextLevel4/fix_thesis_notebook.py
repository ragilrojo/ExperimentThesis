
import json
import os

# Path file V2 sebagai template
template_path = r'g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\experiment_nextLevel4\crypto_strategy_v62_multiyear_comparison_v2.ipynb'
new_notebook_path = r'g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\experiment_nextLevel4\crypto_strategy_v62_multiyear_comparison_v3_thesis_grade.ipynb'

# Header MD
cell_header = [
    "# V62 Crypto: Thesis-Grade Multi-Year Analysis (In-Sample vs Out-of-Sample)\n",
    "\n",
    "## Objective\n",
    "Melakukan evaluasi strategi secara adil (**Fair Evaluation**) dengan memisahkan periode:\n",
    "1. **In-Sample (Training): 2023-2024** - Evaluasi kemampuan model mempelajari pola historis (Overfitting Check).\n",
    "2. **Out-of-Sample (Testing): 2025** - Validasi performa strategi pada data masa depan yang belum pernah dilihat model (**True Performance**).\n",
    "\n",
    "## Methodology\n",
    "- **Model Training:** Menggunakan data hingga akhir 2024.\n",
    "- **Simulation Engine:** Menggunakan 'Realistic Drift' dan 'Turnover Buffer (0.05)' untuk akurasi biaya transaksi.\n",
    "- **Benchmark:** Buy & Hold Market Index (Equal Weight).\n",
    "- **Metrics:** Sharpe Ratio, Max Drawdown, Total Return, Win Rate.\n",
    "- **Robustness:** Menggunakan intersection index untuk menghindari Missing Data Error."
]

# Fungsi Metrics
cell_metrics_function = [
    "# Performance Metrics Calculation Functions\n",
    "def calculate_metrics(portfolio_history, risk_free_rate=0.02):\n",
    "    # Convert to series if dataframe\n",
    "    if isinstance(portfolio_history, pd.DataFrame):\n",
    "        portfolio_history = portfolio_history.iloc[:, 0]\n",
    "    \n",
    "    # Daily Returns\n",
    "    returns = portfolio_history.pct_change().dropna()\n",
    "    if len(returns) == 0: return {'Total Return': '0%', 'Sharpe Ratio': '0', 'Max Drawdown': '0%', 'Volatility': '0%'}\n",
    "    \n",
    "    # 1. Total Return\n",
    "    total_return = (portfolio_history.iloc[-1] / portfolio_history.iloc[0]) - 1\n",
    "    \n",
    "    # 2. Sharpe Ratio (Annualized)\n",
    "    excess_returns = returns - (risk_free_rate / 252)\n",
    "    sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / returns.std()) if returns.std() != 0 else 0\n",
    "    \n",
    "    # 3. Max Drawdown\n",
    "    roll_max = portfolio_history.cummax()\n",
    "    drawdown = (portfolio_history - roll_max) / roll_max\n",
    "    max_drawdown = drawdown.min()\n",
    "    \n",
    "    # 4. Volatility (Annualized)\n",
    "    volatility = returns.std() * np.sqrt(252)\n",
    "    \n",
    "    return {\n",
    "        'Total Return': f\"{total_return:.2%}\",\n",
    "        'Sharpe Ratio': f\"{sharpe_ratio:.2f}\",\n",
    "        'Max Drawdown': f\"{max_drawdown:.2%}\",\n",
    "        'Volatility': f\"{volatility:.2%}\"\n",
    "    }"
]

# Main Loop FIX (Added Intersection Logic to Prevent KeyError)
cell_main_execution = [
    "# --- THESIS EXPERIMENT EXECUTION (ROBUST VERSION) ---\n",
    "phases = {\n",
    "    'In-Sample (Training)': [2023, 2024],\n",
    "    'Out-of-Sample (Testing)': [2025]\n",
    "}\n",
    "\n",
    "fee_rate = 0.0025\n",
    "results_metrics = []\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(18, 6))\n",
    "\n",
    "# Ensure market_index and returns align\n",
    "valid_data_index = returns.index.intersection(market_index.index)\n",
    "\n",
    "for i, (phase_name, years) in enumerate(phases.items()):\n",
    "    print(f\"\\nrunning Phase: {phase_name}...\")\n",
    "    \n",
    "    # 1. Generate Target Dates\n",
    "    target_dates = []\n",
    "    for y in years:\n",
    "        # Get all dates in that year from our valid data index\n",
    "        year_dates = valid_data_index[valid_data_index.year == y]\n",
    "        if len(year_dates) > 0:\n",
    "            target_dates.extend(year_dates)\n",
    "            \n",
    "    phase_dates = pd.DatetimeIndex(target_dates).sort_values().unique()\n",
    "    \n",
    "    print(f\"  > Found {len(phase_dates)} trading days.\")\n",
    "    \n",
    "    if len(phase_dates) > 10:\n",
    "        # Run Strategies\n",
    "        try:\n",
    "            # 1. Benchmark (Static)\n",
    "            res_static = run_simulation_static(phase_dates, returns, get_all_assets, \n",
    "                                             fee=fee_rate, turnover_buffer=0.05)\n",
    "\n",
    "            # 2. AI + Graph Diversify\n",
    "            res_ai_div = run_simulation_ai_gated(phase_dates, returns, all_probs, opt_t, market_index, \n",
    "                                                 get_assets_graph_diversify, override_ma=50, \n",
    "                                                 fee=fee_rate, turnover_buffer=0.05)\n",
    "            \n",
    "            # 3. AI + Graph Cluster\n",
    "            res_ai_clust = run_simulation_ai_gated(phase_dates, returns, all_probs, opt_t, market_index, \n",
    "                                                   get_assets_graph_cluster, override_ma=50, \n",
    "                                                   fee=fee_rate, turnover_buffer=0.05)\n",
    "\n",
    "            # Calc Metrics & Store\n",
    "            for strat_name, res_data in [('Static Markowitz', res_static), \n",
    "                                         ('AI+Diversify', res_ai_div), \n",
    "                                         ('AI+Cluster', res_ai_clust)]:\n",
    "                m = calculate_metrics(res_data['Portfolio_Value'])\n",
    "                m['Phase'] = phase_name\n",
    "                m['Strategy'] = strat_name\n",
    "                results_metrics.append(m)\n",
    "                \n",
    "            # Plotting\n",
    "            df_plot = pd.DataFrame({\n",
    "                'Static Markowitz': res_static['Portfolio_Value'],\n",
    "                'AI + Graph Diversify': res_ai_div['Portfolio_Value'],\n",
    "                'AI + Graph Cluster': res_ai_clust['Portfolio_Value']\n",
    "            })\n",
    "            # Normalize\n",
    "            df_plot = df_plot / df_plot.iloc[0] * 100\n",
    "            \n",
    "            df_plot.plot(ax=axes[i], linewidth=2)\n",
    "            axes[i].set_title(f\"{phase_name}\\n({years[0]}-{years[-1]})\", fontsize=14, fontweight='bold')\n",
    "            axes[i].set_ylabel(\"Portfolio Value (Base 100)\")\n",
    "            axes[i].grid(True, alpha=0.3)\n",
    "            axes[i].legend(loc='upper left')\n",
    "        except Exception as e:\n",
    "             print(f\"  ! Error running simulation for {phase_name}: {e}\")\n",
    "             import traceback\n",
    "             traceback.print_exc()\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.savefig('thesis_grade_comparison.png')\n",
    "plt.show()\n",
    "\n",
    "# Display Metrics Table\n",
    "metrics_df = pd.DataFrame(results_metrics)\n",
    "if not metrics_df.empty:\n",
    "    metrics_df = metrics_df[['Phase', 'Strategy', 'Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Volatility']]\n",
    "    print(\"\\n--- THESIS PERFORMANCE SUMMARY ---\")\n",
    "    display(metrics_df.sort_values(by=['Phase', 'Strategy'], ascending=[True, True]))\n",
    "else:\n",
    "    print(\"No results to display.\")"
]

try:
    with open(template_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # 1. Update Cell 0 (Header)
    if nb['cells'][0]['cell_type'] == 'markdown':
        nb['cells'][0]['source'] = cell_header
    
    # 2. Insert Metrics Function Cell
    func_cell_idx = -1
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code' and "def run_simulation_ai_gated" in "".join(cell['source']):
            func_cell_idx = i
            break
            
    if func_cell_idx != -1:
        new_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": cell_metrics_function
        }
        nb['cells'].insert(func_cell_idx + 1, new_cell)
        
        # 3. Replace Main Loop Cell with ROBUST version
        loop_cell_idx = -1
        # Re-search because index shifted
        for i, cell in enumerate(nb['cells']):
             if cell['cell_type'] == 'code' and "years = [2023, 2024, 2025]" in "".join(cell['source']):
                loop_cell_idx = i
                break
        
        if loop_cell_idx != -1:
            nb['cells'][loop_cell_idx]['source'] = cell_main_execution
            print("Main loop updated with ROBUST date handling.")
        else:
            print("Warning: Could not find main loop cell to replace.")
            
    else:
        print("Warning: Could not find simulation function cell.")

    with open(new_notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"Fixed notebook created: {new_notebook_path}")

except Exception as e:
    print(f"Error: {e}")
