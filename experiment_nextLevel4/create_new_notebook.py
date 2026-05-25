
import json
import os

# Path file asli dan baru
original_notebook_path = r'g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\experiment_nextLevel4\crypto_strategy_v62_multiyear_comparison.ipynb'
new_notebook_path = r'g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\experiment_nextLevel4\crypto_strategy_v62_multiyear_comparison_v2.ipynb'

# Fungsi Simulasi Baru (Realistic Drift & Turnover Buffer)
cell_simulation_functions = [
    "# Simulation Functions (AI-Gated & Static) - Updated with Drift & Turnover Buffer\n",
    "\n",
    "def run_simulation_ai_gated(test_dates, returns, ai_probs, threshold, market_idx, \n",
    "                            selection_func, override_ma=None, fee=0.0025, \n",
    "                            g_lookback=60, rebal_period=20, turnover_buffer=0.05):\n",
    "    \n",
    "    val = 100.0\n",
    "    history = [val]\n",
    "    dates = [test_dates[0]]\n",
    "    \n",
    "    current_weights = {} \n",
    "    risk_status = False \n",
    "    days_since_rebal = 999\n",
    "    \n",
    "    market_ma = None\n",
    "    if override_ma:\n",
    "        market_ma = market_idx.rolling(window=override_ma).mean()\n",
    "    \n",
    "    for i, date in enumerate(test_dates[:-1]):\n",
    "        prev_risk_status = risk_status\n",
    "        prob = ai_probs.loc[date]\n",
    "        \n",
    "        # AI Signal with hysteresis\n",
    "        ai_signal = False\n",
    "        buffer = 0.05\n",
    "        if not risk_status and prob > (threshold + buffer): \n",
    "            ai_signal = True\n",
    "        elif risk_status and prob < (threshold - buffer): \n",
    "            ai_signal = False\n",
    "        else: \n",
    "            ai_signal = risk_status\n",
    "        \n",
    "        # Trend Override\n",
    "        final_signal = ai_signal\n",
    "        if override_ma and market_ma is not None:\n",
    "            try:\n",
    "                current_price = market_idx.loc[date]\n",
    "                ma_price = market_ma.loc[date]\n",
    "                if not pd.isna(ma_price) and not pd.isna(current_price):\n",
    "                    is_uptrend = (current_price > ma_price)\n",
    "                    if ai_signal == False and is_uptrend:\n",
    "                        final_signal = True\n",
    "            except KeyError:\n",
    "                pass\n",
    "        \n",
    "        risk_status = final_signal\n",
    "        regime_changed = (risk_status != prev_risk_status)\n",
    "        \n",
    "        target_weights = current_weights.copy()\n",
    "        \n",
    "        if not risk_status:\n",
    "            if regime_changed or 'CASH' not in current_weights or current_weights.get('CASH', 0) < 0.99:\n",
    "                target_weights = {'CASH': 1.0}\n",
    "                days_since_rebal = 0\n",
    "        else:\n",
    "            if regime_changed or days_since_rebal >= rebal_period:\n",
    "                loc_idx = returns.index.get_loc(date)\n",
    "                window_rets = returns.iloc[max(0, loc_idx-g_lookback):loc_idx]\n",
    "                \n",
    "                selected = selection_func(window_rets)\n",
    "                optimized_weights = optimize_markowitz(window_rets[selected])\n",
    "                \n",
    "                all_keys = set(list(optimized_weights.keys()) + list(current_weights.keys()))\n",
    "                turnover_est = sum(abs(optimized_weights.get(k, 0) - current_weights.get(k, 0)) for k in all_keys)\n",
    "                \n",
    "                if regime_changed or turnover_est > turnover_buffer:\n",
    "                    target_weights = optimized_weights\n",
    "                    days_since_rebal = 0\n",
    "            \n",
    "            days_since_rebal += 1\n",
    "\n",
    "        # Execution with transaction costs\n",
    "        all_keys_exec = set(list(target_weights.keys()) + list(current_weights.keys()))\n",
    "        turnover_actual = sum(abs(target_weights.get(k, 0) - current_weights.get(k, 0)) for k in all_keys_exec)\n",
    "        \n",
    "        if turnover_actual > 0.001:\n",
    "            cost = val * turnover_actual * fee\n",
    "            val -= cost\n",
    "        \n",
    "        next_date = test_dates[i+1]\n",
    "        day_ret = 0\n",
    "        new_weights_drifted = {}\n",
    "        \n",
    "        if 'CASH' in target_weights and target_weights['CASH'] > 0.99:\n",
    "            day_ret = 0\n",
    "            new_weights_drifted = {'CASH': 1.0}\n",
    "        else:\n",
    "            for asset, w in target_weights.items():\n",
    "                if asset in returns.columns:\n",
    "                    r = returns.loc[next_date, asset]\n",
    "                    r_asset = r if not pd.isna(r) else 0\n",
    "                    day_ret += w * r_asset\n",
    "                    new_weights_drifted[asset] = w * (1 + r_asset)\n",
    "                else:\n",
    "                    new_weights_drifted[asset] = w\n",
    "            \n",
    "            total_w = sum(new_weights_drifted.values()) if new_weights_drifted else 0\n",
    "            if total_w > 0:\n",
    "                new_weights_drifted = {k: v/total_w for k, v in new_weights_drifted.items()}\n",
    "        \n",
    "        val *= (1 + day_ret)\n",
    "        history.append(val)\n",
    "        dates.append(next_date)\n",
    "        current_weights = new_weights_drifted\n",
    "    \n",
    "    return pd.DataFrame({'Portfolio_Value': history}, index=dates)\n",
    "\n",
    "def run_simulation_static(test_dates, returns, selection_func, fee=0.0025, \n",
    "                         g_lookback=60, rebal_period=20, turnover_buffer=0.05):\n",
    "    \n",
    "    val = 100.0\n",
    "    history = [val]\n",
    "    dates = [test_dates[0]]\n",
    "    \n",
    "    current_weights = {}\n",
    "    days_since_rebal = 999\n",
    "    \n",
    "    loc_idx = returns.index.get_loc(test_dates[0])\n",
    "    # Initial allocation attempt\n",
    "    try:\n",
    "        window_rets = returns.iloc[max(0, loc_idx-g_lookback):loc_idx]\n",
    "        selected = selection_func(window_rets)\n",
    "        current_weights = optimize_markowitz(window_rets[selected])\n",
    "    except:\n",
    "        current_weights = {}\n",
    "\n",
    "    for i, date in enumerate(test_dates[:-1]):\n",
    "        target_weights = current_weights.copy()\n",
    "        \n",
    "        if days_since_rebal >= rebal_period:\n",
    "            loc_idx = returns.index.get_loc(date)\n",
    "            window_rets = returns.iloc[max(0, loc_idx-g_lookback):loc_idx]\n",
    "            \n",
    "            selected = selection_func(window_rets)\n",
    "            optimized_weights = optimize_markowitz(window_rets[selected])\n",
    "            \n",
    "            all_keys = set(list(optimized_weights.keys()) + list(current_weights.keys()))\n",
    "            turnover_est = sum(abs(optimized_weights.get(k, 0) - current_weights.get(k, 0)) for k in all_keys)\n",
    "            \n",
    "            if turnover_est > turnover_buffer:\n",
    "                target_weights = optimized_weights\n",
    "                days_since_rebal = 0\n",
    "        \n",
    "        days_since_rebal += 1\n",
    "        \n",
    "        all_keys_exec = set(list(target_weights.keys()) + list(current_weights.keys()))\n",
    "        turnover_actual = sum(abs(target_weights.get(k, 0) - current_weights.get(k, 0)) for k in all_keys_exec)\n",
    "        \n",
    "        if turnover_actual > 0.001:\n",
    "            cost = val * turnover_actual * fee\n",
    "            val -= cost\n",
    "        \n",
    "        next_date = test_dates[i+1]\n",
    "        day_ret = 0\n",
    "        new_weights_drifted = {}\n",
    "        \n",
    "        for asset, w in target_weights.items():\n",
    "            if asset in returns.columns:\n",
    "                r = returns.loc[next_date, asset]\n",
    "                r_asset = r if not pd.isna(r) else 0\n",
    "                day_ret += w * r_asset\n",
    "                new_weights_drifted[asset] = w * (1 + r_asset)\n",
    "            else:\n",
    "                new_weights_drifted[asset] = w\n",
    "        \n",
    "        total_w = sum(new_weights_drifted.values()) if new_weights_drifted else 0\n",
    "        if total_w > 0:\n",
    "            new_weights_drifted = {k: v/total_w for k, v in new_weights_drifted.items()}\n",
    "        \n",
    "        val *= (1 + day_ret)\n",
    "        history.append(val)\n",
    "        dates.append(next_date)\n",
    "        current_weights = new_weights_drifted\n",
    "    \n",
    "    return pd.DataFrame({'Portfolio_Value': history}, index=dates)"
]

# Loop Eksekusi Utama (Memanggil fungsi baru dengan parameter baru)
cell_main_loop = [
    "# Main Loop - Year by Year (Updated Logic)\n",
    "years = [2023, 2024, 2025]\n",
    "results_by_year = {}\n",
    "\n",
    "# Fee Rate (0.25%)\n",
    "fee_rate = 0.0025\n",
    "\n",
    "for year in years:\n",
    "    print(f\"Processing Year: {year}...\")\n",
    "    \n",
    "    # Filter data for specific year\n",
    "    year_data = market_index.loc[str(year)]\n",
    "    if len(year_data) > 0:\n",
    "        # Re-index returns and probs for the specific year\n",
    "        year_returns = returns.loc[year_data.index]\n",
    "        year_probs = all_probs.loc[year_data.index]\n",
    "        \n",
    "        if len(year_returns) > 0:\n",
    "            # Run Strategies (Updated with turnover buffer & drift logic)\n",
    "            # 1. Static Markowitz\n",
    "            res_static = run_simulation_static(year_data.index, returns, get_all_assets, \n",
    "                                             fee=fee_rate, turnover_buffer=0.05)\n",
    "\n",
    "            # 2. AI + Graph Diversify (with MA override)\n",
    "            res_ai_div = run_simulation_ai_gated(year_data.index, returns, all_probs, opt_t, market_index, \n",
    "                                                 get_assets_graph_diversify, override_ma=50, \n",
    "                                                 fee=fee_rate, turnover_buffer=0.05)\n",
    "            \n",
    "            # 3. AI + Graph Cluster (with MA override)\n",
    "            res_ai_clust = run_simulation_ai_gated(year_data.index, returns, all_probs, opt_t, market_index, \n",
    "                                                   get_assets_graph_cluster, override_ma=50, \n",
    "                                                   fee=fee_rate, turnover_buffer=0.05)\n",
    "            \n",
    "            # Create comparison dataframe\n",
    "            df_res = pd.DataFrame({\n",
    "                'Static Markowitz': res_static['Portfolio_Value'],\n",
    "                'AI + Graph Diversify': res_ai_div['Portfolio_Value'],\n",
    "                'AI + Graph Cluster': res_ai_clust['Portfolio_Value']\n",
    "            })\n",
    "            \n",
    "            results_by_year[year] = df_res\n",
    "\n",
    "# Plot Results\n",
    "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
    "\n",
    "for i, year in enumerate(years):\n",
    "    if year in results_by_year:\n",
    "        df = results_by_year[year]\n",
    "        # Normalize to 100\n",
    "        df = df / df.iloc[0] * 100\n",
    "        \n",
    "        df.plot(ax=axes[i], linewidth=2)\n",
    "        axes[i].set_title(f\"Performance Year {year}\")\n",
    "        axes[i].set_ylabel(\"Portfolio Value (Base 100)\")\n",
    "        axes[i].grid(True, alpha=0.3)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
]

try:
    with open(original_notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Cari dan ganti sel yang mendefinisikan fungsi simulasi
    # Kita cari sel yang mengandung 'def run_simulation_ai_gated'
    
    simulation_cell_found = False
    loop_cell_found = False
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_code = "".join(cell['source'])
            
            # Ganti sel fungsi simulasi
            if "def run_simulation_ai_gated" in source_code and "def run_simulation_static" in source_code:
                cell['source'] = cell_simulation_functions
                simulation_cell_found = True
                print("Simulation functions updated.")
                
            # Ganti sel loop utama
            if "years = [2023, 2024, 2025]" in source_code and "results_by_year = {}" in source_code:
                cell['source'] = cell_main_loop
                loop_cell_found = True
                print("Main execution loop updated.")

    if not simulation_cell_found:
        print("Warning: Simulation function cell not found by exact match. You may need to inspect manually.")
    if not loop_cell_found:
        print("Warning: Main loop cell not found by exact match. You may need to inspect manually.")

    # Simpan sebagai file baru
    with open(new_notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"New notebook created successfully: {new_notebook_path}")

except Exception as e:
    print(f"Error: {e}")
