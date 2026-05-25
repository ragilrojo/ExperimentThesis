"""
Script to add hybrid strategy to the strategy comparison notebook
Hybrid Strategy: Cluster in Bull regime, Diversify in Bear regime, NO CASH
"""

import json
import sys

# Read the original notebook
notebook_path = r"g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\experiment_nextLevel5\strategy_comparison_diversify_vs_cluster_v2.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell with simulation functions (cell #6)
# We'll add the hybrid function after it

hybrid_cell_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 5b. Hybrid Simulation Function (Bull = Cluster, Bear = Diversify)"
    ]
}

hybrid_cell_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def run_simulation_hybrid(test_dates, returns, ai_probs, threshold, market_idx, \n",
        "                         override_ma=None, fee=0.0025, \n",
        "                         g_lookback=60, rebal_period=20, turnover_buffer=0.05):\n",
        "    \"\"\"Run hybrid simulation - Cluster in Bull, Diversify in Bear, NO CASH\"\"\"\n",
        "    val = 100.0\n",
        "    history = [val]\n",
        "    dates = [test_dates[0]]\n",
        "    current_weights = {} \n",
        "    risk_status = False\n",
        "    days_since_rebal = 999\n",
        "    \n",
        "    market_ma = None\n",
        "    if override_ma:\n",
        "        market_ma = market_idx.rolling(window=override_ma).mean()\n",
        "    \n",
        "    for i, date in enumerate(test_dates[:-1]):\n",
        "        try:\n",
        "            prob = ai_probs.loc[date]\n",
        "        except KeyError:\n",
        "            print(f\"Warning: Missing prob for {date}, using previous\")\n",
        "            prob = ai_probs.iloc[ai_probs.index.get_loc(date, method='ffill')]\n",
        "            \n",
        "        prev_risk_status = risk_status\n",
        "        \n",
        "        # AI Signal with hysteresis\n",
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
        "        target_weights = current_weights.copy()\n",
        "        \n",
        "        # HYBRID: Always invested, switch between Cluster (bull) and Diversify (bear)\n",
        "        if regime_changed or days_since_rebal >= rebal_period:\n",
        "            try:\n",
        "                loc_idx = returns.index.get_loc(date)\n",
        "                window_rets = returns.iloc[max(0, loc_idx-g_lookback):loc_idx]\n",
        "                \n",
        "                if len(window_rets) > 0:\n",
        "                    # Use CLUSTER in BULL, DIVERSIFY in BEAR\n",
        "                    if risk_status:\n",
        "                        selected = get_assets_graph_cluster(window_rets)\n",
        "                    else:\n",
        "                        selected = get_assets_graph_diversify(window_rets)\n",
        "                    \n",
        "                    if len(selected) > 0:\n",
        "                        optimized_weights = optimize_markowitz(window_rets[selected])\n",
        "                        if optimized_weights:\n",
        "                            all_keys = set(list(optimized_weights.keys()) + list(current_weights.keys()))\n",
        "                            turnover_est = sum(abs(optimized_weights.get(k, 0) - current_weights.get(k, 0)) for k in all_keys)\n",
        "                            if regime_changed or turnover_est > turnover_buffer:\n",
        "                                target_weights = optimized_weights\n",
        "                                days_since_rebal = 0\n",
        "            except (KeyError, ValueError) as e:\n",
        "                pass\n",
        "                \n",
        "        days_since_rebal += 1\n",
        "\n",
        "        # Transaction costs\n",
        "        all_keys_exec = set(list(target_weights.keys()) + list(current_weights.keys()))\n",
        "        turnover_actual = sum(abs(target_weights.get(k, 0) - current_weights.get(k, 0)) for k in all_keys_exec)\n",
        "        if turnover_actual > 0.001:\n",
        "            cost = val * turnover_actual * fee\n",
        "            val -= cost\n",
        "        \n",
        "        # Get next date and calculate returns\n",
        "        next_date = test_dates[i+1]\n",
        "        day_ret = 0\n",
        "        new_weights_drifted = {}\n",
        "        \n",
        "        for asset, w in target_weights.items():\n",
        "            if asset in returns.columns:\n",
        "                try:\n",
        "                    r = returns.loc[next_date, asset]\n",
        "                    r_asset = r if not pd.isna(r) else 0\n",
        "                    day_ret += w * r_asset\n",
        "                    new_weights_drifted[asset] = w * (1 + r_asset)\n",
        "                except KeyError:\n",
        "                    new_weights_drifted[asset] = w\n",
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
        "    return pd.DataFrame({'Portfolio_Value': history}, index=dates)\n",
        "\n",
        "print(\"✓ Hybrid simulation function ready\")"
    ]
}

# Insert the new cells after cell index 6 (simulation functions)
# Cells are indexed as: 0-5 setup, 6 simulation functions, we insert at 7
insert_position = 7

nb['cells'].insert(insert_position, hybrid_cell_markdown)
nb['cells'].insert(insert_position + 1, hybrid_cell_code)

print(f"[OK] Added hybrid function cells at position {insert_position}")

# Now update the Run Simulations cell to include Hybrid
# Find the cell with "## 6. Run Simulations"
for idx, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'markdown' and any('## 6. Run Simulations' in line for line in cell.get('source', [])):
        # Next cell should be the code cell
        code_cell_idx = idx + 1
        
        # Update the code to include Hybrid
        new_code = [
            "# Run simulations for each year\n",
            "years = [2023, 2024, 2025]\n",
            "results = {}\n",
            "\n",
            "for year in years:\n",
            "    print(f\"\\nProcessing {year}...\")\n",
            "    test_dates = returns.loc[returns.index.year == year].index\n",
            "    print(f\"  → Test dates: {len(test_dates)} days ({test_dates[0]} to {test_dates[-1]})\")\n",
            "    \n",
            "    # Diversify Strategy\n",
            "    print(\"  → Running Diversify...\")\n",
            "    diversify = run_simulation_ai_gated(\n",
            "        test_dates, returns, all_probs, opt_t, market_index,\n",
            "        get_assets_graph_diversify, override_ma=200\n",
            "    )\n",
            "    \n",
            "    # Cluster Strategy\n",
            "    print(\"  → Running Cluster...\")\n",
            "    cluster = run_simulation_ai_gated(\n",
            "        test_dates, returns, all_probs, opt_t, market_index,\n",
            "        get_assets_graph_cluster, override_ma=200\n",
            "    )\n",
            "    \n",
            "    # Hybrid Strategy (NEW)\n",
            "    print(\"  → Running Hybrid (Cluster in Bull, Diversify in Bear)...\")\n",
            "    hybrid = run_simulation_hybrid(\n",
            "        test_dates, returns, all_probs, opt_t, market_index,\n",
            "        override_ma=200\n",
            "    )\n",
            "    \n",
            "    # Static Markowitz\n",
            "    print(\"  → Running Static baseline...\")\n",
            "    static = run_simulation_static(\n",
            "        test_dates, returns, \n",
            "        lambda x: list(x.columns)\n",
            "    )\n",
            "    \n",
            "    results[year] = {\n",
            "        'Diversify': diversify,\n",
            "        'Cluster': cluster,\n",
            "        'Hybrid': hybrid,\n",
            "        'Static': static\n",
            "    }\n",
            "    \n",
            "    print(f\"  ✓ {year} completed:\")\n",
            "    print(f\"    Diversify: {diversify['Portfolio_Value'].iloc[-1]:.2f}\")\n",
            "    print(f\"    Cluster:   {cluster['Portfolio_Value'].iloc[-1]:.2f}\")\n",
            "    print(f\"    Hybrid:    {hybrid['Portfolio_Value'].iloc[-1]:.2f}\")\n",
            "    print(f\"    Static:    {static['Portfolio_Value'].iloc[-1]:.2f}\")\n",
            "\n",
            "print(\"\\n✓ All simulations completed!\")"
        ]
        
        nb['cells'][code_cell_idx]['source'] = new_code
        print(f"[OK] Updated simulation run cell at index {code_cell_idx}")
        break

# Update metrics calculation
for idx, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'markdown' and any('## 7. Calculate Performance Metrics' in line for line in cell.get('source', [])):
        code_cell_idx = idx + 1
        
        new_code = [
            "def calculate_metrics(portfolio_df):\n",
            "    \"\"\"Calculate comprehensive performance metrics\"\"\"\n",
            "    if len(portfolio_df) == 0:\n",
            "        return {\n",
            "            'Total Return (%)': 0,\n",
            "            'Ann. Return (%)': 0,\n",
            "            'Volatility (%)': 0,\n",
            "            'Sharpe Ratio': 0,\n",
            "            'Max Drawdown (%)': 0\n",
            "        }\n",
            "    \n",
            "    values = portfolio_df['Portfolio_Value']\n",
            "    returns = values.pct_change().dropna()\n",
            "    \n",
            "    total_return = (values.iloc[-1] / values.iloc[0] - 1) * 100\n",
            "    ann_return = ((1 + total_return/100) ** (252/max(len(returns), 1)) - 1) * 100\n",
            "    volatility = returns.std() * np.sqrt(252) * 100 if len(return) > 0 else 0\n",
            "    sharpe = ann_return / volatility if volatility > 0 else 0\n",
            "    \n",
            "    # Max Drawdown\n",
            "    cummax = values.cummax()\n",
            "    drawdown = ((values - cummax) / cummax * 100)\n",
            "    max_dd = drawdown.min()\n",
            "    \n",
            "    return {\n",
            "        'Total Return (%)': total_return,\n",
            "        'Ann. Return (%)': ann_return,\n",
            "        'Volatility (%)': volatility,\n",
            "        'Sharpe Ratio': sharpe,\n",
            "        'Max Drawdown (%)': max_dd\n",
            "    }\n",
            "\n",
            "# Calculate metrics for all strategies\n",
            "metrics_summary = []\n",
            "\n",
            "for year in years:\n",
            "    for strategy_name in ['Diversify', 'Cluster', 'Hybrid', 'Static']:\n",
            "        metrics = calculate_metrics(results[year][strategy_name])\n",
            "        metrics['Year'] = year\n",
            "        metrics['Strategy'] = strategy_name\n",
            "        metrics_summary.append(metrics)\n",
            "\n",
            "metrics_df = pd.DataFrame(metrics_summary)\n",
            "metrics_df = metrics_df[['Year', 'Strategy', 'Total Return (%)', 'Ann. Return (%)', \n",
            "                         'Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)']]\n",
            "\n",
            "print(\"\\n\" + \"=\"*80)\n",
            "print(\"PERFORMANCE METRICS SUMMARY\")\n",
            "print(\"=\"*80)\n",
            "print(metrics_df.to_string(index=False))\n",
            "print(\"=\"*80)"
        ]
        
        nb['cells'][code_cell_idx]['source'] = new_code
        print(f"[OK] Updated metrics calculation cell at index {code_cell_idx}")
        break

# Save the modified notebook
output_path = r"g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\experiment_nextLevel5\strategy_comparison_diversify_vs_cluster_HYBRID.ipynb"

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=4, ensure_ascii=False)

print(f"\n[SUCCESS] Successfully created new notebook with Hybrid strategy:")
print(f"    {output_path}")
print(f"\nHybrid Strategy Logic:")
print(f"  - Bull regime (AI signal = True):  Use CLUSTER")
print(f"  - Bear regime (AI signal = False): Use DIVERSIFY")
print(f"  - NEVER goes to cash - always fully invested")
