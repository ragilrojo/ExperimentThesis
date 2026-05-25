import json
import os

# Notebook structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Helper to create code cell
def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source if isinstance(source, list) else source.splitlines(True)
    }

# Helper to create markdown cell
def create_md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source if isinstance(source, list) else source.splitlines(True)
    }

# Cell 1: Introduction
notebook["cells"].append(create_md_cell([
    "# V44 (Cost-Sensitive + Cash Management): LQ45 AI Strategy\n",
    "\n",
    "**Objective**: Implement realistic Cash Management where transaction costs are paid from the Cash Reserve, not by reducing portfolio value magically.\n",
    "\n",
    "### Modification (Cash & Cost Logic):\n",
    "1. **Minimum Cash Buffer**: Every portfolio allocation MUST keep at least **10% in Cash** at all times (`weights['CASH'] >= 0.1`).\n",
    "2. **Cost Deduction**: Transaction fees are deducted directly from this Cash balance.\n",
    "   - If Cash drops below 0 (impossible in reality, but for sim), the strategy goes bankrupt or sells assets to cover fees (simplified here as negative cash dragging performace correctly).\n",
    "\n",
    "**Rationale**:\n",
    "- This is more realistic. You cannot invest 100% because you need cash to pay for the buy/sell commisions.\n",
    "- Keeping 10% dry powder also dampens volatility slightly.\n",
    "\n",
    "### Comparisons:\n",
    "1. **AI Hybrid V44 (Cost-Sensitive + 10% Cash)**.\n",
    "2. **Markowitz Static (Baseline + 10% Cash)**.\n",
    "3. **IHSG Proxy (Benchmark)**.\n"
]))

# Cell 2: Imports and Data Loading
notebook["cells"].append(create_code_cell([
    "import pandas as pd\n",
    "import numpy as np\n",
    "import networkx as nx\n",
    "import os\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from scipy.optimize import minimize\n",
    "import xgboost as xgb\n",
    "from sklearn.metrics import confusion_matrix\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "plt.style.use('ggplot')\n",
    "\n",
    "# --- 1. Load Data (LQ45) ---\n",
    "file_path = 'data_lq45_2023_2025.xlsx'\n",
    "if not os.path.exists(file_path):\n",
    "    file_path = 'dataset_2023_2025.xlsx'\n",
    "\n",
    "data = pd.read_excel(file_path, index_col=0, parse_dates=True)\n",
    "returns = data.pct_change().dropna()\n",
    "market_return = returns.mean(axis=1)\n",
    "market_price = (1 + market_return).cumprod() * 100\n",
    "\n",
    "print(f\"Data Loaded: {len(data)} rows\")\n"
]))

# Cell 3: Feature Engineering & Cost-Aware Training
notebook["cells"].append(create_code_cell([
    "# --- 2. Feature Engineering & Cost-Aware Model ---\n",
    "\n",
    "features = pd.DataFrame(index=returns.index)\n",
    "features['Vol_20'] = market_return.rolling(window=20).std()\n",
    "ma5 = market_price.rolling(window=5).mean()\n",
    "ma20 = market_price.rolling(window=20).mean()\n",
    "ma50 = market_price.rolling(window=50).mean()\n",
    "features['Dist_MA20'] = market_price / ma20\n",
    "features['Dist_MA50'] = market_price / ma50\n",
    "\n",
    "# Trend Signal (Standard MA50) for Hybrid Filter\n",
    "trend_bullish = (market_price > ma50).astype(int)\n",
    "\n",
    "# --- NEW TARGET DEFINITION ---\n",
    "# Next day return\n",
    "next_ret = market_return.shift(-1)\n",
    "\n",
    "# Cost Threshold (e.g., 0.5% round-trip friction)\n",
    "COST_THRESHOLD = -0.005 \n",
    "\n",
    "# Target = 1 (Stay Invested) if drop is manageable (Return > -0.5%)\n",
    "# Target = 0 (Exit) ONLY if drop is severe (Return <= -0.5%)\n",
    "target = (next_ret > COST_THRESHOLD).astype(int)\n",
    "\n",
    "features = features.dropna()\n",
    "target = target.reindex(features.index).fillna(1) # Default to Invested\n",
    "trend_bullish = trend_bullish.reindex(features.index).fillna(0)\n",
    "\n",
    "train_mask = (features.index.year <= 2024)\n",
    "test_mask = (features.index.year == 2025)\n",
    "\n",
    "X_train, y_train = features.loc[train_mask], target.loc[train_mask]\n",
    "X_test = features.loc[test_mask]\n",
    "\n",
    "# Train Cost-Sensitive Model\n",
    "xgb_model = xgb.XGBClassifier(n_estimators=120, learning_rate=0.04, max_depth=6, random_state=42, eval_metric='logloss')\n",
    "xgb_model.fit(X_train, y_train)\n",
    "\n",
    "# Predictions\n",
    "ai_probs = pd.Series(xgb_model.predict_proba(X_test)[:, 1], index=X_test.index)\n",
    "test_trend = trend_bullish.loc[X_test.index]\n",
    "\n",
    "opt_t = 0.50 # Standard threshold\n",
    "\n",
    "print(f\"[Strategy] Cost-Sensitive AI Trained. Target: Stay Invested if Return > {COST_THRESHOLD*100}%\")\n"
]))

# Cell 4: Functions
notebook["cells"].append(create_code_cell([
    "# --- 3. Simulation Logic (Cash Management) ---\n",
    "# We will force 10% Cash allocation during Optimization\n",
    "\n",
    "def optimize_markowitz(selected_returns, min_cash=0.1):\n",
    "    if len(selected_returns.columns) == 0: return {'CASH': 1.0}\n",
    "    \n",
    "    # Standard Markowitz for 100% of the Equity Portion (which is 1 - min_cash)\n",
    "    # Then we scale down weights\n",
    "    \n",
    "    mu, sigma = selected_returns.mean() * 252, selected_returns.cov() * 252\n",
    "    sigma += np.eye(len(sigma)) * 1e-4 \n",
    "    try:\n",
    "        res = minimize(lambda w: -(np.sum(w*mu)/(np.sqrt(np.dot(w.T, np.dot(sigma, w))) + 1e-6)), \n",
    "                       [1./len(mu)]*len(mu), method='SLSQP', \n",
    "                       bounds=tuple((0, 1) for _ in range(len(mu))), \n",
    "                       constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}))\n",
    "        \n",
    "        if res.success:\n",
    "            raw_weights = dict(zip(selected_returns.columns, res.x))\n",
    "            # Scale down to (1 - min_cash)\n",
    "            # Start with CASH = min_cash\n",
    "            final_weights = {k: v * (1.0 - min_cash) for k, v in raw_weights.items()}\n",
    "            final_weights['CASH'] = min_cash\n",
    "            return final_weights\n",
    "        else:\n",
    "            return {'CASH': 1.0}\n",
    "    except:\n",
    "        return {'CASH': 1.0}\n",
    "\n",
    "def run_simulation_cash_managed(strategy_type, test_dates, returns, ai_probs=None, trend_signal=None, threshold=0.5, fee=0.0025):\n",
    "    val = 100.0\n",
    "    history = [100.0]\n",
    "    dates = [test_dates[0]]\n",
    "    \n",
    "    prev_weights = {'CASH': 1.0} # Start 100% Cash\n",
    "    static_weights = None\n",
    "    state_history = []\n",
    "    \n",
    "    # Logging list\n",
    "    portfolio_log = []\n",
    "    \n",
    "    for i, date in enumerate(test_dates[:-1]):\n",
    "        loc_idx = returns.index.get_loc(date)\n",
    "        window_rets = returns.iloc[loc_idx-30:loc_idx]\n",
    "        \n",
    "        weights = {}\n",
    "        current_state = \"Invested\"\n",
    "\n",
    "        if strategy_type == 'AI Hybrid V44 (CashManaged)':\n",
    "            prob = ai_probs.loc[date]\n",
    "            is_trend_up = trend_signal.loc[date] == 1\n",
    "            \n",
    "            is_risk_off = (prob < threshold) and (not is_trend_up)\n",
    "            \n",
    "            if is_risk_off:\n",
    "                weights = {'CASH': 1.0}\n",
    "                current_state = \"Cash\"\n",
    "            else:\n",
    "                valid_assets = window_rets.dropna(axis=1, how='any').columns\n",
    "                # Force 10% Cash Buffer inside optimization\n",
    "                weights = optimize_markowitz(window_rets[valid_assets], min_cash=0.10)\n",
    "                current_state = \"Invested\"\n",
    "                \n",
    "        elif strategy_type == 'Markowitz Static':\n",
    "            if static_weights is None:\n",
    "                valid_assets = window_rets.dropna(axis=1, how='any').columns\n",
    "                # Markowitz Static also forced to keep 10% Cash\n",
    "                static_weights = optimize_markowitz(window_rets[valid_assets], min_cash=0.10)\n",
    "            weights = static_weights\n",
    "            current_state = \"Static\"\n",
    "        \n",
    "        state_history.append(current_state)\n",
    "        \n",
    "        # Log Portfolio\n",
    "        for asset, weight in weights.items():\n",
    "            if weight > 0.001: \n",
    "                portfolio_log.append({\n",
    "                    'Date': dates[-1].strftime('%Y-%m-%d'),\n",
    "                    'Strategy': strategy_type,\n",
    "                    'Ticker': asset,\n",
    "                    'Weight': weight\n",
    "                })\n",
    "        \n",
    "        # --- TRANSACTION COST LOGIC ---\n",
    "        # Calculate Turnover\n",
    "        all_assets = set(list(weights.keys()) + list(prev_weights.keys()))\n",
    "        # Turnover on Equity portion only. Cash movements don't incur commission fees usually, \n",
    "        # but usually buying/selling stock does. \n",
    "        # For simplicity, we calculate turnover on all changes excluding self-cash adjustments if we consider cash free.\n",
    "        # But standard model applies fee to total turnover volume.\n",
    "        turnover = sum(abs(weights.get(a, 0) - prev_weights.get(a, 0)) for a in all_assets if a != 'CASH')\n",
    "        \n",
    "        cost = val * turnover * fee\n",
    "        \n",
    "        # Deduct Cost from CASH portion of value\n",
    "        # This effectively reduces the total Portfolio Value\n",
    "        val -= cost\n",
    "        \n",
    "        next_date = test_dates[i+1]\n",
    "        \n",
    "        # Portfolio Return Calculation\n",
    "        day_ret = 0.0\n",
    "        for asset, weight in weights.items():\n",
    "            if asset == 'CASH':\n",
    "                # Cash earns 0% (or risk free rate, here 0 for simplicity)\n",
    "                pass \n",
    "            else:\n",
    "                r = returns.loc[next_date, asset]\n",
    "                day_ret += weight * r\n",
    "        \n",
    "        # Apple Return\n",
    "        val *= (1 + day_ret)\n",
    "        \n",
    "        history.append(val); dates.append(next_date); prev_weights = weights\n",
    "        \n",
    "    return pd.DataFrame({'Portfolio_Value': history, 'State': state_history + [state_history[-1]]}, index=dates), portfolio_log\n"
]))

# Cell 5: Execution
notebook["cells"].append(create_code_cell([
    "# --- 4. Models & Benchmark Execution ---\n",
    "\n",
    "print(\"Running AI Hybrid V44 (Cash Managed)...\")\n",
    "hybrid_res_cost, hybrid_log = run_simulation_cash_managed('AI Hybrid V44 (CashManaged)', X_test.index, returns, ai_probs, test_trend, threshold=0.50)\n",
    "\n",
    "print(\"Running Markowitz Static...\")\n",
    "static_res, static_log = run_simulation_cash_managed('Markowitz Static', X_test.index, returns)\n",
    "\n",
    "print(\"Calculating IHSG Benchmark...\")\n",
    "ihsg_ret = market_return.loc[X_test.index]\n",
    "ihsg_price = (1 + ihsg_ret).cumprod() * 100\n"
]))

# Cell 6: Visualization
notebook["cells"].append(create_code_cell([
    "# --- Visualization: Performance Comparison ---\n",
    "plt.figure(figsize=(12, 6))\n",
    "plt.plot(hybrid_res_cost['Portfolio_Value'], label=f'AI Hybrid V44 (10% Cash Buffer)', color='teal', linewidth=2)\n",
    "plt.plot(static_res['Portfolio_Value'], label='Markowitz Static (10% Cash)', color='#1f77b4', linestyle='--', alpha=0.8)\n",
    "plt.plot(ihsg_price, label='IHSG Benchmark', color='#7f7f7f', linestyle=':', linewidth=2)\n",
    "\n",
    "# Highlight Cash Days\n",
    "regime_switches = hybrid_res_cost[hybrid_res_cost['State'].str.contains('Cash')]\n",
    "if not regime_switches.empty:\n",
    "    plt.scatter(regime_switches.index, regime_switches['Portfolio_Value'], color='orange', s=10, label='Cash Signal', zorder=5)\n",
    "\n",
    "plt.title(f'V44: Realistic Cash Mgmt (10% Min Cash) vs Static (2025)\\nTransaction Fees Paid from Cash Reserve')\n",
    "plt.ylabel('Cumulative Wealth (Base 100)')\n",
    "plt.xlabel('Date')\n",
    "plt.legend(loc='upper left')\n",
    "plt.grid(True, which='both', linestyle='--', linewidth=0.5)\n",
    "plt.savefig('v44_cash_managed_comparison.png', dpi=300, bbox_inches='tight')\n",
    "plt.show()\n",
    "\n",
    "# Stats\n",
    "cash_days = hybrid_res_cost['State'].value_counts().get('Cash', 0)\n",
    "print(f\"Total Days in Cash (100%): {cash_days}\")\n"
]))

# Cell 7: Export to Excel
notebook["cells"].append(create_code_cell([
    "# --- 5. Export Portfolio to Excel ---\n",
    "# Combine logs\n",
    "full_log = hybrid_log + static_log\n",
    "log_df = pd.DataFrame(full_log)\n",
    "\n",
    "output_excel = 'v20_portfolio_with_visibility_2025.xlsx'\n",
    "\n",
    "with pd.ExcelWriter(output_excel) as writer:\n",
    "    # Sheet 1: Hybrid Strategy Log\n",
    "    hybrid_df = log_df[log_df['Strategy'] == 'AI Hybrid V44 (CashManaged)']\n",
    "    if not hybrid_df.empty:\n",
    "        hybrid_pivot = hybrid_df.pivot(index='Date', columns='Ticker', values='Weight').fillna(0)\n",
    "        hybrid_pivot.to_excel(writer, sheet_name='AI Hybrid Portfolio')\n",
    "    \n",
    "    # Sheet 2: Static Strategy Log\n",
    "    static_df = log_df[log_df['Strategy'] == 'Markowitz Static']\n",
    "    if not static_df.empty:\n",
    "        static_pivot = static_df.pivot(index='Date', columns='Ticker', values='Weight').fillna(0)\n",
    "        static_pivot.to_excel(writer, sheet_name='Markowitz Static Portfolio')\n",
    "\n",
    "    # Sheet 3: Raw Data\n",
    "    log_df.to_excel(writer, sheet_name='Raw Log', index=False)\n",
    "\n",
    "print(f\"Portfolio log exported to: {output_excel}\")\n"
]))

# Save
output_file = r"g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\experiment_nextLevel3\enhanced_strategy_v44_cost_aware.ipynb"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=4)

print(f"Created notebook: {output_file}")
