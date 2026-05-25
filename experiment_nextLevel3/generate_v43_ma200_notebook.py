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
    "# V43 (Option B): LQ45 AI Strategy with Long-Term Trend Filter (MA200)\n",
    "\n",
    "**Objective**: Reduce 'Whipsaw' (frequent switching) by replacing the sensitive MA50 with a robust **200-Day Moving Average (MA200)**.\n",
    "\n",
    "### Modification (MA200 Logic):\n",
    "The 200-day moving average is the standard for defining long-term bull/bear markets. It is significantly slower and smoother than the MA50.\n",
    "\n",
    "**New Logic:**\n",
    "- **Risk-On Condition (Stay Invested)**:\n",
    "  - IF (AI Prob >= 0.60) **OR** (Market Price > MA200).\n",
    "  - *Interpretation*: We trust the long-term trend. If the market is above MA200, we ignore short-term AI bearishness.\n",
    "- **Risk-Off Condition (Cash)**:\n",
    "  - IF (AI Prob < 0.60) **AND** (Market Price < MA200).\n",
    "  - *Interpretation*: We only exit to Cash if the AI is scared AND the long-term trend is broken (Bear Market).\n",
    "\n",
    "**Rationale**:\n",
    "- MA200 filters out almost all short-term noise and corrections, only exiting during major structural declines.\n",
    "\n",
    "### Comparisons:\n",
    "1. **AI Hybrid V43 (MA200)**.\n",
    "2. **Markowitz Static (Baseline)**.\n",
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

# Cell 3: Feature Engineering & AI Model
notebook["cells"].append(create_code_cell([
    "# --- 2. Feature Engineering & AI Model ---\n",
    "\n",
    "features = pd.DataFrame(index=returns.index)\n",
    "features['Vol_20'] = market_return.rolling(window=20).std()\n",
    "ma5 = market_price.rolling(window=5).mean()\n",
    "ma20 = market_price.rolling(window=20).mean()\n",
    "ma50 = market_price.rolling(window=50).mean()\n",
    "# Add MA200 for V43 Option B\n",
    "ma200 = market_price.rolling(window=200).mean()\n",
    "\n",
    "features['Dist_MA20'] = market_price / ma20\n",
    "features['Dist_MA50'] = market_price / ma50\n",
    "\n",
    "# Calculate Trend Signal: Price > MA200\n",
    "trend_bullish_ma200 = (market_price > ma200).astype(int)\n",
    "\n",
    "# TARGET: MA Slope Direction\n",
    "target = (ma5.shift(-1) > ma5).astype(int)\n",
    "\n",
    "features = features.dropna()\n",
    "target = target.reindex(features.index).fillna(0)\n",
    "trend_bullish_ma200 = trend_bullish_ma200.reindex(features.index).fillna(0)\n",
    "\n",
    "train_mask = (features.index.year <= 2024)\n",
    "test_mask = (features.index.year == 2025)\n",
    "\n",
    "X_train, y_train = features.loc[train_mask], target.loc[train_mask]\n",
    "X_test = features.loc[test_mask]\n",
    "\n",
    "xgb_model = xgb.XGBClassifier(n_estimators=120, learning_rate=0.04, max_depth=6, random_state=42, eval_metric='logloss')\n",
    "xgb_model.fit(X_train, y_train)\n",
    "\n",
    "opt_t = 0.60\n",
    "print(f\"[Strategy] Using Base Threshold: {opt_t:.2f} with MA200 Long-Term Trend Override\")\n"
]))

# Cell 4: Functions
notebook["cells"].append(create_code_cell([
    "# --- 3. Simulation Logic (V43 MA200) ---\n",
    "\n",
    "def optimize_markowitz(selected_returns):\n",
    "    if len(selected_returns.columns) == 0: return {}\n",
    "    if len(selected_returns.columns) == 1: return {selected_returns.columns[0]: 1.0}\n",
    "    mu, sigma = selected_returns.mean() * 252, selected_returns.cov() * 252\n",
    "    sigma += np.eye(len(sigma)) * 1e-4 \n",
    "    try:\n",
    "        res = minimize(lambda w: -(np.sum(w*mu)/(np.sqrt(np.dot(w.T, np.dot(sigma, w))) + 1e-6)), \n",
    "                       [1./len(mu)]*len(mu), method='SLSQP', \n",
    "                       bounds=tuple((0, 1) for _ in range(len(mu))), \n",
    "                       constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}))\n",
    "        return dict(zip(selected_returns.columns, res.x)) if res.success else {}\n",
    "    except:\n",
    "        return {}\n",
    "\n",
    "def run_simulation_ma200(strategy_type, test_dates, returns, ai_probs=None, trend_signal=None, threshold=0.5, fee=0.0025):\n",
    "    val, history, dates = 100.0, [100.0], [test_dates[0]]\n",
    "    prev_weights = {}\n",
    "    static_weights = None\n",
    "    state_history = []\n",
    "    \n",
    "    for i, date in enumerate(test_dates[:-1]):\n",
    "        loc_idx = returns.index.get_loc(date)\n",
    "        window_rets = returns.iloc[loc_idx-30:loc_idx]\n",
    "        \n",
    "        current_state = \"Neutral\"\n",
    "        weights = {}\n",
    "\n",
    "        if strategy_type == 'AI Hybrid V43 (MA200)':\n",
    "            prob = ai_probs.loc[date]\n",
    "            # Trend Signal is Price > MA200\n",
    "            is_trend_up = trend_signal.loc[date] == 1\n",
    "            \n",
    "            # HYBRID LOGIC V43:\n",
    "            # Exit ONLY if AI is scared (< T) AND Long-Term Trend is Broken (< MA200).\n",
    "            is_risk_off = (prob < threshold) and (not is_trend_up)\n",
    "            \n",
    "            if is_risk_off: \n",
    "                weights = {'CASH': 1.0}\n",
    "                current_state = \"Cash (Bear Market)\"\n",
    "            else:\n",
    "                # Risk On\n",
    "                valid_assets = window_rets.dropna(axis=1, how='any').columns\n",
    "                weights = optimize_markowitz(window_rets[valid_assets])\n",
    "                current_state = \"Invested\"\n",
    "                \n",
    "        elif strategy_type == 'Markowitz Static':\n",
    "            if static_weights is None:\n",
    "                valid_assets = window_rets.dropna(axis=1, how='any').columns\n",
    "                static_weights = optimize_markowitz(window_rets[valid_assets])\n",
    "            weights = static_weights\n",
    "            current_state = \"Static\"\n",
    "        \n",
    "        state_history.append(current_state)\n",
    "        \n",
    "        # Transaction Costs\n",
    "        all_assets = set(list(weights.keys()) + list(prev_weights.keys()))\n",
    "        turnover = sum(abs(weights.get(a, 0) - prev_weights.get(a, 0)) for a in all_assets)\n",
    "        val -= (val * turnover * fee)\n",
    "        \n",
    "        next_date = test_dates[i+1]\n",
    "        day_ret = 0 if 'CASH' in weights else sum(w * returns.loc[next_date, a] for a, w in weights.items())\n",
    "        val *= (1 + day_ret)\n",
    "        history.append(val); dates.append(next_date); prev_weights = weights\n",
    "        \n",
    "    return pd.DataFrame({'Portfolio_Value': history, 'State': state_history + [state_history[-1]]}, index=dates)\n"
]))

# Cell 5: Execution
notebook["cells"].append(create_code_cell([
    "# --- 4. Models & Benchmark Execution ---\n",
    "ai_probs = pd.Series(xgb_model.predict_proba(X_test)[:, 1], index=X_test.index)\n",
    "test_trend_ma200 = trend_bullish_ma200.loc[X_test.index]\n",
    "\n",
    "print(\"Running AI Hybrid V43 (MA200)...\")\n",
    "hybrid_res_v43_ma200 = run_simulation_ma200('AI Hybrid V43 (MA200)', X_test.index, returns, ai_probs, test_trend_ma200, threshold=opt_t)\n",
    "\n",
    "print(\"Running Markowitz Static...\")\n",
    "static_res = run_simulation_ma200('Markowitz Static', X_test.index, returns)\n",
    "\n",
    "print(\"Calculating IHSG Benchmark...\")\n",
    "ihsg_ret = market_return.loc[X_test.index]\n",
    "ihsg_price = (1 + ihsg_ret).cumprod() * 100\n"
]))

# Cell 6: Visualization
notebook["cells"].append(create_code_cell([
    "# --- Visualization: Performance Comparison ---\n",
    "plt.figure(figsize=(12, 6))\n",
    "plt.plot(hybrid_res_v43_ma200['Portfolio_Value'], label=f'AI Hybrid V43 (MA200)', color='purple', linewidth=2)\n",
    "plt.plot(static_res['Portfolio_Value'], label='Markowitz Static', color='#1f77b4', linestyle='--', alpha=0.8)\n",
    "plt.plot(ihsg_price, label='IHSG Benchmark', color='#7f7f7f', linestyle=':', linewidth=2)\n",
    "\n",
    "# Highlight Regime Switches\n",
    "regime_switches = hybrid_res_v43_ma200[hybrid_res_v43_ma200['State'].str.contains('Cash')]\n",
    "plt.scatter(regime_switches.index, regime_switches['Portfolio_Value'], color='red', s=10, label='Cash Days', zorder=5)\n",
    "\n",
    "plt.title(f'V43 Option B: MA200 Filter vs Static (2025)\\nRobust Long-Term Trend Filter')\n",
    "plt.ylabel('Cumulative Wealth (Base 100)')\n",
    "plt.xlabel('Date')\n",
    "plt.legend(loc='upper left')\n",
    "plt.grid(True, which='both', linestyle='--', linewidth=0.5)\n",
    "plt.savefig('v43_ma200_comparison.png', dpi=300, bbox_inches='tight')\n",
    "plt.show()\n",
    "\n",
    "# Calculate switches count\n",
    "switches_count = (hybrid_res_v43_ma200['State'] != hybrid_res_v43_ma200['State'].shift(1)).sum()\n",
    "print(f\"Total Regime Switches in V43 (MA200): {switches_count}\")\n"
]))

# Save
output_file = r"g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\experiment_nextLevel3\enhanced_strategy_v43_ma200.ipynb"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=4)

print(f"Created notebook: {output_file}")
