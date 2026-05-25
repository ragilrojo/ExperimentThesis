
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import warnings
from scipy.optimize import minimize

warnings.filterwarnings('ignore')
plt.style.use('ggplot')

# --- 1. Load Data ---
file_path = 'data_lq45_2023_2025.xlsx'
if not os.path.exists(file_path):
    file_path = 'dataset_2023_2025.xlsx'

data = pd.read_excel(file_path, index_col=0, parse_dates=True)
returns = data.pct_change().dropna()
market_return = returns.mean(axis=1)

# --- 2. MIS Strategies ---

def get_mis_random(returns_window, corr_thresh=0.4, seed=None):
    if seed:
        np.random.seed(seed)
    corr_mat = returns_window.corr()
    G = nx.Graph()
    
    # Randomize order
    assets = list(returns_window.columns)
    np.random.shuffle(assets)
    
    G.add_nodes_from(assets)
    for i in range(len(assets)):
        for j in range(i+1, len(assets)):
            if abs(corr_mat.loc[assets[i], assets[j]]) > corr_thresh:
                G.add_edge(assets[i], assets[j])
    return list(nx.approximation.maximum_independent_set(G))

def get_mis_alphabetical(returns_window, corr_thresh=0.4):
    corr_mat = returns_window.corr()
    G = nx.Graph()
    
    # Sort Alphabetical (V54 Logic)
    assets = sorted(returns_window.columns)
    
    G.add_nodes_from(assets)
    for i in range(len(assets)):
        for j in range(i+1, len(assets)):
            if abs(corr_mat.loc[assets[i], assets[j]]) > corr_thresh:
                G.add_edge(assets[i], assets[j])
    return list(nx.approximation.maximum_independent_set(G))

def get_mis_momentum(returns_window, corr_thresh=0.4):
    corr_mat = returns_window.corr()
    G = nx.Graph()
    
    # Sort by Momentum (Mean Return over window) Descending
    # Higher momentum handled first -> likely kept in greedy matching?
    # NetworkX approximation is heuristic. To FORCE priority, we simply iterate and pick.
    # But sticking to nx.approximation for consistency, just feeding nodes in order.
    # Note: nx.approximation.maximum_independent_set behavior depends on node order in internal representation
    
    momentum = returns_window.mean()
    assets = list(momentum.sort_values(ascending=False).index)
    
    G.add_nodes_from(assets)
    for i in range(len(assets)):
        for j in range(i+1, len(assets)):
            if abs(corr_mat.loc[assets[i], assets[j]]) > corr_thresh:
                G.add_edge(assets[i], assets[j])
                
    return list(nx.approximation.maximum_independent_set(G))

# --- 3. Simple Backtest (No AI, Just Risk ON) ---
# Testing PURE selection power

def optimize_markowitz(selected_returns):
    if len(selected_returns.columns) == 0: return {}
    if len(selected_returns.columns) == 1: return {selected_returns.columns[0]: 1.0}
    
    mu, sigma = selected_returns.mean() * 252, selected_returns.cov() * 252
    num_assets = len(mu)
    
    def objective(w):
        ret = np.sum(w * mu)
        risk = np.sqrt(np.dot(w.T, np.dot(sigma, w)))
        if risk < 0.0001: return 0
        return -(ret / risk)
        
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = [1./num_assets] * num_assets
    
    res = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return dict(zip(selected_returns.columns, res.x)) if res.success else {}

def run_test(selector_func, name):
    # Test on 2025
    test_data = returns.loc['2025']
    val = 100.0
    history = [val]
    current_weights = {}
    
    # Rebalance every 20 days
    rebal_period = 20
    days_since = 999
    
    for i in range(len(test_data)):
        date = test_data.index[i]
        
        if days_since >= rebal_period:
            # Lookback 60 days
            # We need data from `returns` (full dataset)
            full_idx = returns.index.get_loc(date)
            window = returns.iloc[full_idx-60:full_idx]
            
            selected_assets = selector_func(window)
            current_weights = optimize_markowitz(window[selected_assets])
            days_since = 0
        
        days_since += 1
        
        # Calculate daily return
        day_ret = 0
        for asset, w in current_weights.items():
            if asset in test_data.columns:
                day_ret += w * test_data.iloc[i][asset]
        
        val *= (1 + day_ret)
        history.append(val)
        
    return history[-1]

# --- 4. Execution ---
print("Running Comparisons (2025 Risk-ON)...")

# 1. V54 Alphabetical
res_alpha = run_test(get_mis_alphabetical, "Alphabetical (V54)")
print(f"V54 (Alphabetical): {res_alpha:.2f}")

# 2. V53 Random (try 5 seeds)
random_results = []
for s in [10, 20, 30, 42, 50]:
    # Lambda to pass seed
    r = run_test(lambda w: get_mis_random(w, seed=s), f"Random {s}")
    random_results.append(r)
print(f"V53 (Random Avg): {np.mean(random_results):.2f} (Min: {min(random_results):.2f}, Max: {max(random_results):.2f})")

# 3. V55 Momentum Sorted
res_mom = run_test(get_mis_momentum, "Momentum Sorted (V55)")
print(f"V55 (Momentum Sorted): {res_mom:.2f}")

# Summary
print("\n--- CONCLUSION ---")
best = max([res_alpha, max(random_results), res_mom])
if res_mom >= res_alpha and res_mom >= max(random_results) * 0.95:
    print("MOMENTUM SORT works well and is stable!")
elif res_alpha > res_mom:
    print("Alphabetical somehow worked better. Investigate why 'A' stocks are good.")
else:
    print("Randomness wins significantly. Selection logic needs diversity.")
