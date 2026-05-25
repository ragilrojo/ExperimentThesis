
import pandas as pd
import numpy as np
import networkx as nx
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import xgboost as xgb
from sklearn.metrics import fbeta_score
import warnings
warnings.filterwarnings('ignore')

# --- 1. Load Data ---
file_path = 'data_lq45_2023_2025.xlsx'
if not os.path.exists(file_path):
    file_path = 'dataset_2023_2025.xlsx'

data = pd.read_excel(file_path, index_col=0, parse_dates=True)
returns = data.pct_change().dropna()
market_return = returns.mean(axis=1)

print(f"Data Loaded: {len(data)} rows")

# --- 2. AI Training ---
features = pd.DataFrame(index=returns.index)
features['Vol_20'] = market_return.rolling(window=20).std()
features['Mom_20'] = market_return.rolling(window=20).mean()
features['Mom_50'] = market_return.rolling(window=50).mean()

sma5 = market_return.rolling(window=5).mean()
sma20 = market_return.rolling(window=20).mean()
target = (sma5 > sma20).astype(int).shift(-5).reindex(features.index).fillna(0)

X_train = features.loc[features.index.year <= 2024].dropna()
y_train = target.loc[X_train.index]
X_test = features.loc[features.index.year == 2025]

xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

train_probs = xgb_model.predict_proba(X_train)[:, 1]
# Simple threshold optimization
opt_t = 0.54 # Hardcoded from previous run for speed, or re-optimize if needed
print(f"Using Threshold: {opt_t:.2f}")

test_probs = pd.Series(xgb_model.predict_proba(X_test)[:, 1], index=X_test.index)

# --- 3. Core Logic ---
def get_mis_assets(returns_window, correlation_threshold=0.4):
    corr_mat = returns_window.corr()
    G = nx.Graph()
    assets = returns_window.columns
    G.add_nodes_from(assets)
    for i in range(len(assets)):
        for j in range(i+1, len(assets)):
            if abs(corr_mat.iloc[i, j]) > correlation_threshold:
                G.add_edge(assets[i], assets[j])
    return list(nx.approximation.maximum_independent_set(G))

def optimize_markowitz(selected_returns):
    if len(selected_returns.columns) == 0: return {}
    if len(selected_returns.columns) == 1: return {selected_returns.columns[0]: 1.0}
    
    mu = selected_returns.mean() * 252
    sigma = selected_returns.cov() * 252
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

def run_simulation(test_dates, returns, ai_probs, threshold, fee=0.0025, g_lookback=60, rebal_period=5, turnover_buffer=0.10):
    val = 100.0
    history = [val]
    dates = [test_dates[0]]
    
    # Track current holdings (weights)
    current_weights = {} 
    
    risk_status = False
    buffer = 0.05
    days_since_rebal = 999 # Force rebalance on first trigger
    
    for i, date in enumerate(test_dates[:-1]):
        # 1. Regime Detection
        prob = ai_probs.loc[date]
        prev_risk_status = risk_status
        
        if not risk_status and prob > (threshold + buffer): risk_status = True
        elif risk_status and prob < (threshold - buffer): risk_status = False
        
        regime_changed = (risk_status != prev_risk_status)
        
        # 2. Rebalancing Decision
        target_weights = current_weights.copy()
        
        if not risk_status:
            # Bear Market -> Cash
            target_weights = {'CASH': 1.0}
            days_since_rebal = 0 # effectively rebalanced
        else:
            # Bull Market
            if regime_changed or days_since_rebal >= rebal_period:
                # Potential Rebalance
                loc_idx = returns.index.get_loc(date)
                window_rets = returns.iloc[loc_idx-g_lookback:loc_idx]
                selected = get_mis_assets(window_rets)
                optimized_weights = optimize_markowitz(window_rets[selected])
                
                # Check Turnover
                # Align keys for calculation
                all_keys = set(list(optimized_weights.keys()) + list(current_weights.keys()))
                turnover_est = sum(abs(optimized_weights.get(k, 0) - current_weights.get(k, 0)) for k in all_keys)
                
                if regime_changed or turnover_est > turnover_buffer:
                    target_weights = optimized_weights
                    days_since_rebal = 0
                else:
                    # Skip rebalance, keep drifting
                    pass
            days_since_rebal += 1
            
        # 3. Execution & Fees
        # Calculate actual turnover from current holding to target
        all_keys_exec = set(list(target_weights.keys()) + list(current_weights.keys()))
        turnover_actual = sum(abs(target_weights.get(k, 0) - current_weights.get(k, 0)) for k in all_keys_exec)
        
        cost = val * turnover_actual * fee
        val -= cost
        
        # 4. Returns & Drift
        next_date = test_dates[i+1]
        
        if 'CASH' in target_weights and target_weights['CASH'] > 0.99:
            day_ret = 0
            current_weights = {'CASH': 1.0}
        else:
            day_ret = 0
            new_weights_drifted = {}
            for asset, w in target_weights.items():
                if asset in returns.columns: # It might be CASH or actual asset
                    r = returns.loc[next_date, asset]
                    day_ret += w * r
                    new_weights_drifted[asset] = w * (1+r)
                else:
                    # CASH or missing asset
                    new_weights_drifted[asset] = w
            
            # Normalize drifted weights
            total_w = sum(new_weights_drifted.values())
            if total_w > 0:
                current_weights = {k: v/total_w for k, v in new_weights_drifted.items()}
            else:
                current_weights = target_weights

        val *= (1 + day_ret)
        history.append(val)
        dates.append(next_date)
        
    return history[-1]

# --- Run Comparisons ---
test_dates = X_test.index

# 1. Ideal (No Fee)
res_nofee = run_simulation(test_dates, returns, test_probs, opt_t, fee=0.0, rebal_period=1, turnover_buffer=0.0)
print(f"V47 Ideal (No Fee): {res_nofee:.2f}")

# 2. Daily Rebal (With Fee)
res_fee_daily = run_simulation(test_dates, returns, test_probs, opt_t, fee=0.0025, rebal_period=1, turnover_buffer=0.0)
print(f"Naive Daily (With Fee): {res_fee_daily:.2f}")

# 3. Smart Control (With Fee)
res_fee_smart = run_simulation(test_dates, returns, test_probs, opt_t, fee=0.0025, rebal_period=5, turnover_buffer=0.05)
print(f"Smart Control (With Fee): {res_fee_smart:.2f}")
