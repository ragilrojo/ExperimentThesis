import pandas as pd
import numpy as np
import networkx as nx
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import xgboost as xgb
from sklearn.metrics import fbeta_score
import random
import warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')

# --- LOCK RANDOMNESS ---
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# --- 1. Load Data ---
file_path = 'data_lq45_2023_2025.xlsx'
if not os.path.exists(file_path):
    file_path = 'dataset_2023_2025.xlsx'
if not os.path.exists(file_path):
    # Fallback absolute path
    file_path = r'g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\experiment_nextLevel3\data_lq45_2023_2025.xlsx'

print(f"Loading data from: {file_path}")
try:
    data = pd.read_excel(file_path, index_col=0, parse_dates=True)
except FileNotFoundError:
    print("Error: Dataset file not found. Please ensure 'data_lq45_2023_2025.xlsx' is in the directory.")
    exit()

returns = data.pct_change().dropna()
market_return = returns.mean(axis=1)
market_index = (1 + market_return).cumprod() * 100

print(f"Data Loaded: {len(data)} rows")

# --- 2. AI Training (Standard V47) ---
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

xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=SEED, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

train_probs = xgb_model.predict_proba(X_train)[:, 1]
opt_t = np.linspace(0.3, 0.7, 41)[np.argmax([fbeta_score(y_train, (train_probs >= t).astype(int), beta=0.5) for t in np.linspace(0.3, 0.7, 41)])]
test_probs = pd.Series(xgb_model.predict_proba(X_test)[:, 1], index=X_test.index)

print(f"Optimized Threshold: {opt_t:.2f}")

# --- 3. Core Logic: Risk Scaling Simulation ---

def get_mis_assets_stable(returns_window, correlation_threshold=0.4):
    corr_mat = returns_window.corr()
    G = nx.Graph()
    assets = sorted(returns_window.columns)
    G.add_nodes_from(assets)
    for i in range(len(assets)):
        for j in range(i+1, len(assets)):
            if abs(corr_mat.loc[assets[i], assets[j]]) > correlation_threshold:
                G.add_edge(assets[i], assets[j])
    return list(nx.approximation.maximum_independent_set(G))

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

def run_simulation_risk_scaling(test_dates, returns, ai_probs, threshold, market_idx, 
                                min_equity=0.5, # Floor: Minimum 50% Equity
                                override_ma=50, 
                                fee=0.0025, 
                                g_lookback=60, 
                                rebal_period=20, 
                                turnover_buffer=0.05):
    val = 100.0
    history = [val]
    dates = [test_dates[0]]
    
    current_weights = {} 
    risk_status = False # False = BEAR Mode (Risk OFF logic), True = BULL Mode
    
    days_since_rebal = 999
    
    # MA for Trend Override
    market_ma = None
    if override_ma:
        market_ma = market_idx.rolling(window=override_ma).mean()
    
    for i, date in enumerate(test_dates[:-1]):
        prev_risk_status = risk_status
        prob = ai_probs.loc[date]
        
        # 1. Base AI Signal
        ai_signal = False
        buffer = 0.05
        if not risk_status and prob > (threshold + buffer): ai_signal = True
        elif risk_status and prob < (threshold - buffer): ai_signal = False
        else: ai_signal = risk_status 
        
        # 2. Trend Override Logic
        final_signal = ai_signal
        try:
            current_price = market_idx.loc[date]
            if market_ma is not None:
                ma_price = market_ma.loc[date]
                if not pd.isna(ma_price):
                    is_uptrend = (current_price > ma_price)
                    if ai_signal == False and is_uptrend:
                        final_signal = True # Force BULL Mode
        except: pass
        
        risk_status = final_signal
        regime_changed = (risk_status != prev_risk_status)
        
        # --- 3. Portfolio Management (Risk Scaling) ---
        target_weights = current_weights.copy()
        
        # Determine Equity Allocation based on Status and Floor
        target_equity_pct = 1.0 if risk_status else min_equity
        
        need_reshuffle = (regime_changed or days_since_rebal >= rebal_period)
        
        # Note: If regime changed to BEAR (risk_status=False), target_equity < 1.0.
        # We need to re-shuffle to ensure we meet this target if we came from 1.0.
        # However, if we are ALREADY in BEAR (risk_status=False) and days < period, 
        # do we re-shuffle? No, we just hold.
        # But wait, holding means prices drift. Equity % might drift.
        # For simplicity, we rebalance only on period or regime change.
        
        if need_reshuffle:
            loc_idx = returns.index.get_loc(date)
            window_rets = returns.iloc[loc_idx-g_lookback:loc_idx]
            selected = get_mis_assets_stable(window_rets)
            raw_stock_weights = optimize_markowitz(window_rets[selected])
            
            # Scale weights
            scaled_weights = {k: v * target_equity_pct for k, v in raw_stock_weights.items()}
            # Assign remainder to CASH
            # If target_equity_pct is 1.0, CASH is 0.
            current_total = sum(scaled_weights.values()) # Should be approx target_equity_pct
            cash_weight = 1.0 - current_total
            if cash_weight > 0.001:
                scaled_weights['CASH'] = cash_weight
            
            # Fee Check
            all_keys = set(list(scaled_weights.keys()) + list(current_weights.keys()))
            turnover_est = sum(abs(scaled_weights.get(k, 0) - current_weights.get(k, 0)) for k in all_keys)
            
            if regime_changed or turnover_est > turnover_buffer:
                target_weights = scaled_weights
                days_since_rebal = 0
            
        days_since_rebal += 1

        # --- 4. Execution ---
        all_keys_exec = set(list(target_weights.keys()) + list(current_weights.keys()))
        turnover_actual = sum(abs(target_weights.get(k, 0) - current_weights.get(k, 0)) for k in all_keys_exec)
        
        if turnover_actual > 0.001:
            cost = val * turnover_actual * fee
            val -= cost
        
        next_date = test_dates[i+1]
        day_ret = 0
        new_weights_drifted = {}
        
        for asset, w in target_weights.items():
            if asset == 'CASH':
                day_ret += 0
                new_weights_drifted['CASH'] = w
            elif asset in returns.columns:
                r = returns.loc[next_date, asset]
                r_asset = r if not pd.isna(r) else 0
                day_ret += w * r_asset
                new_weights_drifted[asset] = w * (1 + r_asset)
            else:
                new_weights_drifted[asset] = w
        
        # Normalize Drifted Weights for next loop
        total_w = sum(new_weights_drifted.values()) if new_weights_drifted else 0
        if total_w > 0:
            new_weights_drifted = {k: v/total_w for k, v in new_weights_drifted.items()}

        val *= (1 + day_ret)
        history.append(val)
        dates.append(next_date)
        current_weights = new_weights_drifted
    
    return pd.DataFrame({'Portfolio_Value': history}, index=dates)

# --- 4. Run Scenarios ---
test_dates = X_test.index
fee_rate = 0.0025

results = {}

print("Running Static (100% Equity Always)...")
static_probs = pd.Series(1.0, index=test_probs.index)
res_static = run_simulation_risk_scaling(test_dates, returns, static_probs, 0.5, market_index, 
                                         min_equity=1.0, override_ma=None, fee=fee_rate)
results['Static Markowitz'] = res_static['Portfolio_Value']

print("Running AI V54 (Floor 0% + SMA50)...")
res_v54 = run_simulation_risk_scaling(test_dates, returns, test_probs, opt_t, market_index, 
                                      min_equity=0.0, override_ma=50, fee=fee_rate)
results['AI V54 (Floor 0%)'] = res_v54['Portfolio_Value']

print("Running AI V55 (Floor 30% + SMA50)...")
res_v55_30 = run_simulation_risk_scaling(test_dates, returns, test_probs, opt_t, market_index, 
                                         min_equity=0.3, override_ma=50, fee=fee_rate)
results['AI V55 (Floor 30%)'] = res_v55_30['Portfolio_Value']

print("Running AI V55 (Floor 50% + SMA50)...")
res_v55_50 = run_simulation_risk_scaling(test_dates, returns, test_probs, opt_t, market_index, 
                                         min_equity=0.5, override_ma=50, fee=fee_rate)
results['AI V55 (Floor 50%)'] = res_v55_50['Portfolio_Value']

benchmark = market_index.reindex(res_static.index).ffill()
benchmark = benchmark / benchmark.iloc[0] * 100
results['Index LQ45'] = benchmark

# --- 5. Visualization ---
plt.figure(figsize=(14, 8))

colors = ['gray', 'blue', 'orange', 'green', 'black']
styles = ['--', '-', '-', '-', ':']

for (name, series), color, style in zip(results.items(), colors, styles):
    plt.plot(series.index, series, label=f"{name} ({series.iloc[-1]:.1f})", color=color, linestyle=style, linewidth=2)

plt.title('V55: Risk Scaling Strategy (Minimum Equity Floor) vs Static')
plt.ylabel('Portfolio Value')
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig('v55_risk_scaling_results.png')
print("Saved plot to v55_risk_scaling_results.png")

# Summary
summary = []
for name, series in results.items():
    total_ret = (series.iloc[-1] - 100)
    peak = series.cummax()
    dd = (series - peak) / peak
    max_dd = dd.min() * 100
    summary.append({'Strategy': name, 'Return %': total_ret, 'Max DD %': max_dd})

df_summary = pd.DataFrame(summary).sort_values('Return %', ascending=False)
print("\n=== Performance ROI & Risk (V55) ===")
print(df_summary.to_string())

df_results = pd.DataFrame(results)
df_results.to_excel('v55_risk_scaling_results.xlsx', index=False)
print("Saved results to v55_risk_scaling_results.xlsx")
