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
plt.style.use('ggplot')

# --- 1. Load Data ---
file_path = 'data_lq45_2023_2025.xlsx'
if not os.path.exists(file_path):
    file_path = 'dataset_2023_2025.xlsx'
if not os.path.exists(file_path):
    # Fallback to absolute path if running from root but file is in subdir
    file_path = r'g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\experiment_nextLevel3\data_lq45_2023_2025.xlsx'

print(f"Loading data from: {file_path}")
data = pd.read_excel(file_path, index_col=0, parse_dates=True)
returns = data.pct_change().dropna()
market_return = returns.mean(axis=1)

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

xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

train_probs = xgb_model.predict_proba(X_train)[:, 1]
opt_t = np.linspace(0.3, 0.7, 41)[np.argmax([fbeta_score(y_train, (train_probs >= t).astype(int), beta=0.5) for t in np.linspace(0.3, 0.7, 41)])]
test_probs = pd.Series(xgb_model.predict_proba(X_test)[:, 1], index=X_test.index)

print(f"Optimized Threshold: {opt_t:.2f}")

# --- 3. Core Logic: Mitigated Simulation ---

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

def run_simulation_mitigated(test_dates, returns, ai_probs, threshold, 
                             mitigation_type='baseline', 
                             mitigation_param=None, 
                             fee=0.0025, 
                             g_lookback=60, 
                             rebal_period=20, 
                             turnover_buffer=0.05):
    val = 100.0
    history = [val]
    dates = [test_dates[0]]
    
    current_weights = {} 
    risk_status = False 
    
    signal_buffer = []
    
    if mitigation_type == 'smoothing' and mitigation_param:
        smoothed_probs = ai_probs.ewm(span=mitigation_param, adjust=False).mean()
    else:
        smoothed_probs = ai_probs

    days_since_rebal = 999
    
    # Pre-calculate signals if needed (not strictly necessary loop-wise)
    
    for i, date in enumerate(test_dates[:-1]):
        prev_risk_status = risk_status
        
        prob = smoothed_probs.loc[date]
        
        buffer = 0.05
        raw_signal_on = False
        if prob > (threshold + buffer):
            raw_signal_on = True
        elif prob < (threshold - buffer):
            raw_signal_on = False
        else:
            raw_signal_on = (prob > threshold)

        if mitigation_type == 'static':
            risk_status = True
        elif mitigation_type == 'confirmation':
            signal_buffer.append(raw_signal_on)
            if len(signal_buffer) > mitigation_param:
                signal_buffer.pop(0)
            
            if len(signal_buffer) == mitigation_param:
                if all(signal_buffer): risk_status = True
                elif not any(signal_buffer): risk_status = False
            else:
                pass # Hold previous
        else: 
            if not risk_status and prob > (threshold + buffer): risk_status = True
            elif risk_status and prob < (threshold - buffer): risk_status = False
                
        regime_changed = (risk_status != prev_risk_status)
        
        target_weights = current_weights.copy()
        
        if not risk_status:
            if regime_changed or 'CASH' not in current_weights or current_weights.get('CASH', 0) < 0.99:
                target_weights = {'CASH': 1.0}
                days_since_rebal = 0
        else:
            if regime_changed or days_since_rebal >= rebal_period:
                loc_idx = returns.index.get_loc(date)
                window_rets = returns.iloc[loc_idx-g_lookback:loc_idx]
                selected = get_mis_assets(window_rets)
                optimized_weights = optimize_markowitz(window_rets[selected])
                
                all_keys = set(list(optimized_weights.keys()) + list(current_weights.keys()))
                turnover_est = sum(abs(optimized_weights.get(k, 0) - current_weights.get(k, 0)) for k in all_keys)
                
                if regime_changed or turnover_est > turnover_buffer:
                    target_weights = optimized_weights
                    days_since_rebal = 0
            
            days_since_rebal += 1

        all_keys_exec = set(list(target_weights.keys()) + list(current_weights.keys()))
        turnover_actual = sum(abs(target_weights.get(k, 0) - current_weights.get(k, 0)) for k in all_keys_exec)
        
        if turnover_actual > 0.001:
            cost = val * turnover_actual * fee
            val -= cost
        
        next_date = test_dates[i+1]
        day_ret = 0
        new_weights_drifted = {}
        
        if 'CASH' in target_weights and target_weights['CASH'] > 0.99:
            day_ret = 0
            new_weights_drifted = {'CASH': 1.0}
        else:
            for asset, w in target_weights.items():
                if asset in returns.columns:
                    r = returns.loc[next_date, asset]
                    day_ret += w * r
                    new_weights_drifted[asset] = w * (1 + r)
                else:
                    new_weights_drifted[asset] = w
            
            total_w = sum(new_weights_drifted.values()) if new_weights_drifted else 0
            if total_w > 0:
                new_weights_drifted = {k: v/total_w for k, v in new_weights_drifted.items()}
        
        current_weights = new_weights_drifted
        val *= (1 + day_ret)
        history.append(val)
        dates.append(next_date)
        
    return pd.DataFrame({'Portfolio_Value': history}, index=dates)

# --- 4. Run Scenarios ---
test_dates = X_test.index
fee_rate = 0.0025

results = {}

print("Running Static...")
static_res = run_simulation_mitigated(test_dates, returns, test_probs, opt_t, 'static', fee=fee_rate)

print("Running Baseline AI...")
base_res = run_simulation_mitigated(test_dates, returns, test_probs, opt_t, 'baseline', fee=fee_rate)

print("Running Smoothing AI(3)...")
smooth3_res = run_simulation_mitigated(test_dates, returns, test_probs, opt_t, 'smoothing', mitigation_param=3, fee=fee_rate)

print("Running Smoothing AI(5)...")
smooth5_res = run_simulation_mitigated(test_dates, returns, test_probs, opt_t, 'smoothing', mitigation_param=5, fee=fee_rate)

print("Running Confirmation AI(3)...")
conf3_res = run_simulation_mitigated(test_dates, returns, test_probs, opt_t, 'confirmation', mitigation_param=3, fee=fee_rate)

print("Running Confirmation AI(5)...")
conf5_res = run_simulation_mitigated(test_dates, returns, test_probs, opt_t, 'confirmation', mitigation_param=5, fee=fee_rate)


benchmark = (1 + market_return.loc[test_dates]).cumprod() * 100
benchmark = benchmark.reindex(static_res.index).fillna(100)

# Combine for Plot
plt.figure(figsize=(14, 8))
plt.plot(static_res.index, static_res['Portfolio_Value'], label=f"Static ({static_res['Portfolio_Value'].iloc[-1]:.1f})", color='gray', linestyle='--')
plt.plot(base_res.index, base_res['Portfolio_Value'], label=f"Baseline AI ({base_res['Portfolio_Value'].iloc[-1]:.1f})", color='red')
plt.plot(smooth3_res.index, smooth3_res['Portfolio_Value'], label=f"Smooth(3) ({smooth3_res['Portfolio_Value'].iloc[-1]:.1f})", color='orange')
plt.plot(smooth5_res.index, smooth5_res['Portfolio_Value'], label=f"Smooth(5) ({smooth5_res['Portfolio_Value'].iloc[-1]:.1f})", color='brown')
plt.plot(conf3_res.index, conf3_res['Portfolio_Value'], label=f"Conf(3) ({conf3_res['Portfolio_Value'].iloc[-1]:.1f})", color='blue')
plt.plot(conf5_res.index, conf5_res['Portfolio_Value'], label=f"Conf(5) ({conf5_res['Portfolio_Value'].iloc[-1]:.1f})", color='cyan')
plt.plot(benchmark.index, benchmark, label=f"LQ45 ({benchmark.iloc[-1]:.1f})", color='black', alpha=0.5, linestyle=':')

plt.title('V51: Fee Mitigation Strategies Comparison (Fee=0.25%)')
plt.legend()
plt.savefig('v51_comparison_plot.png')
print("Saved plot to v51_comparison_plot.png")

# Summary Metric Table
summary = []
for name, series in results.items():
    total_ret = (series.iloc[-1] - 100)
    # Calculate Drawdown
    peak = series.cummax()
    dd = (series - peak) / peak
    max_dd = dd.min() * 100
    summary.append({'Strategy': name, 'Return %': total_ret, 'Max DD %': max_dd})

df_summary = pd.DataFrame(summary).sort_values('Return %', ascending=False)
print("\n=== PERFORMANCE SUMMARY ===")
print(df_summary.to_string())
print("===========================\n")

# Save Results
data_out = pd.DataFrame({
    'Date': static_res.index,
    'Static': static_res['Portfolio_Value'],
    'Baseline': base_res['Portfolio_Value'],
    'Smooth3': smooth3_res['Portfolio_Value'],
    'Smooth5': smooth5_res['Portfolio_Value'],
    'Conf3': conf3_res['Portfolio_Value'],
    'Conf5': conf5_res['Portfolio_Value'],
    'Benchmark': benchmark
})
data_out.to_excel('v51_mitigation_results.xlsx', index=False)
print("Saved results to v51_mitigation_results.xlsx")
