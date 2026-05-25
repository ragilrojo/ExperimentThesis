import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.optimize import minimize
import os

# --- 1. Load Data ---
file_path = 'data_lq45_2023_2025.xlsx'
if not os.path.exists(file_path):
    print("Data file not found.")
    exit()

data = pd.read_excel(file_path, index_col=0, parse_dates=True)
returns = data.pct_change().dropna()
market_return = returns.mean(axis=1)
market_price = (1 + market_return).cumprod() * 100

# --- 2. Feature Engineering ---
features = pd.DataFrame(index=returns.index)
features['Vol_20'] = market_return.rolling(window=20).std()
ma5 = market_price.rolling(window=5).mean()
ma20 = market_price.rolling(window=20).mean()
ma50 = market_price.rolling(window=50).mean()
features['Dist_MA20'] = market_price / ma20
features['Dist_MA50'] = market_price / ma50

# Trend Signal
trend_bullish = (market_price > ma50).astype(int)

# Target
target = (ma5.shift(-1) > ma5).astype(int)

features = features.dropna()
target = target.reindex(features.index).fillna(0)
trend_bullish = trend_bullish.reindex(features.index).fillna(0)

# Split
train_mask = (features.index.year <= 2024)
test_mask = (features.index.year == 2025)

X_train, y_train = features.loc[train_mask], target.loc[train_mask]
X_test = features.loc[test_mask]

# Model
xgb_model = xgb.XGBClassifier(n_estimators=120, learning_rate=0.04, max_depth=6, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Predictions
ai_probs = pd.Series(xgb_model.predict_proba(X_test)[:, 1], index=X_test.index)
test_trend = trend_bullish.loc[X_test.index]

# --- 3. Diagnostics ---
opt_t = 0.60
risk_off_days = 0
total_days = len(X_test)
fee_total = 0.0
turnover_total = 0.0

print(f"\n--- DIAGNOSTICS (2025) ---")
print(f"Total Trading Days: {total_days}")

# Analyze Signals
risk_off_mask = (ai_probs < opt_t) & (test_trend == 0)
risk_on_mask = ~risk_off_mask

print(f"Days Invested (Risk On): {risk_on_mask.sum()} ({risk_on_mask.mean():.1%})")
print(f"Days in Cash (Risk Off): {risk_off_mask.sum()} ({risk_off_mask.mean():.1%})")

# Breakdown of Risk-Off Cause
ai_low = (ai_probs < opt_t).sum()
trend_low = (test_trend == 0).sum()
print(f"  - AI < {opt_t}: {ai_low} days")
print(f"  - Price < MA50: {trend_low} days")
print(f"  - BOTH (Trigger Cash): {risk_off_mask.sum()} days")

# Analyze Fees/Turnover Simulation (Simplified)
# We run a loop to calculate actual fee drag
val = 100.0
fee = 0.0025
prev_weights = {}
fee_cost_cumulative = 0.0

static_weights = None
# Pre-calculate static weights to compaare
def optimize_markowitz(selected_returns):
    if len(selected_returns.columns) == 0: return {}
    mu, sigma = selected_returns.mean() * 252, selected_returns.cov() * 252
    sigma += np.eye(len(sigma)) * 1e-4 
    try:
        res = minimize(lambda w: -(np.sum(w*mu)/(np.sqrt(np.dot(w.T, np.dot(sigma, w))) + 1e-6)), 
                       [1./len(mu)]*len(mu), method='SLSQP', 
                       bounds=tuple((0, 1) for _ in range(len(mu))), 
                       constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}))
        return dict(zip(selected_returns.columns, res.x)) if res.success else {}
    except:
        return {}

test_dates = X_test.index
# Static Optimization
loc_idx_static = returns.index.get_loc(test_dates[0])
window_rets_static = returns.iloc[loc_idx_static-30:loc_idx_static]
static_weights = optimize_markowitz(window_rets_static.dropna(axis=1, how='any'))

print(f"\n--- TURNOVER ANALYSIS ---")
cash_streak = 0
max_cash_streak = 0
switches_to_cash = 0
switches_to_market = 0
prev_state = "Invested" # Assume start

for i, date in enumerate(test_dates[:-1]):
    # Determine State
    prob = ai_probs.loc[date]
    is_trend_up = test_trend.loc[date] == 1
    is_risk_off = (prob < opt_t) and (not is_trend_up)
    
    current_weights = {}
    if is_risk_off:
        current_weights = {'CASH': 1.0}
        current_state = "Cash"
        cash_streak += 1
    else:
        # For simulation, just assume we hold the static portfolio when invested 
        # to isolate the timing effect from the re-optimization effect
        current_weights = static_weights # Approximation for diagnostic
        current_state = "Invested"
        cash_streak = 0
        
    max_cash_streak = max(max_cash_streak, cash_streak)
    
    if prev_state == "Invested" and current_state == "Cash":
        switches_to_cash += 1
    elif prev_state == "Cash" and current_state == "Invested":
        switches_to_market += 1
        
    # turnover calc
    all_assets = set(list(current_weights.keys()) + list(prev_weights.keys()))
    turnover = sum(abs(current_weights.get(a, 0) - prev_weights.get(a, 0)) for a in all_assets)
    
    cost = val * turnover * fee
    fee_cost_cumulative += cost
    val -= cost
    
    # Update value
    next_date = test_dates[i+1]
    # Simple return approx
    day_ret = 0
    if not is_risk_off:
         # Use static weights return for approx
         day_ret = sum(w * returns.loc[next_date, a] for a, w in static_weights.items())
    
    val *= (1 + day_ret)
    prev_weights = current_weights
    prev_state = current_state

print(f"Total Fee Drag (Approx): {fee_cost_cumulative:.2f} points (Start 100)")
print(f"Max Consecutive Days in Cash: {max_cash_streak}")
print(f"Switches Invested -> Cash: {switches_to_cash}")
print(f"Switches Cash -> Invested: {switches_to_market}")
print(f"Total Regime Switches: {switches_to_cash + switches_to_market}")

