import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from scipy.optimize import minimize
import networkx as nx
from sklearn.metrics import fbeta_score
import warnings
import os

warnings.filterwarnings('ignore')
plt.style.use('ggplot')
sns.set_palette("husl")
SEED = 42
np.random.seed(SEED)

print("Starting Evaluation Script...")

# 1. Load Data
file_path = '../experiment_nextLevel2/dataset_2023_2025.xlsx'
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    exit()

data = pd.read_excel(file_path, index_col=0, parse_dates=True)
returns = data.pct_change().dropna()
market_return = returns.mean(axis=1)
market_index = (1 + market_return).cumprod() * 100

print(f"Data Loaded: {len(data)} rows")

# 2. Model Training
features = pd.DataFrame(index=returns.index)
features['Vol_20'] = market_return.rolling(window=20).std()
features['Mom_20'] = market_return.rolling(window=20).mean()
features['Mom_50'] = market_return.rolling(window=50).mean()
target = (market_return.rolling(5).mean() > market_return.rolling(20).mean()).astype(int).shift(-5)

X_train = features.loc[features.index.year <= 2024].dropna()
y_train = target.loc[X_train.index].fillna(0)
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=SEED)
xgb_model.fit(X_train, y_train)

all_probs = pd.Series(xgb_model.predict_proba(features.dropna())[:, 1], index=features.dropna().index)
all_probs = all_probs.reindex(returns.index, method='ffill').bfill()
opt_t = 0.52

print("Model Trained.")

# 3. Functions
def optimize_markowitz_robust(selected_returns, cov_matrix=None):
    clean_returns = selected_returns.dropna(axis=1, how='any')
    if clean_returns.empty: return {}
    if len(clean_returns.columns) == 1: return {clean_returns.columns[0]: 1.0}
    
    mu = clean_returns.mean() * 252
    if cov_matrix is not None:
        sigma = cov_matrix
    else:
        sigma = clean_returns.cov() * 252
        sigma += np.diag(np.ones(len(sigma)) * 1e-4)
    
    num_assets = len(mu)
    def objective(w):
        ret = np.sum(w * mu)
        risk = np.sqrt(np.dot(w.T, np.dot(sigma, w)))
        return -(ret / risk) if risk > 1e-6 else 0
    
    res = minimize(objective, [1./num_assets]*num_assets, method='SLSQP', 
                   bounds=tuple((0, 1) for _ in range(num_assets)), 
                   constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}))
    
    return dict(zip(clean_returns.columns, res.x)) if res.success else {c: 1./num_assets for c in clean_returns.columns}

def get_assets_graph_selection(returns_window, corr_threshold=0.4, top_n=25, strategy_type='diversify'):
    momentum_assets = returns_window.mean().sort_values(ascending=False).head(top_n).index
    returns_mom = returns_window[momentum_assets]
    corr_mat = returns_mom.corr()
    G = nx.Graph()
    G.add_nodes_from(momentum_assets)
    for i, a1 in enumerate(momentum_assets):
        for a2 in momentum_assets[i+1:]:
            if strategy_type == 'diversify':
                if abs(corr_mat.loc[a1, a2]) > corr_threshold: G.add_edge(a1, a2)
            else:
                if abs(corr_mat.loc[a1, a2]) < corr_threshold: G.add_edge(a1, a2)
    return list(nx.approximation.maximum_independent_set(G))

def run_simulation_ai_momentum(test_dates, returns, ai_probs, threshold, market_idx, top_n, strategy_type, fee=0.0025):
    val = 100.0; history = [val]; dates = [test_dates[0]]
    current_weights = {}; risk_status = False; days_since_rebal = 999
    market_ma = market_idx.rolling(window=200).mean()

    for i, date in enumerate(test_dates[:-1]):
        prob = ai_probs.loc[date]
        prev_risk = risk_status
        if not risk_status and prob > (threshold + 0.05): risk_status = True
        elif risk_status and prob < (threshold - 0.05): risk_status = False
        if market_idx.loc[date] > market_ma.loc[date]: risk_status = True
            
        target_weights = current_weights.copy()
        if not risk_status:
            target_weights = {'CASH': 1.0}; days_since_rebal = 0
        else:
            if (risk_status != prev_risk) or days_since_rebal >= 20:
                try:
                    loc_idx = returns.index.get_loc(date)
                    window = returns.iloc[max(0, loc_idx-60):loc_idx]
                    selected = get_assets_graph_selection(window, top_n=top_n, strategy_type=strategy_type)
                    if selected: target_weights = optimize_markowitz_robust(window[selected]); days_since_rebal = 0
                except: pass
        
        days_since_rebal += 1
        turnover = sum(abs(target_weights.get(k, 0) - current_weights.get(k, 0)) for k in set(target_weights)|set(current_weights))
        val -= val * turnover * fee
        next_date = test_dates[i+1]
        day_ret = 0; new_drifted = {}
        if 'CASH' in target_weights: 
            new_drifted = {'CASH': 1.0}
        else:
            for asset, w in target_weights.items():
                r = returns.loc[next_date, asset] if next_date in returns.index else 0
                day_ret += w * r
                new_drifted[asset] = w * (1 + r)
        val *= (1 + day_ret); history.append(val); dates.append(next_date)
        current_weights = {k: v/sum(new_drifted.values()) for k, v in new_drifted.items()} if sum(new_drifted.values()) > 0 else new_drifted
    return pd.DataFrame({'Portfolio_Value': history}, index=dates)

def run_simulation_top15_static_markowitz(test_dates, returns):
    start_date = test_dates[0]
    loc_idx = returns.index.get_loc(start_date)
    pre_window = returns.iloc[max(0, loc_idx-60):loc_idx]
    top15_momentum = pre_window.mean().sort_values(ascending=False).head(15).index
    static_weights = optimize_markowitz_robust(pre_window[top15_momentum])
    val = 100.0; history = [val]; dates = [start_date]
    for i, date in enumerate(test_dates[:-1]):
        next_date = test_dates[i+1]
        day_ret = sum(w * (returns.loc[next_date, asset] if next_date in returns.index else 0) for asset, w in static_weights.items())
        val *= (1 + day_ret); history.append(val); dates.append(next_date)
    return pd.DataFrame({'Portfolio_Value': history}, index=dates)

def get_rmt_cleaned_cov(returns_window):
    T, N = returns_window.shape
    if T < N: return returns_window.cov() * 252 
    corr_mat = returns_window.corr()
    cov_mat = returns_window.cov() * 252
    evals, evecs = np.linalg.eigh(corr_mat)
    Q = T / N
    lambda_max = (1 + 1.0/Q + 2 * np.sqrt(1.0/Q))
    noise_evals = evals[evals < lambda_max]
    if len(noise_evals) > 0:
        mean_noise = np.mean(noise_evals)
        evals[evals < lambda_max] = mean_noise
    corr_clean = evecs @ np.diag(evals) @ evecs.T
    np.fill_diagonal(corr_clean, 1.0)
    std_devs = np.sqrt(np.diag(cov_mat))
    cov_clean = corr_clean * np.outer(std_devs, std_devs)
    return pd.DataFrame(cov_clean, index=cov_mat.index, columns=cov_mat.columns)

def get_network_centrality_weights(returns_window):
    corr = returns_window.corr().abs().fillna(0)
    dist = np.sqrt(2 * (1 - corr))
    G = nx.Graph()
    assets = returns_window.columns
    for i in range(len(assets)):
        for j in range(i + 1, len(assets)):
            G.add_edge(assets[i], assets[j], weight=dist.iloc[i, j])
    mst = nx.minimum_spanning_tree(G)
    try:
        cent = nx.eigenvector_centrality(mst, max_iter=1000)
    except:
        cent = nx.degree_centrality(mst)
    inv_cent = {k: 1.0/(v + 1e-6) for k,v in cent.items()}
    total_inv = sum(inv_cent.values())
    weights = {k: v/total_inv for k,v in inv_cent.items()}
    return weights

def optimize_network_markowitz_proposisi(returns_window, gamma=0.5):
    cov_rmt = get_rmt_cleaned_cov(returns_window)
    w_markowitz = optimize_markowitz_robust(returns_window, cov_matrix=cov_rmt)
    w_network = get_network_centrality_weights(returns_window)
    final_weights = {}
    assets = returns_window.columns
    for asset in assets:
        wm = w_markowitz.get(asset, 0)
        wn = w_network.get(asset, 0)
        final_weights[asset] = (1 - gamma) * wm + gamma * wn
    total_w = sum(final_weights.values())
    if total_w > 0:
        final_weights = {k: v/total_w for k, v in final_weights.items()}
    return final_weights

def run_simulation_network_markowitz(test_dates, returns, gamma=0.5):
    start_date = test_dates[0]
    loc_idx = returns.index.get_loc(start_date)
    pre_window = returns.iloc[max(0, loc_idx-60):loc_idx]
    top15_momentum = pre_window.mean().sort_values(ascending=False).head(15).index
    selected_returns = pre_window[top15_momentum]
    weights = optimize_network_markowitz_proposisi(selected_returns, gamma=gamma)
    val = 100.0; history = [val]; dates = [start_date]
    for i, date in enumerate(test_dates[:-1]):
        next_date = test_dates[i+1]
        day_ret = sum(w * (returns.loc[next_date, asset] if next_date in returns.index else 0) for asset, w in weights.items())
        val *= (1 + day_ret); history.append(val); dates.append(next_date)
    return pd.DataFrame({'Portfolio_Value': history}, index=dates)

# 4. Simulation
test_dates_2025 = returns.loc[returns.index.year == 2025].index
results = {}

print("Simulating Strategies...")
results['Static Markowitz (Top 15)'] = run_simulation_top15_static_markowitz(test_dates_2025, returns)
results['AI-Cls (Top 15)'] = run_simulation_ai_momentum(test_dates_2025, returns, all_probs, opt_t, market_index, 15, 'cluster')

for g in [0, 0.5, 1]:
    results[f"Network Markowitz (gamma={g})"] = run_simulation_network_markowitz(test_dates_2025, returns, gamma=g)

# 5. Evaluation
def calculate_rachev_ratio(returns, alpha=0.95):
    var_lower = np.percentile(returns, (1-alpha)*100)
    etl_lower = returns[returns <= var_lower].mean()
    var_upper = np.percentile(returns, alpha*100)
    etr_upper = returns[returns >= var_upper].mean()
    if np.abs(etl_lower) < 1e-6: return np.nan 
    return etr_upper / np.abs(etl_lower)

def evaluate_portfolio(portfolio_series, name="Strategy"):
    daily_rets = portfolio_series.pct_change().dropna()
    mean_ret = daily_rets.mean()
    std_dev = daily_rets.std()
    sharpe = (mean_ret / std_dev) * np.sqrt(252) if std_dev > 0 else 0
    cum_ret = (1 + daily_rets).cumprod()
    peak = cum_ret.cummax()
    drawdown = (cum_ret - peak) / peak
    mdd = drawdown.min()
    var_95 = np.percentile(daily_rets, 5)
    rachev_95 = calculate_rachev_ratio(daily_rets, 0.95)
    
    return {
        "Strategy": name,
        "Sharpe Ratio": sharpe,
        "Mean Daily Ret (%)": mean_ret * 100,
        "Daily Std Dev (%)": std_dev * 100,
        "Max Drawdown (%)": mdd * 100,
        "VaR (95%) (%)": var_95 * 100,
        "Rachev Ratio (95%)": rachev_95
    }

eval_data = []
for name, res_df in results.items():
    metrics = evaluate_portfolio(res_df['Portfolio_Value'], name)
    eval_data.append(metrics)

eval_df = pd.DataFrame(eval_data).set_index("Strategy")
eval_df = eval_df.sort_values("Sharpe Ratio", ascending=False)

print("\n=== THESIS PERFORMANCE METRICS (2025) ===")
print(eval_df.round(4).to_string())

