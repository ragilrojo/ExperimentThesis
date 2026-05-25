import pandas as pd
import numpy as np
import warnings
import networkx as nx
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

# --- Phase 1: Data Preparation ---
file_old = 'crypto_data_real.xlsx'
file_2024 = 'crypto_data_2024.xlsx'

def load_and_preprocess(f1, f2):
    r_old = pd.read_excel(f1, sheet_name='Returns', index_col=0)
    r_2024 = pd.read_excel(f2, sheet_name='Returns', index_col=0)
    r_old.index = pd.to_datetime(r_old.index)
    r_2024.index = pd.to_datetime(r_2024.index)
    
    # Intersection of assets
    assets = list(set(r_old.columns) & set(r_2024.columns))
    
    # FILTER OUT USDT as requested
    if 'USDT' in assets:
        assets.remove('USDT')
        print("USDT matched and removed from assets.")
    
    assets.sort()
    return r_old[assets], r_2024[assets], assets

print("Loading data...", flush=True)
ret_old, ret_2024, assets = load_and_preprocess(file_old, file_2024)

SET_WINDOW = 30
SET_REBALANCE = 7
print(f"Data loaded. Assets ({len(assets)}): {assets}", flush=True)

# --- Phase 2: Core Functions ---
def apply_rmt_filter(returns_window):
    T, N = returns_window.shape
    corr_mat = returns_window.corr().fillna(0).values
    eigenvalues, eigenvectors = np.linalg.eigh(corr_mat)
    Q = T / N
    lambda_max = (1 + np.sqrt(1/Q))**2
    eigenvalues[eigenvalues < lambda_max] = 0
    corr_denoised = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    np.fill_diagonal(corr_denoised, 1)
    return corr_denoised

def get_centrality_weights(returns_window, gamma=1.0):
    T, N = returns_window.shape
    mu = returns_window.mean().values
    sigma = returns_window.std().values
    corr_f = apply_rmt_filter(returns_window)
    cov_f = np.outer(sigma, sigma) * corr_f
    cov_f += np.eye(N) * 1e-8
    dist_mat = np.sqrt(np.maximum(0, 2 * (1 - corr_f)))
    G_full = nx.from_numpy_array(dist_mat)
    mst = nx.minimum_spanning_tree(G_full)
    try:
        centrality = nx.eigenvector_centrality(mst, max_iter=2000)
        cent_vec = np.array([centrality[i] for i in range(N)])
    except:
        cent_vec = np.array(list(nx.degree_centrality(mst).values()))
    fun = lambda w: w.T @ cov_f @ w + gamma * np.sum(cent_vec * w)
    cons = (
        {'type': 'eq',   'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq', 'fun': lambda w: np.dot(w, mu) - np.mean(mu)},
    )
    bnds = tuple((0, 0.4) for _ in range(N))
    x0 = np.ones(N) / N
    res = minimize(fun, x0, method='SLSQP', bounds=bnds, constraints=cons)
    if not res.success:
        return np.ones(N) / N
    return res.x

def get_network_metrics(returns_window):
    T, N = returns_window.shape
    corr_f = apply_rmt_filter(returns_window)
    density = np.sum(np.abs(corr_f) > 0.1) / (N * N)
    dist_mat = np.sqrt(np.maximum(0, 2 * (1 - corr_f)))
    G_full = nx.from_numpy_array(dist_mat)
    mst = nx.minimum_spanning_tree(G_full)
    mst_dist = sum([d['weight'] for u, v, d in mst.edges(data=True)])
    try:
        centrality = nx.eigenvector_centrality(mst, max_iter=2000)
        cent_vec = np.array([centrality[i] for i in range(N)])
    except:
        cent_vec = np.array(list(nx.degree_centrality(mst).values()))
    st_c = np.std(cent_vec)
    mean_c = np.mean(cent_vec)
    max_c = np.max(cent_vec)
    return np.array([st_c * 10, mean_c * 10, mst_dist * 0.1, max_c, density], dtype=np.float32)

def calculate_metrics(returns_series):
    if len(returns_series) == 0: return {}
    total_ret = (1 + returns_series).prod() - 1
    ann_ret = (1 + total_ret) ** (252 / len(returns_series)) - 1
    ann_vol = returns_series.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cumulative = (1 + returns_series).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()
    return {
        'Total Return': total_ret,
        'Ann. Return': ann_ret,
        'Ann. Volatility': ann_vol,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd,
    }

# --- Phase 3: Fast Environment ---
def fast_centrality_weights(cov_f, cent_vec, mu, gamma):
    N = len(mu)
    fun = lambda w: w.T @ cov_f @ w + gamma * np.sum(cent_vec * w)
    cons = (
        {'type': 'eq',   'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq', 'fun': lambda w: np.dot(w, mu) - np.mean(mu)},
    )
    bnds = tuple((0, 0.4) for _ in range(N))
    x0 = np.ones(N) / N
    res = minimize(fun, x0, method='SLSQP', bounds=bnds, constraints=cons)
    if not res.success:
        return np.ones(N) / N
    return res.x

def precompute_env_data(returns_data, window_size=30):
    n_steps = len(returns_data) - window_size
    obs_cache = {}
    opt_cache = {}
    baseline_ret_cache = {}
    print(f"Precomputing {n_steps} window positions...", flush=True)
    for i in range(window_size, len(returns_data)):
        if i % 100 == 0:
            print(f"Processing position {i}/{len(returns_data)}...", flush=True)
        win = returns_data.iloc[i - window_size : i]
        T, N = win.shape
        nw_features = get_network_metrics(win)
        short_ret = win.iloc[-5:].mean().mean()
        long_ret = win.mean().mean()
        momentum = short_ret - long_ret
        recent_vol = win.iloc[-5:].std().mean()
        market_features = np.array([short_ret * 100, momentum * 100, recent_vol * 100, 0.0], dtype=np.float32)
        obs_cache[i] = (nw_features, market_features)
        
        mu = win.mean().values
        sigma = win.std().values
        corr_f = apply_rmt_filter(win)
        cov_f = np.outer(sigma, sigma) * corr_f
        cov_f += np.eye(N) * 1e-8
        
        dist_mat = np.sqrt(np.maximum(0, 2 * (1 - corr_f)))
        G_full = nx.from_numpy_array(dist_mat)
        mst = nx.minimum_spanning_tree(G_full)
        try:
            centrality = nx.eigenvector_centrality(mst, max_iter=2000)
            cent_vec = np.array([centrality[j] for j in range(N)])
        except:
            cent_vec = np.array(list(nx.degree_centrality(mst).values()))
        opt_cache[i] = (cov_f, cent_vec, mu)
        w_baseline = fast_centrality_weights(cov_f, cent_vec, mu, gamma=1.0)
        baseline_ret_cache[i] = np.dot(w_baseline, returns_data.iloc[i].values)
    return obs_cache, opt_cache, baseline_ret_cache

class GammaPortfolioEnvFast(gym.Env):
    def __init__(self, returns_data, obs_cache, opt_cache, baseline_ret_cache=None, window_size=30, reward_mode='sharpe', gamma_center=1.0, gamma_range=1.0):
        super().__init__()
        self.data = returns_data
        self.obs_cache = obs_cache
        self.opt_cache = opt_cache
        self.baseline_ret_cache = baseline_ret_cache or {}
        self.window_size = window_size
        self.reward_mode = reward_mode
        self.gamma_center = gamma_center
        self.gamma_range = gamma_range
        self.current_step = window_size
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.port_val = 1.0
        self.peak_val = 1.0
    def _action_to_gamma(self, action):
        return self.gamma_center + action * self.gamma_range
    def _get_obs(self):
        nw_features, market_features = self.obs_cache[self.current_step]
        mf = market_features.copy()
        mf[3] = self.port_val - 1
        return np.concatenate([nw_features, mf])
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.port_val = 1.0
        self.peak_val = 1.0
        return self._get_obs(), {}
    def step(self, action):
        gamma = self._action_to_gamma(float(action[0]))
        cov_f, cent_vec, mu = self.opt_cache[self.current_step]
        w = fast_centrality_weights(cov_f, cent_vec, mu, gamma)
        port_ret = np.dot(w, self.data.iloc[self.current_step].values)
        self.port_val *= (1 + port_ret)
        self.peak_val = max(self.peak_val, self.port_val)
        drawdown = (self.port_val - self.peak_val) / self.peak_val
        ew_ret = self.data.iloc[self.current_step].mean()
        nw_ret = self.baseline_ret_cache.get(self.current_step, ew_ret)
        if self.reward_mode == 'excess_nw': reward = (port_ret - nw_ret) * 100
        elif self.reward_mode == 'risk_adjusted_excess':
            excess = (port_ret - nw_ret) * 100
            dd_penalty = min(abs(drawdown) * 10, 2.0)
            reward = excess - dd_penalty
        elif self.reward_mode == 'defensive': reward = (port_ret - 2.0 * abs(drawdown)) * 100
        elif self.reward_mode == 'total_return': reward = port_ret * 100
        elif self.reward_mode == 'excess_return': reward = (port_ret - ew_ret) * 100
        else: reward = port_ret / (self.data.iloc[self.current_step - self.window_size : self.current_step].values.std() + 1e-6) * 10
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        return self._get_obs(), reward, done, False, {}

# --- Phase 4: Backtest Engine ---
def run_backtest_with_frequency(strategy, data, window=30, rebalance_freq=7):
    rets = []
    dates = []
    current_weights = None
    current_gamma = 1.0
    for i in range(window, len(data)):
        if (i - window) % rebalance_freq == 0:
            window_df = data.iloc[i-window:i]
            current_weights = strategy.compute_weights(window_df)
            current_gamma = getattr(strategy, 'last_gamma', 1.0)
        if current_weights is not None:
            daily_ret = np.dot(current_weights, data.iloc[i].values)
            rets.append(daily_ret)
            dates.append(data.index[i])
    return pd.Series(rets, index=dates, name=strategy.name)

class PortfolioStrategy:
    def __init__(self, name): 
        self.name = name
        self.last_gamma = 1.0
    def compute_weights(self, returns_window): raise NotImplementedError()

class NetworkMarkowitz(PortfolioStrategy):
    def __init__(self, name, gamma=1.0): 
        super().__init__(name)
        self.gamma = gamma
        self.last_gamma = gamma
    def compute_weights(self, returns_window): return get_centrality_weights(returns_window, self.gamma)

class RLNetworkMarkowitz(PortfolioStrategy):
    def __init__(self, name, model, gamma_center=1.0, gamma_range=1.0):
        super().__init__(name)
        self.model = model
        self.gamma_center = gamma_center
        self.gamma_range = gamma_range
        self.last_gamma = 1.0
    def compute_weights(self, returns_window):
        nw_features = get_network_metrics(returns_window)
        short_ret = returns_window.iloc[-5:].mean().mean()
        long_ret = returns_window.mean().mean()
        momentum = short_ret - long_ret
        recent_vol = returns_window.iloc[-5:].std().mean()
        market_features = np.array([short_ret * 100, momentum * 100, recent_vol * 100, 0.0], dtype=np.float32)
        obs = np.concatenate([nw_features, market_features])
        obs = np.nan_to_num(obs)
        action, _ = self.model.predict(obs, deterministic=True)
        self.last_gamma = self.gamma_center + float(action[0]) * self.gamma_range
        return get_centrality_weights(returns_window, gamma=self.last_gamma)

# --- EXECUTION ---
print("\nStarting Simulation (No USDT)...", flush=True)
obs_cache, opt_cache, baseline_ret_cache = precompute_env_data(ret_old, window_size=SET_WINDOW)

GAMMA_CENTER = 1.0
GAMMA_RANGE  = 1.0
TRAIN_STEPS  = 20000 

ppo_kwargs = dict(policy="MlpPolicy", verbose=0, ent_coef=0.01, learning_rate=3e-4, n_steps=2048, batch_size=128, n_epochs=10, gamma=0.99, clip_range=0.2)

print("\nTraining RL-Net (Excess NW)...", flush=True)
env_excess = GammaPortfolioEnvFast(ret_old, obs_cache, opt_cache, baseline_ret_cache, window_size=SET_WINDOW, reward_mode='excess_nw', gamma_center=GAMMA_CENTER, gamma_range=GAMMA_RANGE)
model_excess = PPO(env=env_excess, **ppo_kwargs)
model_excess.learn(total_timesteps=TRAIN_STEPS)

print("Training RL-Net (Risk-Adj Excess)...", flush=True)
env_riskadj = GammaPortfolioEnvFast(ret_old, obs_cache, opt_cache, baseline_ret_cache, window_size=SET_WINDOW, reward_mode='risk_adjusted_excess', gamma_center=GAMMA_CENTER, gamma_range=GAMMA_RANGE)
model_riskadj = PPO(env=env_riskadj, **ppo_kwargs)
model_riskadj.learn(total_timesteps=TRAIN_STEPS)

# --- Final Evaluation (Out-of-Sample 2024) ---
print("\nRunning Out-of-Sample Backtests (2024)...", flush=True)
strategies = [
    NetworkMarkowitz("NW (Gamma=1.0)", gamma=1.0),
    NetworkMarkowitz("NW (Gamma=0.0)", gamma=0.0),
    RLNetworkMarkowitz("RL-Net (Excess NW)", model_excess, GAMMA_CENTER, GAMMA_RANGE),
    RLNetworkMarkowitz("RL-Net (Risk-Adj)", model_riskadj, GAMMA_CENTER, GAMMA_RANGE)
]

results = {}
for strat in strategies:
    print(f"Backtesting {strat.name}...")
    res_ret = run_backtest_with_frequency(strat, ret_2024, window=SET_WINDOW, rebalance_freq=SET_REBALANCE)
    results[strat.name] = res_ret

metrics_list = []
for name, rets in results.items():
    m = calculate_metrics(rets)
    m['Strategy'] = name
    metrics_list.append(m)

df_metrics = pd.DataFrame(metrics_list).set_index('Strategy')
print("\nFinal Metrics (2024 Out-of-Sample):")
print(df_metrics.to_string())

# Buy & Hold BTC Benchmark
btc_ret = ret_2024['BTC'].iloc[SET_WINDOW:]
btc_metrics = calculate_metrics(btc_ret)
btc_metrics['Strategy'] = 'BTC Buy & Hold'
print("\nBTC Benchmark:")
print(pd.Series(btc_metrics))
