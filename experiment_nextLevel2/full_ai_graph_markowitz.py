"""
Full Strategy: AI + Graph + Markowitz

Script ini menjalankan strategi lengkap yang menggabungkan:
1. AI Gatekeeper (Random Forest): Untuk deteksi regime pasar (Bullish/Bearish)
2. Graph Theory (MIS): Filter aset berdasarkan korelasi untuk mengurangi redundansi
3. Markowitz Optimization: Optimasi portfolio pada aset yang sudah difilter

Benchmark: Buy and hold Bitcoin (BTC)
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Set style untuk plot
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    plt.style.use('seaborn') # Fallback for older matplotlib versions
sns.set_palette("husl")

def load_data(file_path):
    """Load dataset dari file Excel."""
    if os.path.exists(file_path):
        data = pd.read_excel(file_path, index_col=0, parse_dates=True)
        print(f"Loaded {len(data)} rows of data.")
        return data
    else:
        print(f"File data tidak ditemukan: {file_path}")
        return None

def engineer_features(data):
    """Membuat fitur untuk AI Gatekeeper."""
    # Returns
    returns = data.pct_change().dropna()
    market_return = returns.mean(axis=1)

    # Features
    features = pd.DataFrame(index=returns.index)
    features['Vol_20'] = market_return.rolling(window=20).std()
    features['Mom_20'] = market_return.rolling(window=20).mean()
    features['Mom_50'] = market_return.rolling(window=50).mean()
    features['Target'] = (market_return.shift(-1) > 0).astype(int)
    features = features.dropna()
    
    return features, returns

def train_ai_model(features):
    """Melatih model Random Forest sebagai AI Gatekeeper."""
    # Split
    train_mask = (features.index.year >= 2023) & (features.index.year <= 2024)
    test_mask = (features.index.year == 2025)
    
    if not any(test_mask):
         print("Warning: No test data found for 2025. Using last 20% for testing.")
         split_idx = int(len(features) * 0.8)
         X_train = features.iloc[:split_idx][['Vol_20', 'Mom_20', 'Mom_50']]
         y_train = features.iloc[:split_idx]['Target']
         X_test = features.iloc[split_idx:][['Vol_20', 'Mom_20', 'Mom_50']]
         y_test = features.iloc[split_idx:]['Target']
    else:
        X_train, y_train = features.loc[train_mask, ['Vol_20', 'Mom_20', 'Mom_50']], features.loc[train_mask, 'Target']
        X_test, y_test = features.loc[test_mask, ['Vol_20', 'Mom_20', 'Mom_50']], features.loc[test_mask, 'Target']

    # Train RF
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, min_samples_split=10)
    rf_model.fit(X_train, y_train)
    test_probs = pd.Series(rf_model.predict_proba(X_test)[:, 1], index=X_test.index)
    train_score = accuracy_score(y_train, rf_model.predict(X_train))
    test_score = accuracy_score(y_test, rf_model.predict(X_test))

    print(f"AI Model Train Accuracy: {train_score:.2%}")
    print(f"AI Model Test Accuracy: {test_score:.2%}")
    
    return rf_model, test_probs, X_test

def get_mis_assets(returns_window, correlation_threshold=0.5):
    """Filter aset menggunakan Maximum Independent Set (MIS) berdasarkan korelasi."""
    corr_mat = returns_window.corr()
    G = nx.Graph()
    assets = returns_window.columns
    G.add_nodes_from(assets)
    for i in range(len(assets)):
        for j in range(i+1, len(assets)):
            if corr_mat.iloc[i, j] > correlation_threshold:
                G.add_edge(assets[i], assets[j])
    
    # nx.approximation.maximum_independent_set tidak selalu optimal, tapi cepat.
    # Untuk jumlah aset sedikit, maximal_independent_set mungkin lebih baik atau clique-based approach.
    mis = nx.maximal_independent_set(G) 
    return list(mis)

def optimize_markowitz(selected_assets_returns):
    """Optimasi Markowitz untuk maksimalisasi Sharpe Ratio."""
    if len(selected_assets_returns.columns) == 0: return {}
    if len(selected_assets_returns.columns) == 1: return {selected_assets_returns.columns[0]: 1.0}
        
    mu = selected_assets_returns.mean() * 252
    sigma = selected_assets_returns.cov() * 252
    
    def neg_sharpe_ratio(weights):
        port_return = np.sum(weights * mu)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
        return -port_return / port_volatility if port_volatility > 0 else 0
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(mu)))
    initial_guess = len(mu) * [1. / len(mu)]
    
    try:
        result = minimize(neg_sharpe_ratio, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return dict(zip(selected_assets_returns.columns, result.x))
    except:
        return dict(zip(selected_assets_returns.columns, initial_guess))

def run_full_strategy(test_dates, returns, probs, lookback=30, corr_threshold=0.5):
    """Menjalankan strategi Full (AI + Graph + Markowitz)."""
    current_value = 10000.0
    history = [current_value]
    dates = [test_dates[0]]
    
    # Kita butuh data sebelumnya untuk lookback, jadi pastikan returns cukup panjang
    # test_dates harus subset dari returns.index
    
    weights_history = [] 

    for i, date in enumerate(test_dates[:-1]):
        # AI Gatekeeper: Cek regime pasar
        # probs adalah probabilitas 'Bullish'
        is_bull = probs.loc[date] > 0.5
        
        loc_idx = returns.index.get_loc(date)
        
        if loc_idx < lookback:
            # Not enough data for lookback
            weights = {'CASH': 1.0}
        else:
            window_returns = returns.iloc[loc_idx-lookback:loc_idx]
            
            if not is_bull:
                # Bearish: Pindah ke Cash
                weights = {'CASH': 1.0}
            else:
                # Bullish: Graph Filter + Markowitz
                # Pastikan window_returns hanya berisi aset crypto, bukan 'CASH' jika ada kolom 'CASH'
                # Asumsi returns hanya berisi crypto assets
                selected_assets = get_mis_assets(window_returns, corr_threshold)
                if not selected_assets:
                     weights = {'CASH': 1.0}
                else:
                    weights = optimize_markowitz(window_returns[selected_assets])
        
        weights_history.append(weights)

        # Calculate daily return based on weights for NEXT day
        next_date = test_dates[i+1]
        
        # Jika hari berikutnya tidak ada di returns (misal akhir data), break
        if next_date not in returns.index:
            break
            
        next_day_rets = returns.loc[next_date]
        
        daily_ret = 0
        if 'CASH' in weights:
            # Return CASH = 0 (asumsi stablecoin/fiat tanpa bunga)
            daily_ret = 0 
        else:
            for asset, w in weights.items():
                if asset in next_day_rets:
                    daily_ret += w * next_day_rets[asset]
        
        current_value *= (1 + daily_ret)
        history.append(current_value)
        dates.append(next_date)
        
    return pd.Series(history, index=dates)

def calculate_metrics(series):
    """Hitung metrik performa portfolio."""
    returns = series.pct_change().dropna()
    if len(series) == 0: return {}
    
    total_return = (series.iloc[-1] / series.iloc[0] - 1) * 100
    
    # Handling potential division by zero or short series
    if len(series) > 1:
        annual_return = ((series.iloc[-1] / series.iloc[0]) ** (252 / len(series)) - 1) * 100
        volatility = returns.std() * np.sqrt(252) * 100
        sharpe = (annual_return / volatility) if volatility > 0 else 0
        max_dd = ((series / series.cummax()) - 1).min() * 100
    else:
        annual_return = 0
        volatility = 0
        sharpe = 0
        max_dd = 0

    return {
        'Total Return (%)': total_return,
        'Annual Return (%)': annual_return,
        'Volatility (%)': volatility,
        'Sharpe Ratio': sharpe,
        'Max Drawdown (%)': max_dd,
        'Final Value': series.iloc[-1]
    }

def main():
    file_path = 'dataset_2023_2025.xlsx'
    
    # 1. Load Data
    data = load_data(file_path)
    if data is None: return

    # 2. Feature Engineering & AI Training
    features, returns = engineer_features(data)
    rf_model, test_probs, X_test = train_ai_model(features)
    
    # 3. Full Strategy
    print("Running Full Strategy (AI + Graph + Markowitz)...")
    test_dates = X_test.index
    # Pastikan data crypto (exclude PAXG or others if needed)
    # Disini kita pakai semua kolom sebagai crypto assets
    
    res_full = run_full_strategy(test_dates, returns, test_probs)
    print("Backtest completed!")

    # 4. Benchmark BTC
    if 'BTC-USD' in data.columns:
        btc_series = data.loc[res_full.index, 'BTC-USD']
        btc_norm = btc_series / btc_series.iloc[0] * 10000
    else:
        print("BTC-USD not found in dataset. Using first column as benchmark.")
        btc_series = data.loc[res_full.index, data.columns[0]]
        btc_norm = btc_series / btc_series.iloc[0] * 10000

    # 5. Metrics
    metrics_full = calculate_metrics(res_full)
    metrics_btc = calculate_metrics(btc_norm)

    metrics_df = pd.DataFrame({
        'Full Strategy': metrics_full,
        'BTC Benchmark': metrics_btc
    })

    print("\n=== Performance Metrics ===")
    print(metrics_df.round(2))

    # 6. Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(res_full, label='Full Strategy (AI+Graph+Markowitz)', linewidth=2)
    plt.plot(btc_norm, label='Bitcoin (Buy & Hold)', linestyle='--', alpha=0.7)
    plt.title('Strategy Performance vs Benchmark (2025)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig('strategy_performance_py.png')
    print("Performance plot saved to strategy_performance_py.png")
    
    plt.show()

if __name__ == "__main__":
    main()
