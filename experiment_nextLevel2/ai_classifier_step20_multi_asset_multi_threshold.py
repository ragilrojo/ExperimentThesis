"""
Step 20: Multi-Asset Risk-Averse Regime Analysis (Python Script)

Objective:
Menerapkan 4 varian tingkat konservatisme (Thresholding) menggunakan seluruh dataset (Multi-Asset).
Model ini dirancang untuk meminimalkan False Bullish sesuai prinsip "Capital Preservation".

Variants:
1. Standard (T >= 0.50)
2. Cautious (T >= 0.65)
3. Conservative (T >= 0.80)
4. Ultra-Conservative (T >= 0.90)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import warnings
import os

warnings.filterwarnings('ignore')
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    plt.style.use('seaborn')

def load_and_process_data(file_path):
    """Load and process data for multi-asset analysis."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None

    print(f"Loading data from {file_path}...")
    df = pd.read_excel(file_path, index_col=0, parse_dates=True)
    returns = df.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Filter crypto assets (exclude stablecoins/gold like PAXG if needed)
    crypto_assets = [c for c in df.columns if 'PAXG' not in c]
    crypto_rets = returns[crypto_assets]
    
    print(f"📊 Processing {len(crypto_assets)} assets for Feature Engineering...")
    
    # A. PCA Features (Global Market Drivers)
    scaler = StandardScaler()
    pca = PCA(n_components=5)
    X_hybrid = pd.DataFrame(pca.fit_transform(scaler.fit_transform(crypto_rets)), 
                            index=crypto_rets.index, 
                            columns=[f'PC{i+1}' for i in range(5)])
    
    # B. Volatility & Momentum (BTC Specific)
    if 'BTC-USD' in crypto_rets.columns:
        X_hybrid['BTC_Vol_20'] = crypto_rets['BTC-USD'].rolling(20).std()
        X_hybrid['BTC_Mom_20'] = crypto_rets['BTC-USD'].rolling(20).mean()
    else:
        # Fallback to first column if BTC not found
        first_col = crypto_rets.columns[0]
        X_hybrid['BTC_Vol_20'] = crypto_rets[first_col].rolling(20).std()
        X_hybrid['BTC_Mom_20'] = crypto_rets[first_col].rolling(20).mean()

    # C. Graph Density (Topological Risk)
    print("Computing Graph Density (this may take a moment)...")
    X_hybrid['Graph_Density'] = calculate_graph_density_series(crypto_rets)
    
    # D. Target Construction (BTC Return > 0.5%)
    # Menggunakan BTC sebagai proxy market direction
    target_asset = 'BTC-USD' if 'BTC-USD' in crypto_rets.columns else crypto_rets.columns[0]
    X_hybrid['Target'] = np.where(crypto_rets[target_asset].shift(-1) > 0.005, 1, 0)
    
    X_hybrid = X_hybrid.dropna()
    print(f"✅ Feature Matrix Ready. Shape: {X_hybrid.shape}")
    
    return X_hybrid

def calculate_graph_density_series(ret_df, window=20, threshold=0.5):
    """Calculate rolling graph density based on correlation matrix."""
    rolling_corr = ret_df.rolling(window).corr()
    densities = []
    dates = []
    
    v = len(ret_df.columns)
    max_edges = v * (v - 1) / 2
    
    # Rolling correlation index starts from window-1
    # Note: rolling_corr index is MultiIndex (Date, Asset)
    
    unique_dates = ret_df.index[window-1:]
    
    for date in unique_dates:
        try:
            corr_matrix = rolling_corr.loc[date]
            # Handle potential missing values in correlation
            corr_matrix = corr_matrix.fillna(0)
            
            # Adjacency matrix: 1 if correlation > threshold, else 0
            # Diagonal is always 1, we exclude it in calculation or logic
            adj_matrix = (corr_matrix > threshold).astype(int)
            
            # Sum of adjacency matrix includes diagonal (v ones)
            # Total edges = (Sum - v) / 2
            edges = (adj_matrix.values.sum() - v) / 2
            
            density = edges / max_edges if max_edges > 0 else 0
            densities.append(density)
            dates.append(date)
        except KeyError:
             # Skip dates if rolling corr isn't available
             pass
            
    return pd.Series(densities, index=dates)

def train_model(X_hybrid):
    """Train Random Forest model."""
    print("\nTraining Random Forest Model...")
    
    # Split Data
    # Training: < 2025
    # Testing: 2025
    train_idx = X_hybrid.index.year < 2025
    test_idx = X_hybrid.index.year == 2025
    
    if not any(test_idx):
        print("Warning: No data for 2025 found. Using last 20% split.")
        split_point = int(len(X_hybrid) * 0.8)
        X_train = X_hybrid.iloc[:split_point].drop(columns=['Target'])
        y_train = X_hybrid.iloc[:split_point]['Target']
        X_test = X_hybrid.iloc[split_point:].drop(columns=['Target'])
        y_test = X_hybrid.iloc[split_point:]['Target']
    else:
        X_train, y_train = X_hybrid.loc[train_idx].drop(columns=['Target']), X_hybrid.loc[train_idx, 'Target']
        X_test, y_test = X_hybrid.loc[test_idx].drop(columns=['Target']), X_hybrid.loc[test_idx, 'Target']
        
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    
    probs_bullish = rf_model.predict_proba(X_test)[:, 1]
    
    print("✅ Model trained on Multi-Asset data.")
    return rf_model, X_test, y_test, probs_bullish

def analyze_thresholds(y_test, probs_bullish):
    """Analyze performance across different conservatism thresholds."""
    thresholds = {
        'Standard': 0.50,
        'Cautious': 0.65,
        'Conservative': 0.80,
        'Ultra-Conservative': 0.90
    }
    
    results = []
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (name, thresh) in enumerate(thresholds.items()):
        y_pred = (probs_bullish >= thresh).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        
        # Determine specific CM values
        # Shape might be less than 2x2 if only one class is predicted
        tn, fp, fn, tp = 0, 0, 0, 0
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        elif cm.shape == (1, 1):
            # Only one class present/predicted. Hard to map without context, but assuming:
            pass 
            
        # Metric
        prec = precision_score(y_test, y_pred, zero_division=0)
        
        # Dangerous False Bullish: FP (Predicted Bullish, Actual Bearish)
        # Captured True Bullish: TP
        
        results.append({
            'Variant': name,
            'Threshold': thresh,
            'Precision': prec,
            'Dangerous_False_Bullish(FP)': fp,
            'Captured_True_Bullish(TP)': tp
        })
        
        # Plot Heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges' if name=='Standard' else 'YlGn', ax=axes[i], 
                    xticklabels=['Bearish', 'Bullish'], 
                    yticklabels=['Bearish', 'Bullish'])
        axes[i].set_title(f"{name} (Threshold >= {thresh})\nPrecision: {prec:.1%}")
        axes[i].set_xlabel('Prediksi (AI)')
        axes[i].set_ylabel('Aktual (Market)')

    plt.tight_layout()
    plt.savefig('multi_threshold_confusion_matrix.png')
    print("\nVisualization saved to 'multi_threshold_confusion_matrix.png'")
    # plt.show() # Uncomment if running in environment with display
    
    return pd.DataFrame(results)

def main():
    file_path = 'dataset_2023_2025.xlsx'
    
    # 1. Load & Process
    X_hybrid = load_and_process_data(file_path)
    if X_hybrid is None: return
    
    # 2. Train Model
    rf_model, X_test, y_test, probs_bullish = train_model(X_hybrid)
    
    # 3. Analyze Thresholds
    results_df = analyze_thresholds(y_test, probs_bullish)
    
    print("\n--- SUMMARY TABLE: THE RISK VS OPPORTUNITY ---")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()
