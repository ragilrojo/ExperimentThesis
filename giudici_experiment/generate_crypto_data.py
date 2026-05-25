"""
Generate Dummy Cryptocurrency Data for Giudici et al. (2020) Experiment
Saves to Excel file for later processing
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Parameters (from paper)
n_assets = 10
n_days = 764  # 14 Sept 2017 - 17 Okt 2019
crypto_names = ['BTC', 'ETH', 'XRP', 'USDT', 'BCH', 'LTC', 'BNB', 'EOS', 'XLM', 'TRX']

# Volatilities from paper (Table 1)
volatilities = np.array([0.04, 0.05, 0.07, 0.01, 0.08, 0.06, 0.07, 0.07, 0.10, 0.15])

# Mean returns (slightly positive to negative, realistic for crypto)
mean_returns = np.array([0.0009, -0.0007, 0.0004, 0.0000, -0.0011, -0.0003, 0.0033, 0.0017, 0.0021, 0.0021])

# Create correlation structure
# Crypto markets are highly correlated, except stablecoins
base_corr = 0.6
corr_matrix = np.full((n_assets, n_assets), base_corr)
np.fill_diagonal(corr_matrix, 1.0)

# USDT (stablecoin) has low correlation with others
corr_matrix[3, :] = 0.1
corr_matrix[:, 3] = 0.1
corr_matrix[3, 3] = 1.0

# Add some variation
for i in range(n_assets):
    for j in range(i+1, n_assets):
        if i != 3 and j != 3:  # Not USDT
            variation = np.random.uniform(-0.15, 0.15)
            corr_matrix[i, j] += variation
            corr_matrix[j, i] = corr_matrix[i, j]
            # Ensure valid correlation
            corr_matrix[i, j] = np.clip(corr_matrix[i, j], -0.99, 0.99)
            corr_matrix[j, i] = corr_matrix[i, j]

# Ensure correlation matrix is positive semi-definite
eigenvalues = np.linalg.eigvalsh(corr_matrix)
if np.min(eigenvalues) < 0:
    # Fix by adding small value to diagonal
    corr_matrix = corr_matrix + np.eye(n_assets) * abs(np.min(eigenvalues)) * 1.1

# Generate covariance matrix
cov_matrix = np.outer(volatilities, volatilities) * corr_matrix

# Generate returns
returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)

# Create date range
start_date = datetime(2017, 9, 14)
dates = [start_date + timedelta(days=i) for i in range(n_days)]

# Create DataFrame
df_returns = pd.DataFrame(returns, columns=crypto_names, index=dates)
df_returns.index.name = 'Date'

# Generate prices from returns (starting at 100)
df_prices = pd.DataFrame(index=dates, columns=crypto_names)
df_prices.index.name = 'Date'

for crypto in crypto_names:
    prices = [100]  # Starting price
    for ret in df_returns[crypto].values:
        prices.append(prices[-1] * np.exp(ret))
    df_prices[crypto] = prices[1:]  # Remove initial value

# Save to Excel with multiple sheets
output_file = 'crypto_data.xlsx'

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # Sheet 1: Returns
    df_returns.to_excel(writer, sheet_name='Returns')
    
    # Sheet 2: Prices
    df_prices.to_excel(writer, sheet_name='Prices')
    
    # Sheet 3: Statistics
    stats = pd.DataFrame({
        'Mean': df_returns.mean(),
        'Std': df_returns.std(),
        'Kurtosis': df_returns.kurtosis(),
        'Skewness': df_returns.skew(),
        'Min': df_returns.min(),
        'Max': df_returns.max()
    })
    stats.to_excel(writer, sheet_name='Statistics')
    
    # Sheet 4: Correlation Matrix
    df_corr = df_returns.corr()
    df_corr.to_excel(writer, sheet_name='Correlation')
    
    # Sheet 5: Metadata
    metadata = pd.DataFrame({
        'Parameter': ['Number of Assets', 'Number of Days', 'Start Date', 'End Date', 
                     'Data Source', 'Paper Reference'],
        'Value': [n_assets, n_days, dates[0].strftime('%Y-%m-%d'), dates[-1].strftime('%Y-%m-%d'),
                 'Simulated Data', 'Giudici et al. (2020)']
    })
    metadata.to_excel(writer, sheet_name='Metadata', index=False)

print("[OK] Data generated successfully!")
print(f"[FILE] Saved to: {output_file}")
print(f"\n[DATA] Summary:")
print(f"   - Assets: {n_assets} cryptocurrencies")
print(f"   - Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
print(f"   - Days: {n_days}")
print(f"\n[STATS] Returns Statistics:")
print(stats.round(4))
