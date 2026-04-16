import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

def download_and_format_crypto_data_2024():
    # Assets from paper/dummy
    assets = ['BTC', 'ETH', 'XRP', 'USDT', 'BCH', 'LTC', 'BNB', 'EOS', 'XLM', 'TRX']
    tickers = [f"{a}-USD" for a in assets]
    
    # Dates for 2024
    start_date = "2024-01-01"
    end_date = "2025-01-01"  # Include data up to 31 Dec 2024
    
    print(f"Downloading 2024 data for {assets} from {start_date} to 2024-12-31...")
    
    # Download data
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    
    # Handle missing values (if any)
    # Forward fill then backward fill
    data = data.ffill().bfill()
    
    # Rename columns back to clean names
    data.columns = [c.replace('-USD', '') for c in data.columns]
    # Reorder columns to match original order
    data = data[assets]
    
    # 1. Prices (Normalized to 100 at start)
    df_prices = (data / data.iloc[0]) * 100
    
    # 2. Returns (Log Returns)
    df_returns = np.log(data / data.shift(1)).dropna()
    
    # Adjust df_prices to match the index of returns (drops first day)
    df_prices = df_prices.loc[df_returns.index]
    
    # 3. Statistics
    stats = pd.DataFrame({
        'Mean': df_returns.mean(),
        'Std': df_returns.std(),
        'Kurtosis': df_returns.kurtosis(),
        'Skewness': df_returns.skew(),
        'Min': df_returns.min(),
        'Max': df_returns.max()
    })
    
    # 4. Correlation
    df_corr = df_returns.corr()
    
    # 5. Metadata
    metadata = pd.DataFrame({
        'Parameter': ['Number of Assets', 'Number of Days', 'Start Date', 'End Date', 
                     'Data Source', 'Year focus', 'Data Type'],
        'Value': [len(assets), len(df_returns), df_returns.index[0].strftime('%Y-%m-%d'), 
                 df_returns.index[-1].strftime('%Y-%m-%d'),
                 'Yahoo Finance (Real Data 2024)', '2024 Re-experiment', 'Log Returns']
    })
    
    # Save to Excel
    output_file = 'crypto_data_2024.xlsx'
    print(f"Saving to {output_file}...")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_returns.to_excel(writer, sheet_name='Returns')
        df_prices.to_excel(writer, sheet_name='Prices')
        stats.to_excel(writer, sheet_name='Statistics')
        df_corr.to_excel(writer, sheet_name='Correlation')
        metadata.to_excel(writer, sheet_name='Metadata', index=False)
    
    print(f"[OK] Real data for 2024 downloaded and saved to {output_file}!")

if __name__ == "__main__":
    download_and_format_crypto_data_2024()
