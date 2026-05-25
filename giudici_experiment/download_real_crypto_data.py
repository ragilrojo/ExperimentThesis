import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

def download_and_format_crypto_data():
    # Assets from paper/dummy
    assets = ['BTC', 'ETH', 'XRP', 'USDT', 'BCH', 'LTC', 'BNB', 'EOS', 'XLM', 'TRX']
    tickers = [f"{a}-USD" for a in assets]
    
    start_date = "2017-09-14"
    end_date = "2019-10-18"  # One day after 17 Oct to include it
    
    print(f"Downloading data for {assets} from {start_date} to 2019-10-17...")
    
    # Download data
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    
    # Handle missing values (if any)
    # Forward fill then backward fill
    data = data.ffill().bfill()
    
    # Rename columns back to clean names
    data.columns = [c.replace('-USD', '') for c in data.columns]
    # Reorder columns to match original order
    data = data[assets]
    
    # 1. Prices (Normalized to 100 at start like dummy)
    # The paper uses actual prices usually, but the dummy normalized to 100.
    # To keep format identical, we'll provide both or just normalized.
    # Let's provide normalized prices to match the 'evolution' feel of the dummy.
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
                     'Data Source', 'Paper Reference', 'Data Type'],
        'Value': [len(assets), len(df_returns), df_returns.index[0].strftime('%Y-%m-%d'), 
                 df_returns.index[-1].strftime('%Y-%m-%d'),
                 'Yahoo Finance (Real Data)', 'Giudici et al. (2020)', 'Log Returns']
    })
    
    # Save to Excel
    output_file = 'crypto_data_real.xlsx'
    print(f"Saving to {output_file}...")
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_returns.to_excel(writer, sheet_name='Returns')
        df_prices.to_excel(writer, sheet_name='Prices')
        stats.to_excel(writer, sheet_name='Statistics')
        df_corr.to_excel(writer, sheet_name='Correlation')
        metadata.to_excel(writer, sheet_name='Metadata', index=False)
    
    print("[OK] Real data downloaded and formatted successfully!")

if __name__ == "__main__":
    download_and_format_crypto_data()
