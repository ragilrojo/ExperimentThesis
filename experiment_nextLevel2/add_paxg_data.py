import pandas as pd
import yfinance as yf
import os

# 1. Load existing dataset
file_path = 'G:/My Drive/00_Kuliah/Thesis/sharpenThesis_dpInsya/experiment_nextLevel2/dataset_2023_2025.xlsx'
df = pd.read_excel(file_path, index_col=0)
df.index = pd.to_datetime(df.index)

# 2. Extract date range
start_date = df.index.min().strftime('%Y-%m-%d')
end_date = (df.index.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

print(f"Downloading PAXG-USD from {start_date} to {df.index.max().strftime('%Y-%m-%d')}...")

# 3. Download PAXG-USD
paxg_data = yf.download('PAXG-USD', start=start_date, end=end_date)

# Extraction based on observed MultiIndex structure
# MultiIndex([( 'Close', 'PAXG-USD'), ...], names=['Price', 'Ticker'])
if 'Close' in paxg_data.columns.get_level_values(0):
    paxg = paxg_data['Close']['PAXG-USD']
else:
    # Fallback to a single column if it's not a MultiIndex
    paxg = paxg_data['Close'] if 'Close' in paxg_data.columns else paxg_data.iloc[:, 0]

# 4. Merge
paxg.index = pd.to_datetime(paxg.index)
# Reindex to match the original dataframe's index
df['PAXG-USD'] = paxg.reindex(df.index)

# 5. Check for missing values in PAXG
missing = df['PAXG-USD'].isnull().sum()
if missing > 0:
    print(f"Warning: Found {missing} missing values in PAXG-USD. Filling with ffill...")
    df['PAXG-USD'] = df['PAXG-USD'].ffill().bfill()

# 6. Save back to Excel
df.to_excel(file_path)
print(f"Successfully added PAXG-USD to {file_path}")
print(f"New Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
