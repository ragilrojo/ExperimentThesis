import yfinance as yf
import pandas as pd
import os

# Curated list of major LQ45 stocks (Representative list)
tickers = [
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "TLKM.JK", "ASII.JK", 
    "UNVR.JK", "GOTO.JK", "ADRO.JK", "PTBA.JK", "ANTM.JK", 
    "BRIS.JK", "AMRT.JK", "ICBP.JK", "INDF.JK", "PGAS.JK", 
    "KLBF.JK", "SMGR.JK", "UNTR.JK", "CPIN.JK", "TOWR.JK", 
    "MDKA.JK", "HRUM.JK", "ITMG.JK", "INKP.JK", "MEDC.JK", 
    "AKRA.JK", "EXCL.JK", "TPIA.JK", "MIKA.JK", "BRPT.JK"
]

print(f"Fetching data for {len(tickers)} tickers from yfinance...")

# Download data from 2023 to end of 2025
data = yf.download(tickers, start="2023-01-01", end="2025-12-31")

# Extract Adjusted Close prices
if 'Adj Close' in data.columns:
    df_lq45 = data['Adj Close']
else:
    # yfinance v0.2.x structure might vary or sometimes return Close if Adj Close not available
    df_lq45 = data['Close']

# Drop columns with too many missing values (stocks that might have been delisted or IPOed recently)
df_lq45 = df_lq45.dropna(axis=1, thresh=len(df_lq45) * 0.7)

# Save to Excel
output_file = 'data_lq45_2023_2025.xlsx'
df_lq45.to_excel(output_file)

print(f"Data successfully saved to {output_file}")
print("Final tickers included in the dataset:")
print(df_lq45.columns.tolist())
