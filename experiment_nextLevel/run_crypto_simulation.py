import yfinance as yf
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# 1. Setup Tickers (3 per kategori minimal)
# Format Yahoo Finance
tickers = {
    'Payment': ['BTC-USD', 'LTC-USD', 'BCH-USD'],
    'Layer1': ['ETH-USD', 'SOL-USD', 'BNB-USD', 'ADA-USD', 'XRP-USD'], # XRP sering dianggap payment juga, tapi secara teknis layer 1
    'Stablecoin': ['USDT-USD', 'USDC-USD', 'DAI-USD'],
    'DeFi': ['UNI7083-USD', 'AAVE-USD', 'MKR-USD', 'LINK-USD'],
    'Exchange': ['KCS-USD', 'OKB-USD', 'CRO-USD'], # LEO kurang liquid di YF, diganti yg lain
    'Meme': ['DOGE-USD', 'SHIB-USD', 'PEPE24478-USD'],
    'Privacy': ['XMR-USD', 'ZEC-USD', 'DASH-USD']
}

all_tickers = [ticker for category in tickers.values() for ticker in category]
ticker_to_category = {t: cat for cat, ts in tickers.items() for t in ts}

print(f"Total Aset: {len(all_tickers)}")
print("Categories:", list(tickers.keys()))

# 2. Download Data
# Kita ambil data 1 tahun terakhir (misal 2024-2025)
start_date = "2024-01-01"
end_date = "2025-01-01"

print(f"Downloading data from {start_date} to {end_date}...")
try:
    data = yf.download(all_tickers, start=start_date, end=end_date)['Close']
    
    # Forward fill untuk data weekend/libur (crypto 24/7 tapi kadang ada gap) & Drop NaN awal
    data = data.ffill().dropna()
    print(f"Data shape: {data.shape}")
    
    if data.empty:
        raise ValueError("Data kosong. Cek koneksi internet atau simbol ticker.")

    # 3. Hitung Korelasi pada Window Tertentu (Simulasi 'Crash' vs 'Normal')
    # Kita ambil 2 snapshot berbeda untuk demonstrasi:
    # Snapshot A: Periode 'Tenang' (Low Volatility)
    # Snapshot B: Periode 'Gejolak' (High Volatility/Crash)
    
    # Hitung returns harian
    returns = data.pct_change().dropna()
    
    # Hitung volatilitas pasar rata-rata (rata-rata standar deviasi semua aset) rolling 30 hari
    market_volatility = returns.std(axis=1).rolling(window=7).mean()
    
    # Cari tanggal dengan volatilitas tertinggi dan terendah
    date_calm = market_volatility.idxmin()
    date_crash = market_volatility.idxmax()
    
    print(f"Date Calm: {date_calm}")
    print(f"Date Crash: {date_crash}")

    def analyze_snapshot(target_date, title_prefix):
        # Ambil window 30 hari sebelum target_date
        end_idx = returns.index.get_loc(target_date)
        start_idx = max(0, end_idx - 30)
        
        window_returns = returns.iloc[start_idx : end_idx+1]
        
        # Hitung Korelasi
        corr_matrix = window_returns.corr()
        
        # Bangun Graph
        # Threshold korelasi tinggi > 0.6
        threshold = 0.6
        G = nx.Graph()
        
        # Add nodes with attributes
        for t in all_tickers:
            G.add_node(t, category=ticker_to_category[t])
            
        # Add edges
        edge_count = 0
        for i in range(len(all_tickers)):
            for j in range(i+1, len(all_tickers)):
                t1 = all_tickers[i]
                t2 = all_tickers[j]
                correlation = corr_matrix.loc[t1, t2]
                
                if correlation > threshold:
                    G.add_edge(t1, t2, weight=correlation)
                    edge_count += 1
        
        print(f"\n--- {title_prefix} ({target_date.date()}) ---")
        print(f"Kepadatan Graf (Edges > {threshold}): {edge_count}")
        
        # Algoritma Maximum Independent Set
        # Cari himpunan aset yang TIDAK berkorelasi satu sama lain
        mis = nx.approximation.maximum_independent_set(G)
        print(f"Rekomendasi Aset (Independent Set): {len(mis)} aset")
        print(mis)
        
        # Cek komposisi kategori di Independent Set
        cat_counts = {}
        for node in mis:
            cat = ticker_to_category[node]
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        print("Komposisi Kategori:", cat_counts)
        
        return G, mis

    # Analisis
    G_calm, mis_calm = analyze_snapshot(date_calm, "PASAR TENANG")
    G_crash, mis_crash = analyze_snapshot(date_crash, "PASAR CRASH")

    # Save results to text file for review
    with open("experiment_nextLevel/hasil_simulasi_awal.txt", "w") as f:
        f.write("HASIL SIMULASI TEMPORAL GRAPH CRYPTO PORTFOLIO\n")
        f.write("==============================================\n\n")
        
        f.write(f"Snapshot 1: Pasar Tenang ({date_calm.date()})\n")
        f.write(f"Edges (High Corr): {G_calm.number_of_edges()}\n")
        f.write(f"Independent Set Size: {len(mis_calm)}\n")
        f.write(f"Assets: {', '.join(mis_calm)}\n\n")
        
        f.write(f"Snapshot 2: Pasar Crash ({date_crash.date()})\n")
        f.write(f"Edges (High Corr): {G_crash.number_of_edges()}\n")
        f.write(f"Independent Set Size: {len(mis_crash)}\n")
        f.write(f"Assets: {', '.join(mis_crash)}\n")
        
    print("\nHasil disimpan ke experiment_nextLevel/hasil_simulasi_awal.txt")

except Exception as e:
    print(f"Error: {e}")
