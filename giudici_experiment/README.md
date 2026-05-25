# Giudici et al. (2020) Network Markowitz Experiment

Implementasi paper: **"Network Models to Improve Automated Cryptocurrency Portfolio Management"**  
Authors: Paolo Giudici, Paolo Pagnottoni, Gloria Polinesi (2020)  
Published in: *Frontiers in Artificial Intelligence*, 3, 22

## 📁 File Structure

```
giudici_experiment/
├── README.md                           # Dokumentasi ini
├── generate_crypto_data.py             # Script generate dummy data
├── crypto_data.xlsx                    # Dataset cryptocurrency (5 sheets)
├── giudici_network_markowitz.ipynb     # Main analysis notebook
└── network_markowitz_results.xlsx      # Output hasil optimisasi (generated)
```

## 🎯 Tujuan Eksperimen

Mengimplementasikan dan memahami metodologi **Network Markowitz** yang menggabungkan:

1. **Random Matrix Theory (RMT)** - Filter noise dari correlation matrix
2. **Minimal Spanning Tree (MST)** - Simplifikasi struktur jaringan
3. **Network Centrality** - Ukuran systemic risk setiap aset
4. **Portfolio Optimization** - Markowitz dengan centrality penalty

## 📊 Dataset (`crypto_data.xlsx`)

Dataset berisi **10 cryptocurrency** selama **764 hari** (14 Sept 2017 - 17 Okt 2019):

**Cryptocurrencies:**
- BTC (Bitcoin)
- ETH (Ethereum)
- XRP (Ripple)
- USDT (Tether - Stablecoin)
- BCH (Bitcoin Cash)
- LTC (Litecoin)
- BNB (Binance Coin)
- EOS
- XLM (Stellar)
- TRX (Tron)

**Excel Sheets:**
1. **Returns** - Daily log returns
2. **Prices** - Price evolution (normalized to 100)
3. **Statistics** - Descriptive statistics (mean, std, kurtosis, skewness)
4. **Correlation** - Correlation matrix
5. **Metadata** - Dataset information

## 🚀 Cara Menggunakan

### Step 1: Generate Data (Sudah dilakukan)
```bash
python generate_crypto_data.py
```

Output: `crypto_data.xlsx`

### Step 2: Run Analysis Notebook
```bash
jupyter notebook giudici_network_markowitz.ipynb
```

Atau buka di VS Code / JupyterLab

### Step 3: Review Results

Notebook akan menghasilkan:
- Visualisasi RMT filtering
- MST network graph
- Centrality scores
- Portfolio weights untuk berbagai nilai γ
- Risk-return analysis
- Export ke `network_markowitz_results.xlsx`

## 📐 Metodologi

### 1. Random Matrix Theory (RMT)

**Tujuan:** Memfilter noise dari correlation matrix

**Marchenko-Pastur Threshold:**
```
λ+ = 1 + (1/Q) + 2√(1/Q)
```
dimana Q = T/N (ratio observasi/aset)

**Proses:**
- Hitung eigenvalues dari correlation matrix
- Filter: hanya eigenvalues > λ+ yang dianggap signifikan
- Reconstruct filtered correlation matrix

### 2. Minimal Spanning Tree (MST)

**Tujuan:** Simplifikasi struktur jaringan

**Distance Metric:**
```
d_ij = √(2 - 2c_ij)
```

**Proses:**
- Convert correlation → distance
- Build MST (N-1 edges dari N(N-1)/2 possible edges)
- Visualisasi network structure

### 3. Network Centrality

**Eigenvector Centrality:**
- Mengukur pentingnya node dalam jaringan
- Higher centrality = higher systemic risk
- Lower centrality = peripheral assets

### 4. Portfolio Optimization

**Objective Function:**
```
min_w  w^T Σ* w + γ Σ(x_i w_i)
```

**Where:**
- `w` = portfolio weights
- `Σ*` = filtered covariance matrix
- `γ` = systemic risk aversion parameter
- `x_i` = eigenvector centrality of asset i

**Constraints:**
- Σw_i = 1 (fully invested)
- w_i ≥ 0 (no short selling)
- μ_P ≥ μ̄ (minimum return)

## 🎛️ Parameter γ (Risk Aversion)

Paper menguji 7 nilai γ:

| γ | Interpretasi | Best For |
|---|--------------|----------|
| **0** | Network Markowitz (no centrality penalty) | **Bear market** - proteksi downside |
| 0.005 | Very low risk aversion | Moderate bull market |
| 0.025 | Low risk aversion | Bull market |
| 0.05 | Moderate risk aversion | Balanced |
| 0.15 | Medium risk aversion | Conservative |
| 0.7 | High risk aversion | Very conservative |
| **1.0** | Very high risk aversion | Maximum safety |

## 📈 Key Findings (Paper)

### Bull Market:
- ✅ Model dengan γ > 0 beradaptasi cepat
- ✅ Return lebih tinggi
- ❌ Risiko lebih tinggi

### Bear Market:
- ✅ **γ = 0 (Network Markowitz)** memberikan proteksi terbaik
- ✅ Kerugian lebih rendah vs benchmark
- ✅ VaR lebih rendah

### Overall:
- Sharpe ratio kompetitif dengan Classical Markowitz
- Risk management lebih baik dari Equal Weight
- Kombinasi γ = 0 (bear) dan γ > 0 (bull) optimal

## 🔬 Relevansi dengan Thesis

Paper ini adalah **SOTA baseline** untuk dibandingkan dengan strategi Anda:

| Aspek | Giudici et al. (2020) | Thesis Anda |
|-------|----------------------|-------------|
| **Filtering** | RMT + MST | RMT + MST ✅ |
| **Centrality** | Eigenvector | Network centrality ✅ |
| **Regime Detection** | ❌ Manual | ✅ **Momentum-based** |
| **Asset Selection** | All assets | ✅ **Graph filtering (MIS)** |
| **Adaptive Strategy** | ❌ Fixed γ | ✅ **Bull/Bear adaptive** |
| **Statistical Testing** | ❌ None | ✅ **Significance tests** |

### Kekuatan Thesis Anda:

1. **Automatic Regime Detection** - Momentum-based bull/bear classification
2. **Dynamic Asset Selection** - Graph-based filtering (MIS/Relaxed MIS)
3. **Adaptive Strategy** - Clustering (bull) vs Diversification (bear)
4. **Statistical Rigor** - Significance testing, hypothesis testing
5. **AI-Gated Approach** - Kombinasi momentum + graph + optimization

## 📚 References

**Main Paper:**
```
Giudici, P., Pagnottoni, P., & Polinesi, G. (2020). 
Network models to improve automated cryptocurrency portfolio management. 
Frontiers in Artificial Intelligence, 3, 22.
```

**Key Citations:**
- Markowitz (1952) - Portfolio Selection
- Mantegna (1999) - Hierarchical Structure in Financial Markets
- Marchenko & Pastur (1967) - Random Matrix Theory
- Tola et al. (2008) - Cluster Analysis for Portfolio Optimization

## 💡 Tips Penggunaan

1. **Eksplorasi γ values** - Test berbagai nilai untuk understand trade-off
2. **Compare dengan baseline** - Classical Markowitz, Equal Weight
3. **Analyze centrality** - Understand which assets have high systemic risk
4. **Visualize MST** - See network structure evolution
5. **Export results** - Save to Excel untuk analisis lebih lanjut

## 🔧 Requirements

```python
numpy
pandas
matplotlib
seaborn
scipy
networkx
openpyxl  # For Excel I/O
```

Install:
```bash
pip install numpy pandas matplotlib seaborn scipy networkx openpyxl
```

## 📝 Notes

- Data adalah **dummy/simulated** untuk pembelajaran
- Volatilitas disesuaikan dengan paper (Table 1)
- USDT sebagai stablecoin memiliki volatilitas rendah (0.01)
- Correlation structure realistis untuk crypto market
- Window size = 120 hari (4 bulan) seperti dalam paper

## 🎓 Learning Outcomes

Setelah menjalankan eksperimen ini, Anda akan memahami:

1. ✅ Bagaimana RMT memfilter noise dari correlation matrix
2. ✅ Cara MST menyederhanakan network structure
3. ✅ Interpretasi eigenvector centrality untuk systemic risk
4. ✅ Trade-off antara return dan risk dengan parameter γ
5. ✅ Perbedaan strategi untuk bull vs bear market
6. ✅ Implementasi praktis Network Markowitz

---

**Created:** 2026-02-13  
**Author:** Thesis Experiment  
**Purpose:** Understanding Network Markowitz methodology for cryptocurrency portfolio optimization
