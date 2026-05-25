import json

def code(src): return {"cell_type":"code","execution_count":None,"id":"","metadata":{},"outputs":[],"source":src}
def md(src):   return {"cell_type":"markdown","id":"","metadata":{},"source":src}

# ══════════════════════════════════════════════════════════════════════════════
# CELL 0 – Imports
# ══════════════════════════════════════════════════════════════════════════════
c0 = code(
"import numpy as np\n"
"import pandas as pd\n"
"import matplotlib.pyplot as plt\n"
"import matplotlib.ticker as mticker\n"
"import seaborn as sns\n"
"import networkx as nx\n"
"from scipy import stats\n"
"from itertools import product\n"
"import warnings\n"
"warnings.filterwarnings('ignore')\n"
"\n"
"plt.style.use('seaborn-v0_8-darkgrid')\n"
"sns.set_palette('husl')\n"
"print('Libraries loaded.')"
)

# ══════════════════════════════════════════════════════════════════════════════
# CELL 1 – Load Data
# ══════════════════════════════════════════════════════════════════════════════
c1_md = md(
"## 1. Load Real Crypto Data\n\n"
"Sumber: `crypto_data_real.xlsx` — 9 aset cryptocurrency (log-returns harian).  \n"
"Periode: 2017-11-10 s/d 2019-10-17 (setelah semua koin aktif)."
)

c1 = code(
"df_raw = pd.read_excel('crypto_data_real.xlsx', sheet_name='Returns', index_col=0)\n"
"df_raw.index = pd.to_datetime(df_raw.index)\n"
"df_raw.sort_index(inplace=True)\n"
"df_raw = df_raw.drop(columns=['USDT'], errors='ignore')\n"
"mask = (df_raw != 0).all(axis=1)\n"
"df_returns = df_raw[mask].copy()\n"
"\n"
"print('=== DATASET ===')\n"
"print(f'Aset    : {df_returns.columns.tolist()}')\n"
"print(f'Periode : {df_returns.index.min().date()} s/d {df_returns.index.max().date()}')\n"
"print(f'Obs     : {len(df_returns)} hari')\n"
"print()\n"
"df_returns.describe().round(4)"
)

# ══════════════════════════════════════════════════════════════════════════════
# CELL 2 – Konsep Level 1
# ══════════════════════════════════════════════════════════════════════════════
c2_md = md(
"## 2. Konsep Level 1 ML: Learned Threshold (θ*)\n\n"
"### Graph Diversification Klasik (non-ML)\n"
"```\n"
"θ = 0.4 atau 0.5  ← ditentukan manual oleh peneliti\n"
"```\n\n"
"### Level 1 ML — Threshold Dipelajari dari Data\n"
"```\n"
"θ* = argmax Sharpe(θ)  ← dicari secara otomatis via rolling cross-validation\n"
"```\n\n"
"**Cara kerjanya per periode rebalancing:**\n"
"1. Ambil window training (120 hari)\n"
"2. Bagi: 90 hari pertama = **train**, 30 hari terakhir = **validasi**\n"
"3. Coba semua θ di kandidat {0.1, 0.2, 0.3, ..., 0.9}\n"
"4. Hitung Sharpe Ratio pada data validasi untuk setiap θ\n"
"5. Pilih **θ\\*** yang menghasilkan Sharpe tertinggi\n"
"6. Gunakan θ\\* untuk portofolio periode berikutnya\n\n"
"> Ini adalah bentuk ML paling sederhana: **hyperparameter optimization berbasis data**."
)

# ══════════════════════════════════════════════════════════════════════════════
# CELL 3 – Helper Functions
# ══════════════════════════════════════════════════════════════════════════════
c3_md = md("## 3. Fungsi Pembantu")

c3 = code(
"def build_graph_and_select(returns_window, theta):\n"
"    \"\"\"\n"
"    Bangun graph korelasi dan cari Maximum Independent Set.\n"
"    Returns: list nama aset yang dipilih.\n"
"    \"\"\"\n"
"    corr_mat = returns_window.corr()\n"
"    G = nx.Graph()\n"
"    # Urutkan aset dari return rata-rata tertinggi (preferensi pemilihan)\n"
"    assets = list(returns_window.mean().sort_values(ascending=False).index)\n"
"    G.add_nodes_from(assets)\n"
"    # Tambahkan edge jika korelasi > theta (artinya aset ini 'terlalu mirip')\n"
"    for i, a1 in enumerate(assets):\n"
"        for a2 in assets[i+1:]:\n"
"            if abs(corr_mat.loc[a1, a2]) > theta:\n"
"                G.add_edge(a1, a2)\n"
"    # Maximum Independent Set = aset yang paling 'tidak berkorelasi' satu sama lain\n"
"    selected = list(nx.approximation.maximum_independent_set(G))\n"
"    return selected if selected else assets   # fallback: semua aset\n"
"\n"
"\n"
"def compute_portfolio_return(returns_window, theta):\n"
"    \"\"\"\n"
"    Hitung portfolio return (equal-weighted ke aset terpilih MIS).\n"
"    Dipakai untuk evaluasi θ pada data validasi.\n"
"    \"\"\"\n"
"    selected = build_graph_and_select(returns_window, theta)\n"
"    n = returns_window.shape[1]\n"
"    all_assets = returns_window.columns.tolist()\n"
"    w = np.zeros(n)\n"
"    for a in selected:\n"
"        w[all_assets.index(a)] = 1.0 / len(selected)\n"
"    port_returns = returns_window.values @ w\n"
"    return port_returns\n"
"\n"
"\n"
"def sharpe_ratio(returns, annualize=True):\n"
"    \"\"\"Hitung Sharpe Ratio. Annualized jika annualize=True.\"\"\"\n"
"    if len(returns) < 2 or np.std(returns) == 0:\n"
"        return -999.0\n"
"    sr = np.mean(returns) / np.std(returns)\n"
"    return sr * np.sqrt(252) if annualize else sr\n"
"\n"
"\n"
"def calculate_max_drawdown(cumulative_returns):\n"
"    running_max = np.maximum.accumulate(cumulative_returns)\n"
"    dd = (cumulative_returns - running_max) / running_max\n"
"    return dd.min()\n"
"\n"
"\n"
"def diebold_mariano_test(r_a, r_b):\n"
"    \"\"\"Diebold-Mariano test: H0 = tidak ada perbedaan signifikan.\"\"\"\n"
"    d = r_a - r_b\n"
"    dm = np.mean(d) / (np.std(d) / np.sqrt(len(d))) if np.std(d) > 0 else 0\n"
"    pv = 2 * (1 - stats.norm.cdf(abs(dm)))\n"
"    return dm, pv\n"
"\n"
"\n"
"THETA_CANDIDATES = np.round(np.arange(0.1, 0.95, 0.05), 2)   # [0.10, 0.15, ..., 0.90]\n"
"print(f'Kandidat threshold (θ): {THETA_CANDIDATES}')\n"
"print('Fungsi pembantu siap.')"
)

# ══════════════════════════════════════════════════════════════════════════════
# CELL 4 – Penjelasan Rolling CV
# ══════════════════════════════════════════════════════════════════════════════
c4_md = md(
"## 4. Rolling Cross-Validation untuk Mencari θ*\n\n"
"```\n"
"  |<──── 120 hari window ────>|\n"
"  |<── 90 train ──>|<─ 30 val ─>|  → pilih θ*\n"
"                               |<─ 7 hari test ─>|\n"
"                                  ↑ pakai θ* di sini\n"
"```\n\n"
"Proses ini berulang setiap 7 hari (rolling). Setiap periode mendapat θ* yang berbeda — **adaptif terhadap kondisi pasar**."
)

# ══════════════════════════════════════════════════════════════════════════════
# CELL 5 – Core: Learn θ* per window
# ══════════════════════════════════════════════════════════════════════════════
c5_md = md("## 5. Implementasi: Belajar θ* Secara Otomatis")

c5 = code(
"def find_optimal_theta(train_data, val_data, theta_candidates):\n"
"    \"\"\"\n"
"    Cari θ* yang memaksimalkan Sharpe Ratio pada data validasi.\n"
"\n"
"    Parameters\n"
"    ----------\n"
"    train_data      : DataFrame returns untuk membangun graph\n"
"    val_data        : DataFrame returns untuk evaluasi θ\n"
"    theta_candidates: list/array nilai θ yang akan dicoba\n"
"\n"
"    Returns\n"
"    -------\n"
"    best_theta : float — θ terbaik\n"
"    scores     : dict  — {θ: sharpe_ratio}\n"
"    \"\"\"\n"
"    scores = {}\n"
"    for theta in theta_candidates:\n"
"        # Bangun graph dari data TRAIN, evaluasi return di data VALIDASI\n"
"        selected = build_graph_and_select(train_data, theta)\n"
"        all_assets = train_data.columns.tolist()\n"
"        n = train_data.shape[1]\n"
"        w = np.zeros(n)\n"
"        for a in selected:\n"
"            w[all_assets.index(a)] = 1.0 / len(selected)\n"
"        # Return portfolio pada periode validasi\n"
"        val_returns = val_data.values @ w\n"
"        scores[theta] = sharpe_ratio(val_returns, annualize=True)\n"
"    best_theta = max(scores, key=scores.get)\n"
"    return best_theta, scores\n"
"\n"
"\n"
"# ── Demo: lihat θ* pada satu window ─────────────────────────────────────────\n"
"WINDOW  = 120\n"
"VAL_LEN = 30\n"
"TRAIN_L = WINDOW - VAL_LEN   # 90 hari\n"
"\n"
"demo_train = df_returns.iloc[:TRAIN_L]\n"
"demo_val   = df_returns.iloc[TRAIN_L:WINDOW]\n"
"best_t, score_map = find_optimal_theta(demo_train, demo_val, THETA_CANDIDATES)\n"
"\n"
"print(f'Demo — window pertama:')\n"
"print(f'Train: {demo_train.index.min().date()} s/d {demo_train.index.max().date()} ({TRAIN_L} hari)')\n"
"print(f'Val  : {demo_val.index.min().date()} s/d {demo_val.index.max().date()} ({VAL_LEN} hari)')\n"
"print()\n"
"print('Sharpe per θ di validasi:')\n"
"for t, s in sorted(score_map.items()):\n"
"    marker = ' ← θ*' if t == best_t else ''\n"
"    print(f'  θ = {t:.2f} : Sharpe = {s:7.4f}{marker}')\n"
"print()\n"
"print(f'θ* optimal  : {best_t}')\n"
"print(f'Aset dipilih: {build_graph_and_select(demo_train, best_t)}')"
)

# ══════════════════════════════════════════════════════════════════════════════
# CELL 6 – Backtest 3 strategi
# ══════════════════════════════════════════════════════════════════════════════
c6_md = md(
"## 6. Backtesting: 3 Strategi Dibandingkan\n\n"
"| Strategi | Keterangan |\n"
"|---|---|\n"
"| `GD Fixed θ=0.4` | Baseline — threshold manual 0.4 |\n"
"| `GD Fixed θ=0.5` | Baseline — threshold manual 0.5 |\n"
"| `GD Learned θ*` | **Level 1 ML** — threshold dipelajari dari data |"
)

c6 = code(
"def backtest_gd(df_ret, theta_mode='fixed', theta_value=0.4,\n"
"                window_size=120, val_len=30, rebalance_freq=7,\n"
"                transaction_cost=0.001, theta_candidates=None):\n"
"    \"\"\"\n"
"    Backtest Graph Diversification.\n"
"\n"
"    theta_mode  : 'fixed'  → pakai theta_value konstan\n"
"                  'learned'→ cari θ* optimal via CV setiap rebalancing\n"
"    \"\"\"\n"
"    port_returns  = []\n"
"    dates_out     = []\n"
"    theta_history = []   # catat θ yang dipakai setiap periode\n"
"    n_assets      = df_ret.shape[1]\n"
"    all_assets    = df_ret.columns.tolist()\n"
"\n"
"    for i in range(window_size, len(df_ret), rebalance_freq):\n"
"        window_data = df_ret.iloc[i - window_size : i]\n"
"\n"
"        # ── Pilih θ ──────────────────────────────────────────────────────────\n"
"        if theta_mode == 'learned':\n"
"            train_data = window_data.iloc[:window_size - val_len]\n"
"            val_data   = window_data.iloc[window_size - val_len:]\n"
"            theta_used, _ = find_optimal_theta(train_data, val_data, theta_candidates)\n"
"        else:\n"
"            theta_used = theta_value\n"
"            train_data = window_data   # pakai semua window untuk bangun graph\n"
"\n"
"        theta_history.append(theta_used)\n"
"\n"
"        # ── Seleksi aset via MIS ─────────────────────────────────────────────\n"
"        selected = build_graph_and_select(train_data, theta_used)\n"
"        w = np.zeros(n_assets)\n"
"        for a in selected:\n"
"            w[all_assets.index(a)] = 1.0 / len(selected)\n"
"\n"
"        # ── Hitung return portfolio ──────────────────────────────────────────\n"
"        test_end  = min(i + rebalance_freq, len(df_ret))\n"
"        test_data = df_ret.iloc[i : test_end]\n"
"        for j in range(len(test_data)):\n"
"            dr = np.dot(w, test_data.iloc[j].values)\n"
"            if j == 0 and port_returns:\n"
"                dr -= transaction_cost\n"
"            port_returns.append(dr)\n"
"            dates_out.append(test_data.index[j])\n"
"\n"
"    cum_ret = np.cumprod(1 + np.array(port_returns))\n"
"    return {\n"
"        'returns'           : np.array(port_returns),\n"
"        'cumulative_returns': cum_ret,\n"
"        'dates'             : dates_out,\n"
"        'theta_history'     : theta_history,\n"
"    }\n"
"\n"
"\n"
"# ── Jalankan 3 backtest ───────────────────────────────────────────────────────\n"
"print('Running backtests...')\n"
"\n"
"print('  [1/3] GD Fixed θ=0.4 ...')\n"
"res_04 = backtest_gd(df_returns, theta_mode='fixed', theta_value=0.4)\n"
"\n"
"print('  [2/3] GD Fixed θ=0.5 ...')\n"
"res_05 = backtest_gd(df_returns, theta_mode='fixed', theta_value=0.5)\n"
"\n"
"print('  [3/3] GD Learned θ* (Level 1 ML) ...')\n"
"res_ml = backtest_gd(df_returns, theta_mode='learned',\n"
"                     theta_candidates=THETA_CANDIDATES, val_len=30)\n"
"\n"
"results = {\n"
"    'GD Fixed θ=0.4' : res_04,\n"
"    'GD Fixed θ=0.5' : res_05,\n"
"    'GD Learned θ*'  : res_ml,\n"
"}\n"
"print('\\nSelesai!')"
)

# ══════════════════════════════════════════════════════════════════════════════
# CELL 7 – Metrics
# ══════════════════════════════════════════════════════════════════════════════
c7_md = md("## 7. Metrik Performa")

c7 = code(
"def full_metrics(name, res):\n"
"    r  = res['returns']\n"
"    cr = res['cumulative_returns']\n"
"    ann_r = np.mean(r) * 252\n"
"    ann_v = np.std(r)  * np.sqrt(252)\n"
"    sharpe = ann_r / ann_v if ann_v > 0 else 0\n"
"    mdd    = calculate_max_drawdown(cr)\n"
"    calmar = ann_r / abs(mdd) if mdd != 0 else 0\n"
"    var95  = np.percentile(r, 5)\n"
"    return {\n"
"        'Strategi'            : name,\n"
"        'Total Return (%)'    : round((cr[-1]-1)*100, 2),\n"
"        'Annual Return (%)'   : round(ann_r*100, 2),\n"
"        'Annual Vol (%)'      : round(ann_v*100, 2),\n"
"        'Sharpe Ratio'        : round(sharpe, 4),\n"
"        'Max Drawdown (%)'    : round(mdd*100, 2),\n"
"        'Calmar Ratio'        : round(calmar, 4),\n"
"        'VaR 95% (%)'         : round(var95*100, 4),\n"
"    }\n"
"\n"
"metrics_df = pd.DataFrame([full_metrics(n,r) for n,r in results.items()])\n"
"metrics_df = metrics_df.set_index('Strategi')\n"
"\n"
"# Highlight terbaik per kolom\n"
"def highlight(col):\n"
"    low_better = col.name in ['Annual Vol (%)', 'Max Drawdown (%)', 'VaR 95% (%)']\n"
"    best = col.min() if low_better else col.max()\n"
"    return ['background-color:#d4edda;font-weight:bold' if v==best else '' for v in col]\n"
"\n"
"print('=== METRIK PERFORMA ===' )\n"
"print(metrics_df.to_string())\n"
"print()\n"
"metrics_df.style.apply(highlight).format('{:.4f}')"
)

# ══════════════════════════════════════════════════════════════════════════════
# CELL 8 – DM Test
# ══════════════════════════════════════════════════════════════════════════════
c8_md = md(
"## 8. Uji Statistik Diebold-Mariano\n\n"
"Menguji apakah perbedaan performa antar strategi **signifikan secara statistik** atau hanya kebetulan."
)

c8 = code(
"names  = list(results.keys())\n"
"pairs  = [\n"
"    ('GD Learned θ*', 'GD Fixed θ=0.4'),\n"
"    ('GD Learned θ*', 'GD Fixed θ=0.5'),\n"
"    ('GD Fixed θ=0.5', 'GD Fixed θ=0.4'),\n"
"]\n"
"\n"
"print('DIEBOLD-MARIANO TEST')\n"
"print('='*70)\n"
"print(f'{\"Strategi A\":<22} vs {\"Strategi B\":<22} {\"DM Stat\":>9} {\"P-Value\":>9} {\"Ket\":>14}')\n"
"print('-'*70)\n"
"for a, b in pairs:\n"
"    dm, pv = diebold_mariano_test(results[a]['returns'], results[b]['returns'])\n"
"    sig = 'SIGNIFIKAN*' if pv < 0.05 else 'Tidak Sig.'\n"
"    arah = ''\n"
"    if pv < 0.05:\n"
"        arah = f'(A menang)' if dm > 0 else '(B menang)'\n"
"    print(f'{a:<22} vs {b:<22} {dm:>9.4f} {pv:>9.4f} {sig:>14} {arah}')\n"
"print('='*70)\n"
"print('* signifikan pada alpha = 0.05')"
)

# ══════════════════════════════════════════════════════════════════════════════
# CELL 9 – θ* History
# ══════════════════════════════════════════════════════════════════════════════
c9_md = md(
"## 9. Evolusi θ* Sepanjang Waktu\n\n"
"Salah satu keunggulan Level 1 ML adalah kemampuan **adaptasi** — threshold berubah mengikuti kondisi pasar."
)

c9 = code(
"theta_hist = res_ml['theta_history']\n"
"# Buat timeline tanggal untuk setiap rebalancing point\n"
"rebal_dates = []\n"
"for i in range(120, len(df_returns), 7):\n"
"    rebal_dates.append(df_returns.index[i])\n"
"rebal_dates = rebal_dates[:len(theta_hist)]\n"
"\n"
"fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=False)\n"
"\n"
"# ── Plot 1: Distribusi θ* ───────────────────────────────────────────────────\n"
"ax0 = axes[0]\n"
"unique, counts = np.unique(theta_hist, return_counts=True)\n"
"bars = ax0.bar(unique, counts, width=0.04, color='#457b9d', alpha=0.85, edgecolor='white')\n"
"ax0.axvline(0.4, color='#f4a261', linewidth=2, linestyle='--', label='θ=0.4 (fixed)')\n"
"ax0.axvline(0.5, color='#e76f51', linewidth=2, linestyle='--', label='θ=0.5 (fixed)')\n"
"ax0.axvline(np.mean(theta_hist), color='#2d6a4f', linewidth=2.5,\n"
"            linestyle='-', label=f'Rata-rata θ* = {np.mean(theta_hist):.3f}')\n"
"ax0.set_xlabel('Nilai Threshold (θ)')\n"
"ax0.set_ylabel('Frekuensi dipilih')\n"
"ax0.set_title('Distribusi θ* yang Dipilih oleh Model (Level 1 ML)', fontsize=12)\n"
"ax0.legend(fontsize=9)\n"
"ax0.grid(True, alpha=0.3)\n"
"\n"
"# ── Plot 2: θ* dari waktu ke waktu ──────────────────────────────────────────\n"
"ax1 = axes[1]\n"
"ax1.plot(rebal_dates, theta_hist, color='#457b9d', linewidth=1.5,\n"
"         marker='o', markersize=3, alpha=0.7, label='θ* per rebalancing')\n"
"ax1.plot(rebal_dates,\n"
"         pd.Series(theta_hist).rolling(10).mean().values,\n"
"         color='#e63946', linewidth=2.5, label='Moving avg (10 periode)')\n"
"ax1.axhline(0.4, color='#f4a261', linewidth=1.5, linestyle='--', alpha=0.7, label='θ=0.4 (fixed)')\n"
"ax1.axhline(0.5, color='#e76f51', linewidth=1.5, linestyle='--', alpha=0.7, label='θ=0.5 (fixed)')\n"
"ax1.set_xlabel('Tanggal')\n"
"ax1.set_ylabel('θ* Optimal')\n"
"ax1.set_title('Evolusi θ* Sepanjang Waktu (Adaptif terhadap Kondisi Pasar)', fontsize=12)\n"
"ax1.legend(fontsize=9)\n"
"ax1.grid(True, alpha=0.3)\n"
"# Shading bear market\n"
"ax1.axvspan(pd.Timestamp('2018-01-01'), pd.Timestamp('2018-12-31'),\n"
"            alpha=0.08, color='red', label='Bear 2018')\n"
"\n"
"plt.tight_layout()\n"
"plt.savefig('theta_evolution.png', dpi=150, bbox_inches='tight')\n"
"plt.show()\n"
"\n"
"print(f'Statistik θ*:')\n"
"print(f'  Rata-rata : {np.mean(theta_hist):.4f}')\n"
"print(f'  Std Dev   : {np.std(theta_hist):.4f}')\n"
"print(f'  Min       : {np.min(theta_hist):.2f}')\n"
"print(f'  Max       : {np.max(theta_hist):.2f}')\n"
"print(f'  Modus     : {unique[np.argmax(counts)]:.2f} (dipilih {counts.max()}x)')"
)

# ══════════════════════════════════════════════════════════════════════════════
# CELL 10 – Cumulative Returns
# ══════════════════════════════════════════════════════════════════════════════
c10_md = md("## 10. Perbandingan Cumulative Returns & Drawdown")

c10 = code(
"colors = {\n"
"    'GD Fixed θ=0.4' : '#f4a261',\n"
"    'GD Fixed θ=0.5' : '#e76f51',\n"
"    'GD Learned θ*'  : '#2d6a4f',\n"
"}\n"
"lws = {\n"
"    'GD Fixed θ=0.4' : 1.5,\n"
"    'GD Fixed θ=0.5' : 1.5,\n"
"    'GD Learned θ*'  : 2.8,\n"
"}\n"
"\n"
"fig, axes = plt.subplots(2, 1, figsize=(14, 10))\n"
"\n"
"# ── Plot 1: Cumulative Returns ───────────────────────────────────────────────\n"
"ax1 = axes[0]\n"
"for name, res in results.items():\n"
"    dates = pd.to_datetime(res['dates'])\n"
"    ax1.plot(dates, res['cumulative_returns'],\n"
"             label=name, color=colors[name], linewidth=lws[name])\n"
"\n"
"ax1.axvspan(pd.Timestamp('2018-01-01'), pd.Timestamp('2018-12-31'),\n"
"            alpha=0.07, color='red')\n"
"ax1.axvspan(pd.Timestamp('2019-01-01'), pd.Timestamp('2019-10-17'),\n"
"            alpha=0.07, color='green')\n"
"ax1.axhline(1.0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)\n"
"ax1.set_title('Cumulative Returns: Graph Diversification\\n'\n"
"              'Fixed Threshold vs. Learned Threshold (Level 1 ML)', fontsize=13)\n"
"ax1.set_ylabel('Cumulative Return')\n"
"ax1.legend(fontsize=10)\n"
"ax1.grid(True, alpha=0.3)\n"
"ax1.text(pd.Timestamp('2018-06-01'), ax1.get_ylim()[0]*1.02,\n"
"         'Bear Market 2018', ha='center', color='red', alpha=0.6, fontsize=9)\n"
"ax1.text(pd.Timestamp('2019-05-01'), ax1.get_ylim()[0]*1.02,\n"
"         'Recovery', ha='center', color='green', alpha=0.6, fontsize=9)\n"
"\n"
"# ── Plot 2: Drawdown ─────────────────────────────────────────────────────────\n"
"ax2 = axes[1]\n"
"for name, res in results.items():\n"
"    cr    = res['cumulative_returns']\n"
"    dates = pd.to_datetime(res['dates'])\n"
"    rmax  = np.maximum.accumulate(cr)\n"
"    dd    = (cr - rmax) / rmax * 100\n"
"    ax2.plot(dates, dd, label=name, color=colors[name], linewidth=lws[name])\n"
"\n"
"ax2.axhline(0, color='black', linewidth=0.8)\n"
"ax2.set_title('Drawdown (%)', fontsize=13)\n"
"ax2.set_ylabel('Drawdown (%)')\n"
"ax2.set_xlabel('Tanggal')\n"
"ax2.legend(fontsize=10)\n"
"ax2.grid(True, alpha=0.3)\n"
"\n"
"plt.tight_layout()\n"
"plt.savefig('gd_comparison.png', dpi=150, bbox_inches='tight')\n"
"plt.show()\n"
"print('Saved: gd_comparison.png')"
)

# ══════════════════════════════════════════════════════════════════════════════
# CELL 11 – Sub-period Sharpe
# ══════════════════════════════════════════════════════════════════════════════
c11_md = md("## 11. Analisis per Fase Pasar (Bear / Recovery)")

c11 = code(
"def sharpe_period(res, start, end):\n"
"    dates = pd.to_datetime(res['dates'])\n"
"    mask  = (dates >= start) & (dates <= end)\n"
"    r_sub = res['returns'][mask]\n"
"    if len(r_sub) < 5: return np.nan\n"
"    return sharpe_ratio(r_sub)\n"
"\n"
"phases = [\n"
"    ('Bear (2018)',       '2018-01-01', '2018-12-31'),\n"
"    ('Recovery (2019)',   '2019-01-01', '2019-10-17'),\n"
"    ('Full Period',       df_returns.index.min().strftime('%Y-%m-%d'),\n"
"                          df_returns.index.max().strftime('%Y-%m-%d')),\n"
"]\n"
"\n"
"rows = []\n"
"for name, res in results.items():\n"
"    row = {'Strategi': name}\n"
"    for ph_name, s, e in phases:\n"
"        row[ph_name] = round(sharpe_period(res, s, e), 4)\n"
"    rows.append(row)\n"
"\n"
"phase_df = pd.DataFrame(rows).set_index('Strategi')\n"
"print('SHARPE RATIO PER FASE PASAR')\n"
"print('='*60)\n"
"print(phase_df.to_string())\n"
"print('='*60)\n"
"\n"
"# Bar chart\n"
"fig, ax = plt.subplots(figsize=(10, 5))\n"
"x  = np.arange(len(phase_df))\n"
"bw = 0.25\n"
"ph_colors = ['#e63946', '#2d6a4f', '#457b9d']\n"
"for i, (ph, col) in enumerate(zip(phase_df.columns, ph_colors)):\n"
"    vals = phase_df[ph].values\n"
"    b = ax.bar(x + i*bw, vals, bw, label=ph, color=col, alpha=0.85)\n"
"    for rect, v in zip(b, vals):\n"
"        ax.text(rect.get_x()+rect.get_width()/2,\n"
"                rect.get_height() + (0.02 if v>=0 else -0.08),\n"
"                f'{v:.3f}', ha='center', va='bottom', fontsize=8)\n"
"ax.set_xticks(x + bw)\n"
"ax.set_xticklabels(phase_df.index, fontsize=10)\n"
"ax.axhline(0, color='black', linewidth=0.8)\n"
"ax.set_title('Sharpe Ratio per Fase Pasar\\nGD Fixed vs. GD Learned θ*', fontsize=12)\n"
"ax.set_ylabel('Sharpe Ratio')\n"
"ax.legend()\n"
"ax.grid(axis='y', alpha=0.3)\n"
"plt.tight_layout()\n"
"plt.savefig('sharpe_by_phase.png', dpi=150, bbox_inches='tight')\n"
"plt.show()"
)

# ══════════════════════════════════════════════════════════════════════════════
# CELL 12 – Interpretasi
# ══════════════════════════════════════════════════════════════════════════════
c12_md = md(
"## 12. Interpretasi & Narasi Tesis\n\n"
"### Mengapa θ* Adaptif adalah Kontribusi ML yang Valid?\n\n"
"Graph Diversification klasik mengasumsikan bahwa threshold korelasi **konstan sepanjang waktu** — ini adalah asumsi yang tidak realistis, terutama pada pasar cryptocurrency yang sangat volatile.\n\n"
"Dengan Level 1 ML (learned threshold):\n"
"- Pada **bear market** (volatilitas tinggi): model cenderung memilih θ yang **lebih rendah** → seleksi MIS lebih longgar → lebih banyak aset → diversifikasi lebih lebar untuk proteksi risiko\n"
"- Pada **bull/recovery** (volatilitas rendah): model cenderung memilih θ yang **lebih tinggi** → seleksi MIS lebih ketat → aset dengan korelasi rendah yang benar-benar dikecualikan\n\n"
"### Template Narasi untuk Tesis\n\n"
"> *\"Penelitian ini mengembangkan Graph Diversification dengan pendekatan data-driven dalam penentuan threshold korelasi (θ). Berbeda dari konfigurasi manual yang lazim digunakan (θ = 0.4 atau 0.5), model yang diusulkan mengoptimalkan θ secara otomatis menggunakan rolling cross-validation pada setiap periode rebalancing. Hasil eksperimen menunjukkan bahwa threshold yang dipelajari secara adaptif menghasilkan [masukkan temuan aktual — lebih baik/kompetitif/lebih stabil]. Lebih jauh, analisis evolusi θ* mengungkapkan pola yang konsisten dengan dinamika pasar: threshold cenderung [naik/turun] selama periode [bear/recovery], yang mengindikasikan bahwa model berhasil menangkap perubahan struktur korelasi antar aset kripto.\"*"
)

c12 = code(
"# Ringkasan statistik final\n"
"print('RINGKASAN EKSPERIMEN')\n"
"print('='*65)\n"
"print(f'Dataset         : 9 crypto assets, {len(df_returns)} hari')\n"
"print(f'Periode         : {df_returns.index.min().date()} s/d {df_returns.index.max().date()}')\n"
"print(f'Window          : 120 hari (90 train + 30 validasi)')\n"
"print(f'Rebalancing     : Setiap 7 hari')\n"
"print(f'Kandidat θ      : {list(THETA_CANDIDATES)}')\n"
"print(f'θ* rata-rata    : {np.mean(res_ml[\"theta_history\"]):.4f}')\n"
"print()\n"
"print('METRIK UTAMA:')\n"
"print(metrics_df[['Sharpe Ratio','Max Drawdown (%)','Calmar Ratio']].to_string())\n"
"print()\n"
"\n"
"# Kesimpulan otomatis\n"
"best_sharpe  = metrics_df['Sharpe Ratio'].idxmax()\n"
"best_calmar  = metrics_df['Calmar Ratio'].idxmax()\n"
"best_drawdown= metrics_df['Max Drawdown (%)'].idxmax()  # least negative\n"
"print(f'Sharpe terbaik : {best_sharpe} ({metrics_df.loc[best_sharpe,\"Sharpe Ratio\"]:.4f})')\n"
"print(f'Calmar terbaik : {best_calmar} ({metrics_df.loc[best_calmar,\"Calmar Ratio\"]:.4f})')\n"
"print('='*65)"
)

# ══════════════════════════════════════════════════════════════════════════════
# Assemble
# ══════════════════════════════════════════════════════════════════════════════
cells = [
    c0,
    c1_md, c1,
    c2_md,
    c3_md, c3,
    c4_md,
    c5_md, c5,
    c6_md, c6,
    c7_md, c7,
    c8_md, c8,
    c9_md, c9,
    c10_md, c10,
    c11_md, c11,
    c12_md, c12,
]

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.9.0"}
    },
    "cells": cells
}

out = 'graph_diversification_level1_ml.ipynb'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Notebook dibuat: {out}')
print(f'Total cells    : {len(cells)}')
