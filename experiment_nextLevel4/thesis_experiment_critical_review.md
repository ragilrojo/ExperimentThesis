
# 🎓 Critical Review: Thesis Experiment (V62 Crypto Strategy)

**Date:** 2026-02-12
**Subject:** Evaluasi Akademik terhadap `crypto_strategy_v62_multiyear_comparison_v3_thesis_grade.ipynb`
**Reviewer:** AI Assistant (Acting as Thesis Supervisor/Examiner)

---

## 🟢 1. Kekuatan Eksperimen (Strong Points)
Argumen ini adalah "senjata" utama untuk mempertahankan validitas tesis:

1.  **Metodologi Testing yang Jujur (Split Period):**
    *   Pemisahan eksplisit antara **Training (2023-2024)** dan **Testing (2025)** adalah nilai plus terbesar. Ini menunjukkan pemahaman mendalam tentang **Data Snooping Bias**. Klaim bahwa hasil 2025 adalah performa *real-world* (Out-of-Sample) sangat kuat secara ilmiah.

2.  **Mekanisme Simulasi Realistis (Drift & Fees):**
    *   Implementasi **Realistic Drift** (bobot aset berubah harian mengikuti harga) dan **Transaction Costs (0.25%)** adalah syarat mutlak untuk validitas finansial. Tanpa ini, hasil backtest seringkali dianggap "halusinasi" akademis.

3.  **Filosofi Turnover Buffer:**
    *   Penggunaan `turnover_buffer=0.05` menunjukkan kedewasaan strategi. Tidak hanya mengejar sinyal matematika, tetapi juga memikirkan efisiensi biaya eksekusi. Ini adalah argumen kuat melawan kritik "strategi ini akan bangkrut karena biaya komisi".

4.  **Integrasi Multi-Disiplin (Novelty):**
    *   Menggabungkan **Machine Learning (XGBoost)** untuk *Market Timing* dengan **Graph Theory (NetworkX - MIS/Clustering)** untuk *Asset Selection* adalah kontribusi kebaruan (*novelty*) yang solid dan menarik secara teoritis.

---

## 🔴 2. Kritik Tajam & Celah (Vulnerabilities)
Bersiaplah menjawab pertanyaan-pertanyaan kritis ini di sidang:

### A. Isu "Look-Ahead Bias" pada Grafik Aset (Graph Construction)
*   **Kritik:** "Apakah struktur *Graph* (MIS/Cluster) yang dibangun di hari `t` benar-benar hanya menggunakan data `t-1` ke belakang? Atau ada kebocoran data masa depan?"
*   **Antisipasi:** Pastikan secara eksplisit di naskah bahwa kode menggunakan `window_rets` dengan indeks `loc_idx-g_lookback:loc_idx` (Backward-Looking). Tekankan ini di Metodologi.

### B. Stabilitas vs. Keberuntungan (Robustness Check)
*   **Kritik:** "Hasil Out-of-Sample hanya 1 tahun (2025). Di crypto, 1 tahun bisa saja 'beruntung' (Bull Run). Apakah strategi ini tahan banting di fase *Bear Market* atau *Sideways*?"
*   **Saran Perbaikan:** Lakukan **Sensitivity Analysis** sederhana. Ubah *Lookback Period* (misal: 45 vs 60 vs 90 hari). Jika performa hancur saat parameter diubah sedikit, strategi mungkin *Overfit*. Minimal, tampilkan analisis "Drawdown Duration".

### C. "Black Box" AI Explainability
*   **Kritik:** "Model XGBoost memprediksi rezim pasar. Apa sebenarnya yang dilihat oleh AI? Apakah Volatilitas? Momentum?"
*   **Saran Perbaikan:** Tambahkan visualisasi **Feature Importance** (`xgb.plot_importance`). Ini mengubah model dari *Black Box* menjadi *Grey Box* yang bisa dijelaskan (Explainable AI - XAI).

### D. Benchmark Comparison: Is Markowitz Enough?
*   **Kritik:** "Kenapa hanya dibandingkan dengan Static Markowitz? Di pasar Crypto yang tidak efisien, seringkali strategi naif seperti **Equal Weight (1/N)** atau **Bitcoin Buy & Hold** lebih sulit dikalahkan."
*   **Saran Perbaikan:** Tambahkan satu garis plot lagi di grafik hasil akhir: **BTC Buy & Hold**. Jika strategi Anda bisa mengalahkan BTC (terutama secara *Risk-Adjusted* / Sharpe Ratio), nilai tesis ini akan melonjak.

---

## 📋 3. Rekomendasi Aksi (Action Plan)

Untuk membuat eksperimen ini "Anti-Peluru" dan berpotensi **Cum Laude**:

1.  **Tambahkan Visualisasi Feature Importance:** Tampilkan grafik batang fitur mana (misal: `Vol_20`, `Mom_50`) yang paling dominan bagi keputusan AI.
2.  **Tambahkan Benchmark BTC/ETH:** Di grafik akhir 2025, masukkan kinerja aset tunggal BTC. Ini memberikan konteks pasar yang nyata bagi pembaca.
3.  **Narasi "Regime Identification":** Di pembahasan, jelaskan **"Kapan AI untung?"**. Tunjukkan tanggal spesifik di mana AI memutuskan 'CASH' (keluar pasar) tepat sebelum harga jatuh. Bukti kualitatif ini sangat meyakinkan.

---
*Created by AI Research Assistant for @[experiment_nextLevel4]*
