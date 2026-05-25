# Studi Kasus: Cryptocurrency Portfolio Optimization

Studi kasus ini mendemonstrasikan aplikasi graf temporal di bidang keuangan, khususnya untuk manajemen portofolio aset kripto yang sangat volatil. Kita akan menggunakan analisis korelasi dinamis untuk memitigasi risiko.

## Problem Definition
Tujuannya adalah membangun portofolio yang tangguh (*robust*) terhadap guncangan pasar.
Teori Portofolio Modern (Markowitz) menyarankan diversifikasi. Namun, dalam pasar kripto, korelasi antar aset sering kali "bergeser" mendekati 1 saat pasar jatuh (*market crash*), membuat diversifikasi statis gagal.
Tantangannya: Mendeteksi klaster aset yang bergerak bersamaan secara real-time dan memilih aset perwakilan yang paling tidak berkorelasi.

## Temporal Correlation Graph Construction
Kita menggunakan data harga penutupan per jam dari 50 aset kripto teratas.
- **Nodes**: Aset kripto (BTC, ETH, SOL, dll).
- **Sliding Window Correlation**: Pada setiap jam $t$, kita menghitung matriks korelasi Pearson (atau Dynamic Time Warping) berdasarkan harga 24 jam terakhir.
- **Thresholding**: Sisi $(u, v)$ terbentuk jika korelasi $\rho_{uv}(t) > 0.7$. Ini menghasilkan serangkaian snapshot graf $G_1, G_2, \dots$.

## Strategy Implementation
Strategi investasi berbasis graf temporal (*Graph-Based Asset Selection*):
1. **Centrality Analysis**: Hitung *Eigenvector Centrality* untuk setiap aset pada $G_t$. Aset dengan sentralitas tinggi adalah "penggerak pasar" atau aset yang sangat terpengaruh oleh tren global.
2. **Independent Set Selection**: Temukan *Maximum Independent Set* (atau pendekatan heuristiknya) pada graf $G_t$. Himpunan ini berisi aset-aset yang saat ini tidak berkorelasi satu sama lain.
3. **Rebalancing**: Alokasikan modal hanya pada aset-aset di dalam Independent Set. Ulangi proses ini setiap jam atau setiap hari.

## Backtesting Results
Strategi ini diuji pada data pasar "Bear Market" 2022.
Hasil menunjukan bahwa portofolio berbasis graf temporal memiliki **Drawdown Maksimum** yang jauh lebih rendah (-30%) dibandingkan strategi *Buy-and-Hold* Bitcoin (-70%) atau Equal Weight (-65%). Hal ini karena saat korelasi pasar meningkat (tanda bahaya), graf menjadi sangat padat, sehingga ukuran *Independent Set* mengecil, yang secara otomatis memaksa portofolio untuk memegang lebih banyak uang tunai (jika kas dianggap node terisolasi) atau berkonsentrasi pada sedikit aset yang benar-benar unik.
