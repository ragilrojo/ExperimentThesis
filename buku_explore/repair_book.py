import re

path = r"g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\buku_explore\outline_buku_market_regime.tex"

with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find split points
start_marker = r"\chapter{Advanced Feature Engineering}"
end_marker = r"\chapter{Implementasi Sistem Real-time}"

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx == -1 or end_idx == -1:
    print(f"Error: Markers not found. Start: {start_idx}, End: {end_idx}")
    # Fallback to search for generic chapter if needed, but stick to specific for now.
    # Try searching for Case Studies if Impl not found
    if end_idx == -1:
        end_idx = content.find(r"\chapter{Studi Kasus: Saham}")
        print(f"Fallback End Index (Studi Kasus): {end_idx}")

part1 = content[:start_idx]
part3 = content[end_idx:]

print(f"Part 1 length: {len(part1)}")
print(f"Part 3 length: {len(part3)}")

# Define Part 2 (New Content for Ch 7 - 16)
part2 = r"""
\chapter{Advanced Feature Engineering}
Jika indikator teknikal adalah "kulit" pasar, maka fitur statistik dan mikrostruktur adalah "organ dalam"nya. Di bab ini, kita akan menggali fitur-fitur canggih yang jarang digunakan oleh trader ritel tetapi menjadi makanan sehari-hari algoritma institusi.

\section{Statistical Features}
Fitur statistik menangkap properti distribusi data yang tidak terlihat oleh mata telanjang pada grafik harga.

\subsection{Rolling Statistics (Mean, Variance)}
Pasar yang efisien seharusnya memiliki return dengan mean nol. Pergeseran mean (\textit{drift}) yang signifikan dalam jendela waktu rolling (misal 20 hari) adalah sinyal tren. Lebih penting lagi adalah \textit{Rolling Variance}. Lonjakan varians seringkali terjadi \textit{sebelum} harga jatuh tertembus. Kita menggunakan jendela waktu eksponensial (EWM) untuk memberi bobot lebih pada data terbaru.

\subsection{Higher Moments (Skewness & Kurtosis)}
Dua momen pertama (Mean, Varians) mengasumsikan distribusi Normal. Pasar tidak Normal.
\begin{itemize}
    \item \textbf{Skewness (Kemencengan)}: Mengukur asimetri distribusi return. Skewness negatif yang besar (ekor kiri panjang) menandakan risiko \textit{crash} yang tinggi ("Naga Hitam").
    \item \textbf{Kurtosis (Keruncingan)}: Mengukur ketebalan ekor (\textit{fat tails}). Kurtosis > 3 (Leptokurtic) berarti probabilitas kejadian ekstrem (sangat untung atau sangat rugi) jauh lebih tinggi daripada prediksi model Gaussian.
\end{itemize}

\subsection{Autocorrelation dan Serial Correlation}
Mengukur seberapa kuat harga hari ini dipengaruhi oleh harga kemarin.
\[ \rho_k = \frac{\sum_{t=k+1}^T (y_t - \bar{y})(y_{t-k} - \bar{y})}{\sum_{t=1}^T (y_t - \bar{y})^2} \]
Pasar yang efisien memiliki autokorelasi mendekati nol (Random Walk). Lonjakan autokorelasi positif menandakan tren (momentum), sedangkan negatif menandakan mean-reversion.

\subsection{Fractal Dimension (Hurst Exponent)}
Mengukur "kekasaran" grafik harga untuk menentukan sifat memori jangka panjang (Long-term Memory). Nilai Hurst Exponent ($H$):
\begin{itemize}
    \item $0.5 < H < 1.0$: Trending (Persistent). Sinyal untuk strategi Trend Following.
    \item $0.0 < H < 0.5$: Mean Reverting (Anti-persistent). Sinyal untuk strategi Reversion.
    \item $H = 0.5$: Random Walk.
\end{itemize}

\section{Microstructure Features}
Mengintip ke dalam mesin pencocokan order (matching engine).

\subsection{Bid-Ask Spread dynamics}
Spread bukan biaya statis. Spread yang melebar secara tiba-tiba adalah proksi ketakutan \textit{market maker}. Saat market maker mendeteksi adanya "Informed Trader", mereka melebarkan spread untuk melindungi diri. Ini sinyal volatilitas.

\subsection{Order Imbalance}
Rasio volume beli agresif vs volume jual agresif.
\[ \rho = \frac{V_{buy} - V_{sell}}{V_{buy} + V_{sell}} \]
Jika order beli mendominasi ($\rho > 0$) tetapi harga tidak naik, ini tanda \textit{absorption} (distribusi tersembunyi).

\subsection{Kyle's Lambda (Amihud Illiquidity)}
Mengukur \textit{market impact cost}: seberapa besar harga bergerak untuk setiap unit volume yang diperdagangkan ($ \Delta P / \Delta V $). Lonjakan Lambda adalah tanda bahaya kerapuhan pasar (*fragility*).

\section{Feature Selection Techniques}
\subsection{Correlation Clustering}
Mengelompokkan fitur statistik yang redundan dengan Hierarchical Clustering, lalu mengambil satu perwakilan dari setiap cluster.

\subsection{Feature Importance (MDI)}
Menggunakan Random Forest untuk mengukur *Mean Decrease Impurity*. Fitur yang paling banyak mengurangi entropi adalah fitur terbaik.

\part{Metodologi dan Algoritma}

\chapter{Regime Labeling}
Langkah paling kritis dalam supervised learning bukanlah modelnya, tapi labelnya ("Ground Truth").

\section{Metode Labeling Manual vs Data-Driven}
\subsection{Price-based Rules (Naive)}
"Jika harga di atas SMA200, label = Bull". Terlalu sederhana dan lagging.

\subsection{Drawdown-based Labeling}
Mendifinisikan rezim berdasarkan "rasa sakit" (Drawdown). Berguna untuk analisis risiko historis.

\section{Statistical Change Point Detection}
Mencari titik di mana properti statistik data berubah secara tiba-tiba menggunakan uji statistik (seperti CUSUM atau Chow Test).

\section{Hidden Markov Models (HMM)}
Metode emas (\textit{de facto standard}) untuk pemodelan rezim tanpa pengawasan (\textit{unsupervised}).

\subsection{Konsep Latent States}
HMM berasumsi bahwa pasar digerakkan oleh "mesin keadaan" (\textit{state machine}) yang tidak terlihat. Kita hanya bisa melihat Observasi ($O_t$: return, volatilitas), tapi tidak bisa melihat State ($S_t$: Bull, Bear).

\subsection{Matriks Transisi}
Probabilitas perpindahan antar state. Sifat "lengket" (\textit{sticky}) dari regim pasar dimodelkan di sini (peluang tetap di state yang sama tinggi).

\subsection{Gaussian Mixture Emissions}
Setiap state memancarkan data berdasarkan distribusi probabilitas yang unik.
\begin{itemize}
    \item \textbf{Bull}: Mean Positif, Varians Rendah.
    \item \textbf{Bear}: Mean Negatif, Varians Tinggi.
\end{itemize}

\section{Regime-Switching Models Lainnya}
Selain HMM, kita bisa menggunakan Threshold Auto-Regressive (TAR) atau Markov Switching Auto-Regressive (MSAR).

\chapter{Classification Models untuk Regime Prediction}
Once we have our labels (from HMM or Manual), we can train supervised classifiers to predict them out-of-sample.

\section{Traditional Machine Learning}
\subsection{Logistic Regression}
Model dasar yang memberikan probabilitas linear.

\subsection{Random Forest}
Ensemble dari Decision Trees. Tahan terhadap overfitting dan outlier. Bisa menangani hubungan non-linear antar fitur.

\subsection{Gradient Boosting (XGBoost, LightGBM)}
Model paling populer di kompetisi Kaggle. Sangat akurat tetapi perlu tuning hyperparameter yang hati-hati.

\subsection{Support Vector Machines (SVM)}
Mencari hyperplane pemisah optimal. Bagus untuk dataset kecil dengan dimensi tinggi.

\section{Handling Imbalanced Data}
Data "Crash" sangat jarang.
\begin{itemize}
    \item \textbf{SMOTE}: Oversampling kelas minoritas secara sintetis.
    \item \textbf{Class Weights}: Memberi penalti lebih besar jika salah menebak kelas minoritas.
\end{itemize}

\chapter{Deep Learning untuk Regime Detection}
\section{Feedforward Neural Networks (MLP)}
Jaringan saraf standar. Cocok untuk data tabular. Membutuhkan normalisasi input (Z-score) dan teknik regularisasi (Dropout).

\section{Convolutional Neural Networks (CNN)}
Biasanya untuk gambar, tapi efektif untuk time series 1D. Filter konvolusi bisa belajar mengenali pola lokal seperti "Head and Shoulders" atau lonjakan volatilitas pendek.

\section{Recurrent Neural Networks (RNN)}
\subsection{LSTM (Long Short-Term Memory)}
Dirancang untuk mengingat dependensi jangka panjang. LSTM memiliki "memory cell" yang bisa menyimpan informasi tren dari masa lalu.

\subsection{Attention Mechanisms dan Transformers}
\textit{Self-Attention} memungkinkan model untuk "fokus" pada titik-titik waktu tertentu di masa lalu yang paling relevan. Arsitektur \textit{Transformer} (seperti yang digunakan di LLM) kini mulai mendominasi time-series forecasting (misal: Temporal Fusion Transformer).

\chapter{Time Series Models}
\section{Classical Time Series}
\subsection{ARIMA}
Model parametrik klasik. Bagus untuk data stasioner tetapi gagal menangkap perubahan rezim yang tiba-tiba.

\subsection{GARCH untuk Volatility}
Standar industri untuk memprediksi volatilitas. GARCH memodelkan \textit{volatility clustering}.

\subsection{VAR (Vector Autoregression)}
Perluasan multivariate dari ARIMA. Melihat hubungan timbal balik antar beberapa aset.

\section{Modern Time Series ML}
\subsection{Prophet}
Library dari Facebook. Kuat menangani musiman dan liburan.
\subsection{N-BEATS}
Arsitektur Deep Learning murni untuk forecasting yang mengalahkan model statistika.

\chapter{Unsupervised Learning untuk Regime Discovery}
\section{Clustering Methods}
\subsection{K-Means}
Mempartisi data menjadi K kelompok berdasarkan kemiripan fitur (volatilitas, return).
\subsection{GMM (Gaussian Mixture Models)}
Versi probabilistik dari K-Means. Lebih fleksibel karena mengakomodasi varians yang berbeda tiap cluster.

\section{Dimensionality Reduction}
\subsection{PCA}
Komponen utama pertama seringkali merepresentasikan Beta Pasar.
\subsection{Autoencoders}
Jaringan saraf untuk kompresi data. \textit{Reconstruction Error} dari Autoencoder bisa digunakan sebagai sinyal anomali pasar (Crash Detector).

\chapter{Reinforcement Learning}
\section{Konsep RL untuk Trading}
RL adalah tentang belajar mengambil tindakan (\textit{Action}) untuk memaksimalkan keuntungan jangka panjang (\textit{Reward}).

\section{RL Algorithms}
\subsection{Deep Q-Networks (DQN)}
Menggunakan Neural Network untuk memperkirakan nilai setiap tindakan.
\subsection{PPO (Proximal Policy Optimization)}
Algoritma policy-gradient yang stabil dan menjadi standar industri saat ini untuk trading bot.

\section{RL untuk Regime-aware Trading}
Memasukkan "Label Rezim" (dari HMM) ke dalam state space agen RL, memberinya konteks makro untuk mengambil keputusan yang lebih bijak (misal: lebih konservatif saat Rezim Bear).

\part{Evaluasi dan Optimisasi}

\chapter{Evaluasi Model}
\section{Classification Metrics}
\subsection{Precision, Recall, F1}
Akurasi menyesatkan di finance. Recall (kemampuan mendeteksi Crash) seringkali lebih penting daripada Precision.
\subsection{Confusion Matrix}
Menganalisis tipe kesalahan: False Positive (Alarm Palsu) vs False Negative (Gagal Deteksi Bahaya).

\section{Financial Metrics}
\subsection{Sharpe Ratio & Sortino Ratio}
Mengukur return per unit risiko.
\subsection{Maximum Drawdown}
Resiko kerugian terbesar dari puncak ke lembah.

\section{Backtesting yang Benar}
\subsection{Walk-Forward Validation}
Melatih pada masa lalu, menguji pada masa depan, lalu menggeser jendela waktu.
\subsection{Purged Cross-Validation}
Menghapus data buffer di perbatasan train/test untuk mencegah kebocoran informasi.

\chapter{Model Optimization}
\section{Hyperparameter Tuning}
\subsection{Grid Search vs Random Search}
Random search seringkali lebih efisien.
\subsection{Bayesian Optimization (Optuna)}
Menggunakan probabilitas untuk menebak kombinasi parameter terbaik berikutnya, secara drastis mengurangi waktu pencarian.

\section{Feature Selection}
\subsection{Recursive Feature Elimination (RFE)}
Secara iteratif membuang fitur terlemah.
\subsection{SHAP Values}
Menggunakan Game Theory untuk menjelaskan kontribusi setiap fitur terhadap prediksi model (Explainable AI).

\chapter{Model Interpretability}
\section{Explainable AI (XAI)}
Regulasi menuntut penjelasan. Mengapa model menjual? 
\subsection{SHAP & LIME}
Visualisasi kontribusi fitur lokal dan global.
\subsection{Feature Importance Plot}
Penting untuk sanity check: Apakah fitur yang dipakai model masuk akal secara ekonomi?
"""

new_content = part1 + part2 + "\n" + part3

with open(path, 'w', encoding='utf-8') as f:
    f.write(new_content)

print("File successfully repaired and expanded.")
