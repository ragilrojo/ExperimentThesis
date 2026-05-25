
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

print("Environment Ready for Naive Bayes Analysis!")

# 1. Data Selection (BTC-USD Focus)
# Load data
file_path = 'dataset_2023_2025.xlsx'
try:
    df = pd.read_excel(file_path, index_col=0, parse_dates=True)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    # Create dummy data for testing purposes if file is missing
    dates = pd.date_range(start='2023-01-01', end='2025-12-31')
    df = pd.DataFrame(index=dates)
    df['BTC-USD'] = 100 + np.cumsum(np.random.normal(0, 1, size=len(dates)))
    print("Created dummy data for testing.")

btc_series = df['BTC-USD']
btc_ret = btc_series.pct_change().fillna(0)

print(f"BTC Data Loaded. Range: {btc_series.index[0].date()} to {btc_series.index[-1].date()}")

# 2. Feature Engineering (BTC Specific)
features = pd.DataFrame(index=btc_ret.index)

# Fitur 1: BTC Volatility
features['BTC_Vol_20'] = btc_ret.rolling(20).std()

# Fitur 2: BTC Momentum (Short)
features['BTC_Mom_20'] = btc_ret.rolling(20).mean()

# Fitur 3: BTC Momentum (Long)
features['BTC_Mom_50'] = btc_ret.rolling(50).mean()

# Target: Binary (Bullish if next day return > 0.5%)
features['Target'] = np.where(btc_ret.shift(-1) > 0.005, 1, 0)

features = features.dropna()

print("BTC-Only Features Created.")
print(features.tail())

# 3. Train/Test Split (2025 Prediction)
train_idx = features.index.year < 2025
test_idx = features.index.year == 2025

X_train = features.loc[train_idx].drop(columns=['Target'])
y_train = features.loc[train_idx, 'Target']

X_test = features.loc[test_idx].drop(columns=['Target'])
y_test = features.loc[test_idx, 'Target']

# Scaling Features (Recommended for Gaussian Naive Bayes)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")

if len(X_train) == 0:
    print("Error: Training set is empty.")
    exit()

# 4. Modeling & Results (Naive Bayes)
# Initialize Gaussian Naive Bayes
nb_model = GaussianNB()

# Fit Model
nb_model.fit(X_train_scaled, y_train)

# Predict
y_pred = nb_model.predict(X_test_scaled)
y_proba = nb_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate
acc = accuracy_score(y_test, y_pred)
try:
    roc = roc_auc_score(y_test, y_proba)
except:
    roc = 0.5

print(f"\n--- Naive Bayes Results ---")
print(f"Accuracy: {acc:.2%}")
print(f"ROC-AUC : {roc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualization
plt.figure(figsize=(12, 5))

# Confusion Matrix
plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bearish', 'Bullish'], yticklabels=['Bearish', 'Bullish'])
plt.title('Confusion Matrix (Naive Bayes)')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Probability Distribution
plt.subplot(1, 2, 2)
sns.histplot(y_proba, kde=True, bins=20, color='skyblue')
plt.title('Prediction Probability Distribution')
plt.xlabel('Probability of Bullish')
plt.ylabel('Count')

plt.tight_layout()
plt.savefig('naive_bayes_results.png')
print("Saved naive_bayes_results.png")
