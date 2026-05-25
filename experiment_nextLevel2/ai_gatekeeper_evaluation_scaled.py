
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# Config
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)

print("Libraries loaded successfully!")

# Load Data
file_path = 'dataset_2023_2025.xlsx'
try:
    data = pd.read_excel(file_path, index_col=0, parse_dates=True)
    print(f"Loaded {len(data)} rows of data")
except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    exit(1)

# Calculate returns
returns = data.pct_change().dropna()
market_return = returns.mean(axis=1)

# Create features
features = pd.DataFrame(index=returns.index)
features['Vol_20'] = market_return.rolling(window=20).std()
features['Mom_20'] = market_return.rolling(window=20).mean()
features['Mom_50'] = market_return.rolling(window=50).mean()

# Binary labeling
features['Target'] = (market_return.shift(-1) > 0).astype(int)
features = features.dropna()

print("Feature shape:", features.shape)

# Split by year
train_mask = (features.index.year >= 2023) & (features.index.year <= 2024)
test_mask = (features.index.year == 2025)

X_train_raw = features.loc[train_mask, ['Vol_20', 'Mom_20', 'Mom_50']]
y_train = features.loc[train_mask, 'Target']
X_test_raw = features.loc[test_mask, ['Vol_20', 'Mom_20', 'Mom_50']]
y_test = features.loc[test_mask, 'Target']

print(f"Training set: {len(X_train_raw)} samples")
print(f"Testing set: {len(X_test_raw)} samples")

if len(X_train_raw) == 0:
    print("Error: Training set is empty.")
    exit(1)

# A. Normalisasi/Standardisasi
print("\nApplying StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# Convert back to DataFrame for easier handling
X_train = pd.DataFrame(X_train_scaled, index=X_train_raw.index, columns=X_train_raw.columns)
X_test = pd.DataFrame(X_test_scaled, index=X_test_raw.index, columns=X_test_raw.columns)

print("Features scaled successfully!")
print("Train Mean:\n", X_train.mean().round(4))
print("Train Std:\n", X_train.std().round(4))

# Train Optimized Random Forest
print("Training Random Forest on SCALED features...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Predictions
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)
y_test_proba = rf_model.predict_proba(X_test)[:, 1]

print("Model trained successfully!")

# Calculate metrics
def calculate_metrics(y_true, y_pred, dataset_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\n{'='*50}")
    print(f"{dataset_name} Performance")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1}

train_metrics = calculate_metrics(y_train, y_train_pred, "TRAINING SET (Scaled)")
test_metrics = calculate_metrics(y_test, y_test_pred, "TESTING SET (Scaled)")

try:
    roc_auc = roc_auc_score(y_test, y_test_proba)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
except Exception as e:
    print(f"\nROC-AUC Error: {e}")

print("\n=== Classification Report (Test Set) ===")
print(classification_report(y_test, y_test_pred, target_names=['Bearish', 'Bullish']))

# Confusion matrices
cm_test = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Bearish', 'Bullish'],
            yticklabels=['Bearish', 'Bullish'])
plt.title(f'Testing Set (Scaled Feats)\nAccuracy: {test_metrics["Accuracy"]:.2%}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('scaled_confusion_matrix.png')
print("Saved scaled_confusion_matrix.png")
