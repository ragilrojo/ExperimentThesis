
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
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

print("✅ Libraries loaded successfully!")

# Load Data
file_path = 'dataset_2023_2025.xlsx'
try:
    data = pd.read_excel(file_path, index_col=0, parse_dates=True)
    print(f"📊 Loaded {len(data)} rows of data")
except FileNotFoundError:
    print(f"❌ Error: File '{file_path}' not found.")
    exit(1)

# Calculate returns
returns = data.pct_change().dropna()
market_return = returns.mean(axis=1)

# Create features
features = pd.DataFrame(index=returns.index)
features['Vol_20'] = market_return.rolling(window=20).std()
features['Mom_20'] = market_return.rolling(window=20).mean()
features['Mom_50'] = market_return.rolling(window=50).mean()

# Binary labeling: 0 = Bearish, 1 = Bullish
features['Target'] = (market_return.shift(-1) > 0).astype(int)
features = features.dropna()

print("Feature shape:", features.shape)

# Split by year
train_mask = (features.index.year >= 2023) & (features.index.year <= 2024)
test_mask = (features.index.year == 2025)

X_train = features.loc[train_mask, ['Vol_20', 'Mom_20', 'Mom_50']]
y_train = features.loc[train_mask, 'Target']
X_test = features.loc[test_mask, ['Vol_20', 'Mom_20', 'Mom_50']]
y_test = features.loc[test_mask, 'Target']

print(f"Training set: {len(X_train)} samples")
print(f"Testing set: {len(X_test)} samples")

if len(X_train) == 0:
    print("❌ Error: Training set is empty.")
    exit(1)
if len(X_test) == 0:
    print("❌ Error: Testing set is empty.")
    exit(1)

# Train Optimized Random Forest
print("🔄 Training Optimized Random Forest Classifier...")

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

print("✅ Model trained successfully!")

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

# Training metrics
train_metrics = calculate_metrics(y_train, y_train_pred, "TRAINING SET")

# Testing metrics
test_metrics = calculate_metrics(y_test, y_test_pred, "TESTING SET")

# ROC-AUC
try:
    roc_auc = roc_auc_score(y_test, y_test_proba)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
except Exception as e:
    print(f"\nROC-AUC could not be calculated: {e}")

print("\n=== Detailed Classification Report (Test Set) ===")
print(classification_report(y_test, y_test_pred, target_names=['Bearish', 'Bullish']))

# Confusion matrices
cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Training
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Bearish', 'Bullish'],
            yticklabels=['Bearish', 'Bullish'],
            ax=axes[0])
axes[0].set_title(f'Training Set (Optimized)\nAccuracy: {train_metrics["Accuracy"]:.2%}')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# Testing
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Bearish', 'Bullish'],
            yticklabels=['Bearish', 'Bullish'],
            ax=axes[1])
axes[1].set_title(f'Testing Set (Optimized)\nAccuracy: {test_metrics["Accuracy"]:.2%}')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig('optimized_confusion_matrix.png')
print("Saved optimized_confusion_matrix.png")

# Feature Importance
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X_train.columns

plt.figure(figsize=(10, 6))
plt.title('Feature Importances (Optimized Model)')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), [feature_names[i] for i in indices])
plt.tight_layout()
plt.savefig('optimized_feature_importance.png')
print("Saved optimized_feature_importance.png")
