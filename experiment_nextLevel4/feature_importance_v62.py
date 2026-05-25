"""
Feature Importance Analysis untuk V62 Crypto Strategy
Menampilkan grafik batang untuk fitur-fitur yang paling dominan dalam keputusan AI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# Set style
plt.style.use('ggplot')
sns.set_palette("husl")

# Load data (sama seperti di notebook V62)
file_path = '../experiment_nextLevel2/dataset_2023_2025.xlsx'
data = pd.read_excel(file_path, index_col=0, parse_dates=True)
returns = data.pct_change().dropna()
market_return = returns.mean(axis=1)

# Prepare features (sama seperti di notebook V62)
features = pd.DataFrame(index=returns.index)
features['Vol_20'] = market_return.rolling(window=20).std()
features['Mom_20'] = market_return.rolling(window=20).mean()
features['Mom_50'] = market_return.rolling(window=50).mean()

# Prepare target
sma5 = market_return.rolling(window=5).mean()
sma20 = market_return.rolling(window=20).mean()
target = (sma5 > sma20).astype(int).shift(-5).reindex(features.index).fillna(0)

# Training data (2023-2024)
X_train = features.loc[features.index.year <= 2024].dropna()
y_train = target.loc[X_train.index]

# Train XGBoost model
SEED = 42
xgb_model = xgb.XGBClassifier(
    n_estimators=100, 
    learning_rate=0.05, 
    max_depth=5,
    random_state=SEED, 
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

# Get feature importance
feature_importance = xgb_model.feature_importances_
feature_names = X_train.columns

# Create DataFrame untuk plotting
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\n=== Feature Importance ===")
print(importance_df)
print(f"\nTotal importance: {importance_df['Importance'].sum():.4f}")

# Visualisasi 1: Bar chart horizontal
plt.figure(figsize=(10, 6))
colors = sns.color_palette("viridis", len(importance_df))
bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color=colors)

# Tambahkan nilai di ujung bar
for i, (idx, row) in enumerate(importance_df.iterrows()):
    plt.text(row['Importance'], i, f" {row['Importance']:.3f}", 
             va='center', fontsize=10, fontweight='bold')

plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.title('Feature Importance - AI Decision Factors\n(XGBoost Model V62)', 
          fontsize=14, fontweight='bold', pad=20)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance_horizontal.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualisasi 2: Bar chart vertikal dengan persentase
plt.figure(figsize=(10, 6))
importance_pct = (importance_df['Importance'] / importance_df['Importance'].sum() * 100)

bars = plt.bar(importance_df['Feature'], importance_pct, 
               color=colors, edgecolor='black', linewidth=1.5)

# Tambahkan nilai persentase di atas bar
for bar, pct in zip(bars, importance_pct):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{pct:.1f}%',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.ylabel('Importance (%)', fontsize=12, fontweight='bold')
plt.xlabel('Features', fontsize=12, fontweight='bold')
plt.title('Feature Importance Distribution\n(Percentage Contribution to AI Decisions)', 
          fontsize=14, fontweight='bold', pad=20)
plt.ylim(0, max(importance_pct) * 1.15)  # Tambah ruang untuk label
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance_vertical.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualisasi 3: Pie chart
plt.figure(figsize=(10, 8))
colors_pie = sns.color_palette("Set2", len(importance_df))
wedges, texts, autotexts = plt.pie(
    importance_df['Importance'], 
    labels=importance_df['Feature'],
    autopct='%1.1f%%',
    startangle=90,
    colors=colors_pie,
    textprops={'fontsize': 11, 'fontweight': 'bold'},
    explode=[0.05] * len(importance_df)  # Slight separation
)

# Make percentage text more visible
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')

plt.title('Feature Importance - Proportional View\n(AI Model V62)', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('feature_importance_pie.png', dpi=300, bbox_inches='tight')
plt.show()

# Analisis tambahan: Feature correlation dengan target
print("\n=== Feature Correlation with Target ===")
correlation_df = pd.DataFrame({
    'Feature': feature_names,
    'Correlation': [X_train[col].corr(y_train) for col in feature_names],
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print(correlation_df)

# Save results to Excel
with pd.ExcelWriter('feature_importance_analysis_v62.xlsx') as writer:
    importance_df.to_excel(writer, sheet_name='Feature_Importance', index=False)
    correlation_df.to_excel(writer, sheet_name='Feature_Correlation', index=False)

print("\n✓ Analysis completed!")
print("✓ Grafik disimpan sebagai PNG files")
print("✓ Results disimpan di feature_importance_analysis_v62.xlsx")
