import json
import os
import sys

# Set output encoding to UTF-8 for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

path = r'g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\experiment_nextLevel2\ai_classifier_v18_hierarchical_hmm_gatekeeper.ipynb'

print(f"Opening notebook at: {path}")

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Kode baru yang ingin disisipkan
new_source = [
    "# Plotting the performance summary\n",
    "fig, axes = plt.subplots(2, 1, figsize=(15, 12))\n",
    "\n",
    "sns.barplot(ax=axes[0], x='Symbol', y='Precision', data=h_results_df.sort_values('Precision', ascending=False), palette='viridis')\n",
    "axes[0].set_title('Hierarchical System Precision per Symbol')\n",
    "axes[0].set_ylabel('Precision')\n",
    "axes[0].tick_params(axis='x', rotation=90)\n",
    "\n",
    "sns.barplot(ax=axes[1], x='Symbol', y='Signals', data=h_results_df.sort_values('Precision', ascending=False), palette='viridis')\n",
    "axes[1].set_title('Number of Signals per Symbol')\n",
    "axes[1].set_ylabel('Signals')\n",
    "axes[1].tick_params(axis='x', rotation=90)\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "# --- 5. Visual Comparison: Signal Quality (Confusion Matrix) ---\n",
    "# Kita fokus pada BTC-USD untuk melihat perbandingan kualitas sinyal secara detail\n",
    "\n",
    "symbol_to_plot = 'BTC-USD'\n",
    "if symbol_to_plot in test_data_map:\n",
    "    test_df = test_data_map[symbol_to_plot]\n",
    "    X_test_plot = test_df.drop('Target', axis=1)\n",
    "    y_test_plot = test_df['Target']\n",
    "\n",
    "    # 1. Global Filter (Layer 1)\n",
    "    asset_global_regime_plot = global_regime.reindex(X_test_plot.index, method='ffill').fillna(0).values\n",
    "    \n",
    "    # 2. Local Filter (Layer 2)\n",
    "    asset_ret_test_plot = data[symbol_to_plot].pct_change().reindex(X_test_plot.index)\n",
    "    X_local_test_plot = np.log1p(asset_ret_test_plot.fillna(0)).replace([np.inf, -np.inf], 0).values.reshape(-1, 1)\n",
    "    local_model_plot = local_hmm_map[symbol_to_plot]\n",
    "    bull_local_plot = np.argmax([local_model_plot.means_[i][0] for i in range(2)])\n",
    "    asset_local_regime_plot = (local_model_plot.predict(X_local_test_plot) == bull_local_plot).astype(int)\n",
    "\n",
    "    # 3. XGBoost Probabilities (Layer 3)\n",
    "    probs_plot = universal_xgb.predict_proba(X_test_plot)[:, 1]\n",
    "\n",
    "    # Generate Predictions for different levels\n",
    "    preds_std = (probs_plot >= 0.5).astype(int)\n",
    "    preds_high_prec = (probs_plot >= threshold).astype(int)\n",
    "    preds_hierarchical = (asset_global_regime_plot == 1) & (asset_local_regime_plot == 1) & (probs_plot >= threshold)\n",
    "    preds_hierarchical = preds_hierarchical.astype(int)\n",
    "\n",
    "    # Visualizing Perubahan Kualitas Sinyal\n",
    "    fig, ax = plt.subplots(1, 3, figsize=(18, 5))\n",
    "    \n",
    "    sns.heatmap(confusion_matrix(y_test_plot, preds_std), annot=True, fmt='d', cmap='Greens', ax=ax[0])\n",
    "    ax[0].set_title(f\"Standard Universal XGB\\n(T=0.5)\")\n",
    "    ax[0].set_xlabel(\"Predicted\")\n",
    "    ax[0].set_ylabel(\"Actual\")\n",
    "\n",
    "    sns.heatmap(confusion_matrix(y_test_plot, preds_high_prec), annot=True, fmt='d', cmap='YlGnBu', ax=ax[1])\n",
    "    ax[1].set_title(f\"V17 High-Prec Logic\\n(T={threshold})\")\n",
    "    ax[1].set_xlabel(\"Predicted\")\n",
    "    ax[1].set_ylabel(\"Actual\")\n",
    "\n",
    "    sns.heatmap(confusion_matrix(y_test_plot, preds_hierarchical), annot=True, fmt='d', cmap='Oranges', ax=ax[2])\n",
    "    ax[2].set_title(f\"V18 Hierarchical Gatekeeper\\n(Global + Local + XGB)\")\n",
    "    ax[2].set_xlabel(\"Predicted\")\n",
    "    ax[2].set_ylabel(\"Actual\")\n",
    "\n",
    "    plt.suptitle(f\"Visual Quality Comparison for {symbol_to_plot} (Test Set 2025)\", fontsize=16)\n",
    "    plt.tight_layout(rect=[0, 0.03, 1, 0.95])\n",
    "    plt.show()"
]

# Mencari cell dengan lebih teliti
found = False
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        # Cari berdasarkan teks unik di placeholder atau struktur awal
        if "Comparison: Single vs Hierarchical" in source_str or "Plotting the performance summary" in source_str:
            print(f"Target found at cell index {i}. Updating...")
            cell['source'] = new_source
            found = True
            break

if found:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
    print("Notebook updated successfully.")
else:
    print("Could not find the target cell. Printing cell contents for debugging:")
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            print(f"Cell {i} start: {repr(''.join(cell['source'])[:50])}")
