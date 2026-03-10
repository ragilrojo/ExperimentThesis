import nbformat as nbf
import os

# Path to the notebook
nb_path = r'g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\guidiciti_reexperiment\strategy_comparison.ipynb'

# Read the notebook
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Define the new code for Figure 5
fig5_code = """# --- FIGURE 5: MST Dynamics ---

def calculate_rolling_mst_metrics(returns_df, window=120):
    \"\"\"Calculate rolling Max Link and Residuality from MSTs\"\"\"
    dates = returns_df.index[window:]
    max_links = []
    residualities = []
    
    for i in range(window, len(returns_df)):
        window_data = returns_df.iloc[i-window:i]
        corr = window_data.corr().values
        # Filter with RMT
        corr_filtered = apply_rmt_filter(corr, window, returns_df.shape[1])
        # Build MST
        mst_weights = build_mst(corr_filtered)
        
        # Max Link distance
        max_links.append(np.max(mst_weights))
        
        # Residuality (Sum of MST weights / Total number of assets)
        residuality = np.sum(mst_weights) / (returns_df.shape[1] - 1)
        residualities.append(residuality)
        
    return pd.DataFrame({
        'Max Link': max_links,
        'Residuality': residualities
    }, index=dates)

print("Calculating Figure 5 metrics (Rolling MST)...")
mst_dynamics = calculate_rolling_mst_metrics(df_returns, window=120)

# Plot Figure 5
fig, ax1 = plt.subplots(figsize=(12, 7))

# Max Link (Black line, left axis)
ax1.plot(mst_dynamics.index, mst_dynamics['Max Link'], color='black', label='max link')
ax1.set_xlabel('time')
ax1.set_ylabel('max link distance', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_ylim(1.0, 1.6)

# Residuality (Red line, right axis)
ax2 = ax1.twinx()
ax2.plot(mst_dynamics.index, mst_dynamics['Residuality'], color='red', label='residuality')
ax2.set_ylabel('residuality coefficient', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylim(0.1, 1.0)

# Formatting
plt.title('FIGURE 5 | MST thresholds and residuality coefficients')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + labels1, labels1 + labels2, loc='upper right')

plt.tight_layout()
# Set style and save
plt.style.use('seaborn-v0_8-darkgrid')
plt.savefig('figure_5_mst_dynamics.png', dpi=300)
plt.show()"""

# Create the new cell
new_cell = nbf.v4.new_code_cell(fig5_code)

# Find the right position (after helper functions, before backtesting)
insert_idx = -1
for i, cell in enumerate(nb.cells):
    if 'compute_eigenvector_centrality' in cell.source:
        insert_idx = i + 1
        break

if insert_idx != -1:
    nb.cells.insert(insert_idx, new_cell)
    # Write the notebook back
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Successfully added Figure 5 cell at index {insert_idx}")
else:
    print("Could not find the insertion point.")
