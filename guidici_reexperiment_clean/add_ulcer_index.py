import re

file_path = r'g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\guidici_reexperiment_clean\RLNetworkMarkowitz_thesis_CVaR.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update FOUR_METRICS to EVAL_METRICS
content = content.replace("FOUR_METRICS = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Max Drawdown']", 
                          "EVAL_METRICS = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Max Drawdown', 'Ulcer Index']")
content = content.replace("for m in FOUR_METRICS:", "for m in EVAL_METRICS:")
content = content.replace("enumerate(FOUR_METRICS):", "enumerate(EVAL_METRICS):")

# 2. Add calculate_ulcer_index
ulcer_code = """
def calculate_ulcer_index(ret_series):
    arr = np.array(ret_series)
    if len(arr) == 0: return 0.0
    cumulative = (1 + arr).cumprod()
    peak = np.maximum.accumulate(cumulative)
    dd = (cumulative - peak) / peak
    return float(np.sqrt(np.mean(dd**2)))
"""
cvar_code = "    return float(-np.mean(tail_losses)) if len(tail_losses) > 0 else 0.0\n"
content = content.replace(cvar_code, cvar_code + ulcer_code)

# 3. Add to calculate_all_metrics dict
content = content.replace("'Max Drawdown'    : max_dd,", "'Max Drawdown'    : max_dd,\n        'Ulcer Index'     : calculate_ulcer_index(arr),")

# 4. Update GridSpec and Titles
content = content.replace("gs = gridspec.GridSpec(5, 4, height_ratios=[0.5, 1.5, 1.5, 1.5, 1.0])", 
                          "gs = gridspec.GridSpec(5, 5, height_ratios=[0.5, 1.5, 1.5, 1.5, 1.0])")
content = content.replace("Evaluasi 4 Metrik", "Evaluasi 5 Metrik")
content = content.replace("Rangkuman 4 Metrik", "Rangkuman 5 Metrik")
content = content.replace("INTERPRETASI ABLATION STUDY - 4 METRIK TESIS:", "INTERPRETASI ABLATION STUDY - 5 METRIK TESIS:")

# 5. Update columns
old_cols = "columns = ['Experiment', 'Features', 'Obs\\nDim', 'Sharpe (↑)\\n'+period_name.capitalize(), 'Sortino (↑)\\n'+period_name.capitalize(), 'Calmar (↑)\\n'+period_name.capitalize(), 'Max Drawdown\\n(Mendekati 0)\\n'+period_name.capitalize(), 'Rank']"
new_cols = "columns = ['Experiment', 'Features', 'Obs\\nDim', 'Sharpe (↑)\\n'+period_name.capitalize(), 'Sortino (↑)\\n'+period_name.capitalize(), 'Calmar (↑)\\n'+period_name.capitalize(), 'Max Drawdown\\n(Mendekati 0)\\n'+period_name.capitalize(), 'Ulcer Index\\n(Mendekati 0)\\n'+period_name.capitalize(), 'Rank']"
content = content.replace(old_cols, new_cols)

# 6. Update table row formatting
old_row = """            f"{row['Max Drawdown Mean']:.4f}±{row['Max Drawdown Std']:.4f}" if row['Max Drawdown Std']>1e-4 else f"{row['Max Drawdown Mean']:.4f}",
            f"#{i+1}"
        ]"""
new_row = """            f"{row['Max Drawdown Mean']:.4f}±{row['Max Drawdown Std']:.4f}" if row['Max Drawdown Std']>1e-4 else f"{row['Max Drawdown Mean']:.4f}",
            f"{row['Ulcer Index Mean']:.4f}±{row['Ulcer Index Std']:.4f}" if row['Ulcer Index Std']>1e-4 else f"{row['Ulcer Index Mean']:.4f}",
            f"#{i+1}"
        ]"""
content = content.replace(old_row, new_row)

# 7. Update subplot spans
content = content.replace("ax_interp = fig.add_subplot(gs[4, 2:4])", "ax_interp = fig.add_subplot(gs[4, 2:5])")

# 8. Add Interpretation text for Ulcer Index
old_interp = """    best_mdd_exp = df_metrics.sort_values('Max Drawdown Mean', ascending=False).iloc[0]['Experiment']
    interp_text += f"Best Max Drawdown   : {best_mdd_exp} = {df_metrics.sort_values('Max Drawdown Mean', ascending=False).iloc[0]['Max Drawdown Mean']:.5f} (terbaik)\\n\\n"
"""
new_interp = """    best_mdd_exp = df_metrics.sort_values('Max Drawdown Mean', ascending=False).iloc[0]['Experiment']
    interp_text += f"Best Max Drawdown   : {best_mdd_exp} = {df_metrics.sort_values('Max Drawdown Mean', ascending=False).iloc[0]['Max Drawdown Mean']:.5f} (terbaik)\\n"
    
    best_ulcer_exp = df_metrics.sort_values('Ulcer Index Mean', ascending=True).iloc[0]['Experiment']
    interp_text += f"Best Ulcer Index    : {best_ulcer_exp} = {df_metrics.sort_values('Ulcer Index Mean', ascending=True).iloc[0]['Ulcer Index Mean']:.5f} (terbaik)\\n\\n"
"""
content = content.replace(old_interp, new_interp)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Modification done!")
