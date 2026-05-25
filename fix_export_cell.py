"""
Fix untuk error pada cell export di comparison_baseline_vs_temporal_graph.ipynb

Ganti cell export yang error dengan code berikut:
"""

# Export Comprehensive Results
with pd.ExcelWriter('comparison_baseline_vs_temporal_graph.xlsx', engine='openpyxl') as writer:
    # Summary comparison
    comparison_df.to_excel(writer, sheet_name='Performance_Comparison')
    
    # ML comparison
    ml_comparison.to_excel(writer, sheet_name='ML_Comparison', index=False)
    
    # Baseline portfolio
    baseline_df.to_excel(writer, sheet_name='Baseline_Portfolio', index=False)
    
    # Enhanced portfolio
    enhanced_df.to_excel(writer, sheet_name='Enhanced_Portfolio', index=False)
    
    # Feature importance
    feature_importance.to_excel(writer, sheet_name='Feature_Importance', index=False)
    
    # Daily returns comparison - merge on date to handle different lengths
    baseline_export = baseline_df[['Date', 'Value']].copy()
    baseline_export.columns = ['Date', 'Baseline_Value']
    baseline_export['Baseline_Return'] = baseline_df['Value'].pct_change() * 100
    
    enhanced_export = enhanced_df[['Date', 'Value']].copy()
    enhanced_export.columns = ['Date', 'Enhanced_Value']
    enhanced_export['Enhanced_Return'] = enhanced_df['Value'].pct_change() * 100
    
    # Merge on date to align properly
    daily_comparison = pd.merge(baseline_export, enhanced_export, on='Date', how='outer')
    daily_comparison = daily_comparison.sort_values('Date')
    daily_comparison.to_excel(writer, sheet_name='Daily_Comparison', index=False)

print('\n✅ Comprehensive results exported to: comparison_baseline_vs_temporal_graph.xlsx')
print('\nFiles generated:')
print('  1. comparison_baseline_vs_temporal_graph.xlsx (Excel report)')
print('  2. strategy_comparison.png (Performance charts)')
print('  3. feature_importance_comparison.png (Feature analysis)')
