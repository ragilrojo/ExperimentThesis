import json

notebook_path = r"g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\guidiciti_reexperiment\RLNetworkMarkowitz_4Metrics_CalmarRatioReward.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "ret_te, _ = run_backtest(strat_te, ret_test, SET_WINDOW, SET_REBALANCE)" in source and "m_te = calculate_all_metrics(ret_te, CVAR_LEVEL)" in source:
            old_code = """        # Preview 4 metrik utama
        m_te = calculate_all_metrics(ret_te, CVAR_LEVEL)
        print(f'  seed={seed} | Sharpe={m_te["Sharpe Ratio"]:.3f} | '
              f'Sortino={m_te["Sortino Ratio"]:.3f} | '
              f'Calmar={m_te["Calmar Ratio"]:.3f} | '
              f'CVaR={m_te["CVaR (95%)"]:.4f}')"""
            
            new_code = """        # Preview 4 metrik utama (Train & Test)
        m_tr = calculate_all_metrics(ret_tr, CVAR_LEVEL)
        m_te = calculate_all_metrics(ret_te, CVAR_LEVEL)
        print(f'  [TRAIN] seed={seed} | Sharpe={m_tr["Sharpe Ratio"]:.3f} | Sortino={m_tr["Sortino Ratio"]:.3f} | Calmar={m_tr["Calmar Ratio"]:.3f} | CVaR={m_tr["CVaR (95%)"]:.4f}')
        print(f'  [TEST ] seed={seed} | Sharpe={m_te["Sharpe Ratio"]:.3f} | Sortino={m_te["Sortino Ratio"]:.3f} | Calmar={m_te["Calmar Ratio"]:.3f} | CVaR={m_te["CVaR (95%)"]:.4f}')"""
            
            new_source = source.replace(old_code, new_code)
            
            lines = new_source.split('\n')
            cell['source'] = [line + '\n' for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Backtest cell updated to print train metrics.")
