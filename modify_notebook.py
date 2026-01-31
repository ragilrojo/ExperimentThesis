import nbformat
import os

notebook_path = r'g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\rl_portfolio_ultimate.ipynb'

# The NEW Content for Evaluation Loop Cell
new_eval_source = r'''# --- AI-Gated Classifier Training (Random Forest) ---
# (Re-training on Split to maintain consistency with previous logic, but predicting on ALL data)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print('Training AI Gatekeeper (Bul/Bear Classifier)...')
# Target: Market Bull (1) if Average Return > 0 next day
market_ret = returns_df.mean(axis=1)
feat = pd.DataFrame(index=returns_df.index)
feat['Vol_20'] = market_ret.rolling(20).std()
feat['Mom_20'] = market_ret.rolling(20).mean()
feat['Target'] = (market_ret.shift(-1) > 0).astype(int)
feat = feat.dropna()

X_rf = feat[['Vol_20', 'Mom_20']]
y_rf = feat['Target']

# Align with split indices for Training
# train_df_slice was defined in previous cell
train_mask = X_rf.index.isin(train_df_slice.index)
X_train_rf = X_rf[train_mask]
y_train_rf = y_rf[train_mask]

clf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
clf.fit(X_train_rf, y_train_rf)

# Predict on ALL data for consistency analysis across years
rf_probs_all = clf.predict_proba(X_rf)[:, 1]
rf_probs_series = pd.Series(rf_probs_all, index=X_rf.index)
print(f'AI Gatekeeper Trained. Train Acc: {accuracy_score(y_train_rf, clf.predict(X_train_rf)):.2f}')

# --- Deep Learning (LSTM) Training ---
print('Training LSTM Predictor...')
LOOKBACK = 10
def create_dataset(dataset, look_back=10):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, :])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

train_data_val = returns_df.iloc[:train_split_idx].values
# Create training data
X_train_lstm, y_train_lstm = create_dataset(train_data_val, LOOKBACK)

train_dataset = TensorDataset(torch.from_numpy(X_train_lstm), torch.from_numpy(y_train_lstm))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

lstm_model = LSTMPredictor(input_dim=n_assets, hidden_dim=64, output_dim=n_assets)
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

lstm_model.train()
for epoch in range(20):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = lstm_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

# Predict on ALL data for consistency analysis
X_all_lstm, _ = create_dataset(returns_df.values, LOOKBACK)
lstm_model.eval()
with torch.no_grad():
    pred_all_tensor = lstm_model(torch.from_numpy(X_all_lstm))
    pred_all_numpy = pred_all_tensor.numpy()

# Align predictions with dates
# X_all_lstm[i] ends at index i+lookback. It predicts Y at i+lookback.
# Y corresponds to dataset[i+lookback]. Dataset index matches returns_df.
# So prediction `i` corresponds to returns_df index `i + lookback`.
# We want to map: Date -> Predicted Return for that Date.
valid_lstm_indices = range(LOOKBACK, LOOKBACK + len(pred_all_numpy))
valid_dates = returns_df.index[valid_lstm_indices]
lstm_preds_df = pd.DataFrame(pred_all_numpy, index=valid_dates, columns=assets)

print('LSTM Training Complete. Predictions generated for full period.')

# Helper for Markowitz
def optimize_markowitz(returns_window, cov_matrix):
    n = returns_window.shape[1]
    if n == 0: return []
    def neg_sharpe(weights):
        fed_return = np.sum(returns_window.mean() * weights) * 252
        fed_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        return -(fed_return / fed_vol) if fed_vol > 0 else 0
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    init_guess = n * [1. / n]
    try:
        res = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return res.x
    except:
        return init_guess

# --- Year-by-Year Evaluation Function ---
def run_year_evaluation(year_str, eval_df, eval_returns, eval_vol, eval_mom):
    print(f"\n=== Evaluating Year: {year_str} (Steps: {len(eval_df)}) ===")
    if len(eval_df) < 50:
        print("Not enough data to evaluate.")
        return None

    # Init Env for this specific year
    env_eval = MultiAssetCryptoEnv(eval_df, eval_returns, eval_vol, eval_mom)
    obs, _ = env_eval.reset()

    portfolio_values = {
        'DeepRL PPO (Baseline)': [100.0],
        'DeepRL PPO (Enhanced)': [100.0],
        'Markowitz': [100.0],
        'AI-Gated Markowitz (RF)': [100.0],
        'DL-Markowitz (LSTM)': [100.0],
        'Trend-Based (Markowitz/Cash)': [100.0],
        'Equal Weight': [100.0],
        'Buy and Hold': [100.0]
    }
    bnh_holdings = np.ones(n_assets) * (100.0 / n_assets) 

    # Loop
    for step in range(len(eval_df) - 1):
        current_date = eval_df.index[step]
        
        # 1. PPO Baseline
        action_base, _ = model_baseline.predict(obs, deterministic=True)
        
        # 2. PPO Enhanced
        action_enh, _ = model_enhanced.predict(obs, deterministic=True)
        weights_enh = np.exp(action_enh) / np.sum(np.exp(action_enh))
        asset_weights_enh = weights_enh[:-1]
        
        # Step Env
        # Note: Env steps based on internal index. 'obs' is next state.
        obs, reward, done, _, info = env_eval.step(action_base)
        
        # Update Baseline
        ret_base = info['return']
        portfolio_values['DeepRL PPO (Baseline)'].append(portfolio_values['DeepRL PPO (Baseline)'][-1] * (1 + ret_base))
        
        # Market Data for Manual Calculations
        # Warning: eval_returns is sliced. step+1 refers to valid index in slice.
        if step + 1 < len(eval_returns):
            day_returns = eval_returns.iloc[step + 1].values
        else:
            day_returns = np.zeros(n_assets)
            
        # Update Enhanced
        ret_enh = np.sum(asset_weights_enh * day_returns)
        portfolio_values['DeepRL PPO (Enhanced)'].append(portfolio_values['DeepRL PPO (Enhanced)'][-1] * (1 + ret_enh))
            
        # Baselines
        lookback = 50
        # For covariance, we ideally want previous 50 days even if outside 'eval_df'. 
        # But 'eval_df' here is just a slice.
        # Let's use the main 'returns_df' to get correct lookback window if needed?
        # Simpler: Just rely on passed eval_returns and accept warmup period of 50 days where weights are EW.
        # Or: Can loop from index in main df.
        # Let's stick to simple: if step < lookback, use EW.
        if step >= lookback:
            window_ret = eval_returns.iloc[step-lookback:step]
            cov_mat = window_ret.cov().values
            mw_weights = optimize_markowitz(window_ret, cov_mat)
        else:
            mw_weights = np.ones(n_assets) / n_assets
        
        ret_mw = np.sum(mw_weights * day_returns)
        portfolio_values['Markowitz'].append(portfolio_values['Markowitz'][-1] * (1 + ret_mw))

        # AI-Gated Markowitz
        if current_date in rf_probs_series.index:
            bull_prob = rf_probs_series.loc[current_date]
            if bull_prob > 0.52: ai_mw_w = mw_weights
            elif bull_prob < 0.48: ai_mw_w = 0.0 # Bear
            else: ai_mw_w = 0.5 * mw_weights
        else:
            ai_mw_w = 0.5 * mw_weights
        
        ret_ai = np.sum(ai_mw_w * day_returns)
        portfolio_values['AI-Gated Markowitz (RF)'].append(portfolio_values['AI-Gated Markowitz (RF)'][-1] * (1 + ret_ai))

        # DL-Markowitz (LSTM)
        # We need prediction for "Tomorrow" relative to "Current Step".
        # Current Step time is T. We decide weights for T -> T+1.
        # Return generated is at T+1.
        # LSTM dataframe is indexed by Date of Return.
        # So we want prediction at index Date(T+1).
        target_date_idx = step + 1
        if target_date_idx < len(eval_df):
            target_date = eval_df.index[target_date_idx]
            if target_date in lstm_preds_df.index:
                pred_ret = lstm_preds_df.loc[target_date].values
                
                def optimize_dl(exp_ret, cov):
                    n = len(exp_ret)
                    def neg_sharpe(w):
                        fr = np.sum(exp_ret * w)
                        fv = np.sqrt(np.dot(w.T, np.dot(cov, w)))
                        return -(fr / (fv + 1e-6))
                    c = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                    b = tuple((0, 1) for _ in range(n))
                    i = n * [1. / n]
                    try: res = minimize(neg_sharpe, i, method='SLSQP', bounds=b, constraints=c); return res.x
                    except: return i
                
                if step >= lookback:
                     dl_w = optimize_dl(pred_ret, cov_mat)
                else:
                     dl_w = np.ones(n_assets) / n_assets
                
                ret_dl = np.sum(dl_w * day_returns)
                portfolio_values['DL-Markowitz (LSTM)'].append(portfolio_values['DL-Markowitz (LSTM)'][-1] * (1 + ret_dl))
            else:
                portfolio_values['DL-Markowitz (LSTM)'].append(portfolio_values['DL-Markowitz (LSTM)'][-1])
        else:
             portfolio_values['DL-Markowitz (LSTM)'].append(portfolio_values['DL-Markowitz (LSTM)'][-1])

        # Trend-Based
        current_mom = np.mean(eval_mom.iloc[step].values)
        if current_mom > 0: trend_weights = mw_weights
        else: trend_weights = 0.5 * mw_weights
        
        ret_trend = np.sum(trend_weights * day_returns)
        portfolio_values['Trend-Based (Markowitz/Cash)'].append(portfolio_values['Trend-Based (Markowitz/Cash)'][-1] * (1 + ret_trend))
        
        # Equal Weight
        ew_weights = np.ones(n_assets) / n_assets
        ret_ew = np.sum(ew_weights * day_returns)
        portfolio_values['Equal Weight'].append(portfolio_values['Equal Weight'][-1] * (1 + ret_ew))
        
        # Buy and Hold
        bnh_holdings = bnh_holdings * (1 + day_returns)
        portfolio_values['Buy and Hold'].append(np.sum(bnh_holdings))
        
    return portfolio_values

# Evaluate 2023, 2024, 2025
years_to_test = [2023, 2024, 2025]
results_by_year = {}

for year in years_to_test:
    mask = df.index.year == year
    df_y = df[mask]
    if len(df_y) > 0:
        ret_y = returns_df[mask]
        vol_y = vol_df[mask]
        mom_y = mom_df[mask]
        
        res = run_year_evaluation(str(year), df_y, ret_y, vol_y, mom_y)
        if res is not None:
            results_by_year[year] = res

print("Evaluation of Years 2023, 2024, 2025 Complete.")'''

# The NEW Content for Visualization Cell
new_viz_source = r'''# 5. Results & Visualization
metric_cols = ['Total Return', 'Volatility', 'Sharpe Ratio']

def calculate_metrics(p_values_dict):
    metrics = {}
    for name, values in p_values_dict.items():
        arr = np.array(values)
        if len(arr) < 2: continue
        # daily returns
        rets = arr[1:] / arr[:-1] - 1
        # Prevent division by zero
        if len(rets) == 0: continue
        
        total_ret = (arr[-1] / arr[0]) - 1
        vol = np.std(rets) * np.sqrt(365) # Crypto 365
        if np.std(rets) > 1e-6:
            sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(365)
        else:
            sharpe = 0.0
        
        metrics[name] = [total_ret, vol, sharpe]
    
    return pd.DataFrame(metrics, index=metric_cols).T

# Plot each year
for year in sorted(results_by_year.keys()):
    print(f"\n{'='*20} Results for Year: {year} {'='*20}")
    res = results_by_year[year]
    if not res: continue
    
    # 1. Metrics
    df_metrics = calculate_metrics(res)
    # Formatting
    df_fmt = df_metrics.copy()
    try:
        df_fmt['Total Return'] = df_fmt['Total Return'].apply(lambda x: f"{x:.2%}")
        df_fmt['Volatility'] = df_fmt['Volatility'].apply(lambda x: f"{x:.2%}")
        df_fmt['Sharpe Ratio'] = df_fmt['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
        print(df_fmt)
    except:
        print(df_metrics) # Fallback if empty
    
    # 2. Plot
    plt.figure(figsize=(10, 5))
    for name, values in res.items():
        plt.plot(values, label=name)
    
    plt.title(f"Portfolio Performance - Year {year}")
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value ($)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
'''

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Replace Eval Cell
found_eval = False
for cell in nb.cells:
    if cell.cell_type == 'code':
        # Identify by existing content signature
        if "AI Gatekeeper" in cell.source and "Random Forest" in cell.source:
            cell.source = new_eval_source
            found_eval = True
            break # Modify only the first matching cell

# Replace Viz Cell
found_viz = False
for cell in nb.cells:
    if cell.cell_type == 'code':
        if "Total Return Volatility Sharpe Ratio" in cell.source:
            cell.source = new_viz_source
            found_viz = True
            break

if not found_eval or not found_viz:
    print(f"Warning: Cell finding status - Eval: {found_eval}, Viz: {found_viz}")

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
