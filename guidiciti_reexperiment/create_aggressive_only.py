"""
Create simplified Aggressive Return notebook:
- Remove EquallyWeighted from all simulations
- Remove bootstrap confidence bands
- Remove aggressive_return_dd mode (keep only aggressive_return)
"""
import json, copy

SRC = 'RLNetworkMarkowitz_SAC_XAI_AggressiveReturn.ipynb'
DST = 'RLNetworkMarkowitz_SAC_XAI_AggressiveOnly.ipynb'

with open(SRC, 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb_new = copy.deepcopy(nb)

# Update title
nb_new['cells'][0]['source'] = [
    "# Thesis Experiment: SAC Gamma Controller - Aggressive Return (Loss Penalty)\n",
    "**Setup:** 7-Day Rebalancing | SAC | Multi-seed | XAI\n",
    "\n",
    "**Reward Mode:** `aggressive_return` only\n",
    "- Return positif: reward = `port_ret * 100`\n",
    "- Return negatif (loss): reward = `port_ret * 100 * LOSS_PENALTY_MULT` (penalty 3x)\n",
    "\n",
    "**Simplifications:**\n",
    "- EquallyWeighted benchmark dihilangkan\n",
    "- Bootstrap confidence band dihilangkan\n",
    "- Hanya satu reward mode: `aggressive_return`\n"
]

for i, cell in enumerate(nb_new['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])

    # --- Fix reward_modes: keep only aggressive_return ---
    if "reward_modes = {" in src and "trained_models" in src and "model.learn" in src:
        nb_new['cells'][i]['source'] = [
            "reward_modes = {\n",
            "    'aggressive_return': 'SAC-Net (Aggressive Return)',\n",
            "}\n",
            "\n",
            "trained_models = {}   # {(mode, seed): model}\n",
            "\n",
            "for mode, label in reward_modes.items():\n",
            "    for seed in SEEDS:\n",
            "        model_name = f'sac_{mode}_seed{seed}'\n",
            "        print(f'\\nTraining [{label}] seed={seed}...')\n",
            "\n",
            "        env = GammaPortfolioEnvFast(\n",
            "            ret_old, obs_cache, opt_cache, baseline_ret_cache,\n",
            "            window_size=SET_WINDOW, reward_mode=mode,\n",
            "            gamma_center=GAMMA_CENTER, gamma_range=GAMMA_RANGE,\n",
            "            normalize_reward=True,\n",
            "            loss_penalty_mult=LOSS_PENALTY_MULT\n",
            "        )\n",
            "\n",
            "        model = SAC(env=env, seed=seed, **sac_kwargs)\n",
            "        model.learn(total_timesteps=TRAIN_STEPS, progress_bar=True)\n",
            "        model.save(model_name)\n",
            "\n",
            "        trained_models[(mode, seed)] = model_name\n",
            "        print(f'  Saved: {model_name}.zip')\n",
            "\n",
            "print('\\n=== All SAC models trained ===')\n",
        ]
        nb_new['cells'][i]['outputs'] = []
        nb_new['cells'][i]['execution_count'] = None

    # --- Fix COLORS + remove bootstrap function ---
    if "COLORS = {" in src and "run_multiseed_backtest" in src:
        nb_new['cells'][i]['source'] = [
            "COLORS = {\n",
            "    'Classical Markowitz'          : 'slategray',\n",
            "    'NW (Gamma=1.0)'               : 'steelblue',\n",
            "    'NW (Gamma=0.0)'               : 'lightblue',\n",
            "    'SAC-Net (Aggressive Return)'  : 'red',\n",
            "    'BTC (Buy & Hold)'             : 'black',\n",
            "}\n",
            "\n",
            "def run_multiseed_backtest(mode, label, data, seeds=SEEDS):\n",
            "    seed_results = {}\n",
            "    for seed in seeds:\n",
            "        model_path = f'sac_{mode}_seed{seed}'\n",
            "        strat = RLNetworkMarkowitz(\n",
            "            f'{label} [s{seed}]', model_path,\n",
            "            gamma_center=GAMMA_CENTER, gamma_range=GAMMA_RANGE\n",
            "        )\n",
            "        ret, _, gamma = run_backtest_with_frequency(\n",
            "            strat, data, window=SET_WINDOW, rebalance_freq=SET_REBALANCE\n",
            "        )\n",
            "        seed_results[seed] = (ret, gamma)\n",
            "    return seed_results\n",
            "\n",
            "def aggregate_seed_results(seed_results):\n",
            "    all_rets = pd.DataFrame({s: r for s, (r, _) in seed_results.items()})\n",
            "    return all_rets.mean(axis=1), all_rets.std(axis=1)\n",
            "\n",
            "print('Multi-seed backtest utilities defined (no bootstrap).')\n",
        ]
        nb_new['cells'][i]['outputs'] = []
        nb_new['cells'][i]['execution_count'] = None

    # --- Fix in-sample backtest: remove EW ---
    if "baseline_strategies = [" in src and "EquallyWeighted" in src:
        nb_new['cells'][i]['source'] = [
            "# -- Baseline strategies (tanpa EquallyWeighted) --\n",
            "baseline_strategies = [\n",
            "    ClassicalMarkowitz('Classical Markowitz'),\n",
            "    NetworkMarkowitz('NW (Gamma=1.0)', gamma=1.0),\n",
            "    NetworkMarkowitz('NW (Gamma=0.0)', gamma=0.0),\n",
            "]\n",
            "\n",
            "results_train = {}\n",
            "for s in baseline_strategies:\n",
            "    ret, _, _ = run_backtest_with_frequency(\n",
            "        s, ret_old, window=SET_WINDOW, rebalance_freq=SET_REBALANCE\n",
            "    )\n",
            "    results_train[s.name] = ret\n",
            "\n",
            "if 'BTC' in ret_old.columns:\n",
            "    results_train['BTC (Buy & Hold)'] = ret_old['BTC'].iloc[SET_WINDOW:]\n",
            "\n",
            "# -- SAC strategies (multi-seed) --\n",
            "sac_seed_results_train = {}\n",
            "sac_gamma_train        = {}\n",
            "\n",
            "for mode, label in reward_modes.items():\n",
            "    sr = run_multiseed_backtest(mode, label, ret_old)\n",
            "    sac_seed_results_train[(mode, label)] = sr\n",
            "    mean_ret, _ = aggregate_seed_results(sr)\n",
            "    mean_ret.name = label\n",
            "    results_train[label] = mean_ret\n",
            "    sac_gamma_train[label] = sr[SEEDS[0]][1]\n",
            "\n",
            "print('In-sample backtest complete.')\n",
        ]
        nb_new['cells'][i]['outputs'] = []
        nb_new['cells'][i]['execution_count'] = None

    # --- Fix cumulative return plot: remove bootstrap bands ---
    if "bootstrap_cumret_bands" in src or "Bootstrap confidence bands" in src or "90% CI (bootstrap)" in src:
        new_src = (
            "# -- Plot 1: Cumulative Returns (Training) --\n"
            "fig, ax = plt.subplots(figsize=(13, 5))\n"
            "for name, r in results_train.items():\n"
            "    (1 + r).cumprod().plot(\n"
            "        ax=ax, label=name,\n"
            "        color=COLORS.get(name, 'gray'),\n"
            "        linewidth=2.0,\n"
            "        linestyle='--' if name in ['Classical Markowitz'] else '-'\n"
            "    )\n"
            "\n"
            "ax.set_title('In-Sample Backtest (Training): SAC Aggressive Return vs Baselines')\n"
            "ax.legend(fontsize=8, ncol=2)\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "\n"
            "# -- Plot 2: Gamma Dynamics (Training) --\n"
            "fig, ax = plt.subplots(figsize=(13, 4))\n"
            "for label, g in sac_gamma_train.items():\n"
            "    g.plot(ax=ax, label=f'{label} gamma', color=COLORS.get(label, 'red'), alpha=0.8)\n"
            "ax.axhline(1.0, color='steelblue', linestyle='--', label='NW baseline (gamma=1.0)')\n"
            "ax.axhline(0.0, color='gray', linestyle=':', linewidth=0.8)\n"
            "ax.set_title('Gamma Dynamics - Training Period')\n"
            "ax.set_ylabel('Gamma Value')\n"
            "ax.legend(fontsize=9)\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        )
        nb_new['cells'][i]['source'] = [new_src]
        nb_new['cells'][i]['outputs'] = []
        nb_new['cells'][i]['execution_count'] = None

    # --- Fix OOS backtest: remove EW ---
    if "EquallyWeighted('Equal Weight')" in src and "ret_test" in src:
        new_src = src.replace(
            "    EquallyWeighted('Equal Weight'),\n", ""
        )
        nb_new['cells'][i]['source'] = [new_src]
        nb_new['cells'][i]['outputs'] = []
        nb_new['cells'][i]['execution_count'] = None

    # --- Fix OOS plots: remove bootstrap ---
    if "bootstrap_cumret_bands" in src and "ret_test" in src:
        new_src = src
        # Remove fill_between bootstrap blocks
        lines = new_src.split('\n')
        filtered = []
        skip = False
        for line in lines:
            if 'bootstrap_cumret_bands' in line or '90% CI' in line or 'fill_between' in line:
                skip = True
                continue
            if skip and (line.strip().startswith(')') or line.strip().startswith('label=')):
                skip = False
                continue
            skip = False
            filtered.append(line)
        nb_new['cells'][i]['source'] = ['\n'.join(filtered)]
        nb_new['cells'][i]['outputs'] = []
        nb_new['cells'][i]['execution_count'] = None

# Clear all outputs
for cell in nb_new['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

with open(DST, 'w', encoding='utf-8') as f:
    json.dump(nb_new, f, indent=1, ensure_ascii=False)

print(f"Done! Saved: {DST}")
