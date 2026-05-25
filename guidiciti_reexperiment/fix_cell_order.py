import json

DST = 'RLNetworkMarkowitz_SAC_XAI_AggressiveOnly.ipynb'

with open(DST, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Insert the missing utility functions cell BEFORE cell 19 (the backtest cell)
new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "utilities_fix",
    "metadata": {},
    "outputs": [],
    "source": [
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
        "print('Utilities defined: COLORS, run_multiseed_backtest, aggregate_seed_results')\n",
    ]
}

nb['cells'].insert(19, new_cell)

with open(DST, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Fixed: inserted missing utilities cell at position 19")
