import json

file_path = r'g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\guidiciti_reexperiment\RLNetworkMarkowitz_SAC_ThesisDataTrainingSajaVolatilkanGammaReturnSaja.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The correct training loop code to restore
training_loop_code = [
    "reward_modes = {\n",
    "    'total_return'    : 'SAC-Net (Total Return)',\n",
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
    "            normalize_reward=True\n",
    "        )\n",
    "\n",
    "        model = SAC(env=env, seed=seed, **sac_kwargs)\n",
    "        model.learn(total_timesteps=TRAIN_STEPS, progress_bar=True)\n",
    "        model.save(model_name)\n",
    "\n",
    "        trained_models[(mode, seed)] = model_name\n",
    "        print(f'  Saved: {model_name}.zip')\n",
    "\n",
    "print('\\n=== All SAC models trained ===')\n"
]

# Find the cell and fix it
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'reward_modes = {' in source and 'total_return' in source and 'trained_models' not in source:
            cell['source'] = training_loop_code
            break

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Successfully restored training loop in the simplified notebook.")
