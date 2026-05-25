"""
Script: Create Aggressive Return + Loss Penalty Notebook
=========================================================
Reads RLNetworkMarkowitz_SAC_XAIPenalty.ipynb and creates a modified copy
with a new 'aggressive_return' reward mode that:
  - Positive return: reward = port_ret * 100 (normal)
  - Negative return: reward = port_ret * 100 * LOSS_PENALTY_MULT (amplified penalty)

This makes the SAC agent aggressively pursue returns while heavily penalizing losses.
"""

import json
import copy
import os

SRC = 'RLNetworkMarkowitz_SAC_XAIPenalty.ipynb'
DST = 'RLNetworkMarkowitz_SAC_XAI_AggressiveReturn.ipynb'

with open(SRC, 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb_new = copy.deepcopy(nb)

# ────────────────────────────────────────────────────────────────
# 1. Update the title markdown cell (cell index 0)
# ────────────────────────────────────────────────────────────────
nb_new['cells'][0]['source'] = [
    "# Thesis Experiment: SAC Gamma Controller — Aggressive Return + Loss Penalty\n",
    "**Setup:** 7-Day Rebalancing Frequency | SAC Algorithm | Multi-seed evaluation | **XAI Added**\n",
    "\n",
    "**Key Modification — Aggressive Return Reward Mode:**\n",
    "- Ketika portfolio menghasilkan **return positif**, reward = `port_ret × 100`\n",
    "- Ketika portfolio **mengalami kerugian (loss)**, reward = `port_ret × 100 × LOSS_PENALTY_MULT`\n",
    "- `LOSS_PENALTY_MULT = 3.0` → kerugian dihukum **3× lebih berat** daripada keuntungan\n",
    "- Tujuan: Agent SAC menjadi **sangat agresif untuk return** dan **sangat menghindari loss**\n",
    "\n",
    "**Analogi:** Seperti trader yang menikmati profit biasa, tapi merasakan sakit 3× lipat saat rugi (loss aversion)\n"
]

# ────────────────────────────────────────────────────────────────
# 2. Find and modify the Environment cell (GammaPortfolioEnvFast)
#    Add 'aggressive_return' reward mode + LOSS_PENALTY_MULT param
# ────────────────────────────────────────────────────────────────
for i, cell in enumerate(nb_new['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])

    # Find the environment class cell
    if 'class GammaPortfolioEnvFast' in src and 'def step(self, action)' in src:
        print(f"  Found Environment class at cell index {i}")

        new_source = [
            "# ════════════════════════════════════════════════════════════════\n",
            "# LOSS_PENALTY_MULT: Multiplier untuk penalty saat portfolio RUGI\n",
            "# Nilai 3.0 berarti kerugian dihukum 3x lebih berat dari keuntungan\n",
            "# ════════════════════════════════════════════════════════════════\n",
            "LOSS_PENALTY_MULT = 3.0\n",
            "\n",
            "class GammaPortfolioEnvFast(gym.Env):\n",
            '    """\n',
            "    Environment with gamma centering + reward normalization.\n",
            "    \n",
            "    === AGGRESSIVE RETURN MODE ===\n",
            "    New reward mode: 'aggressive_return'\n",
            "    - Positive return → reward = port_ret * 100 (normal gain)\n",
            "    - Negative return → reward = port_ret * 100 * LOSS_PENALTY_MULT (amplified loss)\n",
            "    \n",
            "    Ini membuat agent SAC sangat agresif mengejar return\n",
            "    dan sangat menghindari kerugian (loss aversion).\n",
            '    """\n',
            "    def __init__(self, returns_data, obs_cache, opt_cache,\n",
            "                 baseline_ret_cache=None, window_size=30,\n",
            "                 reward_mode='aggressive_return',\n",
            "                 gamma_center=1.0, gamma_range=1.0,\n",
            "                 normalize_reward=True,\n",
            "                 loss_penalty_mult=LOSS_PENALTY_MULT):\n",
            "        super().__init__()\n",
            "        self.data               = returns_data\n",
            "        self.obs_cache          = obs_cache\n",
            "        self.opt_cache          = opt_cache\n",
            "        self.baseline_ret_cache = baseline_ret_cache or {}\n",
            "        self.window_size        = window_size\n",
            "        self.reward_mode        = reward_mode\n",
            "        self.gamma_center       = gamma_center\n",
            "        self.gamma_range        = gamma_range\n",
            "        self.normalize_reward   = normalize_reward\n",
            "        self.loss_penalty_mult  = loss_penalty_mult\n",
            "        self.current_step       = window_size\n",
            "        self.port_val           = 1.0\n",
            "        self.peak_val           = 1.0\n",
            "        self._rew_mean  = 0.0\n",
            "        self._rew_M2    = 0.0\n",
            "        self._rew_count = 0\n",
            "        self._ret_buffer = []\n",
            "        self.action_space      = spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)\n",
            "        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)\n",
            "\n",
            "    def _action_to_gamma(self, action):\n",
            "        return float(np.clip(action, -5.0, 5.0)) + self.gamma_center\n",
            "\n",
            "    def _get_obs(self):\n",
            "        nw_feat, mkt_feat = self.obs_cache[self.current_step]\n",
            "        mf = mkt_feat.copy()\n",
            "        mf[3] = self.port_val - 1\n",
            "        return np.concatenate([nw_feat, mf])\n",
            "\n",
            '    def _normalize_reward(self, r):\n',
            '        """Welford online normalization."""\n',
            "        if not self.normalize_reward:\n",
            "            return r\n",
            "        self._rew_count += 1\n",
            "        delta = r - self._rew_mean\n",
            "        self._rew_mean += delta / self._rew_count\n",
            "        delta2 = r - self._rew_mean\n",
            "        self._rew_M2 += delta * delta2\n",
            "        var = self._rew_M2 / self._rew_count if self._rew_count > 1 else 1.0\n",
            "        std = max(np.sqrt(var), 1e-6)\n",
            "        return np.clip(r / std, -10.0, 10.0)\n",
            "\n",
            "    def reset(self, seed=None, options=None):\n",
            "        super().reset(seed=seed)\n",
            "        self.current_step = self.window_size\n",
            "        self.port_val     = 1.0\n",
            "        self.peak_val     = 1.0\n",
            "        self._ret_buffer  = []\n",
            "        self._rew_mean  = 0.0\n",
            "        self._rew_M2    = 0.0\n",
            "        self._rew_count = 0\n",
            "        return self._get_obs(), {}\n",
            "\n",
            "    def step(self, action):\n",
            "        gamma    = self._action_to_gamma(action[0])\n",
            "        cov_f, cent_vec, mu = self.opt_cache[self.current_step]\n",
            "        w        = fast_centrality_weights(cov_f, cent_vec, mu, gamma)\n",
            "        port_ret = np.dot(w, self.data.iloc[self.current_step].values)\n",
            "\n",
            "        self.port_val *= (1 + port_ret)\n",
            "        self.peak_val  = max(self.peak_val, self.port_val)\n",
            "        drawdown = (self.port_val - self.peak_val) / self.peak_val\n",
            "        nw_ret   = self.baseline_ret_cache.get(self.current_step,\n",
            "                        self.data.iloc[self.current_step].mean())\n",
            "        self._ret_buffer.append(port_ret)\n",
            "        if len(self._ret_buffer) > 20:\n",
            "            self._ret_buffer.pop(0)\n",
            "\n",
            "        # ── Reward computation ──────────────────────────────────\n",
            "        if self.reward_mode == 'aggressive_return':\n",
            "            # ★ AGGRESSIVE RETURN + LOSS PENALTY ★\n",
            "            # Jika UNTUNG: reward normal\n",
            "            # Jika RUGI:   reward diperkuat (penalty multiplier)\n",
            "            if port_ret >= 0:\n",
            "                raw = port_ret * 100\n",
            "            else:\n",
            "                raw = port_ret * 100 * self.loss_penalty_mult\n",
            "\n",
            "        elif self.reward_mode == 'aggressive_return_dd':\n",
            "            # ★ AGGRESSIVE + DRAWDOWN PENALTY ★\n",
            "            # Sama seperti aggressive_return, tapi juga menghukum drawdown\n",
            "            if port_ret >= 0:\n",
            "                raw = port_ret * 100\n",
            "            else:\n",
            "                raw = port_ret * 100 * self.loss_penalty_mult\n",
            "            # Tambah drawdown penalty\n",
            "            dd_penalty = min(abs(drawdown) * 20, 5.0)\n",
            "            raw -= dd_penalty\n",
            "\n",
            "        elif self.reward_mode == 'sharpe_incremental':\n",
            "            if len(self._ret_buffer) >= 5:\n",
            "                buf = np.array(self._ret_buffer)\n",
            "                port_sharpe = buf.mean() / (buf.std() + 1e-8)\n",
            "            else:\n",
            "                port_sharpe = 0.0\n",
            "            raw = port_sharpe * 10\n",
            "\n",
            "        elif self.reward_mode == 'excess_nw':\n",
            "            raw = (port_ret - nw_ret) * 100\n",
            "\n",
            "        else:\n",
            "            raw = port_ret\n",
            "\n",
            "        reward = self._normalize_reward(raw)\n",
            "\n",
            "        self.current_step += 1\n",
            "        done = self.current_step >= len(self.data) - 1\n",
            "        obs  = self._get_obs() if not done else np.zeros(self.observation_space.shape)\n",
            "\n",
            "        return obs, float(reward), done, False, {}\n",
            "\n",
            "print(f'GammaPortfolioEnvFast defined — AGGRESSIVE mode with LOSS_PENALTY_MULT={LOSS_PENALTY_MULT}')\n",
        ]
        nb_new['cells'][i]['source'] = new_source
        nb_new['cells'][i]['outputs'] = []
        nb_new['cells'][i]['execution_count'] = None
        break

# ────────────────────────────────────────────────────────────────
# 3. Find and modify the training cell — change reward_modes
# ────────────────────────────────────────────────────────────────
for i, cell in enumerate(nb_new['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])

    if "reward_modes = {" in src and "trained_models" in src and "model.learn" in src:
        print(f"  Found Training loop at cell index {i}")

        new_source = [
            "# ════════════════════════════════════════════════════════════════\n",
            "# REWARD MODES — Aggressive Return fokus\n",
            "# 'aggressive_return'    : loss dihukum 3x, gain normal\n",
            "# 'aggressive_return_dd' : loss dihukum 3x + drawdown penalty\n",
            "# ════════════════════════════════════════════════════════════════\n",
            "reward_modes = {\n",
            "    'aggressive_return'    : 'SAC-Net (Aggressive Return)',\n",
            "    'aggressive_return_dd' : 'SAC-Net (Aggressive + DD Penalty)',\n",
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
        nb_new['cells'][i]['source'] = new_source
        nb_new['cells'][i]['outputs'] = []
        nb_new['cells'][i]['execution_count'] = None
        break

# ────────────────────────────────────────────────────────────────
# 4. Update COLORS dict in backtest utilities cell
# ────────────────────────────────────────────────────────────────
for i, cell in enumerate(nb_new['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])

    if "COLORS = {" in src and "run_multiseed_backtest" in src:
        print(f"  Found COLORS/backtest utilities at cell index {i}")

        old_colors = "'SAC-Net (Sharpe Incr)'  : 'darkorange',\n"
        new_colors = "'SAC-Net (Aggressive Return)'      : 'red',\n"

        old_colors2 = "'SAC-Net (Total Return)' : 'purple',\n"
        new_colors2 = "'SAC-Net (Aggressive + DD Penalty)': 'darkred',\n"

        new_src = src.replace(old_colors, new_colors).replace(old_colors2, new_colors2)
        nb_new['cells'][i]['source'] = [new_src]
        nb_new['cells'][i]['outputs'] = []
        nb_new['cells'][i]['execution_count'] = None
        break

# ────────────────────────────────────────────────────────────────
# 5. Clear ALL outputs to make a fresh notebook
# ────────────────────────────────────────────────────────────────
for cell in nb_new['cells']:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

# ────────────────────────────────────────────────────────────────
# 6. Save the new notebook
# ────────────────────────────────────────────────────────────────
with open(DST, 'w', encoding='utf-8') as f:
    json.dump(nb_new, f, indent=1, ensure_ascii=False)

print(f"\n✅ New notebook saved: {DST}")
print(f"   Key changes:")
print(f"   1. New reward mode: 'aggressive_return' (loss × {3.0}x penalty)")
print(f"   2. New reward mode: 'aggressive_return_dd' (loss penalty + drawdown penalty)")
print(f"   3. LOSS_PENALTY_MULT = 3.0 (configurable)")
print(f"   4. All outputs cleared — ready to run fresh")
