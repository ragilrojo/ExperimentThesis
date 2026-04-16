import json
import os

notebook_path = 'strategy_comparison_coba_toGrid2stage4Matriks_adv.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Define the RL implementation cells
rl_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# REINFORCEMENT LEARNING FOR DYNAMIC GAMMA CONTROL\n",
            "In this section, we implement a Reinforcement Learning (RL) agent that learns to control the `gamma` parameter of the Network Markowitz model dynamically based on market conditions."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import gymnasium as gym\n",
            "from gymnasium import spaces\n",
            "from stable_baselines3 import PPO\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "\n",
            "class GammaPortfolioEnv(gym.Env):\n",
            "    \"\"\"\n",
            "    Custom Environment for RL Gamma Control\n",
            "    State: [avg_return, volatility, avg_correlation, avg_degree_centrality]\n",
            "    Action: Continuous Gamma value (0.0 to 2.0)\n",
            "    \"\"\"\n",
            "    def __init__(self, returns_df, window_size=120):\n",
            "        super(GammaPortfolioEnv, self).__init__()\n",
            "        self.returns_df = returns_df\n",
            "        self.window_size = window_size\n",
            "        self.current_step = window_size\n",
            "        self.n_assets = returns_df.shape[1]\n",
            "        \n",
            "        # Action space: Gamma [0.0, 2.0]\n",
            "        self.action_space = spaces.Box(low=0.0, high=2.0, shape=(1,), dtype=np.float32)\n",
            "        \n",
            "        # Observation space: 4 metrics\n",
            "        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)\n",
            "\n",
            "    def _get_obs(self):\n",
            "        window = self.returns_df.iloc[self.current_step - self.window_size : self.current_step]\n",
            "        avg_ret = window.mean().mean()\n",
            "        vol = window.std().mean()\n",
            "        corr_matrix = window.corr()\n",
            "        avg_corr = corr_matrix.values[np.triu_indices(self.n_assets, k=1)].mean()\n",
            "        \n",
            "        # Simple centrality approximation (avg correlation per asset)\n",
            "        centrality = corr_matrix.mean().mean() \n",
            "        \n",
            "        return np.array([avg_ret, vol, avg_corr, centrality], dtype=np.float32)\n",
            "\n",
            "    def step(self, action):\n",
            "        gamma_val = float(action[0])\n",
            "        \n",
            "        # Execute Network Markowitz with this gamma for the CURRENT step\n",
            "        # (In a real backtest, we'd use this weight for the NEXT period return)\n",
            "        window = self.returns_df.iloc[self.current_step - self.window_size : self.current_step]\n",
            "        \n",
            "        # Instantiate temporary strat to get weights\n",
            "        temp_strat = NetworkMarkowitz(\"Temp\", gamma=gamma_val, window_size=self.window_size)\n",
            "        try:\n",
            "            weights = temp_strat.get_weights(window)\n",
            "            # Calculate return for the NEXT day\n",
            "            next_day_rets = self.returns_df.iloc[self.current_step].values\n",
            "            portfolio_return = np.sum(weights * next_day_rets)\n",
            "        except:\n",
            "            portfolio_return = -0.01 # Penalty for optimization failure\n",
            "        \n",
            "        self.current_step += 1\n",
            "        done = self.current_step >= len(self.returns_df) - 1\n",
            "        \n",
            "        reward = portfolio_return\n",
            "        obs = self._get_obs() if not done else np.zeros((4,), dtype=np.float32)\n",
            "        \n",
            "        return obs, reward, done, False, {}\n",
            "\n",
            "    def reset(self, seed=None, options=None):\n",
            "        super().reset(seed=seed)\n",
            "        self.current_step = self.window_size\n",
            "        return self._get_obs(), {}"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Training the RL Agent on Historical Data\n",
            "print(\"Initializing RL Environment and Agent...\")\n",
            "env = GammaPortfolioEnv(df_returns, window_size=120)\n",
            "model = PPO(\"MlpPolicy\", env, verbose=1, learning_rate=0.0003)\n",
            "\n",
            "print(\"Training RL Agent (this might take a minute)...\")\n",
            "model.learn(total_timesteps=5000) # Small timesteps for demonstration\n",
            "model.save(\"ppo_gamma_controller\")\n",
            "print(\"RL Agent trained and saved.\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "class RLNetworkMarkowitz(NetworkMarkowitz):\n",
            "    def __init__(self, name, model_path, window_size=120):\n",
            "        super().__init__(name, gamma=1.0, window_size=window_size)\n",
            "        self.rl_model = PPO.load(model_path)\n",
            "        self.gamma_history = []\n",
            "\n",
            "    def get_weights(self, returns_data):\n",
            "        # 1. Construct state from returns_data\n",
            "        avg_ret = returns_data.mean().mean()\n",
            "        vol = returns_data.std().mean()\n",
            "        corr_matrix = returns_data.corr()\n",
            "        n_assets = returns_data.shape[1]\n",
            "        avg_corr = corr_matrix.values[np.triu_indices(n_assets, k=1)].mean()\n",
            "        centrality = corr_matrix.mean().mean()\n",
            "        \n",
            "        obs = np.array([avg_ret, vol, avg_corr, centrality], dtype=np.float32)\n",
            "        \n",
            "        # 2. Predict Gamma using RL model\n",
            "        action, _states = self.rl_model.predict(obs, deterministic=True)\n",
            "        current_gamma = float(action[0])\n",
            "        \n",
            "        # 3. Apply Gamma and Optimize\n",
            "        self.gamma = current_gamma\n",
            "        self.gamma_history.append({'date': returns_data.index[-1], 'gamma': current_gamma})\n",
            "        \n",
            "        return super().get_weights(returns_data)\n",
            "    \n",
            "    def get_params(self):\n",
            "        return {'W': self.window_size, 'RL': 'PPO-Gamma'}"
        ]
    }
]

# Insert after class definitions (Search for last class definition or a specific marker)
# I'll insert it before 'Performance Comparison'
for i, cell in enumerate(nb['cells']):
    source_str = "".join(cell.get('source', []))
    if 'def backtest_strategy' in source_str:
        insertion_point = i
        break
else:
    insertion_point = len(nb['cells']) - 1

new_cells = nb['cells'][:insertion_point] + rl_cells + nb['cells'][insertion_point:]
nb['cells'] = new_cells

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("RL Gamma Controller implementation added to notebook successfully.")
