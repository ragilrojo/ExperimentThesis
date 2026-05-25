import json
import os

path = r'g:/My Drive/00_Kuliah/Thesis/sharpenThesis_dpInsya/guidiciti_reexperiment/RLNetworkMarkowitz_Thesis_FinalResettingGamma.ipynb'

if not os.path.exists(path):
    print(f"File not found: {path}")
    exit(1)

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_source = [
    "def apply_rmt_filter(returns_window):\n",
    "    T, N = returns_window.shape\n",
    "    corr_mat = returns_window.corr().fillna(0).values\n",
    "    eigenvalues, eigenvectors = np.linalg.eigh(corr_mat)\n",
    "    Q = T / N\n",
    "    lambda_max = (1 + np.sqrt(1/Q))**2\n",
    "    \n",
    "    # Filter: Set non-significant eigenvalues to 0 (RMT Standard)\n",
    "    # Sesuai Giudici (2020), hanya deviating eigenvalues yang dipertahankan\n",
    "    eigenvalues[eigenvalues < lambda_max] = 0\n",
    "    \n",
    "    # Reconstruction\n",
    "    corr_denoised = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T\n",
    "    np.fill_diagonal(corr_denoised, 1)\n",
    "    return corr_denoised\n",
    "\n",
    "def get_centrality_weights(returns_window, gamma=1.0):\n",
    "    \"\"\"\n",
    "    Computes Network-Regularized Markowitz weights as per Giudici et al (2020):\n",
    "    1. RMT Filter\n",
    "    2. MST Reduction (Langkah krusial paper)\n",
    "    3. Eigenvector Centrality from MST structure\n",
    "    4. Optimization: w' * Sigma * w + gamma * (centrality' * w)\n",
    "    \"\"\"\n",
    "    T, N = returns_window.shape\n",
    "    mu = returns_window.mean().values\n",
    "    sigma = returns_window.std().values\n",
    "    \n",
    "    # 1. Denoised Covariance Matrix (RMT)\n",
    "    corr_f = apply_rmt_filter(returns_window)\n",
    "    cov_f = np.outer(sigma, sigma) * corr_f\n",
    "    cov_f += np.eye(N) * 1e-8 # Numerical stability\n",
    "    \n",
    "    # 2. MST Reduction (Giudici Step)\n",
    "    dist_mat = np.sqrt(np.maximum(0, 2 * (1 - corr_f)))\n",
    "    G_full = nx.from_numpy_array(dist_mat)\n",
    "    mst = nx.minimum_spanning_tree(G_full)\n",
    "    \n",
    "    # 3. Centrality calculation from MST\n",
    "    try:\n",
    "        centrality = nx.eigenvector_centrality(mst, max_iter=2000)\n",
    "        cent_vec = np.array([centrality[i] for i in range(N)])\n",
    "    except:\n",
    "        cent_vec = np.array(list(nx.degree_centrality(mst).values()))\n",
    "\n",
    "    # 4. Optimization\n",
    "    fun = lambda w: w.T @ cov_f @ w + gamma * np.sum(cent_vec * w)\n",
    "    \n",
    "    # Constraint 1: bobot harus sum=1\n",
    "    # Constraint 2: return portofolio >= rata-rata return (mencegah dominasi stablecoin)\n",
    "    cons = (\n",
    "        {'type': 'eq',   'fun': lambda w: np.sum(w) - 1},\n",
    "        {'type': 'ineq', 'fun': lambda w: np.dot(w, mu) - np.mean(mu)},\n",
    "    )\n",
    "    # Bound 0.1 as per paper for crypto portfolios\n",
    "    bnds = tuple((0, 0.1) for _ in range(N))\n",
    "    x0 = np.ones(N) / N\n",
    "    \n",
    "    res = minimize(fun, x0, method='SLSQP', bounds=bnds, constraints=cons)\n",
    "    \n",
    "    if not res.success:\n",
    "        return np.ones(N) / N\n",
    "        \n",
    "    return res.x\n",
    "\n",
    "class PortfolioStrategy:\n",
    "    def __init__(self, name): \n",
    "        self.name = name\n",
    "        self.last_gamma = 1.0\n",
    "    def compute_weights(self, returns_window): raise NotImplementedError()\n",
    "    \n",
    "class EquallyWeighted(PortfolioStrategy):\n",
    "    def compute_weights(self, returns_window): return np.ones(returns_window.shape[1])/returns_window.shape[1]\n",
    "class ClassicalMarkowitz(PortfolioStrategy):\n",
    "    def compute_weights(self, returns_window):\n",
    "        cov = returns_window.cov().values\n",
    "        inv_cov = np.linalg.pinv(cov)\n",
    "        w = np.clip(inv_cov @ np.ones(cov.shape[0]), 0, 1)\n",
    "        return w / np.sum(w)\n",
    "class NetworkMarkowitz(PortfolioStrategy):\n",
    "    def __init__(self, name, gamma=1.0):\n",
    "        super().__init__(name)\n",
    "        self.gamma = gamma\n",
    "    def compute_weights(self, returns_window): return get_centrality_weights(returns_window, self.gamma)\n"
]

found = False
for cell in nb['cells']:
    if 'get_centrality_weights' in ''.join(cell.get('source', [])):
        cell['source'] = new_source
        found = True
        break

if found:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook successfully updated.")
else:
    print("Target cell not found.")
    exit(1)
