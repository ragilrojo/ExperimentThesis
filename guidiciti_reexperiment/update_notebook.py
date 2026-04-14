import nbformat

def fix_and_update_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # 1. Fix imports cell (index 0)
    import_cell = nb.cells[0]
    import_cell.source = """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.optimize import minimize
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.linalg import eigh
from sklearn.covariance import GraphicalLassoCV
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("Libraries imported successfully!")
"""

    # 2. Fix Case Classes Cell
    new_classes = """
class HRPPortfolioStrategy(PortfolioStrategy):
    \"\"\"
    Hierarchical Risk Parity (HRP)
    Uses hierarchical clustering to group assets based on correlation, 
    then performs recursive bisection for weights allocation.
    \"\"\"
    def get_weights(self, returns_data):
        n_assets = returns_data.shape[1]
        std = returns_data.std()
        valid_assets = std > 1e-12
        if not valid_assets.any(): return np.ones(n_assets) / n_assets
        
        filtered_data = returns_data.loc[:, valid_assets]
        assets_names = filtered_data.columns
        cov = filtered_data.cov().fillna(0).values
        corr = filtered_data.corr().fillna(0).values
        
        # 1. Clustering
        dist = np.sqrt(0.5 * (1 - np.clip(corr, -1, 1)))
        link = sch.linkage(sch.distance.squareform(dist), 'single')
        sort_ix = sch.leaves_list(link)
        
        # 2. Recursive Bisection
        weights = pd.Series(1.0, index=assets_names[sort_ix])
        
        def get_cluster_var(cov, cluster_items):
            sub_cov = cov[cluster_items, :][:, cluster_items]
            inv_diag = 1.0 / np.diag(sub_cov)
            w_inv = inv_diag / inv_diag.sum()
            return np.dot(w_inv, np.dot(sub_cov, w_inv))

        def recursive_bisection(cov, sort_ix):
            if len(sort_ix) <= 1:
                return
            mid = len(sort_ix) // 2
            left = sort_ix[:mid]
            right = sort_ix[mid:]
            var_l = get_cluster_var(cov, left)
            var_r = get_cluster_var(cov, right)
            alpha = 1 - var_l / (var_l + var_r)
            weights.iloc[left] *= alpha
            weights.iloc[right] *= (1 - alpha)
            recursive_bisection(cov, left)
            recursive_bisection(cov, right)

        recursive_bisection(cov, list(range(len(sort_ix))))
        full_weights = np.zeros(n_assets)
        full_weights[valid_assets] = weights.reindex(assets_names).values
        return full_weights

class KMeansPortfolioStrategy(PortfolioStrategy):
    \"\"\"
    K-Means Clustering Portfolio
    Clusters assets into N groups, then performs Equal Weighting across 
    and within clusters to ensure structural diversification.
    \"\"\"
    def __init__(self, name="K-Means ML", n_clusters=4, window_size=120):
        super().__init__(name, window_size)
        self.n_clusters = n_clusters

    def get_weights(self, returns_data):
        n_assets = returns_data.shape[1]
        std = returns_data.std()
        valid_assets = std > 1e-12
        if not valid_assets.any(): return np.ones(n_assets) / n_assets
        
        filtered_data = returns_data.loc[:, valid_assets]
        n_filtered = filtered_data.shape[1]
        actual_clusters = min(self.n_clusters, n_filtered)
        corr = filtered_data.corr().fillna(0).values
        
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init='auto').fit(corr)
        labels = kmeans.labels_
        
        weights_sub = np.zeros(n_filtered)
        cluster_weight = 1.0 / actual_clusters
        for i in range(actual_clusters):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) > 0:
                weights_sub[cluster_indices] = cluster_weight / len(cluster_indices)
        
        full_weights = np.zeros(n_assets)
        full_weights[valid_assets] = weights_sub
        return full_weights
"""
    # Find and fix the class definition cell
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'class PortfolioStrategy' in cell.source:
            # We want to keep the original classes and just append our new ones
            # Clean up potential duplicates from previous failed run
            if 'class HRPPortfolioStrategy' in cell.source:
                # Truncate at the first occurrence of our new classes to "reset"
                idx = cell.source.find('class HRPPortfolioStrategy')
                cell.source = cell.source[:idx]
            cell.source += new_classes
            break

    # 3. Fix and Update strategies list cell
    correct_strategies_cell = """strategies = [
    HybridMISNetworkMarkowitz(f"Hybrid MIS-NW Tuned (VaR: W={tuned_hybrid_params['W']}, G={tuned_hybrid_params['G']:.1f}, T={tuned_hybrid_params['T']:.2f})", gamma=tuned_hybrid_params['G'], corr_threshold=tuned_hybrid_params['T'], window_size=tuned_hybrid_params['W']),

    EquallyWeighted("EW"),
    ClassicalMarkowitz("CM"),
    GlassoMarkowitz("GM"),
    NetworkMarkowitz("NW (gamma=0)", gamma=0),
    NetworkMarkowitz("NW (gamma=1.0)", gamma=1.0),
    
    NetworkMarkowitz(f"NW Tuned (VAR: W={tuned_params['VAR']['W']}, G={tuned_params['VAR']['G']:.2f})", 
                     gamma=tuned_params['VAR']['G'], window_size=tuned_params['VAR']['W']),
                     
    NetworkMarkowitz(f"NW Tuned (Sharpe: W={tuned_params['SHARPE']['W']}, G={tuned_params['SHARPE']['G']:.2f})", 
                     gamma=tuned_params['SHARPE']['G'], window_size=tuned_params['SHARPE']['W']),
                     
    NetworkMarkowitz(f"NW Tuned (Rachev: W={tuned_params['RACHEV']['W']}, G={tuned_params['RACHEV']['G']:.2f})", 
                     gamma=tuned_params['RACHEV']['G'], window_size=tuned_params['RACHEV']['W']),
                     
    NetworkMarkowitz(f"NW Tuned (MDD: W={tuned_params['MDD']['W']}, G={tuned_params['MDD']['G']:.2f})", 
                     gamma=tuned_params['MDD']['G'], window_size=tuned_params['MDD']['W']) if 'MDD' in tuned_params else None,
                     
    NetworkMarkowitz(f"NW Tuned (RETURN: W={tuned_params['RETURN']['W']}, G={tuned_params['RETURN']['G']:.2f})", 
                     gamma=tuned_params['RETURN']['G'], window_size=tuned_params['RETURN']['W']) if 'RETURN' in tuned_params else None,

    HRPPortfolioStrategy("HRP (ML-based)"),
    KMeansPortfolioStrategy("K-Means CL (ML-based)", n_clusters=4),
]
strategies = [s for s in strategies if s is not None]

MAX_WINDOW   = max(s.window_size for s in strategies)
global_start = MAX_WINDOW

results = {}
for strat in strategies:
    print(f"Running: {strat.name}...")
    results[strat.name] = backtest_strategy(strat, df_returns, global_start=global_start)

print("\\nAll backtests completed!")
"""
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'strategies = [' in cell.source:
            cell.source = correct_strategies_cell
            break

    with open(file_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Successfully fixed and updated {file_path}")

if __name__ == "__main__":
    fix_and_update_notebook('strategy_comparison_coba_toGrid2stage4Matriks_clean.ipynb')
