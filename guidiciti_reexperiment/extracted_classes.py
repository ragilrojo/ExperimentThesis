class PortfolioStrategy:
    def __init__(self, name, window_size=120):
        self.name            = name
        self.window_size     = window_size
        self.weights_history = []
        self.returns_history = []
    def get_weights(self, returns_data):
        raise NotImplementedError

class EquallyWeighted(PortfolioStrategy):
    def get_weights(self, returns_data):
        n = returns_data.shape[1]
        return np.ones(n) / n

class ClassicalMarkowitz(PortfolioStrategy):
    """Optimasi Mean-Variance tradisional (minimasi varians portofolio)."""
    def get_weights(self, returns_data):
        n_assets = returns_data.shape[1]
        std = returns_data.std()
        valid_assets = std > 1e-12
        if not valid_assets.any(): return np.ones(n_assets) / n_assets
        filtered_data = returns_data.loc[:, valid_assets]
        mu = filtered_data.mean().values
        S  = filtered_data.cov().fillna(0).values
        S  += np.eye(len(mu)) * 1e-8
        objective   = lambda w: w @ S @ w
        constraints = [
            {'type': 'eq',   'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: w @ mu - mu.mean()}
        ]
        res = minimize(objective, np.ones(len(mu)) / len(mu),
                       method='SLSQP',
                       bounds=[(0, 1)] * len(mu),
                       constraints=constraints)
        weights = np.zeros(n_assets)
        weights[valid_assets] = res.x if res.success else (np.ones(len(mu)) / len(mu))
        return weights

class GlassoMarkowitz(PortfolioStrategy):
    """Markowitz dengan matriks presisi yang diestimasi via Graphical Lasso."""
    def get_weights(self, returns_data):
        n_assets = returns_data.shape[1]
        std = returns_data.std()
        valid_assets = std > 1e-12
        if not valid_assets.any(): return np.ones(n_assets) / n_assets
        filtered_data = returns_data.loc[:, valid_assets]
        mu = filtered_data.mean().values
        try:
            glasso = GraphicalLassoCV()
            glasso.fit(filtered_data.values)
            S = glasso.covariance_
        except Exception:
            S = filtered_data.cov().fillna(0).values
        S += np.eye(len(mu)) * 1e-8
        objective   = lambda w: w @ S @ w
        constraints = [
            {'type': 'eq',   'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: w @ mu - mu.mean()}
        ]
        res = minimize(objective, np.ones(len(mu)) / len(mu),
                       method='SLSQP',
                       bounds=[(0, 1)] * len(mu),
                       constraints=constraints)
        weights = np.zeros(n_assets)
        weights[valid_assets] = res.x if res.success else (np.ones(len(mu)) / len(mu))
        return weights

class NetworkMarkowitz(PortfolioStrategy):
    """Network Markowitz: RMT + MST + penalti sentralitas eigenvector."""
    def __init__(self, name="Network Markowitz", gamma=0, window_size=120):
        super().__init__(name, window_size)
        self.gamma = gamma

    def get_weights(self, returns_data):
        n_assets = returns_data.shape[1]
        std = returns_data.std()
        valid_assets = std > 1e-12
        if not valid_assets.any(): return np.ones(n_assets) / n_assets
        filtered_data = returns_data.loc[:, valid_assets]
        mu   = filtered_data.mean().values
        sig  = filtered_data.std().values
        Cf   = apply_rmt_filter(filtered_data)
        dist = build_mst(Cf)
        cent = compute_eigenvector_centrality(dist)
        Sf   = np.outer(sig, sig) * Cf
        Sf += np.eye(len(mu)) * 1e-8
        objective   = lambda w: w @ Sf @ w + self.gamma * np.sum(cent * w)
        constraints = [
            {'type': 'eq',   'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: w @ mu - mu.mean()}
        ]
        res = minimize(objective, np.ones(len(mu)) / len(mu),
                       method='SLSQP',
                       bounds=[(0, 1)] * len(mu),
                       constraints=constraints)
        weights = np.zeros(n_assets)
        weights[valid_assets] = res.x if res.success else (np.ones(len(mu)) / len(mu))
        return weights

class MISNetworkMarkowitz(PortfolioStrategy):
    """
    Two-Stage Portfolio Selection:
    Stage 1: Asset Selection using Maximum Independent Set (MIS) to ensure maximum diversification.
    Stage 2: Asset Weighting using Network Markowitz on the selected subset.
    """
    def __init__(self, name, gamma=0.7, window_size=25, correlation_threshold=0.4):
        super().__init__(name)
        self.gamma = gamma
        self.window_size = window_size
        self.thresh = correlation_threshold
        self.param_history = []

    def get_weights(self, returns_window):
        # Stage 1: Selection via MIS
        corr = returns_window.corr()
        # Build adjunct matrix where edge exists if correlation > threshold
        adj = (corr.abs() > self.thresh).astype(int)
        np.fill_diagonal(adj.values, 0)
        
        G = nx.from_pandas_adjacency(adj)
        selected_assets = list(nx.approximation.maximum_independent_set(G))
        
        # If MIS returns nothing (rare), fallback to all
        if not selected_assets:
            selected_assets = returns_window.columns.tolist()
            
        # Subset returns
        sub_returns = returns_window[selected_assets]
        n_sub = len(selected_assets)
        
        # Stage 2: Network Markowitz on subset
        mu = sub_returns.mean().values
        cov = sub_returns.cov().values
        
        # Distance matrix for subset
        sub_corr = sub_returns.corr().values
        dist = np.sqrt(2 * (1 - np.clip(sub_corr, -1, 1)))
        g_sub = nx.from_numpy_array(dist)
        
        # Centrality
        try:
            centrality = nx.eigenvector_centrality_numpy(g_sub, weight='weight')
            vec_c = np.array([centrality[i] for i in range(n_sub)])
        except:
            vec_c = np.ones(n_sub) / n_sub

        def objective(w):
            port_ret = np.dot(w, mu)
            port_var = np.dot(w.T, np.dot(cov, w))
            penalty = self.gamma * np.dot(w, vec_c)
            return -(port_ret - 0.5 * port_var - penalty)

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = [(0, 1) for _ in range(n_sub)]
        res = minimize(objective, np.ones(n_sub)/n_sub, method='SLSQP', bounds=bounds, constraints=constraints)
        
        # Map subset weights back to full asset list
        full_weights = pd.Series(0.0, index=returns_window.columns)
        if res.success:
            full_weights[selected_assets] = res.x
        else:
            full_weights[selected_assets] = 1.0 / n_sub
            
        return full_weights.values

class HRPPortfolioStrategy(PortfolioStrategy):
    """
    Hierarchical Risk Parity (HRP)
    Uses hierarchical clustering to group assets based on correlation, 
    then performs recursive bisection for weights allocation.
    """
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
        # Distance matrix
        dist = np.sqrt(0.5 * (1 - np.clip(corr, -1, 1)))
        link = sch.linkage(sch.distance.squareform(dist), 'single')
        sort_ix = sch.leaves_list(link)
        
        # 2. Recursive Bisection
        weights = pd.Series(1.0, index=assets_names[sort_ix])
        
        def get_cluster_var(cov, cluster_items):
            # Compute variance of a cluster using inverse-variance weighting proxy
            sub_cov = cov[cluster_items, :][:, cluster_items]
            # Handle singular matrices
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
            
            # Allocation factor based on relative risk
            alpha = 1 - var_l / (var_l + var_r)
            
            # Apply weights to the indices in the series
            weights.iloc[left] *= alpha
            weights.iloc[right] *= (1 - alpha)
            
            recursive_bisection(cov, left)
            recursive_bisection(cov, right)

        recursive_bisection(cov, list(range(len(sort_ix))))
        
        # Map back to full asset list
        full_weights = np.zeros(n_assets)
        full_weights[valid_assets] = weights.reindex(assets_names).values
        return full_weights

class KMeansPortfolioStrategy(PortfolioStrategy):
    """
    K-Means Clustering Portfolio
    Clusters assets into N groups, then performs Equal Weighting across 
    and within clusters to ensure structural diversification.
    """
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
        
        # Use correlation as feature for clustering
        corr = filtered_data.corr().fillna(0).values
        
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init='auto').fit(corr)
        labels = kmeans.labels_
        
        weights_sub = np.zeros(n_filtered)
        cluster_weight = 1.0 / actual_clusters
        
        for i in range(actual_clusters):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) > 0:
                # Equal weight within cluster
                weights_sub[cluster_indices] = cluster_weight / len(cluster_indices)
        
        full_weights = np.zeros(n_assets)
        full_weights[valid_assets] = weights_sub
        return full_weights

