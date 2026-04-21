import json
import uuid

file_path = r'g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\guidiciti_reexperiment\RLNetworkMarkowitz_SAC_ThesisDataTrainingSaja.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Add Markdown Cell for Phase 8
markdown_source = [
    "## Phase 8: Statistical Significance Tests\n",
    "\n",
    "To validate if the performance differences between strategies are statistically significant, we implement:\n",
    "1.  **Paired T-Test**: Compares the mean daily returns.\n",
    "2.  **Diebold-Mariano (DM) Test**: A standard test in forecasting to compare predictive accuracy (or in this case, the time series of returns)."
]

# Add Code Cell for implementation
code_source = [
    "from scipy import stats\n",
    "\n",
    "def diebold_mariano_test(returns1, returns2, h=1):\n",
    "    \"\"\"\n",
    "    Perform Diebold-Mariano test to compare two return series.\n",
    "    h: forecast horizon (default 1 for daily returns)\n",
    "    \"\"\"\n",
    "    d = np.array(returns1) - np.array(returns2)\n",
    "    T = float(len(d))\n",
    "    d_bar = np.mean(d)\n",
    "    \n",
    "    def autocovariance(xi, k):\n",
    "        N = len(xi)\n",
    "        xs = xi - np.mean(xi)\n",
    "        if k == 0: return np.sum(xs**2) / N\n",
    "        return np.sum(xs[k:] * xs[:-k]) / N\n",
    "\n",
    "    var_d_bar = autocovariance(d, 0)\n",
    "    for i in range(1, h):\n",
    "        var_d_bar += 2 * autocovariance(d, i)\n",
    "    var_d_bar /= T\n",
    "    \n",
    "    dm_stat = d_bar / np.sqrt(var_d_bar) if var_d_bar > 0 else 0.0\n",
    "    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))\n",
    "    return dm_stat, p_value\n",
    "\n",
    "# Comparison table\n",
    "results_sig = []\n",
    "target_strategies = [\n",
    "    'SAC-Net (Excess NW)',\n",
    "    'SAC-Net (Sharpe Incr)',\n",
    "    'NW (Gamma=1.0)',\n",
    "    'NW (Gamma=0.0)',\n",
    "    'Equal Weight',\n",
    "    'Classical Markowitz',\n",
    "    'BTC (Buy & Hold)'\n",
    "]\n",
    "\n",
    "print(\"Evaluating statistical significance on Out-of-Sample returns...\")\n",
    "for i in range(len(target_strategies)):\n",
    "    for j in range(i + 1, len(target_strategies)):\n",
    "        s1, s2 = target_strategies[i], target_strategies[j]\n",
    "        if s1 in results_test and s2 in results_test:\n",
    "            r1, r2 = results_test[s1], results_test[s2]\n",
    "            min_len = min(len(r1), len(r2))\n",
    "            r1, r2 = r1.iloc[:min_len], r2.iloc[:min_len]\n",
    "            \n",
    "            t_stat, t_p = stats.ttest_rel(r1, r2)\n",
    "            dm_stat, dm_p = diebold_mariano_test(r1, r2)\n",
    "            \n",
    "            results_sig.append({\n",
    "                'Comparison': f\"{s1} vs {s2}\",\n",
    "                'DM-Stat': dm_stat,\n",
    "                'P-Value': dm_p,\n",
    "                'Significant': 'Yes' if dm_p < 0.05 else 'No'\n",
    "            })\n",
    "\n",
    "df_sig = pd.DataFrame(results_sig)\n",
    "print(\"\\n=== Diebold-Mariano Test Results (OOS) ===\")\n",
    "print(df_sig.round(4).to_string(index=False))\n",
    "\n",
    "# Save significance results\n",
    "df_sig.to_csv(os.path.join(csv_dir, 'statistical_significance.csv'), index=False)\n"
]

# Update the "TO DO" item in markdown cells
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        source = cell['source']
        for i, line in enumerate(source):
            if '2.  **Statistical Significance Tests**: (⏳ TO DO)' in line:
                source[i] = line.replace('(⏳ TO DO)', '(✅ UPDATED)')

# Append the new cells
nb['cells'].append({
    "cell_type": "markdown",
    "id": str(uuid.uuid4())[:8],
    "metadata": {},
    "source": markdown_source
})

nb['cells'].append({
    "cell_type": "code",
    "execution_count": None,
    "id": str(uuid.uuid4())[:8],
    "metadata": {},
    "outputs": [],
    "source": code_source
})

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Successfully implemented Statistical Significance Tests in the notebook.")
