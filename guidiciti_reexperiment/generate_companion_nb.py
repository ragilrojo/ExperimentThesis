import json
import os

# Define the target filename
SOURCE_NB = r'g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\guidiciti_reexperiment\RLNetworkMarkowitz_Thesis_Final_Validated.ipynb'
TARGET_NB = r'g:\My Drive\00_Kuliah\Thesis\sharpenThesis_dpInsya\guidiciti_reexperiment\RLNetworkMarkowitz_Thesis_Documentation_Companion.ipynb'

def generate_companion_nb():
    if not os.path.exists(SOURCE_NB):
        print(f"Source notebook {SOURCE_NB} not found. Please ensure the validated notebook exists.")
        return

    with open(SOURCE_NB, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Modify Title and Introduction based on LaTeX structure
    nb['cells'][0]['source'] = [
        "# RL-Network Markowitz: Experimental Companion Notebook\n",
        "## Synchronized with: RLNetworkMarkowitz_Documentation.tex\n",
        "\n",
        "This notebook contains the complete implementation of the proposed model for the thesis:\n",
        "**'Dynamic Risk Aversion Control using Reinforcement Learning in Network-Based Portfolio Optimization'**\n",
        "\n",
        "### Key Components (as per Documentation):\n",
        "1. **Graph Theory Integration**: Computing Network Density, Centrality, and Louvain Modularity.\n",
        "2. **Feature Engineering**: RMT Filtering for Noise Reduction in Correlation Matrices.\n",
        "3. **Environment Logic**: Delta-Gamma ($\\Delta\\gamma$) accumulation for risk-aversion control.\n",
        "4. **Agent**: SAC (Soft Actor-Critic) with Entropy Regularization.\n",
        "5. **Evaluation**: Sharpe, Sortino, Calmar, and CVaR (95%) ratios.\n"
    ]

    # Add Section Headers to match TeX Document
    new_cells = []
    for cell in nb['cells']:
        source_text = "".join(cell['source'])
        
        if "GLOBAL SETTINGS" in source_text:
            new_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Section III: Research Methodology & System Design\n", "### 3.1. Global Parameter Configuration"]
            })
        
        if "def compute_network_features" in source_text:
            new_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### 3.2. Network Feature Extraction\n", "Implementing Graph Density (Undirected) and Centrality metrics as defined in Section 2.4 of the TeX document."]
            })

        if "class AblationPortfolioEnv" in source_text:
            new_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### 3.3. Reinforcement Learning Environment\n", "Implementing the $\\Delta\\gamma$ accumulation logic: $\\gamma_t = \\text{clip}(\\gamma_{t-1} + \\Delta\\gamma_t, 0, 10)$."]
            })
            
        new_cells.append(cell)

    nb['cells'] = new_cells

    with open(TARGET_NB, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Companion Notebook generated: {TARGET_NB}")

if __name__ == "__main__":
    generate_companion_nb()
