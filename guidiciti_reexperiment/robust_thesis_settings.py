import nbformat as nbf

nb_path = 'RLNetworkMarkowitz_Optimized.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Update global settings for scientific robustness (suitable for a thesis)
for cell in nb.cells:
    if cell.cell_type == 'code' and 'SEEDS         =' in cell.source:
        print("Updating hyperparameters for scientific robustness...")
        # Increase seeds for statistical significance
        cell.source = cell.source.replace("SEEDS         = [42, 123]", "SEEDS         = [42, 123, 77]")
        cell.source = cell.source.replace("SEEDS         = [42]", "SEEDS         = [42, 123, 77]")
        
        # Ensure training steps are sufficient for SAC convergence
        cell.source = cell.source.replace("TRAIN_STEPS   = 1000", "TRAIN_STEPS   = 2000")
        
        # Add a note about statistical robustness in the comments
        if "# Multi-seed for statistical robustness" not in cell.source:
            cell.source = cell.source.replace("SEEDS         =", "# Multi-seed for statistical robustness\nSEEDS         =")
        break

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook hyperparameters updated for thesis standards.")
