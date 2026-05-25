import json
import re

with open('RLNetworkMarkowitz_thesis_CVaR.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix __file__ and matplotlib for Jupyter
content = content.replace("matplotlib.use('Agg')", "# matplotlib.use('Agg')  # Di-comment untuk Jupyter Notebook")
content = content.replace("os.path.dirname(os.path.abspath(__file__))", "os.getcwd()  # Disesuaikan untuk Jupyter Notebook")

sections = re.split(r'# ={64}\n# (.*?)\n# ={64}\n', content)

cells = []

if sections[0].strip():
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in sections[0].strip('\n').split('\n')]
    })

for i in range(1, len(sections), 2):
    header = sections[i]
    code = sections[i+1]
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"## {header}\n"]
    })
    
    if code.strip():
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + '\n' for line in code.strip('\n').split('\n')]
        })

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('RLNetworkMarkowitz_thesis_CVaR.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)
print("Conversion successful!")
