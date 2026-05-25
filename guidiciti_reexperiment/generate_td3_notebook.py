
import json, re, textwrap

# ── Read source ────────────────────────────────────────────────────────────
with open('td3_ablation_portfolio.py', encoding='utf-8') as f:
    raw = f.read()

# ── Helper ─────────────────────────────────────────────────────────────────
def code_cell(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src if isinstance(src, list) else src.splitlines(keepends=True)
    }

def md_cell(src):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src if isinstance(src, list) else src.splitlines(keepends=True)
    }

# ── Split source into labelled sections ───────────────────────────────────
# We cut on the repeated ===...=== comment blocks that delimit sections.
SECTION_RE = re.compile(
    r'(?m)^# =+\n# (.+?)\n# =+\n',
)

splits = list(SECTION_RE.finditer(raw))

def chunk(start_match, end_match):
    start = start_match.start()
    end   = end_match.start() if end_match else len(raw)
    # include the header comment itself
    return raw[start:end].rstrip()

sections = []
for i, m in enumerate(splits):
    title   = m.group(1).strip()
    content = chunk(m, splits[i+1] if i+1 < len(splits) else None)
    sections.append((title, content))

# ── Build cells ────────────────────────────────────────────────────────────
cells = []

# Title markdown
cells.append(md_cell(
    "# TD3 Ablation Study — Portfolio Optimization\n"
    "> **Thesis-Ready** | Twin Delayed DDPG (TD3) · Network-Markowitz\n\n"
    "Script ini adalah versi `.ipynb` dari `td3_ablation_portfolio.py`.\n"
    "Jalankan sel secara berurutan dari atas ke bawah.\n"
))

# pip install cell (commented out by default)
cells.append(md_cell("## 0. Install Dependencies (jika belum)"))
cells.append(code_cell("# %pip install stable_baselines3[extra] scipy shap\n"))

for title, content in sections:
    # Skip the inline pip install line already handled above
    cells.append(md_cell(f"## {title}"))
    # Strip leading section-header comment so it isn't duplicated
    code = re.sub(r'^# =+\n# .+?\n# =+\n', '', content, count=1).strip()
    if code:
        cells.append(code_cell(code + "\n"))

# ── Notebook structure ─────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

out_path = 'td3_ablation_portfolio.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"[OK] Notebook saved -> {out_path}  ({len(cells)} cells)")
