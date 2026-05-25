import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

def draw_flowchart():
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    def draw_box(x, y, w, h, text, color='lightblue', shape='rect'):
        if shape == 'rect':
            box = patches.Rectangle((x-w/2, y-h/2), w, h, linewidth=2, edgecolor='black', facecolor=color)
        elif shape == 'diamond':
            # Create a diamond using coordinates
            pts = [[x, y+h/2], [x+w/2, y], [x, y-h/2], [x-w/2, y]]
            box = patches.Polygon(pts, linewidth=2, edgecolor='black', facecolor=color)
        elif shape == 'capsule':
            box = patches.FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.1,rounding_size=2", linewidth=2, edgecolor='black', facecolor=color)
        
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=11, wrap=True, fontweight='bold' if 'box' not in text.lower() else 'normal')

    def draw_arrow(x1, y1, x2, y2, text=None):
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=20, color='black', linewidth=1.5)
        ax.add_patch(arrow)
        if text:
            ax.text((x1+x2)/2 + 2, (y1+y2)/2, text, fontsize=10, fontweight='bold')

    # 1. Data Source
    draw_box(50, 95, 60, 6, "Multi-Asset Crypto Data\n(dataset_2023_2025.xlsx)", color='#E1F5FE')
    
    # 2. Feature Engineering
    draw_arrow(50, 92, 50, 85)
    draw_box(50, 81, 60, 8, "Feature Engineering\nFeatures: Vol_20, Mom_20, Mom_50\nTarget: Next Day Return (>0 Bull)", color='#B3E5FC')

    # 3. Training/Testing Split
    draw_arrow(50, 77, 30, 70) # To Training
    draw_arrow(50, 77, 70, 70) # To Testing
    draw_box(30, 67, 20, 6, "Training Set\n(2023-2024)", color='#FFF9C4')
    draw_box(70, 67, 20, 6, "Testing Set\n(2025)", color='#FFF9C4')

    # 4. Model Training
    draw_arrow(30, 64, 30, 58)
    draw_box(30, 55, 25, 6, "Train AI Gatekeeper\n(Random Forest)", color='#C8E6C9')

    # 5. Prediction
    draw_arrow(30, 52, 60, 48) # Connection to prediction logic
    draw_arrow(70, 64, 70, 52)
    draw_box(70, 48, 30, 8, "AI Gatekeeper Prediction\nPredict Bull Probability (P)", color='#C8E6C9')

    # 6. Decision Diamond
    draw_arrow(70, 44, 70, 38)
    draw_box(70, 32, 25, 12, "Is Market\nBullish?\n(P > 0.50)", shape='diamond', color='#FFCCBC')

    # 7. Branches
    # No path
    draw_arrow(57.5, 32, 35, 32, "NO")
    draw_box(25, 32, 20, 6, "Risk-Off\n(100% Cash)", color='#EF9A9A')

    # Yes path
    draw_arrow(70, 26, 70, 22, "YES")
    draw_box(70, 18, 35, 8, "Graph Filter:\nMaximum Independent Set (MIS)\n(Find minimally correlated assets)", color='#B2DFDB')

    # 8. Markowitz Optimization
    draw_arrow(70, 14, 70, 8)
    draw_box(70, 4, 35, 8, "Markowitz Optimization\n(Sharpe Ratio Maximization on\nMIS-selected assets)", color='#80CBC4')

    # 9. Convergence to Rebalancing
    # From Risk-Off
    draw_arrow(25, 29, 25, 0) # Just dummy line for routing
    draw_arrow(25, 1, 40, 1)
    
    # From Markowitz
    draw_arrow(52.5, 4, 45, 4)
    draw_arrow(45, 4, 45, 2)
    
    # Portfolio Rebalancing
    draw_box(50, -10, 40, 6, "Portfolio Rebalancing\nUpdate weights and calc returns", color='#E1F5FE')
    
    # Performance Metrics
    draw_arrow(50, -13, 50, -19)
    draw_box(50, -23, 50, 8, "Performance Metrics\nSharpe Ratio, Max Drawdown,\nCumulative Returns Comparison", color='#E1F5FE', shape='capsule')

    plt.title("Enhanced AI-Gated Markowitz Strategy Design (Graph-Integrated)", fontsize=16, pad=20, fontweight='bold')
    
    # Adjust plot for negative coordinates
    ax.set_ylim(-30, 105)
    
    plt.tight_layout()
    plt.savefig('enhanced_markowitz_flowchart.png', dpi=300, bbox_inches='tight')
    print("Flowchart saved as enhanced_markowitz_flowchart.png")

if __name__ == "__main__":
    draw_flowchart()
