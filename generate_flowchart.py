
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_flowchart():
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Color Palette
    bg_color = 'white'
    box_color = '#E1F5FE'
    border_color = '#0277BD'
    text_color = '#01579B'
    arrow_color = '#0288D1'

    # Helper function to draw a box
    def draw_box(x, y, w, h, text, shape='rect', fontsize=10):
        if shape == 'rect':
            box = patches.FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.2", 
                                        linewidth=2, edgecolor=border_color, facecolor=box_color)
        elif shape == 'diamond':
            # Simplified diamond as a rotated rect
            # Actually, standard rect is fine for now, or use Polygon
            pts = [[x, y+h/2], [x+w/2, y], [x, y-h/2], [x-w/2, y]]
            box = patches.Polygon(pts, linewidth=2, edgecolor=border_color, facecolor=box_color)
        
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, color=text_color, fontweight='bold', wrap=True)

    # Title
    plt.text(50, 97, "Graph-Enhanced AI-Gated Markowitz Strategy", ha='center', fontsize=18, fontweight='bold', color='#01579B')

    # 1. Data Input
    draw_box(50, 91, 50, 6, "Multi-Asset Crypto Data (24 Assets)\n(Closing Price Data 2023-2025)")
    
    # 2. Feature Engineering
    draw_box(50, 81, 40, 6, "Feature Engineering & Labeling\n(Features: Volatility, Momentum | Target: Returns)")
    
    # Arrow 1 to 2
    ax.annotate('', xy=(50, 84), xytext=(50, 88), arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))

    # 3. Training/Testing split
    draw_box(25, 71, 25, 6, "Training Set\n(2023-2024)")
    draw_box(75, 71, 25, 6, "Testing Set\n(2025)")
    
    # Arrows to split
    ax.annotate('', xy=(25, 74), xytext=(50, 78), arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2, connectionstyle="angle,angleA=0,angleB=90"))
    ax.annotate('', xy=(75, 74), xytext=(50, 78), arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2, connectionstyle="angle,angleA=0,angleB=90"))
    
    # 4. Train Model
    draw_box(45, 71, 14, 5, "Train AI\n(Random Forest)", fontsize=9)
    ax.annotate('', xy=(37.5, 71), xytext=(40.5, 71), arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))

    # 5. Prediction Diamond
    draw_box(50, 61, 35, 8, "AI Gatekeeper Prediction\n(Predict Bull Probability P)", shape='diamond', fontsize=10)
    
    # Arrows to diamond
    ax.annotate('', xy=(50, 65), xytext=(45, 68.5), arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5, connectionstyle="angle,angleA=0,angleB=-90"))
    ax.annotate('', xy=(50, 65), xytext=(75, 68), arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5, connectionstyle="angle,angleA=0,angleB=-90"))

    # 6. Regimes
    # Bear Path
    draw_box(20, 50, 30, 5, "Bear Regime (P <= 0.5)")
    draw_box(20, 42, 30, 5, "Risk-Off\n(100% Cash/Stablecoins)")
    ax.annotate('', xy=(20, 53), xytext=(32.5, 61), arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5, connectionstyle="angle,angleA=90,angleB=180"))
    ax.annotate('', xy=(20, 44.5), xytext=(20, 47.5), arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))

    # Bull Path
    draw_box(75, 50, 35, 5, "Bull Regime (P > 0.5)")
    draw_box(75, 42, 35, 5, "Temporal Correlation Graph\n(30-day Correlation Window)")
    draw_box(75, 34, 35, 5, "Maximum Independent Set (MIS)\n(Select Uncorrelated Assets)")
    draw_box(75, 26, 35, 5, "Markowitz Optimization\n(Maximize Sharpe for MIS Assets)")
    
    ax.annotate('', xy=(75, 53), xytext=(67.5, 61), arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5, connectionstyle="angle,angleA=90,angleB=0"))
    ax.annotate('', xy=(75, 44.5), xytext=(75, 47.5), arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))
    ax.annotate('', xy=(75, 36.5), xytext=(75, 39.5), arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))
    ax.annotate('', xy=(75, 28.5), xytext=(75, 31.5), arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))

    # 7. Convergence
    draw_box(50, 16, 45, 6, "Portfolio Rebalancing\n(Daily Return Calculation & Value Update)")
    
    # Arrows to convergence
    ax.annotate('', xy=(50, 19), xytext=(20, 39.5), arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5, connectionstyle="angle,angleA=-90,angleB=0"))
    ax.annotate('', xy=(50, 19), xytext=(75, 23.5), arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5, connectionstyle="angle,angleA=-90,angleB=180"))

    # 8. Performance
    draw_box(50, 6, 40, 6, "Performance Analysis\n(Total Return, Sharpe Ratio, Drawdown)")
    ax.annotate('', xy=(50, 9), xytext=(50, 13), arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))

    plt.tight_layout()
    plt.savefig('graph_gated_markowitz_flowchart_new.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    draw_flowchart()
    print("Flowchart saved as graph_gated_markowitz_flowchart_new.png")
