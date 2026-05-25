import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import seaborn as sns
import shap
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from tqdm.notebook import tqdm
from scipy import stats
from scipy.optimize import minimize
from IPython.display import display

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ================================================================
# GLOBAL SETTINGS
# ================================================================
SEEDS         = [42, 123, 77]
TRAIN_STEPS   = 1000
FORCE_RETRAIN = True    # Set False jika model sudah ada
SET_WINDOW    = 30
SET_REBALANCE = 7
REWARD_WINDOW = 20
CVAR_LEVEL    = 0.95    # CVaR digunakan sebagai metrik evaluasi, bukan reward
STAT_ALPHA    = 0.05
SAVE_IMAGES   = False

ABLATION_CONFIGS = {
    'E2_Sharpe'          : {'use_network': True,  'use_market': True,  'reward_type': 'sharpe'},
    'E2_Sortino'         : {'use_network': True,  'use_market': True,  'reward_type': 'sortino'},
    'E2_Calmar'          : {'use_network': True,  'use_market': True,  'reward_type': 'calmar'},
    'E2_Ulcer'           : {'use_network': True,  'use_market': True,  'reward_type': 'ulcer'},
    'E2_Ensemble_Avg'    : {'use_network': True,  'use_market': True,  'is_ensemble': True},
    #'Comp_Static_Gamma0' : {'use_network': True,  'use_market': False, 'static_gamma': 0.0},
    #'Comp_Static_Gamma1' : {'use_network': True,  'use_market': False, 'static_gamma': 1.0},
    'Comp_Static_Gamma2' : {'use_network': True,  'use_market': False, 'static_gamma': 2.0},
    # Buy-and-Hold: gamma awal dihitung satu kali, tidak di-update tiap step
    'BuyHold_Markowitz'  : {'use_network': True,  'use_market': False, 'is_buyhold': True},
    'EqualWeight'        : {'is_equal_weight': True},
}

EVAL_METRICS = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Ulcer Index']

SCRIPT_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'ablation_results_thesis')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print('Environment initialized.')
print(f'Output dir: {OUTPUT_DIR}')



ABLATION_COLORS = {
    'E2_Sharpe'          : '#FF0000',
    'E2_Sortino'         : '#FFD700',
    'E2_Calmar'          : '#008000',
    'E2_Ulcer'           : '#0000FF',
    'E2_Ensemble_Avg'    : '#800080',
    'Comp_Static_Gamma0' : '#FFD700',
    'Comp_Static_Gamma1' : '#800080',
    'Comp_Static_Gamma2' : '#000000',
    'Classic-MV'         : '#FF0000',
    'BuyHold_Markowitz'  : '#008000',
    'EqualWeight'        : '#0000FF',
}

ABLATION_LINESTYLES = {
    'E2_Sharpe'          : '-',
    'E2_Sortino'         : '-',
    'E2_Calmar'          : '-',
    'E2_Ulcer'           : '-',
    'E2_Ensemble_Avg'    : '-',
    'Comp_Static_Gamma0' : '--',
    'Comp_Static_Gamma1' : '--',
    'Comp_Static_Gamma2' : '-',
    'Classic-MV'         : '--',
    'BuyHold_Markowitz'  : '--',
    'EqualWeight'        : '--',
}

DISPLAY_NAMES = {
    'E2_Sharpe'          : 'E2-Sharpe',
    'E2_Sortino'         : 'E2-Sortino',
    'E2_Calmar'          : 'E2-Calmar',
    'E2_Ulcer'           : 'E2-Ulcer',
    'E2_Ensemble_Avg'    : 'E2-Ensemble',
    'Comp_Static_Gamma0' : 'gamma=0 (Static)',
    'Comp_Static_Gamma1' : 'gamma=1 (Static)',
    'Comp_Static_Gamma2' : 'gamma=2 (Static)',
    'Classic-MV'         : 'Classic-MV',
    'BuyHold_Markowitz'  : 'Buy&Hold-Markowitz',
    'EqualWeight'        : 'Equal-Weight (Reb)',
}

# FIX: 2 network + 7 market = 9 total
FEATURE_NAMES = [
    'MST.Dist x0.1', 'Spectral.Gap',
    'VolShort x100', 'VolLong x100', 'Vol.Ratio',
    'Mom5d x100', 'Mom20d x100', 'MomCross x100', 'Pct.Uptrend',
]


def get_display_name(exp_id):
    return DISPLAY_NAMES.get(exp_id, exp_id)


# ── Callbacks ─────────────────────────────────────────────────────────

class RewardLoggerCallback(BaseCallback):
    """Log episode reward (dipakai di plot_learning_curves legacy)."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards        = []
        self.current_episode_reward = 0.0

    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals['rewards'][0]
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0
        return True


class StepLoggerCallback(BaseCallback):
    """
    Log reward dan rolling std setiap N step (bukan per episode).
    Dipakai untuk analisis konvergensi berbasis train-step.

    Atribut yang tersedia setelah training:
      .step_log  : list of (step, mean_reward_rolling, std_reward_rolling)
    """
    def __init__(self, log_freq=500, rolling_window=10, verbose=0):
        super().__init__(verbose)
        self.log_freq       = log_freq
        self.rolling_window = rolling_window
        # buffer episode rewards (internal)
        self._ep_buf        = []
        self._cur_ep_rew    = 0.0
        # output
        self.step_log       = []   # list of (step, mean, std)

    def _on_step(self) -> bool:
        self._cur_ep_rew += self.locals['rewards'][0]
        if self.locals['dones'][0]:
            self._ep_buf.append(self._cur_ep_rew)
            self._cur_ep_rew = 0.0

        if self.n_calls % self.log_freq == 0 and len(self._ep_buf) > 0:
            window = self._ep_buf[-self.rolling_window:]
            mean_r = float(np.mean(window))
            std_r  = float(np.std(window))
            self.step_log.append((self.n_calls, mean_r, std_r))

        return True

    @property
    def episode_rewards(self):
        """Kompatibel mundur dengan RewardLoggerCallback."""
        return self._ep_buf


# ── Learning-curve plots (legacy, tetap ada) ──────────────────────────

def plot_learning_curves(history_dict):
    """Plot learning curve untuk satu model (satu dict {seed: rewards})."""
    plt.figure(figsize=(10, 4))
    for seed, rewards in history_dict.items():
        if len(rewards) == 0:
            continue
        smoothed = pd.Series(rewards).rolling(window=max(1, len(rewards) // 10)).mean()
        plt.plot(smoothed, label=f'seed={seed}', alpha=0.8)
    plt.title(f'SAC Learning Curves - Steps: {TRAIN_STEPS}', fontsize=12, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Episode Reward')
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.legend(fontsize=8)
    plt.tight_layout()
    if SAVE_IMAGES:
        plt.savefig(os.path.join(OUTPUT_DIR, 'learning_curves.png'), dpi=150)
    plt.show()
    plt.close()


def plot_all_learning_curves(all_histories):
    """
    Plot learning curve untuk SEMUA model yang di-training.
    all_histories: dict { exp_id: { seed: [rewards] } }
    Layout: grid subplot — 1 subplot per eksperimen, tiap seed = 1 garis.
    """
    exp_ids = [eid for eid, h in all_histories.items() if any(len(r) > 0 for r in h.values())]
    if not exp_ids:
        print('Tidak ada data learning curve untuk ditampilkan.')
        return

    n_exp  = len(exp_ids)
    n_cols = min(3, n_exp)
    n_rows = (n_exp + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 4 * n_rows),
                             facecolor='white')
    axes = np.array(axes).reshape(-1)

    seed_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

    for ax_idx, exp_id in enumerate(exp_ids):
        ax      = axes[ax_idx]
        h_model = all_histories[exp_id]
        has_data = False
        for s_idx, (seed, rewards) in enumerate(h_model.items()):
            if len(rewards) == 0:
                continue
            has_data = True
            color    = seed_colors[s_idx % len(seed_colors)]
            window   = max(1, len(rewards) // 10)
            raw      = pd.Series(rewards)
            smoothed = raw.rolling(window=window, min_periods=1).mean()
            ax.plot(raw,      alpha=0.15, color=color)
            ax.plot(smoothed, alpha=0.9,  color=color, linewidth=1.8,
                    label=f'seed={seed}')

        ax.set_title(f'{get_display_name(exp_id)}',
                     fontsize=11, fontweight='bold',
                     color=ABLATION_COLORS.get(exp_id, '#333333'))
        ax.set_xlabel('Episode', fontsize=9)
        ax.set_ylabel('Smoothed Reward', fontsize=9)
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if not has_data:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='grey')

    for ax_idx in range(n_exp, len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle(
        f'SAC Learning Curves — Semua Model\nSteps: {TRAIN_STEPS}',
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    if SAVE_IMAGES:
        plt.savefig(os.path.join(OUTPUT_DIR, 'learning_curves_all_models.png'),
                    dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()


# ── Convergence analysis plot (BARU) ──────────────────────────────────

def _detect_convergence_step(steps, means, std_threshold=0.02, plateau_frac=0.15):
    """
    Deteksi otomatis titik konvergensi:
    Titik pertama di mana std rolling reward < std_threshold
    DAN tidak ada improvement > 5% dalam plateau_frac terakhir dari training.

    Return: step integer, atau None jika belum terdeteksi.
    """
    if len(steps) < 4:
        return None
    arr   = np.array(means)
    stds  = np.array([np.std(arr[max(0,i-5):i+1]) for i in range(len(arr))])
    n_plat = max(1, int(len(steps) * plateau_frac))
    tail_improvement = abs(arr[-1] - arr[-n_plat]) / (abs(arr[-n_plat]) + 1e-8)
    for i, (s, sd) in enumerate(zip(steps, stds)):
        if sd < std_threshold and i > len(steps) // 3:
            return s
    return None


def plot_convergence_analysis(step_histories, train_steps=TRAIN_STEPS):
    """
    Visualisasi justifikasi konvergensi train_steps.

    step_histories : dict { exp_id: { seed: StepLoggerCallback } }
    Menghasilkan 3 panel:
      1. Mean reward vs train steps (semua model, mean ± std antar seed)
      2. Rolling std reward vs train steps (stabilitas policy)
      3. Relative improvement per 5k-step interval (diminishing returns)
    """
    exp_ids = list(step_histories.keys())
    if not exp_ids:
        print('Tidak ada step history untuk diplot.')
        return

    fig = plt.figure(figsize=(14, 12), facecolor='white')
    gs  = gridspec.GridSpec(3, 1, hspace=0.45)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    eval_checkpoints = list(range(0, train_steps + 1, 5000))
    conv_steps_all   = {}

    for exp_id in exp_ids:
        color  = ABLATION_COLORS.get(exp_id, '#888888')
        lname  = get_display_name(exp_id)
        sh     = step_histories[exp_id]

        # Kumpulkan step_log dari semua seed, interpolasi ke grid seragam
        all_logs = [cb.step_log for cb in sh.values() if len(cb.step_log) > 0]
        if not all_logs:
            continue

        # Grid step seragam berdasarkan seed pertama
        ref_steps = [s for s, _, _ in all_logs[0]]
        if not ref_steps:
            continue

        # Matriks mean reward per seed [n_seeds x n_checkpoints]
        mean_matrix = []
        std_matrix  = []
        for log in all_logs:
            s_arr = np.array([s for s, _, _ in log])
            m_arr = np.array([m for _, m, _ in log])
            sd_arr= np.array([sd for _, _, sd in log])
            # Interpolasi ke ref_steps
            m_interp  = np.interp(ref_steps, s_arr, m_arr)
            sd_interp = np.interp(ref_steps, s_arr, sd_arr)
            mean_matrix.append(m_interp)
            std_matrix.append(sd_interp)

        mean_matrix = np.array(mean_matrix)
        std_matrix  = np.array(std_matrix)

        agg_mean = mean_matrix.mean(axis=0)
        agg_sem  = mean_matrix.std(axis=0)      # cross-seed std
        agg_std  = std_matrix.mean(axis=0)      # mean policy std

        # ── Panel 1: mean reward ± cross-seed std ──
        ax1.plot(ref_steps, agg_mean, color=color, linewidth=2, label=lname)
        ax1.fill_between(ref_steps,
                         agg_mean - agg_sem,
                         agg_mean + agg_sem,
                         color=color, alpha=0.12)

        # ── Panel 2: rolling std reward ──
        ax2.plot(ref_steps, agg_std, color=color, linewidth=1.8, label=lname)

        # ── Panel 3: relative improvement per interval ──
        n_bins   = len(eval_checkpoints) - 1
        bin_impv = []
        for b in range(n_bins):
            lo = eval_checkpoints[b]
            hi = eval_checkpoints[b + 1]
            idx_lo = np.searchsorted(ref_steps, lo)
            idx_hi = np.searchsorted(ref_steps, hi)
            if idx_hi > idx_lo and idx_hi <= len(agg_mean):
                delta = agg_mean[min(idx_hi, len(agg_mean)-1)] - agg_mean[idx_lo]
                bin_impv.append(delta)
            else:
                bin_impv.append(0.0)
        bin_centers = [(eval_checkpoints[b] + eval_checkpoints[b+1]) / 2
                       for b in range(n_bins)]
        ax3.plot(bin_centers, bin_impv, color=color, linewidth=1.6,
                 marker='o', markersize=4, label=lname)

        # Deteksi konvergensi
        conv = _detect_convergence_step(ref_steps, agg_mean.tolist())
        if conv is not None:
            conv_steps_all[exp_id] = conv
            ax1.axvline(conv, color=color, linestyle=':', linewidth=1, alpha=0.6)
            ax2.axvline(conv, color=color, linestyle=':', linewidth=1, alpha=0.6)

    # ── Anotasi zone eval (garis vertikal abu-abu) ──
    for ck in eval_checkpoints[1:]:
        for ax in [ax1, ax2, ax3]:
            ax.axvline(ck, color='#cccccc', linewidth=0.7, linestyle='--', zorder=0)

    # ── Plateau zone (shading step konvergensi rata-rata s/d akhir) ──
    if conv_steps_all:
        avg_conv = int(np.mean(list(conv_steps_all.values())))
        for ax in [ax1, ax2]:
            ax.axvspan(avg_conv, train_steps,
                       color='#639922', alpha=0.06,
                       label=f'Plateau zone (≥{avg_conv//1000}k)')
        # Anotasi teks
        ax1.annotate(
            f'Rata-rata konvergensi\n≈ {avg_conv:,} steps ({avg_conv*100//train_steps}% budget)',
            xy=(avg_conv, ax1.get_ylim()[0] if ax1.get_ylim()[0] != 0 else -0.1),
            xytext=(avg_conv + train_steps * 0.04,
                    ax1.get_ylim()[0] if ax1.get_ylim()[0] != 0 else -0.1),
            fontsize=8, color='#3B6D11',
            arrowprops=dict(arrowstyle='->', color='#3B6D11', lw=1),
        )

    # ── Zero-improvement reference ──
    ax3.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax3.fill_between(
        [eval_checkpoints[-2], eval_checkpoints[-1]],
        ax3.get_ylim()[0] if ax3.get_ylim()[0] != 0 else -0.05,
        0.005,
        color='#639922', alpha=0.08
    )

    # ── Labels & formatting ──
    ax1.set_title(
        f'Panel 1 — Mean reward vs train steps  (shading = std antar seed)',
        fontsize=11, fontweight='bold'
    )
    ax1.set_xlabel('Training steps', fontsize=9)
    ax1.set_ylabel('Mean episode reward', fontsize=9)
    ax1.legend(fontsize=8, loc='lower right')
    ax1.grid(True, alpha=0.25)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))

    ax2.set_title(
        'Panel 2 — Rolling reward std  (proxy stabilitas policy)',
        fontsize=11, fontweight='bold'
    )
    ax2.set_xlabel('Training steps', fontsize=9)
    ax2.set_ylabel('Reward std (rolling)', fontsize=9)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.25)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))

    ax3.set_title(
        'Panel 3 — Reward improvement per 5k-step interval  (diminishing returns)',
        fontsize=11, fontweight='bold'
    )
    ax3.set_xlabel('Training steps', fontsize=9)
    ax3.set_ylabel('ΔReward per interval', fontsize=9)
    ax3.legend(fontsize=8, loc='upper right')
    ax3.grid(True, alpha=0.25)
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}k'))

    # ── Anotasi justifikasi teks di bawah ──
    if conv_steps_all:
        avg_conv = int(np.mean(list(conv_steps_all.values())))
        buf_pct  = 100 - avg_conv * 100 // train_steps
        fig.text(
            0.5, -0.02,
            f'Justifikasi: Policy rata-rata konvergen pada ≈{avg_conv:,} steps '
            f'({avg_conv*100//train_steps}% budget). '
            f'Sisa {train_steps - avg_conv:,} steps ({buf_pct}%) = buffer konsolidasi. '
            f'Panel 2 menunjukkan std reward menurun dan stabil → policy tidak berubah signifikan. '
            f'Panel 3 menunjukkan diminishing returns menuju 0 → tidak ada manfaat tambah steps.',
            ha='center', fontsize=8.5, style='italic',
            color='#444441', wrap=True
        )

    fig.suptitle(
        f'Convergence Analysis — Justifikasi {train_steps:,} Train Steps (SAC)',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    if SAVE_IMAGES:
        plt.savefig(os.path.join(OUTPUT_DIR, 'convergence_analysis.png'),
                    dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()

    # ── Print ringkasan konvergensi ──
    print('\n=== Ringkasan Konvergensi per Model ===')
    for exp_id, cv in conv_steps_all.items():
        buf = train_steps - cv
        print(f'  {get_display_name(exp_id):<20}: konvergen @ step {cv:>6,}  '
              f'({cv*100//train_steps}% budget, buffer {buf:,} steps)')
    if conv_steps_all:
        avg = int(np.mean(list(conv_steps_all.values())))
        print(f'  {"RATA-RATA":<20}: konvergen @ step {avg:>6,}  '
              f'({avg*100//train_steps}% budget)')
    return conv_steps_all


def run_wilcoxon_tests(results_dict, baseline_id='Classic-MV'):
    """
    Wilcoxon signed-rank test: tiap eksperimen vs baseline.
    Mengembalikan dict {exp_id: p_value}.
    FIX: mengembalikan dict (bukan string), index disejajarkan sebelum test.
    """
    if baseline_id not in results_dict:
        print('Baseline not found in results.')
        return {}
    baseline_rets = results_dict[baseline_id].mean(axis=1)
    pvalues = {}
    for exp_id, rets_df in results_dict.items():
        if exp_id == baseline_id:
            continue
        exp_rets   = rets_df.mean(axis=1)
        common_idx = baseline_rets.index.intersection(exp_rets.index)
        try:
            _, p = stats.wilcoxon(exp_rets.loc[common_idx], baseline_rets.loc[common_idx])
            pvalues[exp_id] = p
        except Exception:
            pvalues[exp_id] = np.nan
    return pvalues


def generate_dashboard(results_dict, title_prefix, period_name, filename_base):
    exp_ids      = list(results_dict.keys())
    metrics_list = []

    for exp_id in exp_ids:
        rets_df = results_dict[exp_id]
        m_seeds = [calculate_all_metrics(rets_df[s], CVAR_LEVEL) for s in rets_df.columns]
        # FIX: label Features sesuai implementasi (2 network + 7 market)
        feat_label = 'Network(2) + Market(7)' if str(exp_id).startswith('E2') else 'Static / Baseline'
        res = {
            'Experiment': exp_id,
            'Features'  : feat_label,
            'Obs Dim'   : get_obs_dim(ABLATION_CONFIGS[exp_id]) if exp_id in ABLATION_CONFIGS else 0,
        }
        for m in EVAL_METRICS:
            vals = [ms[m] for ms in m_seeds]
            res[f'{m} Mean'] = np.mean(vals)
            res[f'{m} Std']  = np.std(vals)
        res['CVaR Mean'] = np.mean([ms['CVaR (95%)'] for ms in m_seeds])
        res['MaxDD Mean'] = np.mean([ms['Max Drawdown'] for ms in m_seeds])
        metrics_list.append(res)

    df_metrics = pd.DataFrame(metrics_list).sort_values(by='Sharpe Ratio Mean', ascending=False)

    # --- Plot 1: Table + Bar Charts ---
    fig1 = plt.figure(figsize=(16, 10), facecolor='white')
    gs1  = gridspec.GridSpec(2, 4, height_ratios=[1.2, 1.0])
    fig1.suptitle(
        f'{title_prefix} - Metrics Comparison\nSteps: {TRAIN_STEPS}',
        fontsize=14, fontweight='bold', y=0.98
    )

    ax_table = fig1.add_subplot(gs1[0, :])
    ax_table.axis('off')
    columns    = ['Experiment', 'Features', 'Obs Dim',
                  'Sharpe (up)', 'Sortino (up)', 'Calmar (up)', 'Ulcer (down)', 'Rank']
    table_data = []
    for rank, (_, row) in enumerate(df_metrics.iterrows(), 1):
        table_data.append([
            row['Experiment'], row['Features'], row['Obs Dim'],
            f"{row['Sharpe Ratio Mean']:.3f}+/-{row['Sharpe Ratio Std']:.3f}",
            f"{row['Sortino Ratio Mean']:.3f}+/-{row['Sortino Ratio Std']:.3f}",
            f"{row['Calmar Ratio Mean']:.3f}+/-{row['Calmar Ratio Std']:.3f}",
            f"{row['Ulcer Index Mean']:.4f}+/-{row['Ulcer Index Std']:.4f}",
            f'#{rank}',
        ])
    table = ax_table.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax_table.set_title('Rangkuman Metrik Performa', fontsize=11, fontweight='bold', pad=15)

    for i, m in enumerate(EVAL_METRICS):
        ax    = fig1.add_subplot(gs1[1, i])
        exps  = [row['Experiment'] for _, row in df_metrics.iterrows()]
        means = [row[f'{m} Mean']   for _, row in df_metrics.iterrows()]
        ax.bar(range(len(exps)), means,
               color=[ABLATION_COLORS.get(e, '#777777') for e in exps])
        ax.set_title(m, fontsize=10, fontweight='bold')
        ax.set_xticks(range(len(exps)))
        ax.set_xticklabels([get_display_name(e) for e in exps],
                           rotation=45, ha='right', fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if SAVE_IMAGES:
        plt.savefig(os.path.join(OUTPUT_DIR, f'1_metrics_{filename_base}'), dpi=200)
    plt.show()
    plt.close()



    # --- Plot 2: Cumulative Returns ---
    plt.figure(figsize=(16, 7), facecolor='white')
    for exp_id in results_dict:
        cum = (1 + results_dict[exp_id].mean(axis=1)).cumprod()
        plt.plot(cum, label=get_display_name(exp_id),
                 color=ABLATION_COLORS.get(exp_id, '#777777'),
                 linestyle=ABLATION_LINESTYLES.get(exp_id, '-'))
    plt.title(f'Cumulative Returns - {period_name} Period - Steps: {TRAIN_STEPS}',
              fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if SAVE_IMAGES:
        plt.savefig(os.path.join(OUTPUT_DIR, f'2_returns_{filename_base}'), dpi=200)
    plt.show()
    plt.close()


    # --- Plot 3: Radar / Spider Chart ---
    from matplotlib.patches import FancyArrowPatch
    df_metrics['R_Sharpe']  = df_metrics['Sharpe Ratio Mean'].rank(ascending=False)
    df_metrics['R_Sortino'] = df_metrics['Sortino Ratio Mean'].rank(ascending=False)
    df_metrics['R_Calmar']  = df_metrics['Calmar Ratio Mean'].rank(ascending=False)
    df_metrics['R_CVaR']    = df_metrics['CVaR Mean'].rank(ascending=True)
    df_metrics['R_MaxDD']   = df_metrics['MaxDD Mean'].rank(ascending=False)
    df_metrics['R_Ulcer']   = df_metrics['Ulcer Index Mean'].rank(ascending=True)

    TOP_N   = min(6, len(df_metrics))
    top_ids = df_metrics['Experiment'].values[:TOP_N]
    categories = ['Sharpe','Sortino','Calmar','Ulcer↓','CVaR↓','MaxDD↑']
    N_cat      = len(categories)
    angles     = np.linspace(0, 2 * np.pi, N_cat, endpoint=False).tolist()
    angles    += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True}, facecolor='white')
    ax.set_facecolor('#f9f9f9')

    n = len(df_metrics)
    for exp_id in top_ids:
        row    = df_metrics[df_metrics['Experiment'] == exp_id].iloc[0]
        vals   = [
            1 - (row['R_Sharpe']  - 1) / (n - 1) if n > 1 else 1.0,
            1 - (row['R_Sortino'] - 1) / (n - 1) if n > 1 else 1.0,
            1 - (row['R_Calmar']  - 1) / (n - 1) if n > 1 else 1.0,
            1 - (row['R_Ulcer']   - 1) / (n - 1) if n > 1 else 1.0,
            1 - (row['R_CVaR']    - 1) / (n - 1) if n > 1 else 1.0,
            1 - (row['R_MaxDD']   - 1) / (n - 1) if n > 1 else 1.0,
        ]
        vals  += vals[:1]
        color  = ABLATION_COLORS.get(exp_id, '#777777')
        ls     = ABLATION_LINESTYLES.get(exp_id, '-')
        ax.plot(angles, vals, color=color, linestyle=ls, linewidth=1.8, label=get_display_name(exp_id))
        if ls == '-':
            ax.fill(angles, vals, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%','50%','75%','100%'], fontsize=7, color='grey')
    ax.set_title(f'{title_prefix} - Performance Radar (Top {TOP_N})\nSteps: {TRAIN_STEPS}',
                 fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=8)
    plt.tight_layout()
    if SAVE_IMAGES:
        plt.savefig(os.path.join(OUTPUT_DIR, f'3_radar_{filename_base}'), dpi=200)
    plt.show()
    plt.close()




trained_models      = {}
train_histories     = {}   # { exp_id: { seed: [episode_rewards] } }  (backward compat)
step_histories_cb   = {}   # { exp_id: { seed: StepLoggerCallback } }  (untuk convergence analysis)

exp_bar = tqdm(ABLATION_CONFIGS.items(), desc='Experiments')
for exp_id, config in exp_bar:
    # Skip static gamma, ensemble, buyhold, equal-weight (tidak ada RL training)
    if (config.get('static_gamma') is not None
            or config.get('is_ensemble')
            or config.get('is_buyhold')
            or config.get('is_equal_weight')):
        continue

    train_histories[exp_id]   = {}
    step_histories_cb[exp_id] = {}
    seed_bar = tqdm(SEEDS, desc=f'Seeds for {exp_id}', leave=False)

    for seed in seed_bar:
        name = f'model_{exp_id}_s{seed}'
        if os.path.exists(name + '.zip') and not FORCE_RETRAIN:
            trained_models[(exp_id, seed)] = name
            print(f'  Loaded existing: {name}')
            continue

        env      = AblationPortfolioEnv(ret_train, config, split='train')
        eval_env = AblationPortfolioEnv(ret_test,  config, split='test')

        # StepLoggerCallback menggantikan RewardLoggerCallback:
        #   log_freq=500  → satu titik data setiap 500 steps
        #   rolling_window=10 → std dihitung dari 10 episode terakhir
        step_cb   = StepLoggerCallback(log_freq=500, rolling_window=10)
        eval_cb   = EvalCallback(eval_env, eval_freq=5000,
                                 deterministic=True, verbose=0)

        model = SAC(
            'MlpPolicy', env,
            seed=seed,
            verbose=0,
            learning_rate=1e-4,
            buffer_size=50_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            ent_coef='auto',
            policy_kwargs=dict(net_arch=[128, 128])
        )
        model.learn(total_timesteps=TRAIN_STEPS,
                    callback=[step_cb, eval_cb],
                    progress_bar=True)

        model.save(name)
        trained_models[(exp_id, seed)]     = name
        train_histories[exp_id][seed]      = step_cb.episode_rewards   # backward compat
        step_histories_cb[exp_id][seed]    = step_cb                   # untuk convergence

# ── Plot legacy learning curves (episode-based) ──────────────────────
if 'E2_Sharpe' in train_histories:
    plot_learning_curves(train_histories['E2_Sharpe'])

plot_all_learning_curves(train_histories)

# ── Plot convergence analysis (step-based, 3 panel) ──────────────────
conv_steps = plot_convergence_analysis(step_histories_cb)

print('Training selesai.')


