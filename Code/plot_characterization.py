"""
Phase 1 plotting script.

Usage (run from Code/ directory after all characterization experiments finish):
    python plot_characterization.py --dataset CIFAR10 --save-dir ./saved_experiment_results

Reads all CSV files matching:
    <save_dir>/csv_files/<dataset>_csv_files/separability_<dataset>*.csv

Produces one figure per metric comparison and a summary overlay figure.
"""

import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.switch_backend('agg')

COLORS = {
    'benign':       '#2196F3',   # blue
    'mal':          '#F44336',   # red
    'mal_all':      '#FF9800',   # orange
    'mal_gamma05':  '#9C27B0',   # purple
    'mal_gamma2':   '#E91E63',   # pink
    'lap_noise':    '#4CAF50',   # green
}

LABEL_MAP = {
    'normal':               'Benign (no attack)',
    'mal':                  'Active attack (Party A only)',
    'mal-all':              'Active attack (all parties)',
    'mal_gamma':            'Active attack',
    'lap_noise':            'Benign + DP noise defense',
    'mal_lap_noise':        'Active + DP noise defense',
}


def infer_condition_label(filename: str) -> str:
    """Map filename → human-readable label for legend."""
    f = os.path.basename(filename)
    if 'mal-all' in f:
        return 'Active attack (all parties)'
    if 'mal' in f and 'lap_noise' in f:
        return 'Active + DP noise defense'
    if 'mal' in f:
        return 'Active attack (Party A only)'
    if 'lap_noise' in f:
        return 'Benign + DP noise defense'
    return 'Benign (no attack)'


def load_all_csvs(csv_dir: str) -> dict[str, pd.DataFrame]:
    """Returns {condition_label: DataFrame}."""
    pattern = os.path.join(csv_dir, 'separability_*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No separability CSVs found in {csv_dir}")
    data = {}
    for f in files:
        label = infer_condition_label(f)
        df = pd.read_csv(f)
        # handle duplicate labels (e.g., two gamma variants): append filename suffix
        if label in data:
            label = label + f" ({os.path.basename(f).split('_lr')[0][-8:]})"
        data[label] = df
        print(f"  Loaded: {label}  ({len(df)} epochs)  ← {os.path.basename(f)}")
    return data


def _color_for(label: str) -> str:
    l = label.lower()
    if 'all parties' in l:
        return COLORS['mal_all']
    if 'active' in l and 'dp' in l:
        return COLORS['lap_noise']
    if 'active' in l:
        return COLORS['mal']
    if 'dp' in l:
        return COLORS['lap_noise']
    return COLORS['benign']


def plot_fisher(data: dict, out_dir: str, dataset: str) -> None:
    """Fisher A vs B per condition — two-panel figure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Fisher Criterion over Training — {dataset}', fontsize=13)

    for label, df in data.items():
        c = _color_for(label)
        axes[0].plot(df['epoch'], df['fisher_a'], label=label, color=c, linewidth=1.8)
        axes[1].plot(df['epoch'], df['fisher_b'], label=label, color=c, linewidth=1.8,
                     linestyle='--')

    axes[0].set_title('Party A (attacker)')
    axes[1].set_title('Party B (benign)')
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Fisher Criterion  J = Tr(S_B)/Tr(S_W)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

    plt.tight_layout()
    path = os.path.join(out_dir, f'{dataset}_fisher_AB.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def plot_fisher_divergence(data: dict, out_dir: str, dataset: str) -> None:
    """Fisher divergence (A − B) is the core detection signal."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(f'Fisher Divergence  (A − B) — {dataset}\n'
                 f'Positive divergence = Party A more class-separable than B', fontsize=11)
    ax.axhline(0, color='black', linewidth=0.8, linestyle=':')

    for label, df in data.items():
        ax.plot(df['epoch'], df['fisher_divergence'],
                label=label, color=_color_for(label), linewidth=1.8)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Fisher_A − Fisher_B')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, f'{dataset}_fisher_divergence.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def plot_silhouette(data: dict, out_dir: str, dataset: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Silhouette Score over Training — {dataset}', fontsize=13)

    for label, df in data.items():
        c = _color_for(label)
        axes[0].plot(df['epoch'], df['silhouette_a'], label=label, color=c, linewidth=1.8)
        axes[1].plot(df['epoch'], df['silhouette_b'], label=label, color=c, linewidth=1.8,
                     linestyle='--')

    axes[0].set_title('Party A (attacker)')
    axes[1].set_title('Party B (benign)')
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Silhouette Score  [-1, 1]')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, f'{dataset}_silhouette.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def plot_intra_var(data: dict, out_dir: str, dataset: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Intra-Class Variance over Training — {dataset}', fontsize=13)

    for label, df in data.items():
        c = _color_for(label)
        axes[0].plot(df['epoch'], df['intra_var_a'], label=label, color=c, linewidth=1.8)
        axes[1].plot(df['epoch'], df['intra_var_b'], label=label, color=c, linewidth=1.8,
                     linestyle='--')

    axes[0].set_title('Party A (attacker)')
    axes[1].set_title('Party B (benign)')
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Intra-class Variance (lower = tighter clusters)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, f'{dataset}_intra_variance.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def plot_grad_norm_ratio(data: dict, out_dir: str, dataset: str) -> None:
    """Gradient norm ratio (sent to A / sent to B) — secondary signal."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(f'Gradient Norm Ratio  (sent to A / sent to B) — {dataset}\n'
                 f'Ratio ≠ 1 may indicate asymmetric learning dynamics', fontsize=11)
    ax.axhline(1.0, color='black', linewidth=0.8, linestyle=':')

    for label, df in data.items():
        ax.plot(df['epoch'], df['grad_norm_ratio'],
                label=label, color=_color_for(label), linewidth=1.8)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('||grad_A|| / ||grad_B||')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, f'{dataset}_grad_norm_ratio.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def print_summary_table(data: dict) -> None:
    """Print final-epoch values for each condition."""
    print("\n─── Final-epoch summary ─────────────────────────────────────────────────")
    header = f"{'Condition':<40} {'Fisher_A':>10} {'Fisher_B':>10} {'Divergence':>12} {'Sil_A':>8}"
    print(header)
    print("─" * len(header))
    for label, df in data.items():
        row = df.iloc[-1]
        print(f"{label:<40} {row['fisher_a']:>10.4f} {row['fisher_b']:>10.4f} "
              f"{row['fisher_divergence']:>+12.4f} {row['silhouette_a']:>8.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description='Phase 1: plot separability characterization')
    parser.add_argument('--dataset', default='CIFAR10', type=str)
    parser.add_argument('--save-dir', default='./saved_experiment_results', type=str)
    args = parser.parse_args()

    csv_dir = os.path.join(args.save_dir, f"csv_files/{args.dataset}_csv_files")
    out_dir = os.path.join(args.save_dir, f"csv_files/{args.dataset}_csv_files/phase1_plots")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nLoading CSVs from: {csv_dir}")
    data = load_all_csvs(csv_dir)

    print(f"\nGenerating plots → {out_dir}")
    plot_fisher(data, out_dir, args.dataset)
    plot_fisher_divergence(data, out_dir, args.dataset)
    plot_silhouette(data, out_dir, args.dataset)
    plot_intra_var(data, out_dir, args.dataset)
    plot_grad_norm_ratio(data, out_dir, args.dataset)
    print_summary_table(data)
    print("Done.")


if __name__ == '__main__':
    main()
