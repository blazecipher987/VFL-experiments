"""
ROC curve for Fisher divergence threshold (tau) detection.

Each epoch of training produces one fisher_divergence value (J_A - J_B).
Label: 0 = benign training, 1 = active attack (MaliciousSGD).
Classifier: flag as attack if divergence > tau.

Datasets used (all seed=0):
  CIFAR-10:  30 benign epochs, 100 attack epochs
  CIFAR-100: 30 benign epochs, 150 attack epochs
  Combined:  60 benign + 250 attack = 310 points

Outputs (saved to saved_experiment_results/figures/):
  roc_curve_cifar10.png    -- CIFAR-10 only
  roc_curve_cifar100.png   -- CIFAR-100 only
  roc_curve_combined.png   -- both datasets together
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ── paths ──────────────────────────────────────────────────────────────
CSV_ROOT = os.path.join(
    "saved_experiment_results", "csv_files"
)
FIG_DIR = os.path.join("saved_experiment_results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

C10_BENIGN = os.path.join(CSV_ROOT, "CIFAR10_csv_files",
    "separability_CIFAR10_lr=0.1_normal_half=16.csv")
C10_ATTACK = os.path.join(CSV_ROOT, "CIFAR10_csv_files",
    "separability_CIFAR10_lr=0.1_mal_half=16.csv")
C100_BENIGN = os.path.join(CSV_ROOT, "CIFAR100_csv_files",
    "separability_CIFAR100_lr=0.1_normal_half=16.csv")
C100_ATTACK = os.path.join(CSV_ROOT, "CIFAR100_csv_files",
    "separability_CIFAR100_lr=0.1_mal_half=16.csv")

TAU = 0.10  # our chosen threshold


def load_divergence(path, label):
    """Return (divergence_array, label_array) from a separability CSV."""
    df = pd.read_csv(path)
    div = df["fisher_divergence"].values
    labels = np.full(len(div), label, dtype=int)
    return div, labels


def compute_roc(div_benign, div_attack):
    """Compute ROC and locate operating point at TAU."""
    scores = np.concatenate([div_benign, div_attack])
    labels = np.concatenate([
        np.zeros(len(div_benign), dtype=int),
        np.ones(len(div_attack),  dtype=int),
    ])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    # find closest threshold to TAU
    idx = np.argmin(np.abs(thresholds - TAU))
    op_fpr = fpr[idx]
    op_tpr = tpr[idx]
    return fpr, tpr, roc_auc, op_fpr, op_tpr, thresholds


def plot_roc(ax, fpr, tpr, roc_auc, op_fpr, op_tpr,
             title, n_benign, n_attack, color):
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random")

    # operating point at tau=0.10
    ax.scatter([op_fpr], [op_tpr], color="red", zorder=5, s=80,
               label=f"τ = {TAU}  (TPR={op_tpr:.3f}, FPR={op_fpr:.3f})")
    ax.annotate(
        f"  τ={TAU}\n  TPR={op_tpr:.3f}\n  FPR={op_fpr:.3f}",
        xy=(op_fpr, op_tpr),
        xytext=(op_fpr + 0.04, op_tpr - 0.12),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="red", lw=1),
        color="red",
    )

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (benign flagged as attack)")
    ax.set_ylabel("True Positive Rate (attack correctly detected)")
    ax.set_title(
        f"{title}\n"
        f"({n_benign} benign epochs  |  {n_attack} attack epochs  |  seed=0)"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)


# ── load data ──────────────────────────────────────────────────────────
c10_b,  _ = load_divergence(C10_BENIGN,  0)
c10_a,  _ = load_divergence(C10_ATTACK,  1)
c100_b, _ = load_divergence(C100_BENIGN, 0)
c100_a, _ = load_divergence(C100_ATTACK, 1)

print(f"CIFAR-10 : {len(c10_b)} benign epochs, {len(c10_a)} attack epochs")
print(f"  benign  divergence: [{c10_b.min():.4f}, {c10_b.max():.4f}]")
print(f"  attack  divergence: [{c10_a.min():.4f}, {c10_a.max():.4f}]")
print()
print(f"CIFAR-100: {len(c100_b)} benign epochs, {len(c100_a)} attack epochs")
print(f"  benign  divergence: [{c100_b.min():.4f}, {c100_b.max():.4f}]")
print(f"  attack  divergence: [{c100_a.min():.4f}, {c100_a.max():.4f}]")
print()

# ── CIFAR-10 ROC ───────────────────────────────────────────────────────
fpr10, tpr10, auc10, op_fpr10, op_tpr10, thr10 = compute_roc(c10_b, c10_a)

fig, ax = plt.subplots(figsize=(6, 5))
plot_roc(ax, fpr10, tpr10, auc10, op_fpr10, op_tpr10,
         "Fisher Divergence Threshold ROC — CIFAR-10",
         len(c10_b), len(c10_a), "#1f77b4")
fig.tight_layout()
out10 = os.path.join(FIG_DIR, "roc_curve_cifar10.png")
fig.savefig(out10, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out10}")

# ── CIFAR-100 ROC ──────────────────────────────────────────────────────
fpr100, tpr100, auc100, op_fpr100, op_tpr100, thr100 = compute_roc(c100_b, c100_a)

fig, ax = plt.subplots(figsize=(6, 5))
plot_roc(ax, fpr100, tpr100, auc100, op_fpr100, op_tpr100,
         "Fisher Divergence Threshold ROC — CIFAR-100",
         len(c100_b), len(c100_a), "#ff7f0e")
fig.tight_layout()
out100 = os.path.join(FIG_DIR, "roc_curve_cifar100.png")
fig.savefig(out100, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out100}")

# ── CIFAR-100 with burn_in=8 applied ──────────────────────────────────
# burn_in=8 means the detector cannot fire in epochs 0-7, so only
# epochs >= 8 are valid detection windows. This matches the actual
# operating condition used in vfl_framework.py.
c100_b_df = pd.read_csv(C100_BENIGN)
c100_a_df = pd.read_csv(C100_ATTACK)
c100_b_burn = c100_b_df[c100_b_df["epoch"] >= 8]["fisher_divergence"].values
c100_a_burn = c100_a_df[c100_a_df["epoch"] >= 8]["fisher_divergence"].values

fpr100b, tpr100b, auc100b, op_fpr100b, op_tpr100b, _ = compute_roc(c100_b_burn, c100_a_burn)

fig, axes = plt.subplots(figsize=(6, 5))
plot_roc(axes, fpr100b, tpr100b, auc100b, op_fpr100b, op_tpr100b,
         "CIFAR-100 (burn_in=8 applied — actual operating condition)",
         len(c100_b_burn), len(c100_a_burn), "#9467bd")
fig.tight_layout()
out100b = os.path.join(FIG_DIR, "roc_curve_cifar100_burnin8.png")
fig.savefig(out100b, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out100b}")

# ── Combined 4-panel figure ────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 5))
plot_roc(axes[0], fpr10,   tpr10,   auc10,   op_fpr10,   op_tpr10,
         "CIFAR-10\n(no burn_in needed)",
         len(c10_b), len(c10_a), "#1f77b4")
plot_roc(axes[1], fpr100,  tpr100,  auc100,  op_fpr100,  op_tpr100,
         "CIFAR-100\n(all epochs, raw)",
         len(c100_b), len(c100_a), "#ff7f0e")
plot_roc(axes[2], fpr100b, tpr100b, auc100b, op_fpr100b, op_tpr100b,
         "CIFAR-100\n(burn_in=8, actual operating condition)",
         len(c100_b_burn), len(c100_a_burn), "#9467bd")

# Combined: CIFAR-10 + CIFAR-100 both with burn_in context
all_b = np.concatenate([c10_b, c100_b_burn])
all_a = np.concatenate([c10_a, c100_a_burn])
fpr_all, tpr_all, auc_all, op_fpr_all, op_tpr_all, _ = compute_roc(all_b, all_a)
plot_roc(axes[3], fpr_all, tpr_all, auc_all, op_fpr_all, op_tpr_all,
         "Combined\n(CIFAR-10 + CIFAR-100 w/ burn_in=8)",
         len(all_b), len(all_a), "#2ca02c")

fig.suptitle(
    "ROC Curve: Fisher Divergence (J_A − J_B) as Attack Detector\n"
    f"Threshold τ = {TAU} marked in red  |  Data: per-epoch measurements, seed=0",
    fontsize=11
)
fig.tight_layout()
out_combined = os.path.join(FIG_DIR, "roc_curve_combined.png")
fig.savefig(out_combined, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_combined}")

# ── Print threshold sensitivity table ─────────────────────────────────
print()
print("Threshold sensitivity (CIFAR-10):")
print(f"{'tau':>8} {'TPR':>8} {'FPR':>8} {'Correctly flagged / missed'}")
for tau_val in [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:
    tp = np.sum(c10_a > tau_val)
    fp = np.sum(c10_b > tau_val)
    fn = np.sum(c10_a <= tau_val)
    tn = np.sum(c10_b <= tau_val)
    tpr_v = tp / len(c10_a)
    fpr_v = fp / len(c10_b)
    print(f"{tau_val:>8.2f} {tpr_v:>8.3f} {fpr_v:>8.3f}   "
          f"TP={tp}/{len(c10_a)}, FP={fp}/{len(c10_b)}, "
          f"FN={fn}, TN={tn}")

print()
print("Threshold sensitivity (CIFAR-100):")
print(f"{'tau':>8} {'TPR':>8} {'FPR':>8} {'Correctly flagged / missed'}")
for tau_val in [0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:
    tp = np.sum(c100_a > tau_val)
    fp = np.sum(c100_b > tau_val)
    fn = np.sum(c100_a <= tau_val)
    tn = np.sum(c100_b <= tau_val)
    tpr_v = tp / len(c100_a)
    fpr_v = fp / len(c100_b)
    print(f"{tau_val:>8.2f} {tpr_v:>8.3f} {fpr_v:>8.3f}   "
          f"TP={tp}/{len(c100_a)}, FP={fp}/{len(c100_b)}, "
          f"FN={fn}, TN={tn}")
