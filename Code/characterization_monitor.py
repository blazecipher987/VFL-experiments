import os
import csv
import numpy as np
import torch
from sklearn.metrics import silhouette_score


# ── core metrics ────────────────────────────────────────────────────────────

def fisher_criterion(embeddings: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Tr(S_B) / Tr(S_W)  — the higher this is, the more class-separable
    the embedding space is.  This is the key signal we track.

    Tr(S_B) = sum_c  n_c * ||mu_c - mu||^2
    Tr(S_W) = sum_c  sum_{x in c}  ||x - mu_c||^2
    """
    with torch.no_grad():
        classes = labels.unique()
        overall_mean = embeddings.mean(dim=0)
        trace_sw = 0.0
        trace_sb = 0.0
        for c in classes:
            mask = (labels == c)
            class_emb = embeddings[mask]
            if len(class_emb) < 2:
                continue
            class_mean = class_emb.mean(dim=0)
            n_c = float(class_emb.shape[0])
            trace_sw += ((class_emb - class_mean) ** 2).sum().item()
            trace_sb += n_c * ((class_mean - overall_mean) ** 2).sum().item()
        return trace_sb / (trace_sw + 1e-8)


def intra_class_variance(embeddings: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean per-class variance averaged across embedding dimensions.
    Lower = embeddings within each class are tighter."""
    with torch.no_grad():
        classes = labels.unique()
        total, count = 0.0, 0
        for c in classes:
            mask = (labels == c)
            class_emb = embeddings[mask]
            if len(class_emb) < 2:
                continue
            total += class_emb.var(dim=0).mean().item()
            count += 1
        return total / max(count, 1)


def inter_class_distance(embeddings: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean pairwise L2 distance between class centroids.
    Higher = class clusters are further apart."""
    with torch.no_grad():
        classes = labels.unique()
        centroids = []
        for c in classes:
            centroids.append(embeddings[labels == c].mean(dim=0))
        if len(centroids) < 2:
            return 0.0
        centroids = torch.stack(centroids)
        dists = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dists.append(torch.dist(centroids[i], centroids[j]).item())
        return float(np.mean(dists))


def compute_silhouette(embeddings: torch.Tensor, labels: torch.Tensor,
                       max_samples: int = 1000) -> float:
    """Silhouette score, subsampled for speed.  Range [-1, 1]; higher = better."""
    emb_np = embeddings.cpu().numpy()
    lab_np = labels.cpu().numpy()
    unique_classes = np.unique(lab_np)
    if len(unique_classes) < 2:
        return 0.0
    if len(emb_np) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(emb_np), max_samples, replace=False)
        emb_np, lab_np = emb_np[idx], lab_np[idx]
        # ensure every class has at least one sample after subsampling
        if len(np.unique(lab_np)) < 2:
            return 0.0
    try:
        return float(silhouette_score(emb_np, lab_np))
    except Exception:
        return 0.0


# ── monitor class ────────────────────────────────────────────────────────────

class SeparabilityMonitor:
    """
    Accumulates per-batch embeddings and gradient norms during training.
    At the end of each epoch, computes separability metrics for Party A
    and Party B separately, then logs them to CSV.

    The core hypothesis:
        Under MaliciousSGD, fisher_a will grow faster and diverge from
        fisher_b compared to benign training.  This divergence is the
        detection signal.
    """

    def __init__(self, save_dir: str, setting_str: str, dataset_name: str):
        self.save_dir = save_dir
        self.setting_str = setting_str
        self.dataset_name = dataset_name

        # per-batch accumulators (cleared after each epoch)
        self._emb_a: list = []
        self._emb_b: list = []
        self._labels: list = []
        self._grad_norms_a: list = []
        self._grad_norms_b: list = []

        # full results across all epochs
        self.results: list = []

        os.makedirs(save_dir, exist_ok=True)

    # called every batch inside simulate_train_round_per_batch
    def collect_batch(self,
                      emb_a: torch.Tensor,
                      emb_b: torch.Tensor,
                      labels: torch.Tensor,
                      grad_a: torch.Tensor,
                      grad_b: torch.Tensor) -> None:
        self._emb_a.append(emb_a.detach().cpu())
        self._emb_b.append(emb_b.detach().cpu())
        self._labels.append(labels.detach().cpu())
        self._grad_norms_a.append(grad_a.detach().norm().item())
        self._grad_norms_b.append(grad_b.detach().norm().item())

    # called once per epoch in main()
    def compute_epoch_metrics(self, epoch: int) -> dict:
        if not self._emb_a:
            return {}

        emb_a = torch.cat(self._emb_a, dim=0)
        emb_b = torch.cat(self._emb_b, dim=0)
        labels = torch.cat(self._labels, dim=0)

        fisher_a = fisher_criterion(emb_a, labels)
        fisher_b = fisher_criterion(emb_b, labels)

        metrics = {
            'epoch':            epoch,
            'fisher_a':         fisher_a,
            'fisher_b':         fisher_b,
            'fisher_divergence': fisher_a - fisher_b,
            'intra_var_a':      intra_class_variance(emb_a, labels),
            'intra_var_b':      intra_class_variance(emb_b, labels),
            'inter_dist_a':     inter_class_distance(emb_a, labels),
            'inter_dist_b':     inter_class_distance(emb_b, labels),
            'silhouette_a':     compute_silhouette(emb_a, labels),
            'silhouette_b':     compute_silhouette(emb_b, labels),
            'grad_norm_a_mean': float(np.mean(self._grad_norms_a)),
            'grad_norm_b_mean': float(np.mean(self._grad_norms_b)),
            'grad_norm_ratio':  (float(np.mean(self._grad_norms_a))
                                 / (float(np.mean(self._grad_norms_b)) + 1e-8)),
        }

        self.results.append(metrics)
        self._clear_buffers()

        print(
            f"[Monitor Ep{epoch:03d}] "
            f"Fisher A={fisher_a:.4f}  B={fisher_b:.4f}  "
            f"Div={metrics['fisher_divergence']:+.4f}  "
            f"Sil_A={metrics['silhouette_a']:.4f}  "
            f"GradRatio={metrics['grad_norm_ratio']:.4f}"
        )
        return metrics

    def save_to_csv(self) -> None:
        if not self.results:
            return
        path = os.path.join(
            self.save_dir,
            f"separability_{self.dataset_name}{self.setting_str}.csv"
        )
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.results[0].keys()))
            writer.writeheader()
            writer.writerows(self.results)
        print(f"[Monitor] Saved to {path}")

    def _clear_buffers(self) -> None:
        self._emb_a.clear()
        self._emb_b.clear()
        self._labels.clear()
        self._grad_norms_a.clear()
        self._grad_norms_b.clear()
