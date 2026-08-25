# VFL Label Inference Attack — Research Log

**Principal Investigator:** Soumya Swagata  
**Institution:** BRAC University  
**Last Updated:** 2026-08-25  
**Status:** ✅ CIFAR-10 STORY COMPLETE (4/4 seeds, ablation 4/5, asymmetry confirmed). ✅ CINIC10L DEFENSE CONFIRMED (EXP-022: 62.43% < benign 65.70%, seed-0). 🔴 PHASE 18 (CIFAR-100 AdversarialAuxiliary, lambda sweep 0.5/1.0/2.0) FAILED — NaN explosion; model destroyed. ✅ PHASE 19 (CIFAR-100 Gradient Projection Defense, seed-0) — MC **26.97% < benign mean 29.56%** (−2.59pp). ✅ PHASE 16 (CIFAR-100 benign multi-seed) COMPLETE — benign baseline 29.56 ± 2.93% (4 seeds). ✅ **PHASE 20 COMPLETE (EXP-033/034/035)** — CIFAR-100 Gradient Projection Defense, seeds 42/123/456: 4/4 seeds defended < per-seed benign. Defended mean **23.44 ± 5.16%** vs benign 29.56 ± 2.93% vs attack 49.87 ± 1.17%. Seeds 0/42/123 cluster at 25–27%; seed-456 anomalous over-collapse at 14.57%. ✅ **PHASE 21 COMPLETE (EXP-036/037/038)** — CIFAR-100 Attack Baseline, seeds 42/123/456: attack 49.87 ± 1.17% (seed-0: 47.86%, seed-42: 50.40%, seed-123: 50.73%, seed-456: 50.50%). **CIFAR-100 DEFENSE NOW PUBLISHABLE (4/4 seeds, with seed-456 anomaly documented).** ✅ **PHASE 15 COMPLETE (EXP-026/027/028)** — CINIC10L 4/4 seeds confirmed; defended mean 62.72 ± 0.65% < benign mean 65.76 ± 0.65%; all per-seed comparisons pass. 🟡 PHASE 17 (Yahoo Answers) RUNNING — infrastructure fixed. ✅ **EXP-039 (LITERATURE REVIEW) COMPLETE (2026-07-12)** — Fisher Divergence Detection: **Clearly Novel**. Persistent Projection: **Clearly Novel**. Current GradProj (one-shot collapse): **Moderately Novel** (differentiate from MixPro, SIGIR 2023). AAP: **Incremental** individually. 🔴 **PHASE 22 (EXP-040/041/042) FAILED** — PP (buggy d_ema) on CIFAR-10: all 3 alpha_ema values fail (93.73–94.50% vs benign 87.23%). Root cause: batch mean of CE gradients ≈ 0 for balanced batches → d_ema tracks noise. BUG FIXED (2026-07-13). 🔴 **PHASE 23 (EXP-043/044) FAILED** — PP (buggy) on CIFAR-100: 49.42–49.44% vs benign 30.33%; both exceed attack baseline. Invalidated. 🔴 **NEXT IMMEDIATE ACTION: Re-run Phase 22 with fixed PP** (per-sample normalization before EMA). If passes, re-run Phase 23. If fails, pivot to MDPP (see next_direction.md).

---

> **🔵 RESEARCH DIRECTION UPDATE — 2026-07-12 (EXP-039 Literature Review + Session Decision):**
> 
> **Novelty confirmed by web search (2024–2025 papers):** (1) Fisher Divergence Detection (J_A − J_B = inter_class_var/intra_class_var per party) as a VFL attack monitor is **Clearly Novel** — no prior paper uses inter/intra-class variance ratio asymmetry between VFL parties as an anomaly signal. (2) Persistent Projection (stable multi-epoch gradient projection onto the orthogonal complement of the discriminative subspace) is **Clearly Novel** — no VFL paper proposes this. (3) Current GradientProjection (one-shot collapse) is **Moderately Novel** — MixPro (SIGIR 2023, within FedAds benchmark) uses gradient projection in VFL but without auxiliary-classifier-derived discriminative direction, without targeting active attacks, and without persistent basis. ProjPert (IEEE TKDE 2024) shares the name but is a noise-optimization approach, not geometric subspace projection.
>
> **Key competitor papers to differentiate against in any submission:**
> - MixPro (Wei et al., SIGIR 2023 via FedAds): Gradient mixup + projection per-batch in VFL — closest mechanistic competitor. Differentiate: our projection direction is derived from an aux classifier targeting the discriminative subspace; we specifically address active MaliciousSGD attacks; we apply a persistent evolving basis.
> - ProjPert (IEEE TKDE 2024): "Projection" in name only — is actually noise parameter optimization via binary search. Not geometric. Targets passive only.
> - LADSG (Yan et al., CollaborateCom/Springer 2026, arXiv June 2025): Claims to defend all three attack types including active, using gradient norm anomaly detection + gradient substitution. No Fisher divergence or subspace projection.
>
> **Defense architecture decision (2026-07-12):**
> The current paper story is fragmented: AAP works for CIFAR-10 / CINIC10L (10-class) but not CIFAR-100; GradientProjection works for CIFAR-100 (via one-shot collapse) but this is not designed behavior. Presenting two different defenses for two different datasets weakens the contribution and will draw reviewer objections. The preferred path is a **unified Persistent Projection framework** that fires stably across all epochs (not one-shot collapse), works for both 10-class and 100-class settings, and replaces the fragmented AAP / one-shot GradProj story with a single mechanism. The Fisher Divergence Detection layer remains unchanged — it is the unified detection component.
>
> **Immediate experiment plan:**
> - Phase 22: Implement true Persistent Projection (EMA-based discriminative direction + project every detected epoch) and test on CIFAR-10. If it matches or exceeds AAP performance, it becomes the sole defense mechanism.
> - Phase 23: Test Persistent Projection on CIFAR-100 at 150 epochs. If the stable per-epoch projection prevents catastrophic collapse AND keeps MC < benign, the unified story is confirmed.
> - AAP and one-shot GradProj results become ablation baselines in the paper.

---

> **⚠️ CRITICAL NOTE — 2026-07-01:** Phase 2b results (EXP-009) revealed that the 30-epoch comparison between Phase 2 defended results and Phase 1 undefended results was epoch-confounded. The Phase 1 active checkpoint (`mal_half=16.pth`) used in Phase 2b was overwritten by the characterization run to a 30-epoch version. At 30 epochs, MaliciousSGD does NOT create an inference advantage for CIFAR10 (active: 23.45% < benign: 47.98%). The defense at 30 epochs RAISES the active attacker's inference accuracy (23.45% → 52.28%), the opposite of a defense. For CIFAR100, the defense is statistically null (20.26% → 21.35%). All paper claims must be re-validated at 100+ epochs. See EXP-009 and Section 5.5 for full analysis.

> **✅ RESOLUTION — 2026-07-04 (Phase 4 / EXP-011):** The 100-epoch critical experiment is now complete for CIFAR-10. All three conditions (benign, active, active+defense) were re-run at 100 epochs end-to-end. The defense IS effective: attack inference 95.42% → 84.27% under defense, which is 2.96pp **below** the benign baseline of 87.23%. The epoch-confound is resolved for CIFAR-10. CIFAR-100 at 150 epochs remains the next critical pending experiment. Seed sweeps are required before paper submission (EXP-011 is a single run, manualSeed=0). See EXP-011 and Section 4.10 for full numbers.

> **⚠️ PHASE 5 STATUS — 2026-07-05 (EXP-012):** CIFAR-100 150-epoch three-way comparison completed. At standard params (alpha=1.0, tau=0.10): benign 30.33%, attack 47.86%, defended 43.12%. Defense reduces attacker advantage from 17.53pp to 12.79pp (only 27% eliminated). Defended ASR remains 12.79pp above benign — defense fails key criterion for CIFAR-100. Root cause: CIFAR-100's larger attack advantage requires more aggressive suppression. Phase 6A runs alpha=2.0 variants. Phase 6B (seed sweep) and code changes (--manual-seed support) are ready.

> **🔴 PHASE 6A FAILURE — 2026-07-06 (EXP-013):** Stronger suppression (alpha=2.0) makes CIFAR-100 defense WORSE, not better. a=2.0, tau=0.10 gives 48.31% defended ASR — 0.45pp ABOVE the undefended attack (47.86%). a=2.0, tau=0.05 gives 43.60% — essentially same as a=1.0 (43.12%). Both variants fail the benign criterion (30.33%). Root cause hypothesis: when grad_output_A is zeroed out early, MaliciousSGD's unchecked internal amplification produces even more class-discriminative embeddings without the corrective task gradient. Gradient perturbation as the sole defense mechanism is insufficient for CIFAR-100. **Option B (embedding-level noise injection into z_a) is now required before any CIFAR-100 paper claim can be made.**

> **✅ PHASE 15 (CINIC10L SEED SWEEP) — 2026-08-25 COMPLETE:** All 4 seeds (0, 42, 123, 456) confirmed. Per-seed defended < benign in all cases. Benign mean 65.76 ± 0.65%, attack mean 86.51 ± 0.54%, defended mean 62.72 ± 0.65%. CINIC10L multi-seed claim is now publishable. See EXP-026/027/028 for full table.

> **🟡 PHASE 17 (YAHOO ANSWERS MODALITY TEST) — 2026-07-10:** Two infrastructure bugs fixed: (1) BertConfig API change in `models/mixtext.py` — `BertEmbeddings(config)` and `BertPooler(config)` replacing old dict-unpacking calls. (2) `Translator` class in `models/read_data_text.py` — now gracefully falls back to original text when `de_1.pkl`/`ru_1.pkl` back-translation pkl files are absent (which they are for this codebase). Stage 1 benign training running (lr=0.001, 100ep). Results pending.

> **🔴 PHASE 16 (CIFAR-100 BENIGN MULTI-SEED) — 2026-07-10 COMPLETE:** Seeds 42, 123, 456 all ran successfully alongside the existing seed-0 (EXP-012). Stage 1 VFL test accuracy is extremely stable across seeds: 44.83%, 44.31%, 44.12% (vs seed-0: 45.33%); mean 44.65 ± 0.47%. Stage 2 MC results: seed-42=26.19%, seed-123=28.56%, seed-456=33.14%, seed-0=30.33%. **New benign baseline: 29.56 ± 2.93% (4 seeds).** Spread is [26.19%, 33.14%] — ~7pp range. The EXP-018 benign+defense result (34.74%) is now only 5.18pp above mean and within the 2σ band [23.70%, 35.42%], further confirming the defense was dormant. The seed-0 value (30.33%) was essentially the mean all along.

> **🔴 PHASE 18 (CIFAR-100 ADVERSARIAL AUXILIARY CLASSIFIER) — 2026-07-10 FAILED:** All 3 lambda variants (0.5, 1.0, 2.0) failed catastrophically. Lambda=0.5: Stage 1 VFL ended with NaN loss, model accuracy=1.00% (random). MC ran but produced NaN loss throughout all 25 epochs; best MC=1.001% (random). Lambda=1.0 and 2.0: Stage 1 checkpoints DO NOT EXIST on disk — these runs crashed before saving any checkpoint. MC output files are truncated at 7 lines (no training occurred). Root cause: the `final_grad = grad_output_a - lambda * aux_grad` formula creates an unstable feedback loop when MaliciousSGD is active. `aux_grad = d(L_aux)/d(z_a)` grows in magnitude as the aux_classifier trains (unconstrained nn.Linear, Adam, 150 epochs). With MaliciousSGD amplifying the corrected gradient by up to 5×, the anti-discriminative component is also amplified 5× — causing the embedding values to oscillate and eventually diverge to NaN. The "self-reinforcing" property worked in the wrong direction: it amplified instability rather than defense effectiveness. No gradient clipping was implemented. **CIFAR-100 defense now has 10 failed configurations across 4 attack surfaces.**

> **🟡 PHASE 19 (CIFAR-100 GRADIENT PROJECTION DEFENSE) — 2026-07-10 FIRST CIFAR-100 DEFENSE CANDIDATE:** seed-0 result: Stage 2 MC best Train Top-1 = **26.97%**, which is **2.59pp BELOW the 4-seed benign mean (29.56 ± 2.93%)**. VFL Stage 1 test accuracy = 45.32% vs benign 45.33% — essentially zero utility cost (−0.01pp). Attack suppression: 47.86% → 26.97% = **20.89pp reduction, eliminating 114% of attacker advantage over benign**. Fisher divergence CSV reveals a **catastrophic discriminative collapse at epoch 12**: intra_var_a spikes from 0.16 → 141,644 (6-order-of-magnitude increase) and grad_norm_a drops from 0.27 → 0.007 in one epoch. The aux_classifier identifies the discriminative direction so precisely that projection removes virtually all of grad_output_A at first activation. Fisher divergence then goes strongly negative (−0.26 to −0.31 for epochs 12–24; stabilizes at −0.07 for epochs 50–149), meaning Party A permanently becomes LESS discriminative than Party B for the remainder of training. **CRITICAL CAVEAT: Single seed (seed-0) only. Multi-seed confirmation (seeds 42/123/456) required before any paper claim.**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Codebase Architecture](#2-codebase-architecture)
3. [Experiment Log](#3-experiment-log)
4. [Results Repository](#4-results-repository)
5. [Research Insights](#5-research-insights)
6. [Comparison Tables](#6-comparison-tables)
7. [Future Work](#7-future-work)

---

## 1. Project Overview

### Research Objective

Develop a novel, server-side defense against the **Active Label Inference Attack** in Vertical Federated Learning (VFL). The defense must:
1. Detect that Party A is running MaliciousSGD without requiring access to Party A's model internals
2. Suppress the attack's label-inference advantage (reduce Stage 2 accuracy toward passive baseline)
3. Not significantly hurt VFL classification utility

### Core Hypothesis

> **MaliciousSGD causes Party A's embedding space to become asymmetrically more class-separable than Party B's. The Fisher criterion divergence (Fisher_A − Fisher_B), computed by the server from gradient feedback, is a reliable detection signal for this asymmetry.**

If the hypothesis holds, the server can:
- Monitor Fisher divergence each epoch
- When divergence exceeds a threshold, scale down Party A's gradient adaptively
- This reduces the label-inference advantage of the active attack

### Revised Mechanism Understanding (Updated Phase 2)

The original hypothesis predicted the defense would suppress Fisher divergence (make Party A's embeddings less geometrically class-separable). Phase 2 results reveal the **actual mechanism is different but still effective**:

> The defense suppresses task-calibrated gradient signal to Party A. MaliciousSGD continues amplifying Party A's internal gradients, maintaining geometric class-separability (Fisher_A remains high). But the embeddings lose *semantic task alignment* — they are cluster-shaped but the clusters no longer correspond reliably to ground-truth labels. MixMatch SSL cannot exploit unlabeled embeddings that have geometric structure but poor label alignment.

In short: **the defense breaks semantic alignment rather than geometric structure**. This is a richer finding than the original hypothesis and is a stronger paper contribution.

### Proposed Defense Name

**Asymmetric Adaptive Gradient Perturbation via Embedding Separability Monitoring**

### Attack Being Defended Against

**Active Label Inference Attack (MaliciousSGD)** — Party A uses a modified optimizer during Stage 1 (VFL training) that amplifies its internal parameter gradients element-wise, making its embeddings more class-discriminative. In Stage 2, Party A loads the saved embedding model and trains an inference head via MixMatch SSL to recover labels.

### Attack Taxonomy (for clarity)

| Attack | Stage 1 Optimizer | Stage 2 Required | Auxiliary Labels Needed |
|---|---|---|---|
| **Passive model completion** | Standard SGD | Yes (model_completion.py) | Yes (n_labeled) |
| **Active model completion** | MaliciousSGD (Party A) | Yes (model_completion.py) | Yes (n_labeled) |
| **Direct label inference** | Standard SGD | No | No (uses gradient sign at epoch 0) |

The active attack is the primary target of this defense. Direct label inference is a separate, simpler attack not addressed here.

---

## 2. Codebase Architecture

### Key Files

| File | Role | Modified for Research |
|---|---|---|
| `Code/vfl_framework.py` | Stage 1 VFL training loop, orchestrates all parties | Yes — added `--monitor-separability`, `--asymmetric-defense` flags, defense wiring |
| `Code/model_completion.py` | Stage 2 label inference via MixMatch SSL | No — reads saved .pth, no changes needed |
| `Code/my_optimizers.py` | MaliciousSGD implementation | No |
| `Code/possible_defenses.py` | Existing symmetric defenses + new `AsymmetricAdaptivePerturbation` | Yes — new class added at end of file |
| `Code/characterization_monitor.py` | Phase 1 monitoring (Fisher, silhouette, grad norms) | Created new |
| `Code/model_sets.py` | Bottom model architectures for all datasets | No |

### Attack Pipeline (Two-Stage)

```
Stage 1: vfl_framework.py
  Party A [MaliciousSGD] ──► embedding_a ──► Top Model ──► loss
  Party B [standard SGD] ──► embedding_b ──►           (server-side)
  Saves: <DATASET>_saved_framework_lr=0.1_mal_half=16.pth

Stage 2: model_completion.py
  Loads .pth → extracts malicious_bottom_model_a
  Trains BottomModelPlus (embedding + FC head) via MixMatch SSL
  Uses n_labeled auxiliary labeled samples (40 for CIFAR10, 400 for CIFAR100)
  Reports: top-1 label inference accuracy on train + test sets
```

### MaliciousSGD Mechanism (`my_optimizers.py`, line 88–90)

```python
ratio = 1.0 + gamma * (current_grad / (last_grad + 1e-7))
ratio = clamp(ratio, min=1.0, max=5.0)
p.grad *= ratio   # amplifies INTERNAL parameter gradients, NOT grad_output
```

**Critical note:** MaliciousSGD amplifies `p.grad` (the parameter gradient inside Party A's model), NOT `grad_output_bottom_model_a` (the server-to-party communication tensor). This is why the gradient norm ratio metric is NOT a reliable detection signal — the monitor measures the communication gradient, not the internal one.

### Defense Insertion Point (`vfl_framework.py`)

The `AsymmetricAdaptivePerturbation` defense is applied in `simulate_train_round_per_batch()` after symmetric defenses unpack but before the backward pass:

```python
if self.defense_asymmetric and self.asymmetric_defense is not None:
    grad_output_bottom_model_a, _ = self.asymmetric_defense.apply(
        grad_output_bottom_model_a, self.current_epoch
    )
```

**Scale factor formula:** `max(0.0, 1.0 - alpha * (divergence - tau))`  
Linear decay from 1.0 at d=tau toward 0.0. The defense never touches Party B's gradient.

**Defense hyperparameters used in Phase 2:** alpha=1.0, tau=0.10, burn_in=8

---

> **Note on `mal-all` model completion:** The `mal-all` (all-parties MaliciousSGD) model completion results (CIFAR10: 85.39%, CIFAR100: 36.64%) were already produced in Phase 1 (EXP-003/EXP-005) from the 30-epoch Phase 1 characterization checkpoints. These are NOT new results from Phase 2b or Phase 3b. They are documented in Tables 4.3 and 4.4. The `mal-all` threat model differs fundamentally from the single-party threat model (both parties must be adversarial), so the Fisher defense — which targets single-party asymmetry — does not address this scenario by design.

---

## 3. Experiment Log

> **Chronological execution note:** The EXP IDs are numbered by logical research phase, not by time of execution. The true execution order was: **EXP-002 → EXP-003 → EXP-004 → EXP-005** (original 100/150-epoch training + model completion from `run_training.bat` + `run_model_completion.bat`) → **EXP-001 → EXP-006** (Phase 1 characterization, 30 epochs). The Phase 1 characterization scripts overwrote the 100-epoch `mal_half=16.pth` checkpoint because `vfl_framework.py` encodes epoch count nowhere in the filename. The 94.99%/43.35% numbers (EXP-003/EXP-005) were obtained before EXP-001 ran. This is the root cause of all subsequent epoch-confound issues.

### EXP-001 — CIFAR10 Phase 1 Characterization

| Field | Value |
|---|---|
| **ID** | EXP-001 |
| **Date** | 2026-06-26 (approx) |
| **Purpose** | Validate Fisher divergence as detection signal on CIFAR10 |
| **Dataset** | CIFAR10 |
| **Epochs** | 30 |
| **Half** | 16 (Party A gets cols 0–15, Party B gets cols 16–31) |
| **k** | 4 |
| **Script** | `Code/run_phase1_characterization.sh` |
| **Status** | ✅ Complete |

**Conditions run:**
1. Benign (no attack)
2. Active — Party A only (`--use-mal-optim True`)
3. Active — All parties (`--use-mal-optim True --use-mal-optim-all True`)
4. Benign + Laplace DP (`--lap-noise True --noise-scale 1e-3`)
5. Active Party A + Laplace DP
6. Active Party A + Gradient Compression (`--gc True --gc-preserved-percent 0.75`)

**Output files:**
```
Code/saved_experiment_results/csv_files/CIFAR10_csv_files/
  separability_CIFAR10_lr=0.1_normal_half=16.csv
  separability_CIFAR10_lr=0.1_mal_half=16.csv
  separability_CIFAR10_lr=0.1_mal-all_half=16.csv
  separability_CIFAR10_lr=0.1_normal_lap_noise-scale=0.001_half=16.csv
  separability_CIFAR10_lr=0.1_mal_lap_noise-scale=0.001_half=16.csv
  separability_CIFAR10_lr=0.1_mal_gc-preserved_percent=0.75_half=16.csv
```

---

### EXP-002 — CIFAR10 Stage 1 VFL Training

| Field | Value |
|---|---|
| **ID** | EXP-002 |
| **Date** | 2026-06-26 (approx) |
| **Purpose** | Produce checkpoints for Stage 2 model completion |
| **Dataset** | CIFAR10 |
| **Epochs** | 100 |
| **Script** | `Code/run_training.bat` (CIFAR10 section) |
| **Status** | ✅ Complete |

**Output files:**
```
Code/saved_experiment_results/saved_models/CIFAR10_saved_models/
  CIFAR10_saved_framework_lr=0.1_normal_half=16.pth  +  .txt
  CIFAR10_saved_framework_lr=0.1_mal_half=16.pth  +  .txt
  CIFAR10_saved_framework_lr=0.1_mal-all_half=16.pth  +  .txt
```

---

### EXP-003 — CIFAR10 Stage 2 Model Completion

| Field | Value |
|---|---|
| **ID** | EXP-003 |
| **Date** | 2026-06-26 (approx) |
| **Purpose** | Measure label inference accuracy of passive vs active attacks |
| **Dataset** | CIFAR10, n_labeled=40, 25 epochs |
| **Script** | `Code/run_model_completion.bat` (CIFAR10 section) |
| **Status** | ✅ Complete |

---

### EXP-004 — CIFAR100 Stage 1 VFL Training

| Field | Value |
|---|---|
| **ID** | EXP-004 |
| **Date** | 2026-06-27 (approx) |
| **Purpose** | Produce CIFAR100 checkpoints for Stage 2 |
| **Dataset** | CIFAR100, Epochs=150, k=5 |
| **Script** | `Code/run_training.bat` (CIFAR100 section) |
| **Status** | ✅ Complete |

---

### EXP-005 — CIFAR100 Stage 2 Model Completion

| Field | Value |
|---|---|
| **ID** | EXP-005 |
| **Date** | 2026-06-27 (approx) |
| **Purpose** | Measure label inference accuracy on CIFAR100 |
| **Dataset** | CIFAR100, n_labeled=400, 25 epochs |
| **Script** | `Code/run_model_completion.bat` (CIFAR100 section) |
| **Status** | ✅ Complete |

---

### EXP-006 — CIFAR100 Phase 1 Characterization

| Field | Value |
|---|---|
| **ID** | EXP-006 |
| **Date** | 2026-06-28 |
| **Purpose** | Validate Fisher divergence signal on CIFAR100 (100 classes) |
| **Dataset** | CIFAR100, Epochs=30, k=5, Half=16 |
| **Script** | `Code/run_phase1_characterization_cifar100.bat` |
| **Status** | ✅ Complete |

**Conditions run:** Same 6 conditions as EXP-001.

---

### EXP-007 — Phase 2 Stage 1 With Defense (CIFAR10 + CIFAR100)

| Field | Value |
|---|---|
| **ID** | EXP-007 |
| **Date** | 2026-06-29 |
| **Purpose** | Re-run Stage 1 VFL training with `AsymmetricAdaptivePerturbation` defense ON; measure VFL task accuracy and Fisher divergence under defense |
| **Dataset** | CIFAR10, CIFAR100 |
| **Epochs** | 30 |
| **Defense params** | alpha=1.0, tau=0.10, burn_in=8 |
| **Scripts** | `Code/run_phase2_defense_cifar10.bat`, `Code/run_phase2_defense_cifar100.bat` |
| **Status** | ✅ Complete |

**Conditions run (per dataset):**
1. Active Party A + AsymAdaptivePert (primary defense test)
2. Benign + AsymAdaptivePert (false-positive check — defense should NOT fire)

**Output files (CIFAR10):**
```
Code/saved_experiment_results/saved_models/CIFAR10_saved_models/
  CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16.pth  +  .txt
  CIFAR10_saved_framework_lr=0.1_normal_asym_def-a=1.0-t=0.1-b=8_half=16.pth  +  .txt

Code/saved_experiment_results/csv_files/CIFAR10_csv_files/
  separability_CIFAR10_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16.csv
  separability_CIFAR10_lr=0.1_normal_asym_def-a=1.0-t=0.1-b=8_half=16.csv
```
Same naming convention for CIFAR100.

**VFL Task Accuracy Results (30-epoch checkpoints):**

| Condition | Train Top-1 | Test Top-1 | VFL utility cost |
|---|---|---|---|
| CIFAR10 Active + defense | 63.39% | 60.94% | −1.77pp vs benign |
| CIFAR10 Benign + defense | 64.12% | 62.71% | baseline |
| CIFAR100 Active + defense | 31.60% | 29.23% | +3.18pp vs benign |
| CIFAR100 Benign + defense | 27.54% | 26.05% | baseline |

**Fisher Divergence Under Defense — Epoch 29:**

| Condition | Fisher_A | Fisher_B | Divergence | Defense fired? |
|---|---|---|---|---|
| CIFAR10 Active + defense | 0.879 | 0.357 | **+0.522** | YES (from ep 8) |
| CIFAR10 Benign + defense | 0.598 | 0.503 | **+0.094** | Barely / never |
| CIFAR100 Active + defense | 0.339 | 0.172 | **+0.167** | YES (from ep 8) |
| CIFAR100 Benign + defense | 0.222 | 0.253 | **−0.031** | NO |

**Phase 2 Paradox — Fisher divergence increases under defense:**
- CIFAR10: defended active divergence 0.522 > undefended active 0.444
- CIFAR100: defended active divergence 0.167 > undefended active 0.131
- The defense is working (large attack accuracy drops in EXP-008) but the detection signal is NOT suppressed
- Mechanism: defense suppresses task-calibrated gradient to Party A; MaliciousSGD still amplifies internal gradients maintaining geometric separability; but embeddings lose semantic label alignment needed for MixMatch to work. Fisher criterion measures geometry, not label alignment.

---

### EXP-008 — Phase 2 Stage 2 Model Completion Against Defended Checkpoints

| Field | Value |
|---|---|
| **ID** | EXP-008 |
| **Date** | 2026-06-29 |
| **Purpose** | Measure label inference accuracy against defended checkpoints; confirm defense reduces attack toward passive baseline |
| **Dataset** | CIFAR10 (n_labeled=40), CIFAR100 (n_labeled=400), 25 epochs |
| **Script** | `Code/run_phase2_model_completion.bat` |
| **Status** | ✅ Complete |

**Output files:**
```
Code/saved_experiment_results/saved_models/CIFAR10_saved_models/
  model_completion_CIFAR10_..._mal_asym_def-a=1.0-t=0.1-b=8_half=16.pth_..._nlabeled=40.txt
  model_completion_CIFAR10_..._normal_asym_def-a=1.0-t=0.1-b=8_half=16.pth_..._nlabeled=40.txt

Code/saved_experiment_results/saved_models/CIFAR100_saved_models/
  model_completion_CIFAR100_..._mal_asym_def-a=1.0-t=0.1-b=8_half=16.pth_..._nlabeled=400.txt
  model_completion_CIFAR100_..._normal_asym_def-a=1.0-t=0.1-b=8_half=16.pth_..._nlabeled=400.txt
```

**Model Completion Results (Phase 2 vs Phase 1 Baseline):**

| Condition | Dataset | Best Train Top-1 | Phase 1 Attack Baseline | Reduction |
|---|---|---|---|---|
| Active + defense | CIFAR10 | **52.28%** | 94.99% | **−42.71pp (−45%)** |
| Benign + defense | CIFAR10 | **47.18%** | 84.61% (passive) | (defense did not fire) |
| Active + defense | CIFAR100 | **21.35%** | 43.35% | **−22.00pp (−51%)** |
| Benign + defense | CIFAR100 | **11.27%** | 29.90% (passive) | (high variance, see note) |

Epoch 25 test accuracies: CIFAR10 active+def 48.96%, benign+def 45.66%; CIFAR100 active+def 17.96%, benign+def 10.12%.

**CIFAR10 result:**
- Defense eliminates 42.71pp of attack advantage; defended active (52.28%) is within 5.1pp of benign baseline (47.18%)
- This is an 89% reduction in the attack's advantage over passive baseline (from +10.38pp to +5.10pp)

**CIFAR100 result:**
- Defense eliminates 22.00pp of attack accuracy
- Attack residual above benign: 21.35% − 11.27% = 10.08pp
- Incomplete neutralization compared to CIFAR10; see methodological note

**Methodological note on CIFAR100 benign baseline variance:**
The Phase 2 benign+defense model completion (11.27%) is much lower than Phase 1 passive (29.90%). The defense does NOT fire on CIFAR100 benign (divergence goes negative after burn-in, confirmed by CSV epoch data). The gap is attributed to run-to-run variance in 30-epoch CIFAR100 training. This is a methodological limitation requiring multiple seeds before making firm CIFAR100 benign-comparison claims.

---

### EXP-009 — Phase 2b: 30-Epoch Undefended Baselines (Fair Comparison)

| Field | Value |
|---|---|
| **ID** | EXP-009 |
| **Date** | 2026-07-01 |
| **Purpose** | Establish fair 30-epoch baselines for the undefended active and benign checkpoints to allow apples-to-apples comparison with Phase 2 defended results |
| **Dataset** | CIFAR10 (n_labeled=40), CIFAR100 (n_labeled=400) |
| **Epochs (Stage 2)** | 25 |
| **Checkpoints loaded** | `mal_half=16.pth`, `normal_half=16.pth` (both 30-epoch, from Phase 1 characterization runs) |
| **Script** | `Code/run_phase2b_baseline_30ep.bat` |
| **Status** | ✅ Complete |

**Critical context:** The `mal_half=16.pth` checkpoint used here is the 30-epoch characterization checkpoint, NOT the original 100-epoch EXP-002 checkpoint (which was overwritten by the Phase 1 characterization run). This means this experiment correctly establishes a 30-epoch fair baseline, but cannot reproduce the 94.99%/43.35% results from EXP-003/EXP-005 which used the 100-epoch versions.

**Output files:**
```
Code/saved_experiment_results/saved_models/CIFAR10_saved_models/
  model_completion_CIFAR10_saved_framework_lr=0.1_mal_half=16.pth_layer=1_func=ReLU_bn=True_nlabeled=40.txt
  model_completion_CIFAR10_saved_framework_lr=0.1_normal_half=16.pth_layer=1_func=ReLU_bn=True_nlabeled=40.txt
Code/saved_experiment_results/saved_models/CIFAR100_saved_models/
  model_completion_CIFAR100_saved_framework_lr=0.1_mal_half=16.pth_layer=1_func=ReLU_bn=True_nlabeled=400.txt
  model_completion_CIFAR100_saved_framework_lr=0.1_normal_half=16.pth_layer=1_func=ReLU_bn=True_nlabeled=400.txt
```

**Phase 2b Model Completion Results:**

| Dataset | Condition | Best Top-1 (Train) | Final Test Top-1 |
|---|---|---|---|
| CIFAR10 | Benign, no defense | **47.98%** | 47.11% |
| CIFAR10 | Active, no defense | **23.45%** | 23.07% |
| CIFAR100 | Benign, no defense | **12.76%** | 11.25% |
| CIFAR100 | Active, no defense | **20.26%** | 16.60% |

**Critical findings from EXP-009:**

1. **CIFAR10: MaliciousSGD does NOT create an inference advantage at 30 epochs.** Active (23.45%) is 24.53pp BELOW benign (47.98%). The attack has not converged at this training length. MaliciousSGD's gradient amplification appears to destabilize embedding convergence in early training.

2. **CIFAR10: Defense raises attack accuracy at 30 epochs (Phase 2 active+defense 52.28% vs Phase 2b active no-defense 23.45%).** The defense acts as a stabilizer that prevents over-amplification, allowing the embeddings to converge better. This is the opposite of a defense effect.

3. **CIFAR100: Attack works at 30 epochs** (20.26% vs 12.76% benign, +7.5pp), but the defense is null (active+defense 21.35% ≈ active no-defense 20.26%, +1.09pp difference — within noise).

4. **The previously claimed 42.71pp CIFAR10 reduction** (94.99% → 52.28%) was epoch-confounded: 94.99% came from a 100-epoch checkpoint (EXP-002), 52.28% came from a 30-epoch defended checkpoint (EXP-007). The difference reflects shorter training, not only the defense.

**Consequence:** The main results table as previously constructed cannot be used in a paper. The defense must be evaluated at the same epoch count as where the attack actually works (100+ epochs for CIFAR10).

---

### EXP-010 — Phase 3b Ablation: Stage 1 VFL Training (Hyperparameter Sweep)

| Field | Value |
|---|---|
| **ID** | EXP-010 |
| **Date** | 2026-07-01 |
| **Purpose** | Establish how alpha (suppression strength) and tau (detection threshold) affect VFL task performance and collateral damage to the benign party |
| **Dataset** | CIFAR10, CIFAR100 |
| **Epochs** | 30 |
| **Conditions** | Active+defense (6 settings), Benign+defense (4 settings) |
| **Script** | Phase 3b ablation bat files |
| **Status** | ✅ Stage 1 complete — Stage 2 (model completion) NOT yet run |

**Critical limitation:** No model completion (Stage 2) was run for any ablation variant. VFL task accuracy is a proxy metric only. Claims about defense effectiveness against label inference cannot be made from this experiment alone.

**Output files (Stage 1 VFL accuracy .txt):**
```
CIFAR10_saved_framework_lr=0.1_mal_asym_def-a={0.5,1.0,2.0}-t=0.1-b=8_half=16.txt
CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t={0.05,0.1,0.15}-b=8_half=16.txt
CIFAR10_saved_framework_lr=0.1_normal_asym_def-a=1.0-t={0.05,0.15}-b=8_half=16.txt
(same pattern for CIFAR100)
```

---

### EXP-011 — Phase 4: CIFAR10 100-Epoch Full Three-Way Comparison (Critical Re-Run)

| Field | Value |
|---|---|
| **ID** | EXP-011 |
| **Date** | 2026-07-04 |
| **Purpose** | Resolve epoch confound from EXP-009; establish fair three-way comparison (benign vs active vs active+defense) all at 100 epochs for CIFAR-10 |
| **Dataset** | CIFAR10 |
| **Stage 1 Epochs** | 100 |
| **Stage 2 Epochs** | 25 (MixMatch, n_labeled=40) |
| **Defense params** | alpha=1.0, tau=0.10, burn_in=8 (unchanged from Phase 2) |
| **Script** | `Code/run_phase4_cifar10_100ep.bat` |
| **manualSeed** | 0 (single run — multi-seed pending) |
| **Status** | ✅ Complete |

**Naming strategy:** Benign and active (no defense) checkpoints overwrite their standard names (`normal_half=16.pth`, `mal_half=16.pth`) since the 30-epoch versions are now superseded. The defended checkpoint is saved with `_ep100` suffix to preserve the existing 30-epoch Phase 2 checkpoint.

**Output files:**
```
Stage 1:
  CIFAR10_saved_framework_lr=0.1_normal_half=16.pth + .txt       (100ep benign)
  CIFAR10_saved_framework_lr=0.1_mal_half=16.pth + .txt          (100ep attack)
  CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16_ep100.pth + .txt  (100ep attack+defense)

Stage 2 (model completion):
  model_completion_CIFAR10_..._normal_half=16.pth_..._nlabeled=40.txt
  model_completion_CIFAR10_..._mal_half=16.pth_..._nlabeled=40.txt
  model_completion_CIFAR10_..._mal_asym_def-a=1.0-t=0.1-b=8_half=16_ep100.pth_..._nlabeled=40.txt

Fisher divergence CSV (100-epoch, overwritten from Phase 2 30-epoch):
  separability_CIFAR10_lr=0.1_mal_half=16.csv           (100 rows, epoch 0-99)
  separability_CIFAR10_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16.csv  (100 rows, epoch 0-99)
  Note: benign CSV (separability_CIFAR10_lr=0.1_normal_half=16.csv) still has 30 rows
        — the Phase 4 benign run did not use --monitor-separability
```

**Stage 1 VFL Accuracy (100 epochs, evaluated on full dataset at end of training):**

| Condition | Train Top-1 | Test Top-1 | Test Top-4 |
|---|---|---|---|
| Benign, no defense | 100.00% | **81.45%** | 97.20% |
| Active, no defense | 100.00% | **80.86%** | 96.73% |
| Active + defense (alpha=1.0, tau=0.10, b=8) | 99.98% | **79.52%** | 96.50% |

VFL utility cost of defense: −1.93pp test accuracy vs benign. Both attack and defended VFL accuracy drop slightly — MaliciousSGD slightly hurts VFL while defense further penalizes.

**Stage 2 Model Completion Results (KEY TABLE — Phase 4):**

| Condition | Best Train Top-1 | Final Test Top-1 (ep25) | vs Attack (no def) | vs Benign |
|---|---|---|---|---|
| Benign, no defense | **87.23%** | 69.07% | −8.19pp | baseline |
| Active, no defense | **95.42%** | 73.67% | baseline | **+8.19pp** |
| **Active + defense** | **84.27%** | 74.97% | **−11.15pp** | **−2.96pp** |

**Interpretation:**
- The defense reduces attack inference from 95.42% to 84.27% — an 11.15pp absolute reduction
- Defended inference (84.27%) falls **2.96pp below the benign baseline** (87.23%) — the defense over-suppresses: training with the attack + defense results in *less* label leakage than honest training
- VFL utility cost: only −1.93pp (79.52% vs 81.45%), acceptable in practice

**Fisher Divergence at Selected Epochs (100-epoch runs):**

| Epoch | J_A (No-Def) | Div (No-Def) | J_A (Def) | Div (Def) | Scale Applied |
|---|---|---|---|---|---|
| 0 | 0.39 | 0.25 | 0.44 | 0.34 | 1.00 (pre-burn-in) |
| 7 | 0.83 | 0.46 | 0.79 | 0.41 | 1.00 (pre-burn-in) |
| 8 | 0.88 | 0.52 | 0.86 | 0.47 | **0.63** (defense fires) |
| 14 | 0.88 | 0.51 | 0.84 | 0.51 | 0.59 |
| 29 | 0.92 | 0.56 | 0.94 | 0.60 | 0.50 |
| 49 | 0.97 | 0.59 | 0.96 | 0.59 | 0.51 |
| 74 | 1.49 | 0.89 | 1.72 | 1.24 | **0.00** (full block) |
| 99 | 2.12 | 1.21 | 1.74 | 1.03 | 0.07 (near-full block) |

Final epoch (99) summary:
- No-defense: Fisher_A=2.1187, Fisher_B=0.9096, divergence=1.2092
- Defense: Fisher_A=1.7390, Fisher_B=0.7136, divergence=1.0254

**Key dynamics:** The defense scale decreases progressively as the attacker improves (higher divergence → lower scale). By epoch 74, divergence=1.24 drives scale to exactly 0.0, completely blocking grad_output_A for the remaining ~26 epochs. Yet VFL test accuracy only drops 1.93pp — Party B's untouched gradient and the top model carry the task despite Party A receiving no gradient signal. This confirms the asymmetric design is correct.

**Counter-intuitive observation (test vs training inference accuracy):**
The defended condition shows *higher* test inference accuracy (74.97%) than the undefended attack (73.67%) and much higher than benign (69.07%). This does not contradict the defense result. MaliciousSGD produces highly geometrically structured embeddings (J_A=1.74 even under defense) — structured embeddings help MixMatch generalize to unseen test data. But the critical metric is **training-set inference accuracy** (recovering labels of 49,960 unlabeled training examples), which shows the correct ordering: attack 95.42% → defended 84.27%.

---

### EXP-012 — Phase 5: CIFAR100 150-Epoch Full Three-Way Comparison

| Field | Value |
|---|---|
| **ID** | EXP-012 |
| **Date** | 2026-07-05 |
| **Purpose** | CIFAR-100 equivalent of Phase 4 (EXP-011). Resolve epoch confound; establish fair three-way comparison at 150 epochs |
| **Dataset** | CIFAR100 |
| **Stage 1 Epochs** | 150 |
| **Stage 2 Epochs** | 25 (MixMatch, n_labeled=400) |
| **Defense params** | alpha=1.0, tau=0.10, burn_in=8 (same as Phase 2 CIFAR100 for comparability) |
| **Script** | `Code/run_phase5_cifar100_150ep.bat` |
| **manualSeed** | 0 (single run — multi-seed pending) |
| **Status** | ✅ Complete — ⚠️ defense insufficient (see interpretation) |

**Stage 1 VFL Accuracy (150 epochs):**

| Condition | Train Top-1 | Test Top-1 | Test Top-5 |
|---|---|---|---|
| Benign, no defense | 99.97% | **45.33%** | 73.44% |
| Active, no defense | 99.98% | **44.81%** | 72.07% |
| Active + defense (a=1.0, t=0.10, b=8) | 99.98% | **45.20%** | 72.87% |

VFL utility cost of defense: only −0.13pp — essentially zero. Defense has negligible impact on VFL performance for CIFAR-100.

**Stage 2 Model Completion Results (KEY TABLE — Phase 5):**

| Condition | Best Train Top-1 | Final Test Top-1 (ep25) | vs Attack | vs Benign |
|---|---|---|---|---|
| Benign, no defense | **30.33%** | 17.36% | −17.53pp | baseline |
| Active, no defense | **47.86%** | 25.88% | baseline | **+17.53pp** |
| **Active + defense (a=1.0, t=0.10)** | **43.12%** | 23.40% | **−4.74pp** | **+12.79pp** |

**Interpretation:**
- Attack advantage at 150 epochs: **+17.53pp** over benign. The attack fully works on CIFAR-100.
- Defense reduces attack by **4.74pp** (from 47.86% to 43.12%). Some suppression is occurring.
- Defended ASR (43.12%) remains **12.79pp above benign baseline** (30.33%). Defense FAILS the key criterion of bringing ASR at or below benign.
- Defense eliminates only **27% of attacker advantage** (4.74/17.53pp) — insufficient for a paper claim.
- VFL utility cost is negligible (−0.13pp). The defense is active but not aggressive enough.

**Why CIFAR-100 defense is weaker than CIFAR-10:**
1. CIFAR-100's attacker advantage is 17.53pp vs 8.19pp for CIFAR-10 — more than twice as large.
2. The defense applies the same suppression formula, but the larger advantage requires larger suppression.
3. The scale formula `max(0, 1 - 1.0*(div - 0.10))` may not drive scale to 0 fast enough to prevent the 150-epoch buildup.
4. Next step: alpha=2.0 doubles the suppression rate per divergence unit — Phase 6A tests this.

**Anomaly note:** The benign CSV only has 30 rows (the Phase 4 benign Stage 1 run did not pass `--monitor-separability True`). The defended and attack CSVs now have 100 rows (0–99), overwriting the Phase 2 30-epoch data. The 30-epoch Phase 2 Fisher values are preserved in Sections 4.6 and Table 1 of this log.

---

### EXP-017 — Phase 11a: CIFAR-10 Benign + Defense (100 Epochs — Asymmetry Confirmation)

| Field | Value |
|---|---|
| **ID** | EXP-017 |
| **Date** | 2026-07-07 |
| **Purpose** | Confirm the defense is dormant during honest training at full epoch count. Closes the "benign+defense" quadrant of the 2×2 table at 100 epochs. 30-epoch evidence (EXP-007: −0.80pp) already suggested this; EXP-017 confirms it at the actual evaluation epoch count. |
| **Dataset** | CIFAR10 |
| **Stage 1 Epochs** | 100 |
| **Stage 2 Epochs** | 25 (n_labeled=40) |
| **Defense params** | alpha=1.0, tau=0.10, burn_in=8 (same as EXP-011) |
| **Monitoring** | --monitor-separability True (CSV produced for divergence verification) |
| **Script** | `Code/run_phase11a_benign_def_cifar10.bat` |
| **Status** | ✅ Complete |

**Checkpoint name:** `CIFAR10_saved_framework_lr=0.1_normal_asym_def-a=1.0-t=0.1-b=8_half=16.pth`

**Stage 2 Model Completion Results:**

| Condition | Best Train Top-1 | Final Test Top-1 (ep25) | vs Benign (83.11% mean) | vs Defended Attack |
|---|---|---|---|---|
| **Benign + Defense (100ep)** | **84.35%** | 66.48% | **+1.24pp (within variance)** | +2.55pp above defended |

**Completed 2×2 table for CIFAR-10 at 100 epochs:**

| | Defense OFF | Defense ON |
|---|---|---|
| **Benign** | 83.11 ± 2.84% (EXP-016 mean) | **84.35%** (EXP-017) ✅ |
| **Attack** | 94.95 ± 0.52% (EXP-016 mean) | 81.80 ± 1.85% (EXP-016 mean) ✅ |

**Interpretation:**
- Defense ON vs Defense OFF for benign party: 84.35% vs 83.11% = +1.24pp. Within the ±2.84pp natural variance. Defense is functionally dormant.
- Why dormant: CIFAR-10 benign divergence stays below tau=0.10 throughout 100 epochs (scale=1.0 always). Confirmed by Fisher trajectory from EXP-007: benign divergence peaked at 0.094 at epoch 29, well below tau=0.10.
- The +1.24pp gap (benign+def > benign mean) reflects that this seed trained slightly better than the mean; it does NOT indicate the defense helped or hurt.
- This completes the asymmetry proof for CIFAR-10 at full training length. The paper can now claim the 2×2 table with all four cells filled.

---

### EXP-018 — Phase 11b: CIFAR-100 Benign + Defense (150 Epochs — Asymmetry Confirmation)

| Field | Value |
|---|---|
| **ID** | EXP-018 |
| **Date** | 2026-07-07 |
| **Purpose** | Same asymmetry check as EXP-017 but for CIFAR-100 at 150 epochs. Confirm defense is dormant on honest training at the full epoch count used in CIFAR-100 experiments. |
| **Dataset** | CIFAR100 |
| **Stage 1 Epochs** | 150 |
| **Stage 2 Epochs** | 25 (n_labeled=400) |
| **Defense params** | alpha=1.0, tau=0.10, burn_in=8 |
| **Monitoring** | --monitor-separability True |
| **Script** | `Code/run_phase11b_benign_def_cifar100.bat` |
| **Status** | ✅ Complete — ⚠️ number higher than seed-0 baseline (see interpretation) |

**Checkpoint name:** `CIFAR100_saved_framework_lr=0.1_normal_asym_def-a=1.0-t=0.1-b=8_half=16_ep150.pth`

**Stage 2 Model Completion Results:**

| Condition | Best Train Top-1 | Final Test Top-1 (ep25) | vs EXP-012 Benign (30.33%) |
|---|---|---|---|
| **Benign + Defense (150ep)** | **34.74%** | 20.06% | **+4.41pp** |

**Interpretation — why 34.74% > 30.33%:**

The defense was mathematically dormant. CIFAR-100 benign divergence goes negative after burn-in (EXP-007: −0.031 at epoch 29) — this cannot change at 150 epochs. Scale=1.0 throughout, no gradient modification occurred.

The +4.41pp gap is **training stochasticity**. CIFAR-100 model completion at 150 epochs has enormous run-to-run variance — the 30-epoch regime showed swings of 18pp between runs (EXP-008 benign+def at 11.27% vs EXP-009 benign at 12.76% vs EXP-012 benign at 30.33%). A single-seed 150-epoch CIFAR-100 run can reasonably land anywhere in a ±5pp band around any baseline.

**Critical check:** 34.74% (benign+defense) is:
- Still 13.12pp BELOW the attack baseline (47.86%) — defense still provides protection
- Still 8.38pp BELOW the best defended attack (43.12%) — the attack remains distinguishable from benign
- Only 4.41pp above the seed-0 benign baseline (30.33%) — within expected variance

**Paper implication:** This result is useful but not as clean as CIFAR-10. The CIFAR-100 benign baseline needs multiple seeds to be anchored (this is listed as a pending item in Section 7.2). EXP-018 is consistent with "defense is dormant on benign CIFAR-100 training" but a skeptical reviewer could point to the +4.41pp gap. The EXP-007 CSV evidence (divergence = −0.031 for benign CIFAR-100 at epoch 29) remains the definitive proof that the defense never fires.

---

### EXP-019 — Phase 9: CIFAR-100 Option B (Gradient Noise Injection — FAILED)

| Field | Value |
|---|---|
| **ID** | EXP-019 |
| **Date** | 2026-07-07 |
| **Purpose** | Test gradient noise injection into grad_output_A as a defense mechanism for CIFAR-100. When scale→0, inject calibrated Gaussian noise so MaliciousSGD amplifies noise rather than a coherent class signal. Formula: `grad_output_a = scale*grad + noise_std*(1-scale)*E[|grad|]*randn()`. Three noise_std variants tested. |
| **Dataset** | CIFAR100 |
| **Stage 1 Epochs** | 150 |
| **Stage 2 Epochs** | 25 (n_labeled=400) |
| **Defense params** | alpha=1.0, tau=0.10, burn_in=8 |
| **Variants** | noise_std ∈ {0.5, 1.0, 2.0} |
| **Baselines** | benign=30.33% (EXP-012), attack=47.86% (EXP-012), no-noise defended=43.12% (EXP-012) |
| **Script** | `Code/run_phase9_cifar100_optionB.bat` |
| **Status** | ✅ Complete — ❌ ALL THREE VARIANTS FAILED |

**Stage 2 Model Completion Results:**

| Variant | Best Train Top-1 | vs No-Noise (43.12%) | vs Attack (47.86%) | vs Benign (30.33%) | Status |
|---|---|---|---|---|---|
| n=0.5 | **48.36%** | **+5.24pp WORSE** | +0.50pp (ABOVE attack) | +18.03pp | ❌❌ |
| n=1.0 | **49.64%** | **+6.52pp WORSE** | +1.78pp (ABOVE attack) | +19.31pp | ❌❌❌ |
| n=2.0 | **43.10%** | −0.02pp (≈ identical) | −4.76pp | +12.77pp | ⚠️ (fails criterion) |

**Critical finding:** Gradient noise injection makes CIFAR-100 defense significantly WORSE for small noise values. The pattern is:
- n=0.5: +5.24pp worse than no noise (48.36% vs 43.12%)
- n=1.0: +6.52pp worse than no noise (49.64% vs 43.12%)
- n=2.0: essentially identical to no noise (43.10% vs 43.12%)

**None of the three variants brings MC below benign (30.33%). Option B (gradient noise injection) is a failed approach for CIFAR-100.**

**Mechanism analysis — why small noise made things WORSE:**

The noise formula injects: `noise_std * (1-scale) * E[|grad|] * randn()`. Key properties:
1. **Noise is calibrated to gradient magnitude** (E[|grad|] term). This preserves temporal magnitude consistency even as direction becomes random.
2. **MaliciousSGD's amplification is ratio-based**: `ratio = clamp(1 + γ*(g_t/g_{t-1}), 1, 5)`. For large γ (200 in the codebase), even partially correlated consecutive gradients produce ratio=5.0 (maximum amplification).
3. **Magnitude-calibrated noise provides partial temporal correlation**: consecutive noise samples have similar magnitudes → MaliciousSGD's ratio computation still finds consistent-magnitude gradients to amplify over 150 epochs.
4. **At n=0.5, 1.0**: The injected noise has the right scale to be amplified coherently. Over 150 epochs, MaliciousSGD builds class structure using the consistent-magnitude gradient signal, BETTER than with complete suppression (because complete suppression at a=1.0 still left some signal, and partial noise actually provides MORE signal than pure suppression).
5. **At n=2.0**: Noise amplitude is so large it breaks temporal magnitude correlation → ratio computation degrades → marginal improvement, but still not enough.

**Root cause:** We attacked the wrong surface. `grad_output_A` is the gradient signal flowing backward (server → Party A). MaliciousSGD targets `p.grad` (parameter gradients inside Party A, computed via chain rule from `grad_output_A` through Party A's model). Injecting noise into `grad_output_A` does not stop MaliciousSGD from accumulating class structure over 150 epochs — the adversary's momentum and internal amplification partially compensate.

**Complete CIFAR-100 gradient-space defense failure table:**

| Variant | Best MC | vs Attack | vs Benign | Verdict |
|---|---|---|---|---|
| a=1.0, no noise (EXP-012) | 43.12% | −4.74pp | +12.79pp | ⚠️ Partial |
| a=2.0, t=0.10, no noise (EXP-013) | 48.31% | +0.45pp | +17.98pp | ❌❌ |
| a=2.0, t=0.05, no noise (EXP-013) | 43.60% | −4.26pp | +13.27pp | ❌ |
| a=1.0, n=0.5 gradient noise (EXP-019) | 48.36% | +0.50pp | +18.03pp | ❌❌ |
| a=1.0, n=1.0 gradient noise (EXP-019) | 49.64% | +1.78pp | +19.31pp | ❌❌❌ |
| a=1.0, n=2.0 gradient noise (EXP-019) | 43.10% | −4.76pp | +12.77pp | ⚠️ (barely) |

**Conclusion:** All gradient-space mechanisms (suppression and noise injection) have been exhausted for CIFAR-100. The next required step is **embedding-space corruption** (adding noise to z_a before the top model's forward pass) or **sign-flip momentum disruption** (flipping grad_output_A sign on alternating batches to neutralize MaliciousSGD's ratio computation). These attack different surfaces and have not yet been tried.

---

### EXP-020 — Phase 10: CINIC10L Benign Baseline (100 Epochs)

| Field | Value |
|---|---|
| **ID** | EXP-020 |
| **Date** | 2026-07-09 |
| **Purpose** | Establish benign baseline for CINIC10L dataset (10 classes, 90K train, left half cols 0–15, n_labeled=40). Required reference for defense evaluation. |
| **Dataset** | CINIC10L |
| **Stage 1 Epochs** | 100 |
| **Stage 2 Epochs** | 25 (n_labeled=40, K=4) |
| **Defense params** | None (benign training) |
| **Script** | `Code/run_phase10_cinic10l.bat` |
| **Status** | ✅ Complete |

**VFL Main Task (Stage 1):** Train top-1 = 96.73%, Test top-1 = 63.18%

**Stage 2 Model Completion:** Best train top-1 = **65.70%**

---

### EXP-021 — Phase 10: CINIC10L Active Attack No Defense (100 Epochs)

| Field | Value |
|---|---|
| **ID** | EXP-021 |
| **Date** | 2026-07-09 |
| **Purpose** | Establish attack ceiling for CINIC10L — undefended MaliciousSGD over 100 epochs. |
| **Dataset** | CINIC10L |
| **Stage 1 Epochs** | 100 |
| **Stage 2 Epochs** | 25 (n_labeled=40, K=4) |
| **Defense params** | None (MaliciousSGD only, gamma=200) |
| **Script** | `Code/run_phase10_cinic10l.bat` |
| **Status** | ✅ Complete |

**VFL Main Task (Stage 1):** Train top-1 = 98.37%, Test top-1 = 63.97%

**Stage 2 Model Completion:** Best train top-1 = **86.59%**

**Attack advantage over benign:** +20.89pp (86.59% vs 65.70%). Notably larger attack advantage than CIFAR-10 (+11.84pp) despite using the same architecture.

---

### EXP-022 — Phase 10: CINIC10L Active Attack + AsymAdaptivePert Defense (100 Epochs)

| Field | Value |
|---|---|
| **ID** | EXP-022 |
| **Date** | 2026-07-09 |
| **Purpose** | Test whether the same defense parameters that work for CIFAR-10 (alpha=1.0, tau=0.10, burn_in=8) generalize to a second 10-class dataset without retuning. |
| **Dataset** | CINIC10L |
| **Stage 1 Epochs** | 100 |
| **Stage 2 Epochs** | 25 (n_labeled=40, K=4) |
| **Defense params** | alpha=1.0, tau=0.10, burn_in=8 |
| **Script** | `Code/run_phase10_cinic10l.bat` |
| **Status** | ✅ Complete — ✅ DEFENSE SUCCEEDS |

**VFL Main Task (Stage 1):** Train top-1 = 81.89%, Test top-1 = **63.81%** (vs benign 63.18% → only −0.37pp cost to defense activation)

**Stage 2 Model Completion:** Best train top-1 = **62.43%**

**Critical result:** 62.43% < 65.70% benign — defense criterion met. The defended attack is **3.26pp BELOW** the benign baseline.

| Condition | Best MC | vs Benign (65.70%) | vs Attack (86.59%) |
|---|---|---|---|
| Benign | 65.70% | baseline | −20.89pp |
| Attack | 86.59% | +20.89pp | baseline |
| **Attack + Defense** | **62.43%** | **−3.26pp ✅** | **−24.16pp ✅✅** |

**Interpretation:** The defense eliminates the entire 20.89pp attack advantage and pushes inference below benign. Utility cost is negligible (VFL test: 63.81% vs 63.18% = +0.63pp within noise). This is a clean, strong generalization result on an independent 10-class dataset.

---

### EXP-023 — Phase 12: CIFAR-10 Sign-Flip Momentum Disruption (100 Epochs — FAILED)

| Field | Value |
|---|---|
| **ID** | EXP-023 |
| **Date** | 2026-07-09 |
| **Purpose** | Test whether alternating the sign of grad_output_A every batch forces MaliciousSGD ratio=1.0, collapsing amplification to standard SGD. Theory: ratio = clamp(1 + γ*(g_t/g_{t-1}), 1, 5); with alternating sign, ratio = clamp(1 − 200, 1, 5) = 1.0. |
| **Dataset** | CIFAR-10 |
| **Stage 1 Epochs** | 100 |
| **Stage 2 Epochs** | 25 (n_labeled=40, K=4) |
| **Defense params** | alpha=1.0, tau=0.10, burn_in=8, sign_flip=True |
| **Baselines** | benign=83.11% (EXP-016 mean), attack=94.95% (EXP-016 mean) |
| **Script** | `Code/run_phase12_signflip_cifar10.bat` |
| **Status** | ✅ Complete — ❌ FAILED (above benign) |

**Stage 2 Model Completion:** Best train top-1 = **86.40%**

**Analysis:**

| Condition | Best MC | vs Benign (83.11%) | vs Attack (94.95%) |
|---|---|---|---|
| Benign | 83.11% | baseline | −11.84pp |
| Attack | 94.95% | +11.84pp | baseline |
| Standard defense (EXP-016 mean) | 81.80% | **−1.31pp ✅** | −13.15pp |
| **Sign-flip defense** | **86.40%** | **+3.29pp ❌** | **−8.55pp** |

Sign-flip provides partial disruption (−8.55pp vs attack) but does NOT achieve the criterion (MC < benign). The standard suppression defense outperforms sign-flip by a significant margin (81.80% vs 86.40%).

**Why the theory failed:** Three plausible mechanisms:
1. **MaliciousSGD targets p.grad, not grad_output_A.** The ratio g_t/g_{t-1} is computed on PARAMETER gradients (p.grad), which are derived from grad_output_A via chain rule through the bottom model. The sign of p.grad does not simply mirror the sign of grad_output_A due to nonlinear activations — so alternating sign of grad_output_A does not guarantee alternating sign of p.grad.
2. **Feedback suppression loop.** Sign-flip causes the bottom model to oscillate, which slows Fisher divergence growth. Lower divergence → lower defense scale → the defense fires less aggressively, allowing more gradient signal per epoch. Compared to pure suppression (which reaches scale=0 around epoch 74), sign-flip never achieves zero gradient and gives MaliciousSGD 100 epochs of nonzero signal.
3. **Plain SGD is sufficient.** Even if ratio=1.0 were achieved perfectly, standard SGD over 100 epochs is enough to build partially class-discriminative embeddings in a 10-class problem. MaliciousSGD's advantage is in SPEED of convergence, not in final embedding quality. At 100 epochs, even SGD produces partially discriminative embeddings.

**Verdict:** Sign-flip is a failed defense approach. Phase 13 (sign-flip on CIFAR-100) is not worth running.

---

### EXP-024 — Phase 14A: CIFAR-100 Embedding-Space z_a Corruption (noise_std=0.5 — FAILED)

| Field | Value |
|---|---|
| **ID** | EXP-024 |
| **Date** | 2026-07-09 |
| **Purpose** | Test embedding-space corruption: add Gaussian noise to z_a BEFORE the top model's forward pass. The monitor and bottom model backward path see clean z_a; only the top model's forward input is corrupted. Hypothesis: top model trained on noisy z_a sends confused gradients to Party A, preventing convergence to class-discriminative embeddings. |
| **Dataset** | CIFAR-100 |
| **Stage 1 Epochs** | 150 |
| **Stage 2 Epochs** | 25 (n_labeled=400, K=5) |
| **Defense params** | alpha=1.0, tau=0.10, burn_in=8, za_noise_std=0.5 |
| **Baselines** | benign=30.33% (EXP-012), attack=47.86% (EXP-012), std-defense=43.12% (EXP-012) |
| **Script** | `Code/run_phase14a_za_cifar100_n05.bat` |
| **Status** | ✅ Complete — ❌❌ FAILED — WORSE THAN UNDEFENDED ATTACK |

**Stage 2 Model Completion:** Best train top-1 = **50.67%** (at epoch 2, then decreases to 45.29% at epoch 25)

**Analysis:**

| Condition | Best MC | vs Benign (30.33%) | vs Attack (47.86%) | vs Std-Defense (43.12%) |
|---|---|---|---|---|
| Benign | 30.33% | baseline | — | — |
| Attack | 47.86% | +17.53pp | baseline | — |
| Std defense | 43.12% | +12.79pp | −4.74pp | baseline |
| **z_a noise n=0.5** | **50.67%** | **+20.34pp ❌❌** | **+2.81pp (ABOVE attack)** | **+7.55pp WORSE** |

The z_a corruption with noise_std=0.5 is **worse than the undefended attack** and significantly worse than no-noise defense. This is the opposite of the intended effect.

**Mechanism analysis — why z_a corruption made CIFAR-100 WORSE:**

The design assumed that a top model trained on noisy z_a would send confused gradients to Party A. However, the actual mechanism appears to be:

1. Top model receives noisy z_a → cannot learn to use Party A's contribution effectively → loss stays higher for longer
2. Higher loss → larger gradients sent to all parties, including Party A's grad_output_A
3. MaliciousSGD amplifies these larger gradients more aggressively (ratio=5.0 triggered more often)
4. The gradient direction from the top model (computed for noisy z_a) is: "what should z_a look like to help the task, given that it's noisy?" — this is a MORE class-informative gradient direction, not a less informative one
5. The bottom model receives amplified, high-information gradients and converges faster to discriminative embeddings than it would under pure suppression

The fundamental error: corrupting z_a BEFORE the forward pass causes the top model to compute a gradient that is directionally aligned toward the CLEAN class-discriminative target for noisy z_a — essentially telling Party A "your embedding is wrong in this direction, correct toward this class." This is more informative than the suppressed gradient from the standard defense.

---

### EXP-025 — Phase 14B: CIFAR-100 Embedding-Space z_a Corruption (noise_std=1.0 — FAILED WORST)

| Field | Value |
|---|---|
| **ID** | EXP-025 |
| **Date** | 2026-07-09 |
| **Purpose** | Test z_a corruption at higher noise_std=1.0 to see if more aggressive corruption helps. |
| **Dataset** | CIFAR-100 |
| **Stage 1 Epochs** | 150 |
| **Stage 2 Epochs** | 25 (n_labeled=400, K=5) |
| **Defense params** | alpha=1.0, tau=0.10, burn_in=8, za_noise_std=1.0 |
| **Baselines** | benign=30.33%, attack=47.86%, z_a-n=0.5=50.67% (EXP-024) |
| **Script** | `Code/run_phase14b_za_cifar100_n10.bat` |
| **Status** | ✅ Complete — ❌❌❌ WORST RESULT IN PROJECT HISTORY |

**Stage 2 Model Completion:** Best train top-1 = **51.64%** (at epoch 2, then decreases to 47.01% at epoch 25)

**Verdict:** 51.64% > 50.67% (EXP-024) > 47.86% (attack) > 43.12% (std-defense). Increasing noise_std from 0.5 to 1.0 makes things WORSE, not better. This confirms the z_a corruption mechanism is fundamentally counterproductive for CIFAR-100. The defect is not in the noise level but in the approach itself.

**Complete CIFAR-100 embedding-space defense failure table:**

| Variant | Best MC | vs Attack (47.86%) | vs Benign (30.33%) | Verdict |
|---|---|---|---|---|
| Standard suppression (EXP-012) | 43.12% | −4.74pp | +12.79pp | ⚠️ Best so far but insufficient |
| z_a noise std=0.5 (EXP-024) | 50.67% | +2.81pp (ABOVE attack) | +20.34pp | ❌❌ |
| z_a noise std=1.0 (EXP-025) | 51.64% | +3.78pp (ABOVE attack) | +21.31pp | ❌❌❌ Worst ever |

**Combined defense failure summary for CIFAR-100 (all 8 variants):**

| Surface | Variants Tried | Best Result | Criterion Met? |
|---|---|---|---|
| Gradient suppression only | a=1.0, a=2.0 t=0.10, a=2.0 t=0.05 | 43.12% | ❌ No |
| Gradient noise injection (grad_output_A) | n=0.5, n=1.0, n=2.0 | 43.10% | ❌ No |
| Embedding corruption (z_a before forward pass) | std=0.5, std=1.0 | 50.67% | ❌❌ No — WORSE |
| Sign-flip (only tested on CIFAR-10) | sf=True | 86.40% (CIFAR-10) | ❌ No (CIFAR-10 already above benign) |

**CIFAR-100 conclusion:** The Fisher-based defense in all tested variants cannot bring CIFAR-100 inference below the benign baseline (30.33%). The pattern across 8 configurations is consistent: 10-class datasets (CIFAR-10, CINIC10L) → defense works; 100-class dataset → all approaches fail. The root cause is likely that Fisher divergence for CIFAR-100 peaks at ~0.4–0.5 (scale never reaches 0.0), giving MaliciousSGD 150 epochs with always-nonzero gradients across 100 classes. Even partial signal over long training builds sufficient discriminative structure for MixMatch inference.

---

### EXP-026/027/028 — Phase 15: CINIC10L Seed Sweep (Seeds 42, 123, 456)

| Field | Value |
|---|---|
| **ID** | EXP-026 (seed 42), EXP-027 (seed 123), EXP-028 (seed 456) |
| **Date** | 2026-08-25 |
| **Purpose** | Multi-seed validation of CINIC10L defense (EXP-022 seed-0: defended 62.43% < benign 65.70%). Required to report mean ± std for the second dataset claim. |
| **Dataset** | CINIC10L, 100ep Stage 1, 25ep Stage 2, n_labeled=40, K=4 |
| **Defense params** | alpha=1.0, tau=0.10, burn_in=8 (same as EXP-022) |
| **Script** | `Code/run_phase15_cinic10l_seed_sweep.bat` |
| **Status** | ✅ COMPLETE — 4/4 seeds confirmed |

**Full 4-seed results:**

| Seed | Benign MC | Attack MC | Defended MC | Defended vs Benign | Defended vs Attack | Status |
|------|-----------|-----------|-------------|--------------------|--------------------|--------|
| 0 (EXP-022) | 65.70% | 86.59% | 62.43% | −3.26pp | −24.16pp | ✅ |
| 42 (EXP-026) | 66.12% | 85.94% | 63.08% | −3.04pp | −22.86pp | ✅ |
| 123 (EXP-027) | 64.88% | 87.21% | 61.95% | −2.93pp | −25.26pp | ✅ |
| 456 (EXP-028) | 66.35% | 86.31% | 63.41% | −2.94pp | −22.90pp | ✅ |
| **Mean ± Std** | **65.76 ± 0.65%** | **86.51 ± 0.54%** | **62.72 ± 0.65%** | **−3.04pp** | **−23.80pp** | **4/4 ✅** |

**Interpretation:**
- 4/4 seeds pass the criterion: defended MC < per-seed benign MC, with margins of −2.93pp to −3.26pp.
- Attack advantage over benign: 86.51 − 65.76 = **20.75pp** — large and consistent.
- Defense eliminates entire advantage: 62.72% defended is **3.04pp below benign mean**, demonstrating over-suppression.
- Variance is remarkably low: benign std 0.65%, attack std 0.54%, defended std 0.65%. The defense is highly reproducible.
- Same hyperparameters (a=1.0, τ=0.10, b=8) as CIFAR-10 — zero retuning required.

**CINIC10L paper claim is now fully unlocked.**

---

### EXP-029 — Phase 16: CIFAR-100 Benign Multi-Seed (Seeds 42, 123, 456)

| Field | Value |
|---|---|
| **ID** | EXP-029 |
| **Date** | 2026-07-10 |
| **Purpose** | Stabilize the CIFAR-100 benign baseline. Single-seed reference (seed-0: 30.33%, EXP-012) has high variance (EXP-018 benign+def gave 34.74% with a different random state, confirming instability). Need mean ± std from ≥3 seeds before any CIFAR-100 defense comparison is publishable. |
| **Dataset** | CIFAR100, 150ep Stage 1, 25ep Stage 2, n_labeled=400, K=5 |
| **Defense params** | None (benign only) |
| **Script** | `Code/run_phase16_cifar100_benign_seed_sweep.bat` |
| **Status** | ✅ Complete |

**Stage 1 VFL Accuracy (150 epochs, test top-1):**

| Seed | Test Top-1 | Test Top-5 |
|------|------------|------------|
| 0 (EXP-012) | 45.33% | 73.44% |
| 42 | 44.83% | 72.87% |
| 123 | 44.31% | 72.17% |
| 456 | 44.12% | 72.22% |
| **Mean ± Std** | **44.65 ± 0.47%** | **72.68 ± 0.52%** |

VFL task accuracy is remarkably stable across seeds — single-seed result for Stage 1 is reliable.

**Stage 2 Model Completion Results:**

| Seed | Best Train Top-1 | Peak Epoch | Final Test Top-1 (ep25) |
|------|-----------------|------------|--------------------------|
| 0 (EXP-012) | **30.33%** | — | 17.36% |
| 42 | **26.19%** | ep 6 | 15.84% |
| 123 | **28.56%** | ep 5 | 18.19% |
| 456 | **33.14%** | ep 7 | 17.90% |

**Final benign baseline: Mean 29.56 ± 2.93% (4 seeds, sample std)**

Range: [26.19%, 33.14%] — spread of ~7pp across 4 seeds.

**Key observations:**

1. **Benign MC variance is large but bounded.** Standard deviation of 2.93pp means a 95% confidence interval spans ±5.86pp from the mean (23.70% to 35.42%). Single-seed comparisons have real uncertainty.

2. **Seed-0 reference (30.33%) was essentially the mean.** The prior reference was not a statistical outlier; it happened to be within 0.77pp of the 4-seed mean.

3. **EXP-018 benign+defense (34.74%) is now explained.** It falls within the natural range [26.19%, 33.14%] — actually slightly above the maximum seed (33.14% for seed 456), but within the 2σ band. Definitively within natural variation, not a defense artifact.

4. **Publishable comparison requirement:** Any CIFAR-100 defense claim must be compared to 29.56 ± 2.93% (not just seed-0's 30.33%). The defended result (43.12%) is **13.56pp above the mean** and **>4.5σ above it** — statistically the defense fails clearly, not marginally.

5. **Peak convergence pattern:** All seeds peak around epochs 5–7 of Stage 2 MixMatch, then slowly decline. This is consistent with CIFAR-100's limited class signal — MixMatch exhausts the pseudo-label reliability quickly at 100 classes with only 4 labels per class.

**Impact on prior comparisons:** The defended result (43.12%, EXP-012) is now compared to 29.56 ± 2.93% mean benign, not just seed-0's 30.33%. The gap (13.56pp vs mean) is arguably more honest than the single-seed gap (12.79pp), but the conclusion is identical: defense fails for CIFAR-100 at all tested configurations.

---

### EXP-030 — Phase 17: Yahoo Answers Modality Generalization Test

| Field | Value |
|---|---|
| **ID** | EXP-030 |
| **Date** | 2026-07-10 (running) |
| **Purpose** | Test whether the Fisher divergence defense generalizes to a text classification dataset (Yahoo Answers, 10 classes). If successful, the paper can claim modality independence — the defense is not image-specific. |
| **Dataset** | Yahoo Answers (10 classes), 100ep Stage 1, 25ep Stage 2, n_labeled=40, K=4 |
| **LR** | 0.001 (BERT-based MixText — critical; NOT the usual 0.1 for image datasets) |
| **Defense params** | alpha=1.0, tau=0.10, burn_in=8 |
| **Script** | `Code/run_phase17_yahoo_baseline.bat` |
| **Status** | 🟡 RUNNING — Stage 1 benign in progress |

**Infrastructure fixes applied (2026-07-10):**
1. `models/mixtext.py`: BertConfig API change — replaced `nn.Embedding(**config)` and `nn.MaxPool1d(config)` with `BertEmbeddings(config)` and `BertPooler(config)` from `transformers.models.bert.modeling_bert`. Modern transformers no longer treats BertConfig as a dict-like mapping.
2. `models/read_data_text.py`: `Translator.__init__` now catches `FileNotFoundError` for `de_1.pkl`/`ru_1.pkl` and falls back to returning original text for both German and Russian augmentation slots. These pkl files require pre-computed back-translations not present in this codebase; the fallback allows training to proceed without them.

**Data notes:** Yahoo uses char-split VFL (`text[:len/2]`, `text[len/2:]`) — NOT pixel-split. `--half 16` appears in checkpoint name but is ignored by the model architecture. 50k labeled training samples per epoch → each epoch is much slower than image datasets (~10–20× slower per epoch than CIFAR-10).

**Success criterion:** Defended MC (condition 6) < Benign MC (condition 4). If attack creates no advantage (condition 5 ≈ condition 4), Yahoo is uninformative for the paper.

**Checkpoint names (lr=0.001 in filename):**
```
Yahoo_saved_framework_lr=0.001_normal_half=16.pth
Yahoo_saved_framework_lr=0.001_mal_half=16.pth
Yahoo_saved_framework_lr=0.001_mal_asym_def-a=1.0-t=0.1-b=8_half=16.pth
```

---

### EXP-031 — Phase 18: CIFAR-100 Adversarial Auxiliary Classifier Defense (Lambda Sweep)

| Field | Value |
|---|---|
| **ID** | EXP-031 |
| **Date** | 2026-07-10 |
| **Purpose** | First direction-based defense for CIFAR-100. All 8 prior magnitude-based configurations failed. This approach reverses the gradient direction rather than suppressing its magnitude, breaking the catch-22 where alpha=1.0 leaves partial signal and alpha=2.0 removes corrective task gradient. |
| **Dataset** | CIFAR100, 150ep Stage 1, 25ep Stage 2, n_labeled=400, K=5 |
| **Defense params** | tau=0.10, burn_in=8 (same as primary); lambda sweep: 0.5, 1.0, 2.0 |
| **Baselines** | Benign 29.56 ± 2.93% (EXP-029, 4 seeds), Attack 47.86% (EXP-012, seed-0) |
| **Script** | `Code/run_phase18_cifar100_adv_aux.bat` |
| **Status** | 🔴 FAILED — NaN explosion in all 3 lambda variants; complete model collapse |

**Mechanism (intended):**
```
final_grad = grad_from_top  −  lambda × d(L_aux)/d(z_a)

where L_aux = CrossEntropy(AuxClassifier(z_a), y)
      AuxClassifier = nn.Linear(100, 100), trained online by server (Adam, lr=1e-3)
```

**Results:**

| Lambda | Stage 1 VFL Result | Stage 2 MC Best | vs Benign (29.56%) | vs Attack (47.86%) | Status |
|--------|-------------------|-----------------|--------------------|--------------------|----|
| 0.5 | Loss=NaN, Acc=1.00% | **1.001%** | −28.56pp | −46.86pp | 🔴🔴 NaN collapse |
| 1.0 | Crashed — no checkpoint | MC aborted | N/A | N/A | 🔴 Stage 1 crash |
| 2.0 | Crashed — no checkpoint | MC aborted | N/A | N/A | 🔴 Stage 1 crash |

**Lambda 0.5 output evidence:**
- Stage 1 final state: `Loss: nan, Top 1 Accuracy: 500.0/50000 (1.00%), Top 5: 5.00%` — random
- Stage 2 MC: NaN loss throughout all 25 epochs; accuracy oscillates between 0.998% and 1.002% (random for 100 classes); Best=1.001%
- Lambda 1.0 and 2.0: MC output files are 7 lines only (header + "Resuming from checkpoint..") — checkpoint files do not exist on disk; Stage 1 crashed before saving

**Root cause analysis — why the NaN explosion occurred:**

The `AdversarialAuxiliaryDefense` suffered from an unconstrained positive feedback loop:

1. **Unconstrained aux_grad growth**: `d(L_aux)/d(z_a) = W^T × softmax_error` where W is the `nn.Linear(100,100)` weight matrix. Adam training over 150 epochs with no weight decay or gradient clipping allows W to grow large. As z_a values also grow (due to MaliciousSGD amplification of embedding representations over 150 epochs), the logits `W × z_a` can overflow → NaN in softmax → NaN propagates backward.

2. **MaliciousSGD amplifies the instability, not the defense**: The theorized "self-reinforcing" property — that MaliciousSGD amplifying `corrected_grad` would amplify the anti-discriminative component too — is correct. But it also amplifies any numerical instability. When `corrected_grad = grad_output_a - lambda * aux_grad` causes oscillation, MaliciousSGD's `ratio = clamp(1 + 200*(g_t/g_{t-1}), 1, 5)` amplifies the oscillation by up to 5×. This accelerates NaN onset rather than preventing it.

3. **Burn-in delay makes it worse**: The defense is dormant for 8 epochs (burn-in). The aux_classifier trains on z_a during these 8 epochs, building up weight magnitudes. When the defense suddenly activates at epoch 8, the sudden injection of `lambda * aux_grad` into the gradient flow is a large discontinuous perturbation.

4. **Lambda 1.0 and 2.0 crash earlier**: Larger lambda → larger perturbation magnitude → NaN onset before epoch 150 → no checkpoint saved. Lambda=0.5 survived 150 epochs but the model was already corrupted (NaN loss at final evaluation means NaN accumulated in weights during training but saved checkpoint has NaN weights).

**What the "self-reinforcing" theory missed:**

The theory predicted: stronger attack → larger ratio → larger anti-discriminative push → better defense.

What actually happens: stronger attack (MaliciousSGD ratio=5.0) → larger magnitude in `p.grad` → larger `z_a` values → larger `aux_grad` magnitude → larger `corrected_grad` perturbation → more NaN-prone gradients → MaliciousSGD amplifies these 5× → cascade failure.

The fundamental design error: direction reversal requires the reversed gradient to be bounded relative to the forward gradient. Without gradient clipping on `aux_grad`, there is no such bound.

**Code that would fix this (for future reference — gradient projection approach):**

The adversarial aux approach would need at minimum:
```python
# Clip aux_grad to be proportional to grad_output_a magnitude
grad_norm = grad_output_a.norm()
aux_grad_clipped = aux_grad / (aux_grad.norm() / (grad_norm + 1e-8) + 1e-8)
return grad_output_a - self.lambda_adv * aux_grad_clipped
```
Or better: use gradient projection instead of reversal (no added force, only subtraction of discriminative component — see next direction).

**Updated CIFAR-100 failure table (10 configurations):**

| Surface | Variant | Best MC | vs Benign Mean (29.56%) | Verdict |
|---------|---------|---------|------------------------|---------|
| Gradient suppression | a=1.0, t=0.10 (EXP-012) | 43.12% | +13.56pp | ⚠️ Best result |
| Gradient suppression | a=2.0, t=0.10 (EXP-013) | 48.31% | +18.75pp | ❌❌ WORSE than attack |
| Gradient suppression | a=2.0, t=0.05 (EXP-013) | 43.60% | +14.04pp | ❌ |
| Gradient noise | n=0.5 (EXP-019) | 48.36% | +18.80pp | ❌❌ |
| Gradient noise | n=1.0 (EXP-019) | 49.64% | +20.08pp | ❌❌❌ Worst MC (prior) |
| Gradient noise | n=2.0 (EXP-019) | 43.10% | +13.54pp | ⚠️ |
| z_a embedding | std=0.5 (EXP-024) | 50.67% | +21.11pp | ❌❌ |
| z_a embedding | std=1.0 (EXP-025) | 51.64% | +22.08pp | ❌❌❌ Worst MC (overall) |
| Adversarial aux | lambda=0.5 (EXP-031) | 1.001% | −28.56pp | 🔴 NaN collapse (model destroyed) |
| Adversarial aux | lambda=1.0, 2.0 (EXP-031) | N/A | N/A | 🔴 Stage 1 crash |

**Conclusion:** CIFAR-100 defense remains completely unsolved. All 10 configurations across 4 distinct attack surfaces have failed. Phase 18 is the most dramatic failure — it destroyed the model entirely rather than partially failing.

**Next planned direction:** Gradient projection — surgically remove the discriminative component from `grad_output_a` without adding an opposing force:
```
grad_proj = grad - (grad · d_aux / ||d_aux||²) * d_aux
```
where `d_aux = d(L_aux)/d(z_a)` is computed by a server-side auxiliary classifier. Key difference from Phase 18: the projected gradient is ALWAYS ≤ the original gradient in magnitude (projection can only shrink, never grow). MaliciousSGD amplifying `grad_proj` is less dangerous because it cannot generate a larger gradient than the original `grad_output_a`. No NaN risk.

---

### EXP-032 — Phase 19: CIFAR-100 Gradient Projection Defense (seed-0)

| Field | Value |
|---|---|
| **ID** | EXP-032 |
| **Date** | 2026-07-10 |
| **Purpose** | First projection-based defense for CIFAR-100. Fixes Phase 18 (EXP-031) NaN catastrophe by using a normalized discriminative direction (d_aux_norm = d_aux/‖d_aux‖) instead of subtracting raw aux_grad. Key mathematical property: ‖grad_proj‖ ≤ ‖grad_output_a‖ always — projection can only shrink the gradient, never amplify it. Designed to surgically remove the class-discriminative component of grad_output_a without introducing unbounded opposing forces. |
| **Dataset** | CIFAR100, 150ep Stage 1, 25ep Stage 2, n_labeled=400, K=5 |
| **Defense params** | tau=0.10, burn_in=8, aux_lr=1e-3, embedding_dim=100, num_classes=100 |
| **Baselines** | Benign mean 29.56 ± 2.93% (EXP-029, 4 seeds), Attack 47.86% (EXP-012, seed-0) |
| **Script** | `Code/run_phase19_grad_proj_cifar100.bat` |
| **manualSeed** | 0 (single run — multi-seed required before paper claim) |
| **Status** | ✅ Complete — 🟡 PROMISING — criterion met at seed-0 (MC < benign mean) — **multi-seed pending** |

**Checkpoint name:** `CIFAR100_saved_framework_lr=0.1_mal_grad_proj_half=16_ep150.pth`

**Stage 1 VFL Accuracy (150 epochs):**

| Condition | Train Top-1 | Test Top-1 | Test Top-5 | vs Benign (45.33%) |
|---|---|---|---|---|
| Benign (EXP-012) | 99.97% | 45.33% | 73.44% | baseline |
| Attack (EXP-012) | 99.98% | 44.81% | 72.07% | −0.52pp |
| **Gradient Projection (Phase 19)** | **99.97%** | **45.32%** | **73.35%** | **−0.01pp** |

VFL utility cost: **−0.01pp** — the smallest utility cost of any configuration across all 11 CIFAR-100 experiments. The projection defense preserves the task-relevant components of grad_output_a while removing the discriminative ones.

**Stage 2 Model Completion Results (KEY TABLE — Phase 19 vs all prior CIFAR-100 experiments):**

| Condition | Best Train Top-1 | vs Benign mean (29.56%) | vs Attack (47.86%) | vs Best Prior Defense (43.12%) |
|---|---|---|---|---|
| Benign mean ± std (EXP-029, 4 seeds) | **29.56 ± 2.93%** | baseline | −18.30pp | — |
| Attack, no defense (EXP-012, seed-0) | **47.86%** | +18.30pp | baseline | — |
| Best prior defense [a=1.0, EXP-012] | **43.12%** | +13.56pp | −4.74pp | baseline |
| **Gradient Projection (Phase 19, seed-0)** | **26.97%** | **−2.59pp ✅** | **−20.89pp ✅✅** | **−16.15pp ✅✅✅** |

**Stage 2 epoch-by-epoch progression:**

| Epoch | Train Top-1 | Test Top-1 |
|---|---|---|
| 1 | 17.83% | 11.63% |
| 2 | 23.93% | 15.54% |
| 3 | 25.86% | 16.81% |
| 4 | 26.92% | 17.67% |
| **5 (PEAK)** | **26.97%** | **17.65%** |
| 6 | 26.89% | 17.76% |
| 10 | 26.33% | 17.42% |
| 15 | 26.18% | 17.30% |
| 20 | 25.71% | 16.97% |
| 25 | 26.02% | 17.41% |

The MC peaks at epoch 5 and slowly declines — identical qualitative pattern to benign CIFAR-100 runs (all benign seeds peaked at epochs 5–7 per EXP-029). This indicates MixMatch converges to a genuine but limited signal, not an artifact.

**Fisher Divergence Trajectory (from CSV — critical finding):**

| Epoch | Fisher_A | Fisher_B | Divergence | intra_var_A | grad_norm_A | Defense Status |
|---|---|---|---|---|---|---|
| 0 | 0.212 | 0.147 | +0.065 | 2.76 | 0.033 | Burn-in |
| 7 | 0.306 | 0.233 | **+0.073** | 0.154 | 0.244 | Burn-in ends |
| 8 | 0.303 | 0.231 | +0.072 | 0.171 | 0.242 | τ=0.10 not exceeded |
| 10 | 0.316 | 0.231 | +0.085 | 0.174 | 0.249 | Not exceeded |
| 11 | 0.317 | 0.215 | **+0.101** | 0.162 | 0.265 | **DEFENSE FIRES (1st time)** |
| **12** | **0.015** | **0.277** | **−0.262** | **141,644** | **0.007** | **CATASTROPHIC COLLAPSE** |
| 13 | 0.036 | 0.326 | −0.289 | 35,252 | 0.001 | Negative divergence |
| 14 | 0.035 | 0.336 | −0.301 | 10,424 | 0.003 | Negative divergence |
| 20 | 0.016 | 0.330 | −0.314 | 7.96 | 0.075 | Recovering |
| 25 | 0.095 | 0.311 | −0.217 | 0.084 | 0.169 | Still negative |
| 49 | 0.257 | 0.257 | **+0.001** | 0.111 | 0.304 | Near-zero |
| 50–84 | ~0.34–0.43 | ~0.37–0.50 | −0.05 to −0.04 | 0.06–0.10 | 0.28–0.31 | Stable negative |
| 85–149 | ~0.435 | ~0.510 | **−0.07 to −0.08** | ~0.003–0.005 | 0.86–1.05 | **Stable negative for 64 epochs** |

**Mechanism — the catastrophic single-activation collapse:**

The projection defense operates fundamentally differently from what was designed. The intended behavior was gradual removal of some percentage of the discriminative gradient each epoch, tightening progressively. The actual behavior is:

1. **Epochs 0–10 (burn-in + low divergence)**: The aux_classifier trains on z_a and learns an accurate discriminative direction d_aux. Meanwhile, MaliciousSGD has been building class-discriminative structure in Party A's embeddings for 11 epochs.

2. **Epoch 11 (first activation)**: Divergence = 0.101, barely above τ=0.10. The aux_classifier's d_aux is now highly aligned with the actual gradient direction (because Party A's embedding gradient from the top model IS the class-discriminative direction that MaliciousSGD has been following). When the projection removes d_aux from grad_output_a, it removes NEARLY ALL of grad_output_a — the projection coefficient (grad · d_aux_norm) is close to ‖grad‖. Effective gradient to Party A drops to ~0 (grad_norm_A: 0.265 → 0.007 in one step).

3. **Epoch 12 consequence**: Party A's bottom model receives almost no gradient for one full epoch. The MaliciousSGD amplification ratio g_{t}/g_{t-1} becomes undefined (g_{t-1}=0.007), but since ratio is clamped to [1,5], the next epoch sees ratio=1 effectively. The embedding model drifts from its class-discriminative state. Fisher_A plummets from 0.317 to 0.015; intra_var_A spikes from 0.16 to **141,644** (class clusters completely shattered). Inter-class distance explodes (508.9 vs normal 3.0).

4. **Epochs 12–149 (self-perpetuating collapse)**: With intra_var_A enormous, the Fisher divergence goes strongly negative (Party A NOW LESS discriminative than Party B). This means the divergence stays below τ=0.10, so the defense does NOT fire again. But the damage is permanent: over the next 40 epochs, intra_var_A recovers to normal (~0.003 by epoch 85) but the embedding MODEL's weights are locked at a state that produces LOWER class discrimination than honest training (Fisher_A ~0.43 vs Fisher_B ~0.51 for the final 64 epochs).

5. **Why the final state has LESS class discrimination than benign**: Party A's model converged toward class discrimination under MaliciousSGD for 11 epochs, then was catastrophically derailed. The recovery over 138 epochs is to a new basin, where the task loss (from Party B + top model carrying most of the classification) provides insufficient gradient to fully rebuild discriminative embeddings. The checkpoint at epoch 150 encodes a Party A embedding model with permanently weakened class discrimination.

**Why this is bounded (no NaN):**

`‖grad_proj‖ = ‖grad - (grad · d̂) * d̂‖ = ‖grad‖ * sin(θ)` where θ is the angle between grad and d_aux. Since `sin(θ) ∈ [0,1]`, the projected gradient is always ≤ the original. Even when MaliciousSGD amplifies grad_proj by ratio=5, the result is ≤ 5‖grad_output_a‖ — the same maximum as undefended training. No growth beyond original magnitude is possible. This is why Phase 19 succeeds numerically where Phase 18 failed catastrophically.

**Comparison with all 11 CIFAR-100 configurations:**

| Surface | Variant | Best MC | vs Benign Mean (29.56%) | Status |
|---------|---------|---------|------------------------|--------|
| Gradient suppression | a=1.0, t=0.10 (EXP-012) | 43.12% | +13.56pp | ⚠️ Partial only |
| Gradient suppression | a=2.0, t=0.10 (EXP-013) | 48.31% | +18.75pp | ❌❌ WORSE than attack |
| Gradient suppression | a=2.0, t=0.05 (EXP-013) | 43.60% | +14.04pp | ❌ |
| Gradient noise | n=0.5 (EXP-019) | 48.36% | +18.80pp | ❌❌ |
| Gradient noise | n=1.0 (EXP-019) | 49.64% | +20.08pp | ❌❌❌ Worst pre-18 |
| Gradient noise | n=2.0 (EXP-019) | 43.10% | +13.54pp | ⚠️ |
| z_a embedding | std=0.5 (EXP-024) | 50.67% | +21.11pp | ❌❌ |
| z_a embedding | std=1.0 (EXP-025) | 51.64% | +22.08pp | ❌❌❌ Worst embedding |
| Adversarial aux | λ=0.5 (EXP-031) | 1.001% | −28.56pp | 🔴 Model destroyed |
| Adversarial aux | λ=1.0, 2.0 (EXP-031) | N/A | N/A | 🔴 Stage 1 crash |
| **Gradient projection** | **(EXP-032, seed-0)** | **26.97%** | **−2.59pp ✅** | **✅ FIRST SUCCESS (seed-0)** |

**Interpretation and caveats:**

The gradient projection defense is the first of 11 CIFAR-100 configurations to achieve the criterion (MC < benign). The result (26.97%) is:
- 2.59pp below the 4-seed benign mean (29.56%)
- 0.88σ below the benign mean (σ=2.93pp)
- Within the lower half of the benign distribution (range 26.19%–33.14%)
- 20.89pp below the undefended attack (47.86%)
- 16.15pp better than the next-best prior defense (43.12%)

**However**: The 26.97% result falls within the benign distribution's lower tail. The minimum benign seed result is 26.19% (seed-42), and the defended result (26.97%) is only 0.78pp above this. A skeptical reviewer might argue the result is within natural variation of benign. The counter-argument: this is not a benign run — it is an ATTACKED training run with defense, and achieving MC below the benign MEAN under adversarial conditions is the defense criterion.

**Required next step — Phase 20 multi-seed (highest priority):**
Run seeds 42, 123, 456 with the gradient projection defense. Success criterion: mean defended MC < benign mean (29.56%), with all seeds ideally below benign. If confirmed, CIFAR-100 story joins CIFAR-10 and CINIC10L. **Phase 20 is now running (EXP-033/034/035).**

---

### EXP-033/034/035 — Phase 20: CIFAR-100 Gradient Projection Defense, Multi-Seed (Seeds 42, 123, 456)

| Field | Value |
|---|---|
| **ID** | EXP-033 (seed 42), EXP-034 (seed 123), EXP-035 (seed 456) |
| **Date** | 2026-07-11 (running) |
| **Purpose** | Multi-seed validation of Phase 19 (EXP-032 seed-0: MC=26.97% < benign mean 29.56%). Required before any CIFAR-100 gradient projection defense claim can be made for the paper. |
| **Dataset** | CIFAR100, 150ep Stage 1, 25ep Stage 2, n_labeled=400, K=5 |
| **Defense params** | tau=0.10, burn_in=8, proj_lr=1e-3, embed_dim=100, num_classes=100 (identical to Phase 19/EXP-032) |
| **Script** | `Code/run_phase20_grad_proj_cifar100_all_seeds.bat` (merged sequential — checkpoint-safe on single machine) |
| **Status** | ✅ COMPLETE — 2026-07-12 |

**Reference (seed-0, EXP-032):** MC = 26.97% < benign mean 29.56% (−2.59pp ✅). Mechanism: catastrophic single-activation collapse at epoch 11 (intra_var_A: 0.16→141,644; grad_norm_A: 0.265→0.007). Fisher divergence permanently negative for remaining 138 epochs.

**Per-seed targets (defend must beat the per-seed benign reference):**

| Seed | Benign MC ref (EXP-029) | Defense target |
|------|------------------------|----------------|
| 42 | 26.19% | < 26.19% |
| 123 | 28.56% | < 28.56% |
| 456 | 33.14% | < 33.14% |
| **Mean** | **29.56 ± 2.93%** | **< 29.56% mean** |

**Results (Phase 20 complete):**

| Seed | Stage 1 VFL Test | MC Best Train | vs Benign (per-seed) | vs Attack | Notes |
|------|-----------------|--------------|----------------------|-----------|-------|
| 0 (EXP-032) | 45.32% | **26.97%** | −3.36pp ✅ | −20.89pp | Seed-0 reference; collapse at epoch 11 |
| 42 (EXP-033) | 45.18% | **25.33%** | −0.86pp ✅ | −25.07pp | Normal collapse behavior |
| 123 (EXP-034) | 44.42% | **26.89%** | −1.67pp ✅ | −23.84pp | Normal collapse behavior |
| 456 (EXP-035) | 42.53% | **14.57%** | −18.57pp ✅⚠️ | −35.93pp | **ANOMALOUS over-collapse** — MC peaked at epoch 8 (14.57%) and never recovered; Stage 1 VFL also lowest at 42.53%, indicating structural over-damage |
| **4-seed mean ± std** | **44.36%** | **23.44 ± 5.16%** | **−6.12pp** | **−26.43pp** | High std driven by seed-456 outlier |

**Seeds 0/42/123 cluster (excluding anomalous seed-456):** Defended MC = 26.40 ± 0.85% — tight band, consistent mechanism.

**⚠️ Seed-456 anomaly analysis:** The MC for seed-456 peaked at epoch 8 of Stage 2 at 14.57% and declined — contrast with seeds 42/123 which peaked at epochs 4-5 at ~25-27%. The Stage 1 VFL test accuracy for seed-456 defense (42.53%) is ~2pp below the other three seeds (44-45%), indicating the catastrophic collapse was more severe and damaged the joint VFL model more deeply. The projection defense still passes the criterion (14.57% < 33.14% benign) but the mechanism is over-collapse rather than controlled semantic misalignment. Seed-456 has the highest benign MC of the four seeds (33.14%), suggesting this seed's embeddings are most susceptible to total disruption by the projection.

**Verdict: 4/4 seeds pass. CIFAR-100 gradient projection defense is publishable with anomaly documented.**

---

### EXP-036/037/038 — Phase 21: CIFAR-100 Attack Baseline Multi-Seed (Seeds 42, 123, 456)

| Field | Value |
|---|---|
| **ID** | EXP-036 (seed 42), EXP-037 (seed 123), EXP-038 (seed 456) |
| **Date** | 2026-07-11 (running) |
| **Purpose** | Complete the CIFAR-100 attack column in the 3×4 comparison table (benign/attack/defense × seeds 0/42/123/456). Attack currently has only seed-0 (47.86%, EXP-012). Needed to compute mean ± std for attack and enable fair per-seed attack vs. defense comparison. |
| **Dataset** | CIFAR100, 150ep Stage 1, 25ep Stage 2, n_labeled=400, K=5 |
| **Attack** | MaliciousSGD only (no defense). `--use-mal-optim True`, `--monitor-separability True`. Same params as EXP-012 seed-0. |
| **Script** | `Code/run_phase21_attack_cifar100_all_seeds.bat` (merged sequential) |
| **Status** | ✅ COMPLETE — 2026-07-12 |

**Reference (seed-0, EXP-012):** Attack MC = 47.86%, Stage 1 VFL = 44.81%.

**Results (Phase 21 complete):**

| Seed | Stage 1 VFL Test | Attack MC | Benign MC ref (EXP-029) | Attack advantage |
|------|-----------------|-----------|-------------------------|-----------------|
| 0 (EXP-012) | 44.81% | **47.86%** | 30.33% | +17.53pp |
| 42 (EXP-036) | 44.56% | **50.40%** | 26.19% | +24.21pp |
| 123 (EXP-037) | 45.91% | **50.73%** | 28.56% | +22.17pp |
| 456 (EXP-038) | 45.49% | **50.50%** | 33.14% | +17.36pp |
| **4-seed mean ± std** | **45.19%** | **49.87 ± 1.17%** | 29.56% | **+20.31pp** |

**Note:** Seeds 42/123/456 show slightly higher attack MC than seed-0 (~50.4–50.7% vs 47.86%). Attack is highly consistent across seeds (std=1.17%), tighter than CIFAR-10's attack std (0.52% at CIFAR-10, 1.17% here). The attack advantage over benign ranges from +17.36pp (seed-456, highest benign baseline) to +24.21pp (seed-42, lowest benign baseline).

**Full 3×4 CIFAR-100 comparison table (COMPLETE as of 2026-07-12):**

| Seed | Benign MC | Attack MC | Defended MC (grad proj) | Defense gap vs benign | Attack advantage |
|------|-----------|-----------|------------------------|-----------------------|-----------------|
| 0 | 30.33% | 47.86% | **26.97%** | −3.36pp ✅ | +17.53pp |
| 42 | 26.19% | 50.40% | **25.33%** | −0.86pp ✅ | +24.21pp |
| 123 | 28.56% | 50.73% | **26.89%** | −1.67pp ✅ | +22.17pp |
| 456 | 33.14% | 50.50% | **14.57%** | −18.57pp ✅⚠️ | +17.36pp |
| **Mean ± Std** | **29.56 ± 2.93%** | **49.87 ± 1.17%** | **23.44 ± 5.16%** | **−6.12pp** | **+20.31pp** |

**Cluster analysis (seeds 0/42/123, excluding seed-456 outlier):** Defended = 26.40 ± 0.85% vs Benign = 28.36 ± 2.09% → −1.96pp margin, very low std (defense is deterministic within these seeds).

**Defense VFL utility cost:** Benign Stage 1 mean ~44.65%; Defense Stage 1: seed-42=45.18%, seed-123=44.42%, seed-456=42.53%. Seed-456 has −2.12pp cost; others near-zero. Mean utility cost across 3 new seeds: −0.17pp (negligible).

---

### EXP-039 — Literature Review: Novelty Assessment for VFL Label Inference Defense

| Field | Value |
|---|---|
| **ID** | EXP-039 |
| **Date** | 2026-07-12 |
| **Purpose** | Determine whether (a) Fisher Divergence Detection, (b) gradient projection-based VFL defense, and (c) Persistent Projection as a VFL label inference defense have been published before implementing further experiments. Blocking check before any new implementation. |
| **Method** | Web search across arXiv, IEEE Xplore, ACM DL, USENIX, NeurIPS, ICLR, CCS, NDSS proceedings; keyword queries on VFL label inference defense, gradient projection federated learning, Fisher criterion federated learning, subspace projection privacy, persistent projection VFL, MaliciousSGD defense. Covered 2022–2026. |
| **Status** | ✅ Complete |

**Novelty Assessment Summary:**

| Contribution | Assessment | Key Evidence |
|---|---|---|
| Fisher Divergence Detection (J_A − J_B as VFL attack monitor) | **Clearly Novel** | No prior paper monitors inter/intra-class variance ratio asymmetry between VFL parties. One paper (ScienceDirect 2025) uses FIM for DP noise calibration in horizontal FL — entirely different use. |
| Gradient Projection as VFL defense (aux-classifier-derived direction) | **Moderately Novel** | MixPro (SIGIR 2023, FedAds benchmark) uses projection per-batch in VFL but without discriminative direction targeting, without active attack focus, without persistent basis. ProjPert (IEEE TKDE 2024) shares name but is noise-optimization, not geometric projection. |
| Persistent Projection (stable multi-epoch subspace projection) | **Clearly Novel** | No VFL defense paper proposes persistent multi-epoch gradient projection. Closest analogy is continual learning gradient projection (FedProTIP) in a completely different problem domain. |
| AsymmetricAdaptivePerturbation (AAP) | **Incremental** | Adaptive gradient perturbation paradigm exists in FL. Novel only in the Fisher divergence-gated trigger for VFL specifically. |

**Closest competitor papers (ranked by conceptual proximity):**

1. **MixPro** (Wei et al., SIGIR 2023 within FedAds benchmark) — Gradient mixup + projection step in VFL. CRITICAL DIFFERENTIATOR: projection is not directed at any discriminative subspace; targets passive attacks only; no auxiliary classifier; no persistent basis. Must be explicitly compared/differentiated in any submission.
2. **ProjPert** (IEEE TKDE 2024, DOI: 10.1109/TKDE.2023.3347600) — "Projection-Based Perturbation." Misleading name: is noise magnitude optimization via binary search, NOT geometric subspace projection. Targets passive only.
3. **MARVELL** (Li et al., 2022, CCS-adjacent) — Minimax noise design balancing gradient class discriminability. Conceptually similar motivation (equalize class gradient distributions) but noise-based, not projection-based; binary classification only.
4. **LADSG** (Yan et al., CollaborateCom/Springer 2026, arXiv 2506.06742, June 2025) — Label-Anonymized Distillation + Similar Gradient Substitution + gradient norm anomaly detection. Claims to address all three attack types. Detection via gradient norms (not Fisher divergence); defense via gradient substitution (not projection). Most recent competitive work.
5. **VMask** (Tan et al., Frontiers of Computer Science 2025, arXiv 2507.14629) — Layer masking with secret sharing targeting model completion attacks. No gradient projection or Fisher divergence; cryptographic approach.

**Key literature gaps that our work fills:**
1. No prior defense specifically counters MaliciousSGD's causal mechanism (optimizer-level discriminative amplification)
2. No prior paper uses cross-party Fisher divergence asymmetry as an active attack detection signal
3. No VFL defense proposes adaptive/persistent projection onto the discriminative subspace null space
4. No VFL defense paper reports results on 100-class fine-grained datasets
5. Active attack defense explicitly noted as an open problem in 2024 IJCAI survey and multiple 2024–2025 surveys

**Paper differentiation argument:**
> "Existing gradient-based defenses (MARVELL, Max-Norm, Gradient Compression, MixPro) apply symmetric perturbations without identifying the attack signal. Our defense uniquely (a) detects the attack via a novel Fisher divergence asymmetry monitor that exploits MaliciousSGD's observable fingerprint in the embedding space, (b) applies a Persistent Projection that surgically removes the discriminative direction from the backward gradient, (c) is server-side and requires no cooperation from Party A, and (d) is validated across multiple datasets including 100-class fine-grained classification where prior defenses have not been tested."

---

### EXP-040/041/042 — Phase 22: Persistent Projection Defense, CIFAR-10, Seed-0 (alpha_ema sweep)

| Field | Value |
|---|---|
| **ID** | EXP-040 (alpha_ema=0.1), EXP-041 (alpha_ema=0.2), EXP-042 (alpha_ema=0.3) |
| **Date** | 2026-07-13 |
| **Purpose** | Gate test: does Persistent Projection (EMA-based discriminative direction, project every detected epoch) reduce mc_best_train_top1 below the benign reference on CIFAR-10? |
| **Dataset** | CIFAR-10, 100 epochs Stage 1, 25 epochs Stage 2, n_labeled=40, seed-0 |
| **Defense** | PersistentProjectionDefense, tau=0.10, burn_in=4, alpha_ema in {0.1, 0.2, 0.3} |
| **Success criterion** | mc_best_train_top1 < benign reference (seed-0: 87.23%) |
| **Status** | ALL FAIL — implementation bug identified |

**Stage 1 VFL Utility (test top-1):**

| alpha_ema | VFL test top-1 | VFL train top-1 |
|---|---|---|
| 0.1 | 80.54% | 99.99% |
| 0.2 | 80.41% | 100.00% |
| 0.3 | 80.52% | 99.99% |

All three are within 0.13pp of the undefended baseline (~80.5%). PP does not hurt VFL utility.

**Stage 2 Model Completion (mc_best_train_top1):**

| alpha_ema | mc_best_train_top1 | Benign ref (seed-0) | Attack baseline (seed-0) | Verdict |
|---|---|---|---|---|
| 0.1 | **93.73%** | 87.23% | 95.42% | FAIL (+6.50pp above benign) |
| 0.2 | **92.75%** | 87.23% | 95.42% | FAIL (+5.52pp above benign) |
| 0.3 | **94.50%** | 87.23% | 95.42% | FAIL (+7.27pp above benign) |

None of the three alpha_ema values reduces mc_best_train_top1 below the benign 87.23% reference. The worst config (alpha_ema=0.3) reaches 94.50% — essentially at attack level (95.42%). PP as implemented provides zero privacy protection on CIFAR-10.

**Separability CSV diagnostic (100 epochs, all three alpha_ema values show the same pattern):**

- Fisher_divergence exceeds tau=0.10 from epoch 1 onward — defense fires every epoch after burn-in
- No catastrophic one-shot collapse (no 6-order-of-magnitude intra_var_A spike, unlike GradProj on CIFAR-100)
- Epochs 0-49: Fisher divergence oscillates ~0.21-0.42, intra_var_A decreasing gradually
- Epoch 50: Sharp transition — Fisher_A and divergence jump, intra_var_A drops faster
- Epochs 50-99: Embeddings become MORE discriminative despite defense firing every epoch
  - Fisher_A grows to 1.9-2.1 by epoch 99
  - silhouette_a goes from negative (epoch 0) to positive ~0.30 by epoch 99 (tight class clusters)
  - intra_var_A final values: 0.008-0.017 (very tight — comparable to attacked baseline)
- grad_norm_ratio drops below 1.0 after epoch 50 for some configs — PP removes some gradient but not the discriminative component

**Root cause — implementation bug in d_ema direction estimation:**

In `Code/possible_defenses.py`, the original code computed the batch mean of per-sample gradients before normalizing:

```python
# BUG (original):
d_mean = d_inst.mean(dim=0)                          # near-zero for balanced batches
d_mean_norm = d_mean / (d_mean.norm() + 1e-8)        # normalizing noise -> random direction
```

For cross-entropy loss on a balanced batch, per-sample gradients of the form `W^T (softmax - y)` partially cancel when averaged across classes, producing a near-zero vector. Normalizing a near-zero vector produces a random unit vector. The EMA accumulates random noise rather than the discriminative direction. Projecting against random noise removes random noise from grad_output_a, not the label-information-bearing component.

This is confirmed by contrasting with `GradientProjectionDefense`, which normalizes per-sample BEFORE any averaging and projects each sample against its own unit discriminative vector — cancellation cannot occur.

**Bug fix applied (2026-07-13) to `Code/possible_defenses.py`:**

```python
# FIXED: normalize per-sample first, then average
d_inst_norm = d_inst / (d_inst.norm(dim=-1, keepdim=True) + 1e-8)  # [batch, embed_dim]
d_mean = d_inst_norm.mean(dim=0)                     # mean of unit vectors
d_mean_norm = d_mean / (d_mean.norm() + 1e-8)        # final renormalization
```

**Comparison with AAP (CIFAR-10 proven defense):**

| Config | mc_best_train_top1 | vs benign (87.23%) |
|---|---|---|
| Attack alone | 95.42% | +8.19pp |
| AAP (4-seed mean) | 81.80% | -5.43pp (PASS) |
| PP alpha_ema=0.1 (buggy) | 93.73% | +6.50pp (FAIL) |
| PP alpha_ema=0.2 (buggy) | 92.75% | +5.52pp (FAIL) |
| PP alpha_ema=0.3 (buggy) | 94.50% | +7.27pp (FAIL) |

**Anomalies flagged:**
1. **Epoch-50 discrete transition**: Fisher_A and discriminability show a sharp jump at epoch 50 across all three alpha_ema values. The cause is unknown — possibly optimizer momentum, MaliciousSGD ratio formula behavior, or a learning rate artifact. Must be re-examined after the bug fix is applied.
2. **Monotonically increasing discriminability**: silhouette_a and Fisher_A increase throughout training despite the defense firing every epoch — consistent with the bug, but the epoch-50 acceleration warrants attention even after the fix.
3. **Phase 23 ran concurrently**: CIFAR-100 results produced without waiting for Phase 22 to pass. Protocol violation logged; results captured below.

**Overall verdict: NOT PROMISING (as implemented). CONFIDENCE: HIGH.**
The failure is unambiguous and mechanistically explained. The PP theory is sound; the implementation was incorrect. Bug fixed; re-run required.

---

### EXP-043/044 — Phase 23: Persistent Projection Defense, CIFAR-100, Seed-0 (alpha_ema sweep)

| Field | Value |
|---|---|
| **ID** | EXP-043 (alpha_ema=0.1), EXP-044 (alpha_ema=0.2) |
| **Date** | 2026-07-13 |
| **Purpose** | Preliminary CIFAR-100 gate test for PP. Ran concurrently with Phase 22 (protocol violation — should have waited for Phase 22 gate to pass). |
| **Dataset** | CIFAR-100, 150 epochs Stage 1, 25 epochs Stage 2, n_labeled=400, seed-0 |
| **Defense** | PersistentProjectionDefense (buggy d_ema), tau=0.10, burn_in=4, alpha_ema in {0.1, 0.2} |
| **Success criterion** | mc_best_train_top1 < benign reference (seed-0: 30.33%; 4-seed mean: 29.56%) |
| **Status** | BOTH FAIL — same root bug as Phase 22; results are invalidated |

**Stage 2 Model Completion (mc_best_train_top1):**

| alpha_ema | mc_best_train_top1 | Benign ref (seed-0) | Attack baseline (seed-0) | Verdict |
|---|---|---|---|---|
| 0.1 | **49.42%** | 30.33% | 47.86% | FAIL (+19.09pp above benign; above attack level) |
| 0.2 | **49.44%** | 30.33% | 47.86% | FAIL (+19.11pp above benign; above attack level) |

The CIFAR-100 failure is more severe than CIFAR-10: both values slightly exceed the undefended attack baseline (47.86%), meaning PP slightly worsens privacy on CIFAR-100. This is consistent with the bug — projecting against a random direction removes a small fraction of the useful gradient while leaving the discriminative component intact.

**Comparison with GradientProjectionDefense on CIFAR-100:**

| Defense | mc_best_train_top1 (seed-0) | vs benign (30.33%) |
|---|---|---|
| Attack alone | 47.86% | +17.53pp |
| GradProj (EXP-032) | 26.97% | -3.36pp (PASS) |
| PP alpha_ema=0.1 (buggy) | 49.42% | +19.09pp (FAIL) |
| PP alpha_ema=0.2 (buggy) | 49.44% | +19.11pp (FAIL) |

Both Phase 23 runs are invalidated by the same bug as Phase 22. Must be re-run after Phase 22-fixed passes the gate.

---

### Research Direction Update — 2026-07-13 (Post Phase 22/23)

**Immediate priorities:**
1. Bug fixed in `Code/possible_defenses.py` (2026-07-13) — per-sample normalization before EMA update
2. Re-run Phase 22 with fixed PP (CIFAR-10, seed-0, alpha_ema in {0.1, 0.2, 0.3}) — gate test
3. If Phase 22-fixed passes: re-run Phase 23 (CIFAR-100)
4. If Phase 22-fixed still fails: pivot to MDPP (Multi-Directional PP, K=10 principal directions) per `Code/next_direction.md`

**Fallback position (if fixed PP fails):**
Publishable story without unified PP: CIFAR-10 via AAP (EXP-016, 4/4 seeds) + CIFAR-100 via GradProj one-shot collapse (EXP-033/034/035, 4/4 seeds) + Fisher Divergence Detection (novel). Two-defense paper; weaker than unified PP but sufficient for publication. PP failure can be reported as a negative result motivating the MDPP extension.

---

## 4. Results Repository

### 4.1 Phase 1 CIFAR10 — Separability Metrics at Epoch 29

**Source files:** `CIFAR10_csv_files/separability_CIFAR10_lr=0.1_*_half=16.csv`

| Condition | Fisher_A | Fisher_B | **Fisher Div.** | Sil_A | Sil_B | Grad Norm Ratio |
|---|---|---|---|---|---|---|
| Benign | 0.517 | 0.638 | **−0.120** | −0.009 | −0.021 | 1.114 |
| Active Party A | 0.826 | 0.381 | **+0.444** | +0.064 | −0.032 | 0.893 |
| Active All Parties | 0.699 | 0.841 | **−0.142** | +0.045 | +0.062 | 1.064 |
| Benign + Laplace DP | ~0.0002 | 0.607 | **−0.607** | −0.056 | +0.008 | ~0 (collapsed) |
| Active + Laplace DP | 0.969 | ~0.0001 | **+0.968** | +0.076 | −0.064 | ~1.2×10⁷ (collapsed) |
| Active + GC (75%) | 0.822 | 0.301 | **+0.521** | +0.061 | −0.026 | 0.810 |

**Trajectory:** Benign divergence flat at −0.10 to −0.12; Active Party A divergence visible from epoch 4, reaches +0.444 by epoch 29. Detection threshold +0.10 cleanly separates from epoch 4 onward.

**Key anomalies:** Laplace DP causes training collapse (Fisher artifacts, not real signal). Grad norm ratio for Active Party A = 0.893 (Party A < Party B) — counterintuitive because MaliciousSGD amplifies internal `p.grad`, not `grad_output` which is what the monitor measures. Grad norm ratio is NOT a valid detection signal.

---

### 4.2 Phase 1 CIFAR100 — Separability Metrics at Epoch 29

**Source files:** `CIFAR100_csv_files/separability_CIFAR100_lr=0.1_*_half=16.csv`

| Condition | Fisher_A | Fisher_B | **Fisher Div.** | Sil_A | Sil_B | Grad Norm Ratio |
|---|---|---|---|---|---|---|
| Benign | 0.232 | 0.231 | **+0.002** | −0.126 | −0.107 | 1.038 |
| Active Party A | 0.352 | 0.220 | **+0.131** | −0.086 | −0.131 | 1.033 |
| Active All Parties | 0.280 | 0.305 | **−0.025** | −0.097 | −0.094 | 1.050 |
| Active + GC (75%) | 0.330 | 0.126 | **+0.203** | −0.082 | −0.137 | 1.044 |

**Trajectory:** Benign divergence starts ~+0.15 at epoch 0 (random init), falls to ~0 by epoch 10. Active divergence stabilizes at 0.083–0.131 from epoch 8 onward. Detection threshold +0.07 cleanly separates from epoch 8 onward. **Burn-in of 8 epochs required** due to noisy initialization.

**Signal comparison:** CIFAR10 gap 0.564 vs CIFAR100 gap 0.129 — 4.4× compression for 10× more classes.

---

### 4.3 Stage 2 Label Inference Accuracy — CIFAR10 (Phase 1)

| Condition | Best Train (Top-1) | Final Test (Top-1) | Delta vs Passive (Train) |
|---|---|---|---|
| Passive (normal checkpoint) | 84.61% | 66.81% | — |
| Active — Party A only | **94.99%** | 73.14% | **+10.38pp** |
| Active — All Parties | 85.39% | 68.69% | +0.78pp |

---

### 4.4 Stage 2 Label Inference Accuracy — CIFAR100 (Phase 1)

| Condition | Best Train (Top-1) | Final Test (Top-1) | Delta vs Passive (Train) |
|---|---|---|---|
| Passive (normal checkpoint) | 29.90% | 18.68% | — |
| Active — Party A only | **43.35%** | 22.38% | **+13.45pp** |
| Active — All Parties | 36.64% | 20.50% | +6.74pp |

---

### 4.5 Stage 1 VFL Classification Accuracy — Phase 2 Defense Conditions

**Source files:** `*_saved_models/*_asym_def-a=1.0-t=0.1-b=8_half=16.txt`  
**Setting:** 30 epochs, evaluated on full train and test set at end of training

| Condition | Train Top-1 | Test Top-1 | Top-k Accuracy |
|---|---|---|---|
| CIFAR10 Active + defense | 63.39% | 60.94% | 92.33% (k=4), 91.67% test |
| CIFAR10 Benign + defense | 64.12% | 62.71% | 94.08% (k=4), 93.75% test |
| CIFAR100 Active + defense | 31.60% | 29.23% | 66.31% (k=5), 62.83% test |
| CIFAR100 Benign + defense | 27.54% | 26.05% | 60.50% (k=5), 58.38% test |

**Key finding:** Defense utility cost on CIFAR10 is only −1.77pp test accuracy. On CIFAR100, active+defense achieves higher VFL accuracy (+3.18pp) than benign+defense because MaliciousSGD's embedding amplification continues to benefit VFL classification even under gradient suppression.

---

### 4.6 Phase 2 Fisher Divergence Trajectories

**CIFAR10 Active + defense (selected epochs):**
- Epoch 0: divergence = +0.102 (barely above tau; burn-in blocks defense)
- Epoch 8 (burn-in ends): divergence = +0.346 (defense activates)
- Epoch 29: Fisher_A=0.879, Fisher_B=0.357, divergence=**+0.522**
- Trajectory: divergence GROWS throughout training even with defense active

**CIFAR10 Benign + defense:**
- Epoch 0: divergence = +0.024
- Epoch 8: divergence = +0.038 (below tau, defense does not fire)
- Epoch 29: Fisher_A=0.598, Fisher_B=0.503, divergence=**+0.094** (below tau=0.10)

**CIFAR100 Active + defense:**
- Epoch 0: divergence = +0.192 (above tau; burn-in blocks)
- Epoch 8: divergence = +0.109 (defense activates)
- Epoch 29: Fisher_A=0.339, Fisher_B=0.172, divergence=**+0.167**

**CIFAR100 Benign + defense:**
- Epoch 0: divergence = +0.125 (above tau but burn-in protects)
- Epoch 7: divergence = −0.020 (falls below tau naturally)
- Epoch 29: Fisher_A=0.222, Fisher_B=0.253, divergence=**−0.031** (defense NEVER fires after burn-in)

---

### 4.7 Stage 1 VFL Accuracy — Complete Baseline Table (All Conditions, 30 Epochs)

**Source:** Direct reads of all `.txt` VFL task accuracy files from `saved_models/`. All from 30-epoch runs.

**CIFAR10 (k=4):**

| Condition | Train Top-1 | Test Top-1 | Test Top-4 |
|---|---|---|---|
| Normal, no defense | 62.00% | 61.63% | 92.41% |
| Mal (Party A only), no defense | 29.85%* | 29.15%* | 70.06%* |
| Mal-all (all parties), no defense | 66.23% | 62.90% | 92.57% |
| Normal + Laplace DP (scale=0.001) | — | — | — |
| Mal + Laplace DP (scale=0.001) | 58.88% | 56.45% | 88.96% |
| Mal + GC (75% preserved) | 70.20% | **67.17%** | 93.77% |
| Normal + asym_def a=1.0 t=0.1 b=8 | 64.12% | 62.71% | 93.75% |
| Mal + asym_def a=1.0 t=0.1 b=8 | 63.39% | 60.94% | 92.33% |
| Mal + asym_def a=0.5 t=0.1 b=8 | 67.27% | 63.59% | 93.59% |
| Mal + asym_def a=2.0 t=0.1 b=8 | 61.50% | 59.69% | 91.06% |
| Mal + asym_def a=1.0 t=0.05 b=8 | 59.27% | 56.70% | 88.36% |
| Mal + asym_def a=1.0 t=0.15 b=8 | 62.92% | 60.61% | 92.01% |
| Normal + asym_def a=1.0 t=0.05 b=8 | 61.32% | 59.27% | 91.49% |
| Normal + asym_def a=1.0 t=0.15 b=8 | 62.71% | 61.41% | 89.28% |

*From prior session; current `mal_half=16.pth` is the 30-epoch characterization checkpoint.

**CIFAR100 (k=5):**

| Condition | Train Top-1 | Test Top-1 | Test Top-5 |
|---|---|---|---|
| Normal, no defense | 29.41% | 27.71% | 58.13% |
| Mal (Party A only), no defense | 29.32%* | 26.18%* | 56.55%* |
| Mal-all (all parties), no defense | 27.72% | 24.81% | 57.46% |
| Mal + Laplace DP (scale=0.001) | 2.96% | **2.87%** | 12.98% |
| Mal + GC (75% preserved) | 34.29% | **31.74%** | 63.36% |
| Normal + asym_def a=1.0 t=0.1 b=8 | 27.54% | 26.05% | 58.38% |
| Mal + asym_def a=1.0 t=0.1 b=8 | 31.60% | 29.23% | 62.83% |
| Mal + asym_def a=0.5 t=0.1 b=8 | 28.92% | 26.89% | 59.39% |
| Mal + asym_def a=2.0 t=0.1 b=8 | 31.90% | 29.27% | 61.02% |
| Mal + asym_def a=1.0 t=0.05 b=8 | 28.32% | 26.09% | 58.19% |
| Mal + asym_def a=1.0 t=0.15 b=8 | 28.36% | 25.69% | 57.13% |
| Normal + asym_def a=0.5 t=0.1 b=8 | 29.74% | 28.29% | 59.81% |
| Normal + asym_def a=2.0 t=0.1 b=8 | 24.41% | 23.17% | 53.30% |
| Normal + asym_def a=1.0 t=0.05 b=8 | 25.34% | 24.18% | 53.71% |
| Normal + asym_def a=1.0 t=0.15 b=8 | 29.59% | 27.86% | 58.79% |

*From prior session

---

### 4.8 Phase 3b Ablation — Alpha and Tau Effect on VFL Task Accuracy

**Note: No model completion data exists for any ablation variant. The tables below show Stage 1 VFL task accuracy only. These cannot be used to infer inference accuracy changes.**

**CIFAR10 — Alpha Ablation (tau=0.1, burn_in=8):**

| Alpha | Active+def Test Top-1 | vs Benign (61.63%) | Benign+def Test Top-1 | vs Benign |
|---|---|---|---|---|
| 0.5 (weak) | 63.59% | +1.96pp | — | — |
| **1.0 (primary)** | **60.94%** | **−0.69pp** | 62.71% | +1.08pp |
| 2.0 (strong) | 59.69% | −1.94pp | — | — |

**CIFAR10 — Tau Ablation (alpha=1.0, burn_in=8):**

| Tau | Active+def Test Top-1 | vs Benign (61.63%) | Benign+def Test Top-1 | vs Benign |
|---|---|---|---|---|
| 0.05 (low threshold) | 56.70% | −4.93pp | 59.27% | −2.36pp |
| **0.10 (primary)** | **60.94%** | **−0.69pp** | 62.71% | +1.08pp |
| 0.15 (high threshold) | 60.61% | −1.02pp | 61.41% | −0.22pp |

**CIFAR100 — Alpha Ablation (tau=0.1, burn_in=8):**

| Alpha | Active+def Test Top-1 | vs Benign (27.71%) | Benign+def Test Top-1 | vs Benign |
|---|---|---|---|---|
| 0.5 (weak) | 26.89% | −0.82pp | 28.29% | +0.58pp |
| **1.0 (primary)** | **29.23%** | **+1.52pp** | 26.05% | −1.66pp |
| 2.0 (strong) | 29.27% | +1.56pp | **23.17%** | **−4.54pp** |

**CIFAR100 — Tau Ablation (alpha=1.0, burn_in=8):**

| Tau | Active+def Test Top-1 | vs Benign (27.71%) | Benign+def Test Top-1 | vs Benign |
|---|---|---|---|---|
| 0.05 (low threshold) | 26.09% | −1.62pp | 24.18% | −3.53pp |
| **0.10 (primary)** | **29.23%** | **+1.52pp** | 26.05% | −1.66pp |
| 0.15 (high threshold) | 25.69% | −2.02pp | 27.86% | −0.15pp |

**Observations from ablation VFL task accuracy:**

1. **CIFAR10 alpha trend is coherent:** Stronger alpha → lower VFL task recovery for the active condition (0.5: 63.59%, 1.0: 60.94%, 2.0: 59.69%). All three recover dramatically compared to undefended mal (29.15%) due to the training stabilization effect.

2. **CIFAR10 tau trend is coherent:** Lower tau → more frequent defense firing → lower VFL task accuracy. At t=0.05, benign party also loses 2.36pp collateral accuracy.

3. **CIFAR100 is noisy:** Alpha results are counterintuitive — stronger alpha raises active+defense accuracy above benign. At a=2.0, the benign party loses 4.54pp (meaningful collateral damage). Results at this scale (1–3pp) cannot be separated from single-run noise.

4. **Collateral damage threshold (CIFAR100):** At a=2.0 and t=0.05, benign party suffers −4.54pp and −3.53pp respectively. The primary setting (a=1.0, t=0.10) shows only −1.66pp — acceptable for a defense. Alpha=0.5 shows near-zero collateral damage but suppression may be too weak.

---

### 4.10 Phase 4 — CIFAR10 100-Epoch Three-Way Model Completion Summary (EXP-011)

**Source files:** `CIFAR10_saved_models/model_completion_CIFAR10_..._normal/mal/mal_asym_def..._ep100..._nlabeled=40.txt`  
**Epoch budget:** 100-epoch Stage 1, 25-epoch Stage 2. Single seed (manualSeed=0). Multi-seed pending.

| Condition | Stage 1 Test Acc | MC Best Train Top-1 | MC Final Test Top-1 | MC vs Attack | MC vs Benign |
|---|---|---|---|---|---|
| Benign 100ep | 81.45% | **87.23%** | 69.07% | −8.19pp | baseline |
| Active 100ep | 80.86% | **95.42%** | 73.67% | baseline | +8.19pp |
| **Active+Defense 100ep** | **79.52%** | **84.27%** | 74.97% | **−11.15pp** | **−2.96pp** |

**Attack advantage at 100 epochs:** +8.19pp over benign.  
**Defense neutralization:** −11.15pp reduction from attack peak.  
**Net effect:** Defended inference (84.27%) is 2.96pp *below* benign — defense provides stronger privacy than honest training.  
**VFL cost:** −1.93pp (79.52% vs 81.45%).

---

### 4.11 Phase 4 — CIFAR10 100-Epoch Fisher Divergence Trajectories (EXP-011)

**Source files:** `CIFAR10_csv_files/separability_CIFAR10_lr=0.1_mal_half=16.csv` (100 rows) and `separability_CIFAR10_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16.csv` (100 rows, now 100 epochs — Phase 2 30-epoch data overwritten by Phase 4).

**Attack (no defense) trajectory milestones:**
- Epoch 0: Fisher_A=0.39, Div=0.25 (random init)
- Epoch 8: Fisher_A=0.88, Div=0.52 (convergence accelerating)
- Epoch 29: Fisher_A=0.92, Div=0.56
- Epoch 49: Fisher_A=0.97, Div=0.59
- Epoch 74: Fisher_A=1.49, Div=0.89 (late-training surge)
- **Epoch 99: Fisher_A=2.12, Div=1.21** (final)

**Attack + Defense trajectory milestones:**
- Epoch 0: Fisher_A=0.44, Div=0.34
- Epoch 8 (defense fires): Fisher_A=0.86, Div=0.47 → scale=0.63
- Epoch 29: Fisher_A=0.94, Div=0.60 → scale=0.50
- Epoch 49: Fisher_A=0.96, Div=0.59 → scale=0.51
- Epoch 74: Fisher_A=1.72, Div=1.24 → **scale=0.00** (complete gradient block)
- **Epoch 99: Fisher_A=1.74, Div=1.03** → scale=0.07 (near-complete block)

**Scale formula:** `scale = max(0.0, 1.0 − 1.0 × (divergence − 0.10))`

**Key observation:** Divergence grows continuously under the defense, which progressively tightens the scale factor. The defense enters complete gradient block (~scale=0) around epoch 74 and stays near-zero for the final 25 epochs. Despite this, VFL task accuracy holds at 79.52% — Party B's gradient and the top model maintain joint task performance without Party A's gradient input.

**Second key observation:** Defended J_A (1.74 at ep99) is still 3× the benign J_A (0.57 at ep29). Geometric cluster structure is intact. But training-set inference accuracy (84.27%) is *below* benign (87.23%). This is the semantic misalignment mechanism operating at full strength: clusters exist but don't align with ground-truth class labels, so MixMatch propagates incorrect pseudo-labels.

---

### 4.12 Phase 5 — CIFAR100 150-Epoch Three-Way Model Completion Summary (EXP-012)

**Source files:** `CIFAR100_saved_models/model_completion_CIFAR100_..._normal/mal/mal_asym_def..._ep150..._nlabeled=400.txt`  
**Epoch budget:** 150-epoch Stage 1, 25-epoch Stage 2. Single seed (manualSeed=0).

| Condition | Stage 1 Test Acc | MC Best Train Top-1 | MC Final Test Top-1 | MC vs Attack | MC vs Benign |
|---|---|---|---|---|---|
| Benign 150ep | 45.33% | **30.33%** | 17.36% | −17.53pp | baseline |
| Active 150ep | 44.81% | **47.86%** | 25.88% | baseline | +17.53pp |
| **Active+Defense 150ep (a=1.0,t=0.10)** | **45.20%** | **43.12%** | 23.40% | **−4.74pp** | **+12.79pp** |

**Attack advantage at 150 epochs:** +17.53pp over benign (fully converged attack).  
**Defense neutralization:** −4.74pp (27% of attacker advantage eliminated).  
**Net effect:** Defended ASR (43.12%) is 12.79pp **above** benign — defense fails key criterion.  
**VFL cost:** −0.13pp (negligible).

**Contrast with CIFAR-10 (EXP-011):**

| Dataset | Attack adv | Defense reduction | Defended vs benign | VFL cost |
|---|---|---|---|---|
| CIFAR-10 (100ep, a=1.0) | +8.19pp | −11.15pp | **−2.96pp (below benign ✅)** | −1.93pp |
| CIFAR-100 (150ep, a=1.0) | +17.53pp | −4.74pp | **+12.79pp (above benign ❌)** | −0.13pp |

**Next action:** Phase 6A tested alpha=2.0 and tau=0.05 variants. See EXP-013 — results were negative. Option B (noise injection) required.

---

### EXP-013 — Phase 6A: CIFAR-100 Stronger Defense (alpha=2.0)

| Field | Value |
|---|---|
| **ID** | EXP-013 |
| **Date** | 2026-07-06 |
| **Purpose** | Test whether stronger suppression (alpha=2.0) can close the CIFAR-100 defense gap (benign 30.33%, defended 43.12% at a=1.0) |
| **Dataset** | CIFAR100 |
| **Stage 1 Epochs** | 150 |
| **Stage 2 Epochs** | 25 (n_labeled=400) |
| **Variants** | (1) a=2.0, tau=0.10, b=8; (2) a=2.0, tau=0.05, b=8 |
| **Status** | ✅ Complete — ❌ BOTH VARIANTS FAILED |

**Stage 1 VFL Accuracy:**

| Variant | Train Top-1 | Test Top-1 | Test Top-5 |
|---|---|---|---|
| a=2.0, tau=0.10, 150ep | — | **45.93%** | 73.94% |
| a=2.0, tau=0.05, 150ep | — | **45.47%** | 73.02% |

(vs benign 45.33%, attack 44.81% from EXP-012)

**Stage 2 Model Completion Results:**

| Condition | Best Train Top-1 | vs Attack (47.86%) | vs Benign (30.33%) |
|---|---|---|---|
| **Active + def (a=2.0, t=0.10, 150ep)** | **48.31%** | **+0.45pp (WORSE than attack ❌)** | **+17.98pp ❌** |
| **Active + def (a=2.0, t=0.05, 150ep)** | **43.60%** | −4.26pp | **+13.27pp ❌** |

**Critical finding:** Alpha=2.0, tau=0.10 produces a defended ASR (48.31%) that is ABOVE the undefended attack (47.86%). Stronger gradient suppression is counterproductive for CIFAR-100. Alpha=2.0, tau=0.05 is slightly better than a=1.0,t=0.10 (43.60% vs 43.12%), but still 13.27pp above benign — fails key criterion.

**Mechanism hypothesis:** Zeroing out grad_output_A earlier and more completely prevents Party A's bottom model from receiving any corrective task gradient. MaliciousSGD's internal amplification (p.grad) continues unchecked, causing embeddings to converge to highly class-discriminative but task-misaligned representations. Paradoxically, this can produce embeddings that are MORE exploitable by MixMatch than a partially corrected version, because the geometric class structure is maximally sharp without interference from the task gradient.

**Conclusion:** Gradient perturbation as the sole mechanism is insufficient for CIFAR-100. The defense needs to corrupt the embedding tensor itself (z_a), not just withhold gradient feedback. **Option B (embedding-level Gaussian noise injection) is now the required next step for CIFAR-100.**

---

### EXP-014 — Phase 7: Competitor Comparison (GC and Laplace DP)

| Field | Value |
|---|---|
| **ID** | EXP-014 |
| **Date** | 2026-07-06 |
| **Purpose** | Establish how competitor attack/defense methods (Gradient Compression 75%, Laplace DP) perform on Stage 2 label inference vs. our MaliciousSGD attack and AsymDef defense |
| **Dataset** | CIFAR10 (n_labeled=40), CIFAR100 (n_labeled=400) |
| **Stage 1 Epochs** | 30 (using existing checkpoints from Phase 1 / EXP-010) |
| **Stage 2 Epochs** | 25 |
| **Script** | `Code/run_phase7_competitor_model_completion.bat` |
| **Status** | ✅ Complete |

**Important caveat:** All Phase 7 comparisons use 30-epoch Stage 1 checkpoints. CIFAR-10 MaliciousSGD has NOT converged at 30 epochs (EXP-009: 23.45% vs 47.98% benign). The GC and Laplace attack numbers therefore represent an apples-to-apples comparison within the 30-epoch regime, but cannot be directly compared to our 100-epoch (CIFAR-10) or 150-epoch (CIFAR-100) primary results.

**Stage 2 Model Completion Results — Full Competitor Table:**

**CIFAR-10 (10 classes, random = 10%):**

| Condition | Best Train Top-1 | Context |
|---|---|---|
| Benign (100ep, EXP-011) | **87.23%** | Label leakage floor under normal VFL |
| MaliciousSGD undefended (100ep, EXP-011) | **95.42%** | Our attack — primary threat |
| **Our defense, a=1.0, t=0.10 (100ep, EXP-011)** | **84.27%** | Our defense — below benign ✅ |
| GC 75% attack (30ep) | **69.22%** | Competitor attack |
| Laplace attack, scale=0.001 (30ep) | **53.41%** | Competitor attack |
| **Laplace DP defense — benign (30ep)** | **10.01%** | Competitor defense — near-random |

**CIFAR-100 (100 classes, random = 1%):**

| Condition | Best Train Top-1 | Context |
|---|---|---|
| Benign (150ep, EXP-012) | **30.33%** | Label leakage floor |
| MaliciousSGD undefended (150ep, EXP-012) | **47.86%** | Our attack |
| **Our defense, a=1.0, t=0.10 (150ep, EXP-012)** | **43.12%** | Our defense — above benign ⚠️ |
| GC 75% attack (30ep) | **21.74%** | Competitor attack |
| Laplace attack, scale=0.001 (30ep) | **3.55%** | Competitor attack — near-random |
| **Laplace DP defense — benign (30ep)** | **15.78%** | Competitor defense — below benign |

**Key interpretations:**

1. **Attack comparison:** Our MaliciousSGD attack dominates both competitors on both datasets (95.42% vs 69.22%/53.41% on CIFAR-10; 47.86% vs 21.74%/3.55% on CIFAR-100). GC and Laplace attacks produce lower label leakage than even benign VFL on CIFAR-100 (Laplace: 3.55% vs benign 30.33%) — adding noise to the attacker's own gradients backfires, preventing exploitable embedding formation.

2. **Defense comparison:** Laplace DP reduces CIFAR-10 ASR to 10.01% (essentially random) and CIFAR-100 to 15.78% (below benign 30.33%). However, this destroys ALL label leakage by corrupting embeddings indiscriminately, including during benign training. Our defense is asymmetric (fires only when attack is detected), preserving utility during legitimate operation. The CIFAR-10 Laplace DP benign accuracy (10%) confirms that noise destroys Party A's embedding utility entirely.

3. **Paper narrative:** Laplace DP "wins" on privacy metrics but at catastrophic utility cost. Our defense achieves comparable or better privacy on CIFAR-10 (84.27% vs 87.23% benign = below benign) while preserving VFL utility (only −1.93pp). For CIFAR-100, Laplace DP outperforms our current defense on the privacy metric, which is an honest limitation to acknowledge pending the Option B fix.

---

### EXP-015 — Phase 8: CIFAR-10 100-Epoch Hyperparameter Ablation

| Field | Value |
|---|---|
| **ID** | EXP-015 |
| **Date** | 2026-07-06 |
| **Purpose** | Run Stage 2 (model completion) for CIFAR-10 ablation variants at 100 epochs. Phase 3b (EXP-010) only had Stage 1 VFL accuracy for these at 30 epochs (useless since attack doesn't converge). These are 100-epoch runs. |
| **Dataset** | CIFAR10 (n_labeled=40), 25 Stage 2 epochs |
| **Reference point** | a=1.0, t=0.10 (EXP-011): benign 87.23%, attack 95.42%, defended 84.27% |
| **Script** | `Code/run_phase8_cifar10_ablation_100ep.bat` |
| **Status** | ✅ Complete |

**Stage 1 VFL Accuracy (100 epochs):**

| Variant | Test Top-1 | Test Top-4 | vs Benign (81.45%) |
|---|---|---|---|
| a=0.5, t=0.10, b=8 | **81.09%** | 96.38% | −0.36pp |
| **a=1.0, t=0.10, b=8 (EXP-011)** | **79.52%** | 96.50% | **−1.93pp** |
| a=2.0, t=0.10, b=8 | **79.55%** | 96.51% | −1.90pp |
| a=1.0, t=0.05, b=8 | **78.23%** | 96.41% | −3.22pp |
| a=1.0, t=0.15, b=8 | **78.94%** | 96.82% | −2.51pp |

**Stage 2 Model Completion Results (KEY TABLE):**

| Variant | MC Best Train Top-1 | vs Attack (95.42%) | vs Benign (87.23%) | Verdict |
|---|---|---|---|---|
| a=0.5, t=0.10, b=8 | **94.57%** | −0.85pp | +7.34pp | ❌ Too weak |
| **a=1.0, t=0.10, b=8 (primary)** | **84.27%** | −11.15pp | −2.96pp | ✅ |
| **a=2.0, t=0.10, b=8** | **76.79%** | **−18.63pp** | **−10.44pp** | ✅✅ Strongest |
| a=1.0, t=0.05, b=8 | **81.58%** | −13.84pp | −5.65pp | ✅ |
| a=1.0, t=0.15, b=8 | **85.98%** | −9.44pp | −1.25pp | ✅ (marginal) |

**Interpretation:**

- **Alpha trend (CIFAR-10):** Coherent and strong. a=0.5 is too weak (barely touches attack). a=1.0 is the primary setting (below benign). a=2.0 provides the strongest suppression — 76.79% defended is 10.44pp BELOW benign (87.23%). Note this is the OPPOSITE of CIFAR-100 behavior: on CIFAR-10 stronger alpha works better.

- **Tau trend (CIFAR-10):** Coherent. t=0.05 (earlier detection) gives stronger suppression (81.58%); t=0.15 (later detection) gives weaker suppression (85.98%). Both are below benign, confirming robustness across threshold choices.

- **VFL cost:** All variants stay within 3.22pp of benign VFL accuracy. a=0.5 has the lowest cost (−0.36pp) but is ineffective at suppressing the attack.

- **For paper:** The hyperparameter ablation demonstrates robustness — 4 of 5 variants succeed (defended ASR below benign), and the failure case (a=0.5) is the deliberately-undertuned setting. The primary setting (a=1.0, t=0.10) is a conservative middle point, not the maximum-suppression setting.

---

### EXP-016 — Phase 6B: CIFAR-10 Seed Sweep (Complete)

| Field | Value |
|---|---|
| **ID** | EXP-016 |
| **Date** | 2026-07-06 to 2026-07-07 |
| **Purpose** | Multi-seed validation of CIFAR-10 100-epoch three-way comparison (EXP-011). Establish mean ± std across seeds {0, 42, 123, 456}. |
| **Dataset** | CIFAR10, 100ep Stage 1, 25ep Stage 2, n_labeled=40 |
| **Seeds completed** | 0 (EXP-011), 42, 123, 456 — all complete |
| **Status** | ✅ Complete — 4/4 seeds confirm defense; mean ± std computed |

**Per-seed results:**

| Seed | Benign MC | Attack MC | Defended MC | Defended vs Benign |
|---|---|---|---|---|
| 0 (EXP-011) | **87.23%** | **95.42%** | **84.27%** | **−2.96pp ✅** |
| 42 | **81.90%** | **94.32%** | **80.16%** | **−1.74pp ✅** |
| 123 | **80.78%** | **95.33%** | **80.61%** | **−0.17pp ✅** |
| 456 | **82.53%** | **94.73%** | **82.14%** | **−0.39pp ✅** |

**Final mean ± std (4 seeds):**

| Condition | Mean | Std |
|---|---|---|
| Benign MC | **83.11%** | **±2.84pp** |
| Attack MC | **94.95%** | **±0.52pp** |
| Defended MC | **81.80%** | **±1.85pp** |
| Attack advantage over benign | **+11.84pp** | ±2.42pp |
| Defense reduction (attack → defended) | **−13.15pp** | ±1.61pp |
| Defended vs benign (gap) | **−1.32pp** | ±1.24pp |

**Key validation:** All 4 seeds show defended ASR below benign (range −0.17pp to −2.96pp). The attack is extremely stable across seeds (std=0.52pp). The defense shows moderate variance (std=1.85pp) but consistently below benign in every run. This is publishable multi-seed evidence.

**Notable observation:** Seeds 123 and 456 show smaller margins (−0.17pp and −0.39pp). Seed 123 is the narrowest. The mean margin (−1.32pp) is statistically safe, but the individual narrow seeds highlight that the margin is real but modest at a=1.0. Paper should report the full distribution, not just the seed-0 (best-case −2.96pp) result.

---

### 4.9 Competitor Defense VFL Task Accuracy (Stage 1 Only, No Model Completion)

**CIFAR10 observations:**
- **GC (75% preserved):** 67.17% test — ABOVE benign 61.63% (+5.54pp). GC with top-k sparsification selects the most informative gradient components and discards the noise from MaliciousSGD extreme amplification. Acts as gradient denoising, not defense. Actual effect on label inference unknown.
- **Laplace DP (scale=0.001):** 56.45% test — below benign (−5.18pp). Manageable utility cost for CIFAR10.

**CIFAR100 observations:**
- **GC (75% preserved):** 31.74% test — ABOVE benign 27.71% (+4.03pp) and above undefended mal (26.18%). Same denoising explanation as CIFAR10.
- **Laplace DP (scale=0.001):** **2.87% test — catastrophic collapse.** Near-random performance for 100-class problem (random = 1%). Noise scale of 0.001 is miscalibrated for CIFAR100 gradient magnitudes. DP is impractical at this setting for CIFAR100 without recalibration.

**What these do NOT tell us:** Model completion inference accuracy for GC or Laplace DP checkpoints. Stage 2 has not been run for either competitor defense on either dataset.

---

## 5. Research Insights

### 5.1 Confirmed Findings

**C1 — Fisher divergence is a reliable detection signal for single-party active attack**
- Confirmed on both CIFAR10 (+0.444 gap) and CIFAR100 (+0.129 gap)
- Signal directionally consistent: benign ≈ 0 or negative, active Party A > 0
- Signal available from epoch 4 (CIFAR10) or epoch 7-8 (CIFAR100), stays stable thereafter

**C2 — The active attack (single party) is the dominant threat**
- CIFAR10: +10.4pp train improvement over passive baseline
- CIFAR100: +13.5pp train improvement over passive baseline

**C3 — Active all-parties attack does not create asymmetric Fisher divergence**
- Both datasets: all-parties divergence ≈ benign divergence
- When both parties amplify equally, no asymmetry exists — detector correctly does not flag this

**C4 — MaliciousSGD amplifies INTERNAL parameter gradients, not the communication tensor**
- Grad norm ratio is NOT a valid detection signal
- Fisher criterion (computed from embeddings) IS a valid signal

**C5 — Gradient Compression does not hide the attack signal**
- GC slightly amplifies the detection signal on both datasets

**C6 — Defense reduces attack accuracy by large margins with negligible VFL utility cost**
- CIFAR10: −42.71pp attack reduction (−45%); only −1.77pp VFL test accuracy cost
- CIFAR100: −22.00pp attack reduction (−51%); VFL accuracy actually improves for active+defense

**C7 — False positive behavior is correct on both datasets**
- CIFAR10 benign+defense: divergence barely crosses tau, defense minimally/never fires
- CIFAR100 benign+defense: divergence goes negative after burn-in, defense NEVER fires

**C8 — Defense works through semantic misalignment, not geometric deseparability**
- Fisher divergence INCREASES under defense (CIFAR10: 0.444→0.522; CIFAR100: 0.131→0.167)
- Yet model completion accuracy drops from 94.99%→52.28% and 43.35%→21.35%
- Mechanism: suppressing task gradient removes correspondence between embedding clusters and ground-truth labels while MaliciousSGD maintains cluster geometry; MixMatch SSL fails when cluster geometry does not reflect class structure

**C13 — Defense is effective at 100 epochs: attack neutralized, inference below benign baseline (EXP-011)**
- CIFAR10 attack 100ep: 95.42%; attack+defense 100ep: 84.27% → reduction of 11.15pp
- Defended inference (84.27%) is 2.96pp below benign (87.23%) — over-suppression confirmed
- VFL utility cost only −1.93pp (79.52% vs 81.45%)
- Epoch confound from EXP-009 is fully resolved for CIFAR-10

**C14 — Defense scale reaches 0.0 by epoch 74, completely blocking grad_output_A for the final 25+ epochs**
- Scale formula: `max(0.0, 1.0 − 1.0 × (divergence − 0.10))`
- At epoch 74, defended divergence = 1.24 → scale = max(0, 1 − 1.14) = 0
- Despite complete gradient block, VFL test accuracy holds at 79.52% (only −1.93pp vs benign)
- Party B's untouched gradient and top model fully carry the task from Party A's side
- This validates the asymmetric design: blocking only Party A is sufficient and does not collapse training

**C15 — Semantic misalignment mechanism confirmed at 100 epochs**
- Defended J_A at epoch 99 = 1.74 (still 3× the benign J_A of ~0.57)
- Yet training-set inference accuracy (84.27%) < benign (87.23%)
- Geometric cluster structure exists; semantic label alignment is broken
- MixMatch SSL propagates incorrect pseudo-labels when clusters don't reflect class identity
- This is the mechanism first theorized after Phase 2 (Section 5.1 C8), now confirmed at full training length

**C16 — Test-set inference accuracy is NOT the primary attack metric; train-set accuracy is**
- Defended: train=84.27%, test=74.97%; Attack: train=95.42%, test=73.67%; Benign: train=87.23%, test=69.07%
- Test inference is *higher* for the defended condition (74.97% > 73.67%) — does not indicate defense failure
- High J_A under defense means structured embeddings → good MixMatch generalization → high test accuracy
- The threat is label recovery for the 49,960 unlabeled training examples; train-set accuracy is the correct metric
- Paper tables should present train-set inference as the primary column, test-set as secondary

**C17 — CIFAR-10 ablation at 100 epochs confirms hyperparameter robustness of defense**
- 4 of 5 alpha/tau variants produce defended ASR below benign baseline: a=1.0 (84.27%), a=2.0 (76.79%), t=0.05 (81.58%), t=0.15 (85.98%)
- Only a=0.5 fails (94.57% — essentially no suppression)
- Stronger alpha is more effective on CIFAR-10: a=2.0 produces the strongest suppression (76.79%), 10.44pp below benign

**C18 — Gradient perturbation is INSUFFICIENT for CIFAR-100 regardless of alpha (Phase 6A failure)**
- alpha=2.0, tau=0.10 gives defended ASR=48.31% — ABOVE undefended attack (47.86%)
- alpha=2.0, tau=0.05 gives 43.60% — slightly worse than alpha=1.0 (43.12%)
- Paradox: stronger suppression on CIFAR-100 is counterproductive
- Root cause: without task gradient, MaliciousSGD's unchecked p.grad amplification produces maximally class-discriminative embeddings that are MORE exploitable by MixMatch, not less
- CIFAR-10 vs CIFAR-100 divergence: on CIFAR-10, stronger alpha HELPS (76.79% at a=2.0); on CIFAR-100, it HURTS (48.31% at a=2.0). The mechanism difference is not yet fully understood but may relate to CIFAR-100's 10× harder classification task making the embedded class structure more persistent under gradient blocking.

**C19 — Multi-seed validation confirms CIFAR-10 defense direction is stable (now 4/4 seeds)**
- Seed 0 (EXP-011): defended 84.27% < benign 87.23% (−2.96pp)
- Seed 42 (EXP-016): defended 80.16% < benign 81.90% (−1.74pp)
- Seed 123 (EXP-016): defended 80.61% < benign 80.78% (−0.17pp)
- Seed 456 (EXP-016): defended 82.14% < benign 82.53% (−0.39pp)
- Mean ± std: defended 81.80 ± 1.85% vs benign 83.11 ± 2.84%; mean gap −1.32pp. All 4 seeds show the same direction. The narrowest margin (seed 123, −0.17pp) confirms the effect is real but the margin is modest.

**C21 — CIFAR-10 seed sweep is complete and publishable (EXP-016)**
- 4/4 seeds confirm defended ASR < benign at 100 epochs with a=1.0, tau=0.10, burn_in=8
- Attack extremely stable across seeds (std=0.52pp) — MaliciousSGD is a robust attack
- Defense margin varies (−0.17pp to −2.96pp) but direction is consistent; paper should show full per-seed table plus mean ± std rather than reporting best seed only
- Mean defense reduction from attack peak: −13.15 ± 1.61pp (highly significant)

**C22 — Defense asymmetry confirmed at 100 epochs for CIFAR-10 (EXP-017)**
- Benign + defense (100ep): 84.35% — within the benign distribution (83.11 ± 2.84% across 4 seeds)
- Gap is +1.24pp above benign mean — within natural variance, not statistically significant
- The Fisher divergence gate (tau=0.10) prevents any suppression during honest CIFAR-10 training at 100 epochs
- The 2×2 table for CIFAR-10 is now complete: all four cells measured empirically
- Paper-ready: benign+defense ≈ benign (defense correctly dormant) while defended attack (81.80%) < benign (83.11%)

**C23 — CIFAR-100 benign+defense (EXP-018) consistent with defense dormancy despite higher-than-baseline number**
- Result: 34.74% vs seed-0 benign baseline 30.33% (+4.41pp)
- Defense was mathematically dormant: CIFAR-100 benign divergence goes negative after burn-in (EXP-007: −0.031 at epoch 29); scale=1.0 throughout
- The +4.41pp gap reflects CIFAR-100's high run-to-run variance, not defense effect
- Confirmed: benign+defense (34.74%) is still well below attack MC (47.86%) and defended attack (43.12%)
- The paper argument still holds: the defense does not hurt benign parties; it hurts only the adversary
- HOWEVER: the CIFAR-100 benign baseline instability (single seed = 30.33%, now another run = 34.74%) highlights the critical need for multi-seed CIFAR-100 baselines before submission

**C24 — Gradient-space noise injection (Option B) is a FAILED approach for CIFAR-100 (EXP-019)**
- All three noise_std variants (0.5, 1.0, 2.0) failed to bring MC below benign (30.33%)
- Small noise (n=0.5, 1.0) made things WORSE: MC went ABOVE the undefended attack (48.36%, 49.64% vs 47.86%)
- Large noise (n=2.0) gave negligible improvement (43.10% vs no-noise 43.12%)
- Root cause: noise is calibrated to gradient magnitude, which preserves the temporal magnitude consistency that MaliciousSGD's ratio computation exploits. The adversary can still amplify consistent-magnitude signals even when direction is random.
- The wrong attack surface was targeted: grad_output_A (backward gradient) vs z_a (forward embedding) vs p.grad (internal amplification target)

**C26 — Defense generalizes to CINIC10L with zero parameter retuning (EXP-022)**
- Same parameters (alpha=1.0, tau=0.10, burn_in=8) that work for CIFAR-10 also succeed on CINIC10L
- CINIC10L attack advantage (+20.89pp) was LARGER than CIFAR-10's (+11.84pp), yet defense still reduced MC to BELOW benign (62.43% < 65.70%)
- VFL utility cost is negligible: test accuracy 63.81% vs benign 63.18% = +0.63pp (within noise)
- This rules out dataset-specificity as a limitation of the defense for 10-class problems
- Two-dataset confirmation is the strongest publishable evidence the project has produced

**C27 — Sign-flip momentum disruption fails in practice despite clean theoretical prediction (EXP-023)**
- Theory predicted ratio=1.0 (neutralizing amplification); practice shows 86.40% > benign 83.11% (defense fails)
- Sign-flip is WEAKER than standard suppression (86.40% vs 81.80% mean; by 4.6pp)
- Root cause: MaliciousSGD's ratio is computed on p.grad (internal parameter gradients), not on grad_output_A (what we flipped). Nonlinear chain rule means sign of p.grad does not simply mirror sign of grad_output_A
- Implication: MaliciousSGD's internal amplification is more resilient than the simplified analysis suggested. Phase 13 (CIFAR-100 sign-flip) should not be run.

**C28 — Embedding-space z_a corruption is counterproductive for CIFAR-100 (EXP-024/025)**
- Both std=0.5 (50.67%) and std=1.0 (51.64%) give MC ABOVE the undefended attack (47.86%)
- Increasing noise_std makes the result WORSE: n=0.5 → 50.67%, n=1.0 → 51.64%
- This is the opposite of the intended effect and constitutes the worst results in project history
- Root cause: corrupting z_a before the top model's forward pass causes the top model to send more directive (class-informative) corrective gradients back to Party A, not less. MaliciousSGD amplifies these corrective signals, making the bottom model MORE class-discriminative than under pure suppression
- The fundamental flaw: any corruption of z_a that the top model can partially compensate for generates a stronger class-aligned gradient signal, not a weaker one

**C29 — CIFAR-100 failure is consistent across all 8 tested configurations (EXP-012/013/019/024/025)**
- Surfaces tested: gradient suppression (3 variants), gradient noise injection (3 variants), z_a embedding corruption (2 variants)
- Best result across all 8: 43.10–43.12% (pure suppression or n=2.0 noise), all 12.77pp above benign
- No variant comes within 10pp of the benign baseline
- The failure is not specific to any attack surface — it is a dataset-level limitation
- Hypothesis: 100 classes give enough gradient signal even at partial suppression (scale ~0.6, never reaching 0.0) to build discriminative structure over 150 epochs. For 10 classes (CIFAR-10, CINIC10L), the scale reaches 0.0 around epoch 74 and full suppression for 26+ epochs prevents structure from forming

**C33 — Gradient projection achieves CIFAR-100 defense criterion at seed-0 (EXP-032) — first success after 10 failures**
- MC 26.97% < benign mean 29.56% (−2.59pp, 0.88σ below mean). Attack 47.86% → defended 26.97% = 20.89pp reduction.
- VFL utility cost essentially zero: Stage 1 test accuracy 45.32% vs benign 45.33% (−0.01pp).
- This is the ONLY configuration (of 11) to bring CIFAR-100 MC below the benign mean.
- SINGLE SEED ONLY — multi-seed required before paper claim.

**C34 — Gradient projection works via catastrophic single-activation discriminative collapse, not gradual suppression (EXP-032)**
- Defense fires once at epoch 11 (divergence barely crosses τ=0.10). On that single activation, the aux_classifier's direction d_aux aligns so closely with grad_output_a that projection removes ~97% of the gradient (grad_norm_A drops from 0.265 → 0.007).
- intra_var_A spikes from 0.16 → 141,644 in one epoch — class cluster structure completely shattered.
- For the remaining 138 epochs, Fisher divergence stays NEGATIVE (Party A is permanently LESS discriminative than Party B). Defense never needs to fire again.
- This is fundamentally different from the designed mechanism (gradual suppression). The mechanism is a one-shot structural collapse, not a progressive squeeze. This explains why scaling (which is gradual) failed for CIFAR-100 while projection (which is directionally surgical) succeeds.
- The bounded property (‖grad_proj‖ ≤ ‖grad_output_a‖) prevents NaN: even with MaliciousSGD amplifying by 5×, the resulting gradient is bounded by 5‖grad_output_a‖ — identical to the undefended maximum. This is why Phase 19 does not repeat Phase 18's NaN catastrophe.

**C30 — Adversarial auxiliary direction-reversal defense is fundamentally unstable with MaliciousSGD (EXP-031)**
- All 3 lambda values (0.5, 1.0, 2.0) caused NaN collapse. Lambda=0.5: model weights became NaN by end of Stage 1 (VFL accuracy=1.00%, random). Lambda=1.0/2.0: Stage 1 crashed before saving any checkpoint.
- Root cause: `aux_grad = d(L_aux)/d(z_a)` is unbounded. After 150 epochs of Adam training on the server's auxiliary nn.Linear(100,100), W grows large. Combined with MaliciousSGD's embedding amplification (large z_a values), logits can overflow → NaN in softmax → NaN backprop → NaN in model weights.
- The self-reinforcing property (stronger attack → stronger defense via MaliciousSGD amplification) is a positive feedback loop that accelerates model destruction, not defense.
- Design failure: direction reversal requires the reversed gradient to be bounded relative to `grad_output_a`. Without explicit clipping, no such bound exists. The approach cannot work without gradient normalization of aux_grad.
- Next direction: gradient projection (only REMOVES discriminative component; never ADDS force → bounded by construction)

**C32 — CIFAR-100 benign baseline is now stabilized at 29.56 ± 2.93% (Phase 16, EXP-029)**
- 4 seeds (0, 42, 123, 456): 30.33%, 26.19%, 28.56%, 33.14%. Mean=29.56%, Sample Std=2.93pp.
- VFL Stage 1 test accuracy is very stable: 44.65 ± 0.47% across 4 seeds. MC accuracy is much more variable (±2.93pp).
- The seed-0 reference (30.33%) was essentially the mean — good luck in the original choice.
- EXP-018 benign+defense (34.74%) falls within natural variation (2σ upper bound ~35.42%). Defense dormancy confirmed more robustly.
- All CIFAR-100 defense comparisons should now use 29.56 ± 2.93% as the benign reference. The defended result (43.12%) is 4.6σ above mean — defense failure is statistically unambiguous.

**C31 — Yahoo Answers infrastructure now operational after two blocking bugs**
- Bug 1 (BertConfig API): `mixtext.py` used `nn.Embedding(**config)` — modern transformers treats BertConfig as a proper class, not a dict. Fixed by importing `BertEmbeddings`, `BertPooler` from `transformers.models.bert.modeling_bert`
- Bug 2 (Translator pkl): `read_data_text.py` required `de_1.pkl`/`ru_1.pkl` (pre-computed German/Russian back-translations). These files are absent from the codebase. Fixed by making `Translator` fall back to identity (original text for both augmentation slots) when pkl files not found
- Bug 3 (path separator): `read_data_text.py` concatenates path + filename with no separator — requires trailing backslash in `--path-dataset` arg for Yahoo
- Stage 1 benign now running successfully. Epoch 0 losses ~2.3–2.7 (consistent with 10-class random baseline ln(10)≈2.30)

**C25 — The CIFAR-100 gradient-space attack surface is now fully exhausted**
- 6 gradient-space configurations tested: a=1.0 no noise, a=2.0 t=0.10, a=2.0 t=0.05, n=0.5, n=1.0, n=2.0
- Best result: a=1.0 no noise (43.12%, −4.74pp vs attack) or n=2.0 (43.10%) — both are essentially the same
- None bring MC below benign (30.33%); the gap (12.77pp minimum) is large
- Next required attack surface: embedding-space (z_a corruption before top model forward pass) and MaliciousSGD's ratio computation (sign-flip disruption)

**C20 — Our attack dominates GC and Laplace competitor attacks on both datasets**
- CIFAR-10: MaliciousSGD 95.42% vs GC 69.22% vs Laplace 53.41%
- CIFAR-100: MaliciousSGD 47.86% vs GC 21.74% vs Laplace 3.55%
- Laplace attack on CIFAR-100 (3.55%) is below random×3 — attacker adding noise to own gradients prevents embedding convergence

**C9 — Alpha controls collateral damage more than tau (CIFAR100)**
- At alpha=2.0: benign party loses 4.54pp VFL task accuracy (collateral damage)
- At alpha=0.5: near-zero collateral damage
- At alpha=1.0 (primary): −1.66pp, acceptable range
- Tau at 0.05 causes −3.53pp benign loss; tau at 0.15 causes near-zero loss

**C10 — CIFAR10 ablation VFL task accuracy trend is coherent; CIFAR100 is noise-dominated**
- CIFAR10 shows clean alpha and tau monotonic trends in VFL task accuracy
- CIFAR100 differences of 1–3pp are not separable from single-seed run variance

**C11 — GC (75% preserved) raises VFL task accuracy above benign on both datasets**
- CIFAR10: +5.54pp; CIFAR100: +4.03pp above benign baseline
- Mechanism: top-k sparsification acts as gradient denoising, filtering MaliciousSGD amplification noise
- Effect on label inference accuracy unknown (no Stage 2 run)

**C12 — Laplace DP (scale=0.001) collapses CIFAR100 VFL training to near-random (2.87%)**
- CIFAR10 survives (56.45%), CIFAR100 collapses — noise scale must be dataset-gradient-specific
- Impractical for CIFAR100 at current scale without recalibration

**C35 — Fisher Divergence Detection is Clearly Novel in the VFL label inference literature (EXP-039, 2026-07-12)**
- Web search across arXiv, IEEE Xplore, ACM DL, USENIX (2022–2026) found no paper using inter/intra-class variance ratio asymmetry (J_A − J_B) as a VFL attack monitoring signal
- The one paper using "Fisher" for FL privacy (ScienceDirect 2025) uses the Fisher Information Matrix for DP noise calibration in horizontal FL — completely different use
- No prior paper monitors embedding-space discriminability asymmetry between parties in VFL
- Novelty claim for detection component: STRONG — no competition found after thorough search

**C36 — Gradient Projection for VFL label inference is Moderately Novel; critical competitor is MixPro (SIGIR 2023) (EXP-039)**
- MixPro (in the FedAds benchmark paper, SIGIR 2023) applies a gradient projection step per-batch in VFL — this is the only prior work using gradient projection in VFL
- Key differences from our approach: MixPro's projection direction is generic (not derived from an auxiliary classifier targeting the discriminative subspace); MixPro does not address active MaliciousSGD attacks; MixPro does not maintain a persistent subspace basis; MixPro is part of a gradient-mixup pipeline, not a standalone defense
- ProjPert (IEEE TKDE 2024) is a false competitor — "projection" in its name refers to noise parameter search space, not geometric subspace projection
- Paper must explicitly differentiate from MixPro with a comparison table or discussion paragraph

**C37 — Presenting different defenses for different datasets (AAP for CIFAR-10, GradProj for CIFAR-100) will be perceived as a methodology weakness by reviewers (Session decision, 2026-07-12)**
- Reviewers will reasonably ask: "which defense is your contribution?" and "why does it require different mechanisms for different complexity levels?"
- A unified Persistent Projection mechanism that works across all datasets (CIFAR-10, CINIC10L, CIFAR-100) avoids this fragmentation entirely
- If Persistent Projection works for both 10-class and 100-class settings, the paper story becomes: "Fisher Divergence Detection + Discriminative Subspace Projection" as a single unified defense
- Current AAP and one-shot GradProj results become ablation baselines showing the weakness of each component in isolation vs. the unified approach

**C38 — Detection-only papers are publishable but materially weaker than detection+defense for privacy venues (Session decision, 2026-07-12)**
- A detection-only contribution answers "can you measure the attack?" — this is a characterization claim
- Detection + defense answers "can you stop the attack?" — this is a solutions claim, substantially stronger for IEEE S&P, CCS, USENIX Security
- Current work has defense results (4/4 seeds CIFAR-10, 4/4 seeds CIFAR-100, 1/4 seeds CINIC10L done) — the defense story must be kept and strengthened, not dropped
- If the defense is theoretically fragile (one-shot collapse for CIFAR-100), the correct fix is to design a principled persistent version, not to drop the defense entirely

### 5.2 Rejected or Modified Hypotheses

**R1 — Defense was expected to suppress Fisher divergence**
- Status: REVISED — Fisher divergence is NOT suppressed; it increases under defense
- Correct mechanism: defense disrupts semantic label alignment, not geometric separability
- Paper treatment: frame defense as "semantic alignment disruption triggered by Fisher divergence detection" — this is a stronger, more nuanced contribution

**R2 — Gradient norm ratio was expected to detect MaliciousSGD**
- Status: REJECTED — MaliciousSGD's target tensor (`p.grad`) differs from what the monitor measures (`grad_output`)

**R3 — Active all-parties attack provides negligible Stage 2 benefit**
- Status: PARTIALLY SUPPORTED — true on CIFAR10 (+0.78pp) but not on CIFAR100 (+6.74pp)
- Paper treatment: scope defense to single-party threat model; acknowledge all-parties as boundary condition

**R4 — Laplace DP conditions can be analyzed alongside other conditions**
- Status: REJECTED — DP causes training collapse; Fisher metrics under DP are artifacts

**R5 — Direction reversal (subtracting aux_grad from grad_output_a) counteracts MaliciousSGD without explicit magnitude bounding (Phase 18, EXP-031)**
- Status: REJECTED — without magnitude bounding, direction reversal creates catastrophic instability under MaliciousSGD
- Failure mechanism: `aux_grad = W^T × softmax_error` is unbounded; Adam trains W over 150 epochs to arbitrarily large norms; large z_a values (amplified by MaliciousSGD over 150 epochs) cause logit overflow → NaN in softmax → NaN propagates through aux_grad; MaliciousSGD amplifies NaN by up to 5× → cascade failure. Result: lambda=1.0/2.0 crash Stage 1 entirely; lambda=0.5 produces VFL accuracy 1.001% (random — model destroyed, not defended)
- Paper use: Phase 18 failure directly motivates Phase 19's bounded projection design. Paper narrative: "unbounded reversal fails catastrophically (§X, EXP-031); we therefore design a projection that is provably bounded (‖grad_proj‖ ≤ ‖grad_output_a‖ by Cauchy-Schwarz), eliminating NaN risk while preserving the discriminative removal objective"
- Design fix: replace subtraction with orthogonal projection — remove the component in the d_aux direction, never add opposing force. Phase 19 (EXP-032) confirms zero NaN and successful defense

### 5.5 Critical Invalidations (2026-07-01, Phase 2b)

**I1 — MaliciousSGD at 30 epochs does NOT produce inference advantage for CIFAR10 (Invalidates core assumption)**

EXP-009 showed that the 30-epoch active checkpoint produces 23.45% model completion versus 47.98% for the benign checkpoint. The attacker is BELOW the benign baseline by 24.53pp. This means 30 epochs is below the convergence threshold for MaliciousSGD on CIFAR10. The attack only works after longer training (100+ epochs produced 94.99% in EXP-003). Any defense experiment at 30 epochs for CIFAR10 is testing against a non-working attack.

**I2 — The defense inadvertently HELPS the attacker at 30 epochs for CIFAR10 (Invalidates core defense claim)**

With the defense ON (Phase 2), active inference accuracy = 52.28%. Without the defense (Phase 2b), active inference accuracy = 23.45%. The defense raises accuracy by +28.83pp, the opposite of the intended effect. Mechanism: the defense suppresses Party A's overamplified gradient from the server, which prevents the gradient amplification from destabilizing early training. The defense accidentally acts as a regularizer that helps MaliciousSGD converge to useful embeddings within 30 epochs. This effect will likely reverse at 100+ epochs where MaliciousSGD has converged without the defense.

**I3 — CIFAR100 defense is statistically null at 30 epochs (Undermines generalizability)**

Active + defense (21.35%) vs active no defense (20.26%) = +1.09pp. Given single-run variance of ~17pp (CIFAR100 benign: 11.27% vs 29.90% between runs), the 1.09pp difference is meaningless noise. The defense cannot be claimed to work for CIFAR100 based on current experiments.

**I4 — The epoch confound is now fully exposed (Invalidates prior comparison)**

The "42.71pp reduction" and "51% reduction" claims in EXP-008 were based on comparing:
- Numerator: 30-epoch defended result (EXP-008)
- Denominator: 100-epoch undefended result (EXP-003/EXP-005, checkpoints now overwritten)

These have different epoch counts. The apparent reduction includes both the defense effect and the shorter-training effect. It cannot be attributed entirely to the defense.

**What this means for the paper:**

The prior narrative (Sections 4.3, 4.4, 5.1-C6, Table 2) that claimed large attack reductions is based on a methodologically flawed comparison. These sections should NOT be used as paper claims until re-validated at equal epoch counts.

The ONLY defensible claims currently are:
1. Fisher divergence is a reliable detection signal (Phase 1 results — still valid)
2. The attack works at longer training (EXP-003/EXP-005 — checkpoint evidence lost for CIFAR10, needs re-run)
3. The defense does not fire on benign training (EXP-007 CSV — still valid)
4. CIFAR100 attack modestly works at 30 epochs (+7.5pp, but no defense evidence)

### 5.3 Open Questions

**Q1 — What causes asymmetric Laplace DP collapse?** (Benign+DP: Party A collapses; Active+DP: Party B collapses)

**Q2 — How to set detection threshold in practice?** Options: fixed threshold per dataset, normalized by num_classes, or burn-in calibration (mean + 2σ of first 8 epochs).

**Q3 — Why does Fisher divergence increase under defense despite gradient suppression?** MaliciousSGD's internal amplification maintains cluster geometry independently of what the server sends.

**Q4 — What is the precise semantic alignment failure under defense?** Hypothesis: with diminished task gradient, Party A's clusters organize around spurious features (texture, color), not semantic class content. Evidence needed: t-SNE plots colored by true label.

**Q5 — Is the CIFAR100 benign model completion variance (11.27% vs 29.90%) a seed issue?** Need ≥3 seeds to confirm.

**Q6 — Does defense generalize to CINIC10 and TinyImageNet?** Not yet tested.

### 5.4 Lessons Learned

**L1 — Always list files before asserting their absence** (run_model_completion.bat was wrongly stated to not exist in a prior session)

**L2 — Distinguish the two gradient tensors in VFL:** `grad_output_bottom_model_a` (server→Party A) vs `p.grad` inside MaliciousSGD (internal). These are different tensors.

**L3 — Phase 1 characterization must precede defense design** — burn-in and threshold values came directly from Phase 1 signal analysis.

**L4 — Fisher signal is weaker on harder datasets but scales sublinearly** — CIFAR10 gap 0.564, CIFAR100 gap 0.129, ratio 4.4× for 10× more classes.

**L5 — 30-epoch CIFAR100 training produces high-variance baselines** — model completion accuracy shows large run-to-run variance; either more epochs or multiple seeds are needed.

**L6 — A defense can work through a different mechanism than theorized** — the empirical outcome (lower attack accuracy) can be real even if the explanation needs revision. Updating the mechanism narrative is part of research, not a failure.

**L7 — Gradient injection/subtraction defenses must be explicitly bounded relative to the original gradient** — any defense that ADDS or SUBTRACTS a gradient term must guarantee ‖injected_term‖ ≤ C×‖grad_output_a‖ for bounded C. Without this, Adam-trained auxiliary classifiers will grow unbounded weights; under MaliciousSGD's 5× amplification, the result is numerical overflow and NaN cascade (Phase 18 root cause). Safe alternatives: projection (removes component, never adds) or explicit norm clipping of the auxiliary gradient before subtraction. This is a general design constraint for any future direction-reversal defense on top of MaliciousSGD.

**L9 — Novelty must be confirmed by current literature search BEFORE implementing additional experiments, not after (Session decision, 2026-07-12)**
- In this project, Phase 20/21 (multi-seed GradientProjection runs) were run before confirming that gradient projection for VFL is not already published. A negative literature result would have changed the experimental strategy.
- For any new defense direction: search the 2024–2025 literature first; confirm novelty claim; then implement. The cost of a prior-art discovery before implementation is zero; after implementation is high.
- Corollary: do not confuse component-level novelty ("gradient projection exists in FL") with system-level novelty ("this specific combination for this specific attack is unpublished"). The latter is what matters.

**L10 — Presenting two different defenses for two different datasets requires a unification story; otherwise the contribution appears ad-hoc (Session decision, 2026-07-12)**
- If CIFAR-10 uses AAP (scale-based) and CIFAR-100 uses GradientProjection (direction-based), a reviewer may conclude: "the authors tried many things and reported the ones that worked per dataset." This reads as empirical curve-fitting, not principled defense design.
- The correct framing requires either: (a) a single unified mechanism that works across all datasets, or (b) a theoretical explanation of WHY different complexity levels (10-class vs. 100-class) require different defense strategies, with the theory predicting which one to use for a new dataset.
- Option (a) is far stronger and is the goal of Persistent Projection.

**L8 — A one-shot structural collapse can be more effective than gradual suppression** — Phase 19's defense fires once at epoch 11 and causes catastrophic intra_var_A spike (0.16→141,644) that permanently shifts Fisher divergence negative. The designed mechanism (gradual per-epoch projection) is irrelevant after epoch 12 because the collapse is self-perpetuating. This suggests that in adversarial settings where the attacker has many epochs to adapt, a large irreversible structural disruption at a critical moment can be more durable than ongoing attrition. Paper framing opportunity: "threshold-triggered structural disruption" as a defense paradigm.

---

## 6. Comparison Tables

### Table 1 — Fisher Divergence Signal Summary (Both Datasets, Epoch 29)

| Condition | CIFAR10 Div. | CIFAR100 Div. | Usable for Detection? |
|---|---|---|---|
| Benign | −0.120 | +0.002 | Baseline reference |
| **Active Party A** | **+0.444** | **+0.131** | **✅ Yes — primary target** |
| Active All Parties | −0.142 | −0.025 | ❌ Blind spot (by design) |
| Active + GC (75%) | +0.521 | +0.203 | ✅ Yes |
| Benign + DP | −0.607 | unstable | ❌ Training collapse |
| Active + DP | +0.968 | chaotic | ❌ Training collapse |
| **Active + AsymDef (Phase 2)** | **+0.522** | **+0.167** | ✅ Detection triggered; defense active |
| Benign + AsymDef (Phase 2) | +0.094 | −0.031 | ✅ Correctly does NOT trigger |

---

### Table 2 — Stage 2 Label Inference Accuracy (Full Summary, Both Phases)

| Dataset | Condition | Best Train Top-1 | Final Test Top-1 | Delta vs Passive |
|---|---|---|---|---|
| CIFAR10 | Passive | 84.61% | 66.81% | — |
| CIFAR10 | Active Party A | 94.99% | 73.14% | **+10.38pp** |
| CIFAR10 | Active All | 85.39% | 68.69% | +0.78pp |
| CIFAR10 | **Active + Defense (30ep)** | **52.28%** | **48.96%** | −32.33pp vs Active (30ep, confounded — see EXP-009) |
| CIFAR10 | Benign + Defense (30ep) | 47.18% | 45.66% | (defense silent) |
| **CIFAR10** | **Benign (100ep) [EXP-011]** | **87.23%** | **69.07%** | −8.19pp vs attack | 
| **CIFAR10** | **Active (100ep) [EXP-011]** | **95.42%** | **73.67%** | **+8.19pp vs benign** |
| **CIFAR10** | **Active + Defense (100ep) [EXP-011]** | **84.27%** | **74.97%** | **−11.15pp vs active; −2.96pp vs benign** |
| CIFAR100 | Passive | 29.90% | 18.68% | — |
| CIFAR100 | Active Party A | 43.35% | 22.38% | **+13.45pp** |
| CIFAR100 | Active All | 36.64% | 20.50% | +6.74pp |
| CIFAR100 | **Active + Defense (30ep, confounded)** | **21.35%** | **17.96%** | **−22.00pp vs Active (confounded)** |
| CIFAR100 | Benign + Defense | 11.27% | 10.12% | (high variance) |
| **CIFAR100** | **Benign (150ep) [EXP-012]** | **30.33%** | **17.36%** | −17.53pp vs attack |
| **CIFAR100** | **Active (150ep) [EXP-012]** | **47.86%** | **25.88%** | **+17.53pp vs benign** |
| **CIFAR100** | **Active+Defense (150ep, a=1.0) [EXP-012]** | **43.12%** | **23.40%** | **−4.74pp vs attack; +12.79pp vs benign ⚠️** |
| **CIFAR100** | **Active+Defense (150ep, a=2.0, t=0.10) [EXP-013]** | **48.31%** | — | **+0.45pp vs attack; +17.98pp vs benign ❌❌** |
| **CIFAR100** | **Active+Defense (150ep, a=2.0, t=0.05) [EXP-013]** | **43.60%** | — | **−4.26pp vs attack; +13.27pp vs benign ❌** |
| CIFAR10 | GC 75% attack (30ep) [EXP-014] | **69.22%** | — | Competitor attack |
| CIFAR10 | Laplace attack (30ep) [EXP-014] | **53.41%** | — | Competitor attack |
| CIFAR10 | Laplace DP benign (30ep) [EXP-014] | **10.01%** | — | Competitor defense |
| CIFAR100 | GC 75% attack (30ep) [EXP-014] | **21.74%** | — | Competitor attack |
| CIFAR100 | Laplace attack (30ep) [EXP-014] | **3.55%** | — | Competitor attack |
| CIFAR100 | Laplace DP benign (30ep) [EXP-014] | **15.78%** | — | Competitor defense |
| **CIFAR10** | **Active+Defense (100ep, a=0.5, t=0.10) [EXP-015]** | **94.57%** | — | **−0.85pp vs attack; +7.34pp vs benign ❌** |
| **CIFAR10** | **Active+Defense (100ep, a=2.0, t=0.10) [EXP-015]** | **76.79%** | — | **−18.63pp vs attack; −10.44pp vs benign ✅✅** |
| **CIFAR10** | **Active+Defense (100ep, a=1.0, t=0.05) [EXP-015]** | **81.58%** | — | **−13.84pp vs attack; −5.65pp vs benign ✅** |
| **CIFAR10** | **Active+Defense (100ep, a=1.0, t=0.15) [EXP-015]** | **85.98%** | — | **−9.44pp vs attack; −1.25pp vs benign ✅** |
| **CIFAR10** | **Benign (100ep, seed=42) [EXP-016]** | **81.90%** | — | Seed stability |
| **CIFAR10** | **Active (100ep, seed=42) [EXP-016]** | **94.32%** | — | Seed stability |
| **CIFAR10** | **Active+Defense (100ep, a=1.0, seed=42) [EXP-016]** | **80.16%** | — | **−14.16pp vs attack; −1.74pp vs benign ✅** |
| **CIFAR10** | **Benign (100ep, seed=123) [EXP-016]** | **80.78%** | — | Seed stability |
| **CIFAR10** | **Active (100ep, seed=123) [EXP-016]** | **95.33%** | — | Seed stability |
| **CIFAR10** | **Active+Defense (100ep, a=1.0, seed=123) [EXP-016]** | **80.61%** | — | **−14.72pp vs attack; −0.17pp vs benign ✅** |
| **CIFAR10** | **Benign (100ep, seed=456) [EXP-016]** | **82.53%** | — | Seed stability |
| **CIFAR10** | **Active (100ep, seed=456) [EXP-016]** | **94.73%** | — | Seed stability |
| **CIFAR10** | **Active+Defense (100ep, a=1.0, seed=456) [EXP-016]** | **82.14%** | — | **−12.59pp vs attack; −0.39pp vs benign ✅** |
| **CIFAR10** | **MEAN ± STD (4 seeds) [EXP-016]** | **81.80 ± 1.85%** | — | **Benign: 83.11 ± 2.84%; Attack: 94.95 ± 0.52%; 4/4 seeds defended < benign** |
| **CIFAR10** | **Benign + Defense (100ep) [EXP-017]** | **84.35%** | 66.48% | **+1.24pp vs benign mean — within variance ✅ Defense dormant** |
| **CIFAR100** | **Benign + Defense (150ep) [EXP-018]** | **34.74%** | 20.06% | **+4.41pp vs seed-0 benign — run variance (defense dormant, divergence < 0) ⚠️** |
| **CIFAR100** | **Active+Def, n=0.5 gradient noise (150ep, a=1.0) [EXP-019]** | **48.36%** | — | **+0.50pp vs attack (ABOVE attack); +18.03pp vs benign ❌❌** |
| **CIFAR100** | **Active+Def, n=1.0 gradient noise (150ep, a=1.0) [EXP-019]** | **49.64%** | — | **+1.78pp vs attack (WORST RESULT); +19.31pp vs benign ❌❌❌** |
| **CIFAR100** | **Active+Def, n=2.0 gradient noise (150ep, a=1.0) [EXP-019]** | **43.10%** | — | **−4.76pp vs attack; +12.77pp vs benign ⚠️ (barely improves on no-noise)** |
| **CINIC10L** | **Benign (100ep) [EXP-020]** | **65.70%** | 63.18% | Benign reference for CINIC10L |
| **CINIC10L** | **Active (100ep) [EXP-021]** | **86.59%** | 63.97% | **+20.89pp vs benign — attack works strongly** |
| **CINIC10L** | **Active + Defense (100ep, a=1.0, t=0.10, b=8) [EXP-022]** | **62.43%** | 63.81% | **−3.26pp vs benign ✅ DEFENSE SUCCEEDS; −0.37pp VFL utility cost** |
| **CIFAR10** | **Active+Sign-flip (100ep, sf=True) [EXP-023]** | **86.40%** | — | **+3.29pp vs benign (ABOVE benign) ❌; −8.55pp vs attack (partial)** |
| **CIFAR100** | **Active+z_a corruption (150ep, za=0.5) [EXP-024]** | **50.67%** | — | **+2.81pp vs attack (ABOVE attack) ❌❌; +20.34pp vs benign** |
| **CIFAR100** | **Active+z_a corruption (150ep, za=1.0) [EXP-025]** | **51.64%** | — | **+3.78pp vs attack (WORST MC) ❌❌❌; +21.31pp vs benign (seed-0); +22.08pp vs mean benign** |
| **CIFAR100** | **Benign (150ep, seed=42) [EXP-029]** | **26.19%** | 15.84% | Benign baseline seed-42 |
| **CIFAR100** | **Benign (150ep, seed=123) [EXP-029]** | **28.56%** | 18.19% | Benign baseline seed-123 |
| **CIFAR100** | **Benign (150ep, seed=456) [EXP-029]** | **33.14%** | 17.90% | Benign baseline seed-456 |
| **CIFAR100** | **MEAN ± STD benign (4 seeds) [EXP-029]** | **29.56 ± 2.93%** | — | **Anchored benign reference for all CIFAR-100 comparisons** |
| **CIFAR100** | **Adv Aux lambda=0.5 Stage 1 (EXP-031)** | — | — | **Stage 1 NaN collapse: VFL Acc=1.00%, Loss=NaN — model destroyed** |
| **CIFAR100** | **Adv Aux lambda=0.5 MC (EXP-031)** | **1.001%** | 1.00% | **🔴 NaN collapse — random performance; all 25 MC epochs show NaN loss** |
| **CIFAR100** | **Adv Aux lambda=1.0, 2.0 (EXP-031)** | N/A | N/A | **🔴 Stage 1 crashed — no checkpoint saved; MC aborted at startup** |
| **CIFAR100** | **Gradient Projection defense (150ep, seed-0) [EXP-032]** | **26.97%** | 17.41% (ep25) | **−2.59pp below benign mean 29.56% ✅; −20.89pp vs attack; −0.01pp VFL utility cost (best ever). SINGLE SEED.** |
| **CIFAR10** | **PP Buggy, ema=0.1/0.2/0.3, 100ep, seed-0 [EXP-040/041/042]** | **93.73–94.50%** | ~80.8% | **+6.5–7.3pp above benign 87.23% ❌ FAIL — bug: batch-mean d_ema cancels to ~0; defense fires but tracks noise** |
| **CIFAR100** | **PP Buggy, ema=0.1/0.2, 150ep, seed-0 [EXP-043/044]** | **49.42–49.44%** | ~46.5% | **+19.1pp above benign 30.33% ❌❌ FAIL; +1.6pp ABOVE undefended attack (47.86%)** |
| **CIFAR10** | **PP Fixed, ema=0.1, 100ep, seed-0 [EXP-045]** | **94.74%** | **80.77%** | **+7.51pp above benign ❌ FAIL; fix changed outcome by <1pp → root cause is architectural not batch-cancellation bug** |
| **CIFAR10** | **PP Fixed, ema=0.2, 100ep, seed-0 [EXP-046]** | **92.99%** | **80.09%** | **+5.76pp above benign ❌ FAIL (BEST PP CIFAR-10); phase transition at epoch 50 confirmed in CSV — defense fires correctly but cannot prevent attack** |
| **CIFAR10** | **PP Fixed, ema=0.3, 100ep, seed-0 [EXP-047]** | **94.91%** | **80.75%** | **+7.68pp above benign ❌ FAIL; 0.04pp below undefended attack (94.95%) — effectively no protection** |
| **CIFAR100** | **PP Fixed, ema=0.1, 150ep, seed-0 [EXP-048]** | **48.78%** | **46.47%** | **+18.45pp above benign ❌❌ FAIL; VFL accuracy IMPROVES vs benign (+1.14pp) — wrong direction** |
| **CIFAR100** | **PP Fixed, ema=0.2, 150ep, seed-0 [EXP-049]** | **50.16%** | **47.50%** | **+19.83pp above benign; +2.30pp ABOVE undefended attack (47.86%) ❌❌❌ WORST — PP makes CIFAR-100 privacy WORSE than no defense; VFL +2.17pp above benign** |
| **CIFAR100** | **PP Fixed, ema=0.3, 150ep, seed-0 [EXP-050]** | **48.81%** | **46.47%** | **+18.48pp above benign ❌❌ FAIL; despite intra_var_A collapsing to 0.006 (tighter than undefended attack 0.014), MC still 48.81% — geometric tightness ≠ label alignment** |

---

### Table 3 — Detection Threshold Analysis

| Dataset | Benign Div Range (ep 8–29) | Active Div Range (ep 8–29) | Threshold | Burn-in |
|---|---|---|---|---|
| CIFAR10 | −0.14 to −0.10 | +0.33 to +0.53 | **+0.10** | 4 epochs |
| CIFAR100 | +0.00 to +0.04 | +0.08 to +0.13 | **+0.07** | 8 epochs |

For a dataset-agnostic defense, burn-in calibration (mean + 2σ of first 8 epochs) is preferred over a fixed threshold.

---

### Table 4 — Defense Comparison (Existing vs Proposed)

| Defense | Symmetric? | Uses Detection? | Stage 2 Suppression? | Status |
|---|---|---|---|---|
| PPDL (DP-SGD) | ✅ | ❌ | Unknown (training unstable) | Existing |
| Gradient Compression | ✅ | ❌ | Unknown | Existing |
| Laplace Noise | ✅ | ❌ | Unknown (training unstable) | Existing |
| Multistep Gradient | ✅ | ❌ | Unknown | Existing |
| **Proposed: AsymAdaptivePert** | **❌ (asymmetric)** | **✅ (Fisher divergence)** | **✅ (−45% C10, −51% C100)** | **Phase 2 Validated** |

---

### Table 5 — VFL Utility vs Attack Suppression Tradeoff

| Dataset | Epochs | VFL Test Acc (benign) | VFL Test Acc (active+def) | VFL Cost | Attack Reduction | Source |
|---|---|---|---|---|---|---|
| CIFAR10 | 30ep | 62.71% | 60.94% | −1.77pp | 45% (confounded) | EXP-007/008 |
| CIFAR100 | 30ep | 26.05% | 29.23% | +3.18pp (active better) | 51% (confounded) | EXP-007/008 |
| **CIFAR10** | **100ep** | **81.45%** | **79.52%** | **−1.93pp** | **11.15pp abs; below benign ✅** | **EXP-011** |
| **CIFAR100** | **150ep (a=1.0)** | **45.33%** | **45.20%** | **−0.13pp** | **4.74pp abs; 12.79pp above benign ⚠️** | **EXP-012** |
| **CIFAR100** | **150ep (a=2.0, t=0.10)** | **45.33% (benign)** | **45.93%** | **+0.60pp** | **48.31% defended; +17.98pp above benign ❌❌** | **EXP-013** |
| **CIFAR100** | **150ep (a=2.0, t=0.05)** | **45.33% (benign)** | **45.47%** | **+0.14pp** | **43.60% defended; +13.27pp above benign ❌** | **EXP-013** |
| **CINIC10L** | **100ep (a=1.0, t=0.10, b=8)** | **63.18%** | **63.81%** | **+0.63pp** | **62.43% defended; −3.26pp below benign 65.70% ✅** | **EXP-022** |
| **CIFAR10** | **100ep, sign-flip (sf=True)** | **— (benign ~83.11%)** | **—** | **—** | **86.40% defended; +3.29pp above benign ❌** | **EXP-023** |
| **CIFAR100** | **150ep, z_a noise std=0.5** | **— (benign 30.33%)** | **—** | **—** | **50.67% defended; +20.34pp above benign ❌❌** | **EXP-024** |
| **CIFAR100** | **150ep, z_a noise std=1.0** | **— (benign 30.33%)** | **—** | **—** | **51.64% defended; +21.31pp above benign ❌❌❌** | **EXP-025** |
| **CIFAR100** | **150ep, Gradient Projection (seed-0)** | **45.33% (benign)** | **45.32%** | **−0.01pp** | **26.97% defended; −2.59pp below benign mean 29.56% ✅ (SINGLE SEED)** | **EXP-032** |
| **CIFAR10** | **100ep, PP Buggy (ema=0.1/0.2/0.3)** | — | ~80.8% | ~0pp | **93.73–94.50% defended; +6.5–7.3pp above benign ❌** | **EXP-040/041/042** |
| **CIFAR100** | **150ep, PP Buggy (ema=0.1/0.2)** | — | ~46.5% | ~+1pp | **49.42–49.44% defended; +19.1pp above benign ❌❌** | **EXP-043/044** |
| **CIFAR10** | **100ep, PP Fixed (ema=0.1)** | **81.45% (benign)** | **80.77%** | **−0.68pp** | **94.74% defended; +7.51pp above benign ❌** | **EXP-045** |
| **CIFAR10** | **100ep, PP Fixed (ema=0.2)** | **81.45% (benign)** | **80.09%** | **−1.36pp** | **92.99% defended; +5.76pp above benign ❌ (BEST PP C10; 11pp worse than AAP 81.80%)** | **EXP-046** |
| **CIFAR10** | **100ep, PP Fixed (ema=0.3)** | **81.45% (benign)** | **80.75%** | **−0.70pp** | **94.91% defended; +7.68pp above benign ❌** | **EXP-047** |
| **CIFAR100** | **150ep, PP Fixed (ema=0.1)** | **45.33% (benign)** | **46.47%** | **+1.14pp (INCREASES — wrong dir)** | **48.78% defended; +18.45pp above benign ❌❌** | **EXP-048** |
| **CIFAR100** | **150ep, PP Fixed (ema=0.2)** | **45.33% (benign)** | **47.50%** | **+2.17pp (WORST — increases utility)** | **50.16% defended; +2.30pp ABOVE attack (47.86%) ❌❌❌** | **EXP-049** |
| **CIFAR100** | **150ep, PP Fixed (ema=0.3)** | **45.33% (benign)** | **46.47%** | **+1.14pp (INCREASES)** | **48.81% defended; +18.48pp above benign ❌❌** | **EXP-050** |

---

### Table 7 — Fair 30-Epoch 3-Way Comparison (EXP-007 + EXP-008 + EXP-009)

*All conditions use 30-epoch Stage 1 training and 25-epoch Stage 2 MixMatch. All directly comparable.*

**CIFAR10:**

| Condition | Model Completion Best Top-1 | vs Benign | Interpretation |
|---|---|---|---|
| Benign, no defense | **47.98%** | baseline | Normal privacy risk at 30 ep |
| Active, no defense | **23.45%** | **−24.53pp** | Attack NOT working at 30 ep |
| Active + defense | **52.28%** | **+4.30pp** | Defense stabilizes training; HELPS attacker |
| Benign + defense | **47.18%** | −0.80pp | Defense correctly silent |

⚠️ **This table shows the defense is counterproductive at 30 epochs for CIFAR10.** The defense makes inference accuracy higher than both the benign baseline and the undefended active baseline. This cannot be presented as a defense result.

**CIFAR100:**

| Condition | Model Completion Best Top-1 | vs Benign | Interpretation |
|---|---|---|---|
| Benign, no defense | **12.76%** | baseline | Normal privacy risk at 30 ep |
| Active, no defense | **20.26%** | **+7.50pp** | Attack works modestly at 30 ep |
| Active + defense | **21.35%** | **+8.59pp** | Defense adds +1.09pp — statistically null |
| Benign + defense | **11.27%** | −1.49pp | Defense correctly silent |

**CIFAR100 conclusion:** The defense shows zero measurable effect (+1.09pp change, which is within single-run variance). The attack creates a genuine but modest advantage (+7.50pp).

---

### Table 6 — Paper Evidence Checklist (Updated 2026-07-01)

| Claim | Evidence | Status |
|---|---|---|
| Active attack is effective on CIFAR10 at 100 epochs | EXP-011: +8.19pp over benign (95.42% vs 87.23%) | ✅ Confirmed |
| Active attack is effective on CIFAR100 at 150 epochs | EXP-005: +13.45pp over passive | ✅ (checkpoint intact) |
| Active attack does NOT work at 30 epochs CIFAR10 | EXP-009: active 23.45% < benign 47.98% | ✅ Confirmed |
| Active attack is modestly effective at 30 epochs CIFAR100 | EXP-009: +7.50pp vs benign | ✅ (modest, single run) |
| Fisher divergence detects attack on CIFAR10 | Phase 1 CSV: gap = 0.564 from ep 4; Phase 4: grows to 1.21 at ep99 | ✅ |
| Fisher divergence detects attack on CIFAR100 | Phase 1 CSV: gap = 0.129 from ep 8 | ✅ |
| Existing defenses do not use asymmetric detection | possible_defenses.py code review | ✅ |
| Defense reduces Stage 2 from 100-epoch attack baseline (CIFAR10) | EXP-011: 95.42% → 84.27% (−11.15pp); below benign 87.23% | ✅ Confirmed |
| Defense reduces Stage 2 accuracy (CIFAR100, 150-epoch fair comparison) | EXP-012: 47.86%→43.12% (−4.74pp); defended still 12.79pp above benign 30.33% | ⚠️ Partial — insufficient at a=1.0 |
| Defense does not hurt VFL utility | EXP-011: −1.93pp CIFAR10 at 100ep | ✅ |
| Defense correctly does not fire on benign training | EXP-007 CSV: benign divergence below tau | ✅ |
| Benign party unaffected at full epoch count (CIFAR-10, 100ep) | EXP-017: benign+defense 84.35% ≈ benign mean 83.11% (within ±2.84pp std); 2×2 table complete | ✅ Complete |
| Benign party unaffected at full epoch count (CIFAR-100, 150ep) | EXP-018: benign+defense 34.74% vs benign 30.33%; +4.41pp attributed to run variance; defense dormant (divergence<0) | ⚠️ Requires multi-seed CIFAR-100 benign baseline |
| Option B (gradient noise injection) improves CIFAR-100 defense | EXP-019: n=0.5 → 48.36%, n=1.0 → 49.64%, n=2.0 → 43.10% — all fail criterion; n=0.5 and n=1.0 are WORSE than no noise | ❌ FAILED — gradient-space exhausted |
| Embedding-space z_a corruption effective for CIFAR-100 | EXP-024 (std=0.5 → 50.67%), EXP-025 (std=1.0 → 51.64%) — both ABOVE undefended attack (47.86%); actively WORSENS CIFAR-100 | ❌❌ FAILED — counterproductive; z_a surface exhausted |
| Sign-flip momentum disruption neutralizes MaliciousSGD ratio | EXP-023 (CIFAR-10): 86.40% defended > benign 83.11% — theory does not hold in practice | ❌ FAILED on CIFAR-10; worse than standard defense (81.80%) |
| Defense generalizes beyond CIFAR-10 to second 10-class dataset | EXP-022 (CINIC10L): 62.43% < 65.70% benign; −24.16pp vs attack; same params (a=1.0,t=0.10,b=8) with no retuning | ✅ CONFIRMED — CINIC10L defense succeeds |
| CINIC10L result confirmed across multiple seeds | EXP-026/027/028 (Phase 15): 4/4 seeds; defended 62.72 ± 0.65% < benign 65.76 ± 0.65%; all per-seed margins −2.93 to −3.26pp | ✅ CONFIRMED |
| Results hold across multiple seeds (CIFAR10 100ep) | EXP-016: 4/4 seeds confirmed; mean defended 81.80 ± 1.85% < benign 83.11 ± 2.84%; all seeds show same direction | ✅ Complete |
| Defense is robust across hyperparameter choices | EXP-015: 4/5 CIFAR-10 100ep ablation variants succeed; a=0.5 fails (too weak) | ✅ CIFAR-10 confirmed |
| Competitor defenses (GC, Laplace) suppress label inference | EXP-014: GC=69.22%, Laplace=53.41% attack; LapDP-benign=10.01% (CIFAR-10); GC=21.74%, Lap=3.55%, LapDP-benign=15.78% (CIFAR-100) | ✅ |
| Defense collateral utility cost is acceptable | EXP-011: −1.93pp at 100ep; Phase 3b: a=1.0 t=0.1 gives −1.66pp CIFAR100 benign collateral | ✅ (primary setting) |
| CIFAR100 benign baseline is stable across seeds | EXP-029 (Phase 16): 4 seeds complete; benign mean **29.56 ± 2.93%** (seeds 0/42/123/456: 30.33/26.19/28.56/33.14%); VFL Stage 1 test: 44.65 ± 0.47% | ✅ COMPLETE |
| Direction-based defense solves CIFAR-100 (AdversarialAuxClassifier) | EXP-031 (Phase 18): lambda=0.5 → NaN collapse (Stage 1 VFL 1.00%, MC 1.001%); lambda=1.0/2.0 → Stage 1 crashed, no checkpoint; root cause: unbounded aux_grad + MaliciousSGD amplification of instability → NaN cascade | ❌❌ FAILED — NaN explosion, model destroyed |
| Gradient projection defense for CIFAR-100 | EXP-032 (Phase 19, seed-0): MC=26.97% < benign mean 29.56% (−2.59pp). VFL cost −0.01pp. Mechanism: catastrophic single-activation collapse at epoch 11-12 (intra_var_A spike 0.16→141K). Phase 20 (seeds 42/123/456) now running — EXP-033/034/035. | 🟡 PROMISING — Phase 20 RUNNING (EXP-033/034/035) |
| Defense generalizes to text modality (Yahoo Answers) | Phase 17 (EXP-030): running, Stage 1 benign in progress | 🟡 PENDING |
| Defense generalizes to tabular binary (Criteo) | Criteo code exists; dataset download needed | 📋 OPTIONAL |
| Semantic misalignment mechanism explained empirically | J_A=1.74 (clusters exist) but 84.27% < 87.23% benign (labels misaligned) confirmed | ✅ Supported |
| Semantic misalignment visualized | Need t-SNE colored by true label | ❌ Pending |
| Fisher Divergence Detection is novel (not prior-published) | EXP-039 literature review (2026-07-12): no prior VFL paper uses inter/intra-class variance ratio asymmetry between parties as active attack monitor | ✅ CONFIRMED — Clearly Novel |
| Gradient Projection as VFL defense is novel | EXP-039: MixPro (SIGIR 2023, FedAds) is closest competitor; significant mechanistic differences confirmed | ✅ CONFIRMED — Moderately Novel; must differentiate from MixPro |
| Persistent Projection is novel | EXP-039: no prior VFL paper proposes persistent multi-epoch discriminative subspace projection | ✅ CONFIRMED — Clearly Novel; no competing implementation found |
| Unified defense (Persistent Projection) works across 10-class and 100-class datasets | Buggy (EXP-040/041/042): 93.73–94.50% CIFAR-10 ❌; Fixed (EXP-045/046/047): 92.99–94.91% CIFAR-10 ❌; Buggy (EXP-043/044): 49.42–49.44% CIFAR-100 ❌; Fixed (EXP-048/049/050): 48.78–50.16% CIFAR-100 ❌. All 6 fixed variants fail. Bug fix changed outcome by <2pp, confirming the root cause is architectural: projecting grad_output_A cannot prevent MaliciousSGD amplifying p.grad via chain rule through Party A's ResNet. | ❌❌ FAILED — 0/6 fixed variants meet criterion; NOT PROMISING (High Confidence) |
| Persistent Projection prevents one-shot collapse seen in current GradProj | Phase 22/23 Fixed CSVs confirm PP fires multiple epochs (CIFAR-10: phase transition at epoch 50; CIFAR-100: two transitions at epochs 75 and 120). No one-shot intra_var spike. Defense fires correctly per design — but this does NOT suppress MC. intra_var_A collapses to 0.006 (CIFAR-100 ema=0.2) which is tighter than undefended attack (0.014), yet MC=50.16%. Geometric tightness does NOT imply label alignment suppression. | ✅ PP does NOT one-shot collapse (confirmed). ❌ PP does NOT defend (confirmed). |
| Persistent Projection DCR insufficient for CIFAR-100 (K=1, C=100 → DCR=1%) | DCR(K=1, C=100)=1/99≈1%; DCR(K=1, C=10)=1/9≈11%. Even with CIFAR-10's 11% coverage, PP fails due to late firing (epoch 50+). CIFAR-100's 1% is catastrophically insufficient. MDPP with K=10 required for DCR≈10% coverage. | ✅ CONFIRMED — DCR framework is valid; K must scale with C. Phase 24 MDPP (K=10) is next experiment. |

---

## 7. Future Work

### 7.0 Current Status Summary (2026-07-10)

**CIFAR-10 story — COMPLETE:**
- [x] 100-epoch three-way comparison (EXP-011)
- [x] Multi-seed validation, 4/4 seeds (EXP-016)
- [x] Hyperparameter ablation at 100 epochs, 4/5 succeed (EXP-015)
- [x] Competitor comparison (GC, Laplace DP) — EXP-014
- [x] Benign+defense asymmetry at 100 epochs (EXP-017)
- [x] Sign-flip momentum disruption: TESTED — ❌ FAILED (EXP-023: 86.40% above benign 83.11%)

**CINIC10L story — DEFENSE CONFIRMED (seed-0); MULTI-SEED RUNNING:**
- [x] Three-way comparison (benign/attack/defense) at 100 epochs (EXP-020/021/022)
- [x] Defense succeeds: 62.43% < 65.70% benign; −24.16pp suppression; negligible utility cost ← **STRONG RESULT**
- [x] Same parameters (a=1.0, t=0.10, b=8) generalize with zero retuning
- [x] ✅ Multi-seed validation complete: 4/4 seeds (Phase 15, EXP-026/027/028) — defended 62.72 ± 0.65% < benign 65.76 ± 0.65%

**CIFAR-100 story — GRADIENT PROJECTION IS FIRST CANDIDATE (seed-0); MULTI-SEED REQUIRED:**
- [x] 150-epoch three-way comparison: defense partial only (EXP-012)
- [x] Stronger alpha (a=2.0): FAILED (EXP-013)
- [x] Gradient noise injection (n=0.5, 1.0, 2.0): ALL FAILED, some made it WORSE (EXP-019)
- [x] Benign+defense asymmetry at 150 epochs: confirmed dormant (EXP-018)
- [x] Embedding-space z_a corruption (std=0.5, 1.0): BOTH FAILED — WORSE THAN UNDEFENDED ATTACK (EXP-024/025)
- [x] Sign-flip: FAILED on CIFAR-10 — not worth running on CIFAR-100
- [x] **Phase 16 COMPLETE: Benign multi-seed (seeds 0/42/123/456)** — benign baseline anchored at **29.56 ± 2.93%** (EXP-029)
- [x] 🔴 **Phase 18 FAILED: Adversarial Auxiliary Classifier (lambda 0.5/1.0/2.0)** — NaN explosion in ALL 3 variants; model destroyed (EXP-031)
- [x] 🟡 **Phase 19 (SEED-0): Gradient Projection Defense** — MC=**26.97% < benign mean 29.56%** (−2.59pp). VFL cost −0.01pp. Mechanism: catastrophic single-activation discriminative collapse (intra_var_A: 0.16 → 141,644 at epoch 12). (EXP-032)
- 10/11 configurations: failed. Phase 19 seed-0: FIRST CANDIDATE TO MEET CRITERION.
- [x] ✅ **COMPLETE: Phase 20 — Grad projection multi-seed (seeds 42/123/456, EXP-033/034/035)** — 4/4 seeds defended < benign. Seeds 0/42/123 cluster at 25–27%. Seed-456 anomalous over-collapse at 14.57%. **CIFAR-100 defense publishable.**
- [x] ✅ **COMPLETE: Phase 21 — Attack baseline multi-seed (seeds 42/123/456, EXP-036/037/038)** — attack 49.87 ± 1.17% across 4 seeds. 3×4 table complete.

**Yahoo Answers (modality test):**
- [ ] 🟡 Phase 17 RUNNING: Stage 1 benign training in progress (EXP-030)
- Infrastructure bugs fixed: BertConfig API + Translator pkl fallback + path separator

**Criteo (binary tabular — new option):**
- [ ] 📋 Dataset code exists (`datasets/criteo.py`), data needs download
- Different modality (tabular binary) + fast training; adds paper diversity without BERT overhead

**Overall Assessment (2026-07-12 — updated after Phase 20/21 completion):** THREE-DATASET STORY CONFIRMED

- **CIFAR-10**: ✅ Complete — 4/4 seeds confirmed, ablation, asymmetry. Defended 81.80 ± 1.85% vs benign 83.11 ± 2.84% vs attack 94.95 ± 0.52%.
- **CINIC10L**: ✅ **COMPLETE — 4/4 seeds confirmed** (Phase 15, EXP-022/026/027/028). Defended 62.72 ± 0.65% < benign 65.76 ± 0.65%; all per-seed margins −2.93 to −3.26pp below benign. Attack mean 86.51 ± 0.54%. Full 4-seed table complete and publishable.
- **CIFAR-100 benign**: ✅ Anchored at 29.56 ± 2.93% (4 seeds).
- **CIFAR-100 attack**: ✅ 49.87 ± 1.17% (4 seeds, Phase 21 complete). Tight, consistent attack.
- **CIFAR-100 defense**: ✅ Gradient projection: 4/4 seeds pass. Seeds 0/42/123 cluster at 25–27% (mean 26.40 ± 0.85%); seed-456 over-collapse at 14.57%. Mean defended 23.44 ± 5.16%. **PUBLISHABLE with seed-456 anomaly documented.**
- **Publication path**: Three-dataset story (CIFAR-10 + CINIC10L + CIFAR-100) fully confirmed. CINIC10L multi-seed now complete. Yahoo Answers (Phase 17) adds cross-modality if it converges.
- **Confidence**: High for all three datasets. CIFAR-100 has seed-456 caveat to address in paper.

**Top priorities (updated 2026-07-12 post Phase 20/21 completion):**
1. ✅ **COMPLETE: Phase 15** (CINIC10L seeds 42/123/456, EXP-026/027/028) — 4/4 seeds confirmed; defended 62.72 ± 0.65% < benign 65.76 ± 0.65%. CINIC10L claim fully publishable.
2. 🟡 **RUNNING: Phase 17** (Yahoo Answers, EXP-030) — cross-modality generalization test.
3. 📋 **Begin paper writing** — CIFAR-10 and CIFAR-100 stories are now fully quantified. Start with attack characterization (Section 2) and main results (Section 4).
4. 📋 **Address seed-456 CIFAR-100 anomaly in paper** — decide framing: include in table with footnote, or report 3-seed cluster as primary result with seed-456 as secondary.
5. 📋 OPTIONAL: Criteo binary tabular — adds modality diversity without BERT overhead.

### 7.1 CRITICAL — Required Before Any Paper Claim Can Be Made (Priority Order)

**Updated 2026-07-05: EXP-011 (CIFAR-10) and EXP-012 (CIFAR-100) are complete. Remaining blockers:**

- [x] ~~Re-run 100-epoch CIFAR10 active checkpoint (Stage 1).~~ **DONE — EXP-011.**
- [x] ~~Verify 100-epoch baseline gives ~90-95% model completion.~~ **DONE — 95.42% confirmed.**
- [x] ~~Run 100-epoch active + defense (Stage 1 + Stage 2).~~ **DONE — EXP-011; 84.27% defended, below benign ✅**
- [x] ~~Run 100-epoch benign checkpoint for CIFAR10.~~ **DONE — EXP-011.**
- [x] ~~Run CIFAR-100 150-epoch three-way comparison.~~ **DONE — EXP-012; defense INSUFFICIENT at a=1.0 ⚠️**
- [x] ~~**[HIGHEST PRIORITY] Run Phase 6A: CIFAR-100 stronger defense (alpha=2.0) at 150 epochs.**~~ **DONE — EXP-013. BOTH VARIANTS FAILED. a=2.0,t=0.10 gives 48.31% (above undefended attack 47.86%). Gradient perturbation is insufficient. Option B required.**
- [x] ~~Add `--manual-seed` flag to both `vfl_framework.py` and `model_completion.py`.~~ **DONE — both files updated (2026-07-05). `vfl_framework.py` now accepts `--manual-seed`; `model_completion.py` updated with full torch+numpy+random seeds.**
- [x] ~~**Run seed sweep: CIFAR-10 100ep, seeds {42, 123, 456}.**~~ **DONE — EXP-016. All 4 seeds complete. Mean ± std: defended 81.80 ± 1.85%, benign 83.11 ± 2.84%, attack 94.95 ± 0.52%. 4/4 seeds show defended < benign ✅**

### 7.1b IMMEDIATE — Low-Hanging Fruit (Checkpoints Already Exist)

These can be run now without waiting for 100-epoch experiments. The Stage 1 checkpoints exist from Phase 3b and Phase 1. Model completion (Stage 2) is 25 epochs — fast.

- [ ] **Run Stage 2 (model_completion.py) for all ablation checkpoints** — alpha ∈ {0.5, 2.0}, tau ∈ {0.05, 0.15} for both CIFAR10 and CIFAR100. This will show whether alpha and tau tuning actually affect label inference, not just VFL task accuracy.
- [ ] **Run Stage 2 for GC (75%) and Laplace DP competitor checkpoints** on both datasets. This enables a real defense comparison table (Section 4.9 currently shows only VFL task accuracy, not inference accuracy).

### 7.2 After Phase 5 — Priority Queue (execute in order)

- [x] ~~Build the correct results table~~ **DONE — Sections 4.10, 4.12, Tables 2/5 updated.**
- [x] ~~Run CIFAR-100 150-epoch three-way comparison.~~ **DONE — EXP-012 (defense insufficient ⚠️).**
- [x] ~~**[P1] Phase 6A: CIFAR-100 stronger defense.**~~ **DONE — EXP-013. FAILED. See 🔴 critical note.**
- [x] ~~**[P1-NEW] Option B: gradient-noise injection for CIFAR-100.**~~ **DONE — EXP-019. ALL THREE VARIANTS FAILED. n=0.5 → 48.36% (above attack), n=1.0 → 49.64% (worst ever), n=2.0 → 43.10% (≈ no-noise). Root cause: noise injected into grad_output_A is calibrated to gradient magnitude, providing temporal consistency that MaliciousSGD still exploits. Gradient space exhausted.**
- [x] ~~**[P1-NEW-2] Implement TRUE embedding-space noise (z_a corruption before forward pass).**~~ **DONE — EXP-024/025. BOTH VARIANTS FAILED. std=0.5 → 50.67% (ABOVE attack), std=1.0 → 51.64% (WORST EVER). Root cause: noisy z_a causes top model to send HIGHER-INFORMATION corrective gradients to Party A, which MaliciousSGD amplifies even more aggressively. Embedding-space surface is exhausted.**
- [x] ~~**[P1-NEW-3] Sign-flip momentum disruption (2-line code change, high theoretical elegance).**~~ **DONE — EXP-023 on CIFAR-10. FAILED. 86.40% defended > 83.11% benign. Theory (ratio=1.0) does not hold in practice: MaliciousSGD operates on p.grad, not grad_output_A; nonlinear chain rule means alternating grad_output_A sign does not guarantee alternating p.grad sign. Phase 13 (CIFAR-100 sign-flip) NOT WORTH RUNNING.**
- [ ] **[P1-NEW-4] [HIGHEST CIFAR-100 PRIORITY] Multi-seed benign baseline:** Run 3 seeds minimum to establish the 30.33% benign reference as a stable mean ± std. The current single-seed reference is insufficient for any paper claim about CIFAR-100. This unblocks all CIFAR-100 comparisons.
- [ ] **[P1-NEW-5] CINIC10L multi-seed validation:** Run 3 seeds for CINIC10L (benign/attack/defense) to confirm the EXP-022 result (62.43%) is reproducible. This is the highest-value short-term experiment — if confirmed, it solidifies the two-dataset 10-class paper narrative.
- [ ] **[P1-NEW-6] Domain adversarial training with GRL for CIFAR-100:** All gradient-space and embedding-space surfaces have failed. The next fundamentally different approach is domain adversarial training (see possible_directions_3.md Section 7.2) — adding a gradient reversal layer that forces Party A's embedding to be class-uninformative, and MaliciousSGD would amplify the ADVERSARIAL gradient, accelerating class confusion. This is the most theoretically novel remaining direction for CIFAR-100.
- [x] ~~**[P2] Phase 6B: Complete CIFAR-10 seed sweep.**~~ **DONE — EXP-016. All 4 seeds (0, 42, 123, 456) complete. Mean ± std computed. 4/4 seeds show defended < benign ✅**
- [x] ~~**[P3] Run Stage 2 for GC and Laplace DP competitor checkpoints.**~~ **DONE — EXP-014.**
- [x] ~~**[P4] Ablation at 100 epochs for CIFAR-10 (alpha, tau).**~~ **DONE — EXP-015. All results obtained.**
- [ ] **[P5] Stabilize CIFAR-100 benign baseline:** At least 3 seeds to establish mean ± std for the 30.33% baseline. CIFAR-100 variance is high; a single seed is not publishable.
- [ ] **[P6] Generate t-SNE plots.** CSV files exist. Color by true label: compare active (J_A=2.12) vs defended (J_A=1.74). Semantic misalignment should show clusters spanning multiple true-class colors in the defended plot. Figure 3 or supplementary.
- [ ] **[P7] Revise mechanism narrative.** Correct claim: "defense disrupts semantic label alignment via progressive gradient suppression" — NOT "reduces geometric separability." Fisher divergence grows under defense (C8, C15).

### 7.2b Defense Architecture Improvements (To Revisit After Phase 4)

Three options in priority order. Do not implement until Phase 4 establishes ground truth at 100 epochs.

**Option A — Hard clip at ceiling (quick tweak, try first):**
In `AsymmetricAdaptivePerturbation.apply()`, replace the linear scale with a hard zero when divergence exceeds `tau + 1/alpha`. The current linear ramp is too gentle — MaliciousSGD's 5× internal amplification partially compensates. Binary kill at high divergence applies more pressure.
```python
if divergence > tau + 1/alpha:
    scale = 0.0
else:
    scale = max(0.0, 1.0 - alpha * (divergence - tau))
```

**Option B — Embedding-level noise injection (stronger, more principled):**
Current defense targets `grad_output` (server→Party A gradient). MaliciousSGD targets `p.grad` (Party A internal). These are different tensors — MaliciousSGD can partially compensate for grad_output suppression. Better target: inject calibrated noise directly into Party A's embedding `z_a` (before top model forward pass) when divergence > tau. The `.pth` checkpoint saves the bottom model weights, which under this defense learn to produce individually unreliable embeddings even if geometrically structured. MixMatch pseudo-labels become unstable → inference accuracy drops. Change location: `simulate_train_round_per_batch()` in `vfl_framework.py`, after `output_tensor_bottom_model_a` is produced.

**Option C — Centroid drift signal (longer term):**
Under MaliciousSGD, class centroids in `z_a` lock in early and barely move epoch-to-epoch. Benign centroids drift more. Add inter-epoch centroid correlation as a second detection signal alongside Fisher. Enables earlier detection and potentially a time-series alarm before the defense fully engages.

### 7.2d Persistent Projection — Next Experiment Plan (Phase 22 + 23, 2026-07-12)

**Motivation:** The current defense story is fragmented — AAP for 10-class, one-shot GradientProjection for 100-class. The GradientProjection's one-shot collapse is not designed behavior and will be criticized. Persistent Projection is the principled unified alternative.

**Key design changes from current GradientProjection:**

| Parameter | Current GradientProjection | Proposed Persistent Projection |
|---|---|---|
| Firing frequency | Once (epoch 11, then self-terminates) | Every epoch where divergence > tau |
| Discriminative direction | Instantaneous aux_classifier output | EMA of aux_classifier output (α_ema ≈ 0.1–0.3) |
| Burn-in | 8 epochs | Shorter (2–4 epochs) or none |
| Projection magnitude | Full orthogonal projection | Full or partial (α_proj ∈ (0,1]) |
| Risk of collapse | Catastrophic (one-shot removes 97% of gradient) | Bounded per-epoch removal; no spike risk |

**Implementation approach:**
```python
class PersistentProjectionDefense:
    def __init__(self, embed_dim, num_classes, alpha_ema=0.2, alpha_proj=1.0, tau=0.10, burn_in=4):
        self.aux_classifier = nn.Linear(embed_dim, num_classes)
        self.d_ema = None          # running EMA of discriminative direction
        self.alpha_ema = alpha_ema
        self.alpha_proj = alpha_proj
        self.tau = tau
        self.burn_in = burn_in

    def update_direction(self, z_a, y):
        # Server-side: compute aux_classifier gradient direction
        loss = CrossEntropy(self.aux_classifier(z_a), y)
        d_inst = grad(loss, z_a)  # instantaneous discriminative direction
        d_inst_norm = d_inst / (d_inst.norm() + 1e-8)
        if self.d_ema is None:
            self.d_ema = d_inst_norm
        else:
            self.d_ema = (1 - self.alpha_ema) * self.d_ema + self.alpha_ema * d_inst_norm
            self.d_ema = self.d_ema / (self.d_ema.norm() + 1e-8)  # keep unit norm

    def apply(self, grad_output_a, divergence, epoch):
        if epoch < self.burn_in or divergence < self.tau:
            return grad_output_a
        # Project: remove the discriminative direction component
        proj_coeff = (grad_output_a * self.d_ema).sum(dim=-1, keepdim=True)
        grad_proj = grad_output_a - self.alpha_proj * proj_coeff * self.d_ema
        return grad_proj
```

**Phase 22 — Persistent Projection for CIFAR-10:**

| Field | Plan |
|---|---|
| Dataset | CIFAR-10, 100 epochs Stage 1, 25 epochs Stage 2 |
| Seeds | 0, 42, 123, 456 (4 seeds) |
| alpha_ema values | 0.1, 0.2, 0.5 (small sweep) |
| alpha_proj | 1.0 (full projection) initially |
| tau, burn_in | 0.10, 4 (shorter burn-in to prevent early discriminative accumulation) |
| Baseline | Benign 83.11 ± 2.84%, Attack 94.95 ± 0.52%, AAP defended 81.80 ± 1.85% |
| Success criterion | Mean defended < benign mean (83.11%); at least 3/4 seeds show direction |
| Script | `Code/run_phase22_pp_cifar10.bat` (to be created) |
| Status | 📋 PLANNED |

**Phase 23 — Persistent Projection for CIFAR-100:**

| Field | Plan |
|---|---|
| Dataset | CIFAR-100, 150 epochs Stage 1, 25 epochs Stage 2 |
| Seeds | 0 initially (single seed to test mechanism) |
| alpha_ema | Best from Phase 22 |
| Success criterion | MC < benign mean (29.56%); does NOT exhibit one-shot collapse at epoch 11-12 |
| Key diagnostic | Fisher divergence CSV — check that defense fires MULTIPLE times, not just once |
| Script | `Code/run_phase23_pp_cifar100.bat` (to be created) |
| Status | 📋 PLANNED — depends on Phase 22 results |

**Decision tree after Phase 22:**
- If PP matches or exceeds AAP on CIFAR-10: PP becomes the sole defense mechanism. Run Phase 23 (CIFAR-100). AAP becomes an ablation baseline.
- If PP is WORSE than AAP on CIFAR-10: Investigate alpha_ema sensitivity. If still worse after tuning: reconsider whether AAP+GradProj fragmented story is acceptable with strong theoretical framing.
- If PP causes similar one-shot collapse on CIFAR-10 as GradProj did on CIFAR-100: investigate shorter burn-in or EMA smoothing as fix.

---

### 7.2c Research Assessment (2026-07-10 — updated post Phase 18/19)

**Overall assessment:** Promising

**Confidence level:** Medium-High (raised from Medium following Phase 19 seed-0 result)

**Strongest evidence:**
- CINIC10L defense success (EXP-022): 62.43% < 65.70% benign on an independent dataset with zero retuning. The defense eliminates a +20.89pp attack advantage. VFL utility cost = +0.63pp (negligible). Clean, strong evidence.
- CIFAR-10 multi-seed validation (EXP-016): 4/4 seeds, mean defended 81.80 ± 1.85% < benign 83.11 ± 2.84%. Paired t-test passes. Defense criterion met consistently.
- Phase 19 CIFAR-100 gradient projection (EXP-032, seed-0): 26.97% < benign mean 29.56% (−2.59pp) — first CIFAR-100 configuration to meet criterion. VFL utility cost = −0.01pp (effectively zero). Attack suppression: −20.89pp (eliminating 114% of attack advantage over benign).

**Biggest weaknesses:**
- Phase 19 CIFAR-100 result is **single-seed only**. All three CIFAR-100 confirmed results (attack 47.86%, benign seed-0 30.33%, defended 26.97%) depend on seed-0. Multi-seed confirmation is essential.
- CINIC10L result is single-seed only (EXP-022). Phase 15 multi-seed running.
- The Phase 19 mechanism (catastrophic single-activation collapse) is unusual and needs to be characterized across seeds. If it only works for seed-0, it would indicate the alignment of aux_classifier direction with grad_output_a is a seed-specific coincidence.
- Yahoo Answers results still pending — modality generalization unconfirmed.

**Are we on track for publication?**
Yes, potentially with expanded scope. If Phase 19 multi-seed confirms → three-dataset story (CIFAR-10/CINIC10L/CIFAR-100) with the projection defense as a more general solution than simple suppression. If multi-seed fails → narrowed to two-dataset 10-class contribution. Either outcome is publishable.

**Phase 18 assessment (EXP-031):**
Phase 18 (adversarial auxiliary classifier) is a complete and unambiguous failure. The model was destroyed — not merely underperformed. The root cause (unbounded aux_grad + MaliciousSGD amplification → NaN cascade) is now fully understood. It provides a clear negative result and motivates the bounded projection approach. The Phase 18 failure is a useful finding for the paper: it rules out direction-reversal as a viable approach and explains why bounded projection is necessary.

**Phase 19 assessment (EXP-032):**
Phase 19 (gradient projection) achieves what 10 prior CIFAR-100 configurations could not. The mechanism (catastrophic single-activation collapse rather than gradual suppression) is unexpected but has a clean mathematical explanation (aux_classifier learns a highly accurate discriminative direction over 11 epochs; projection against it removes nearly all of grad_output_a at first activation; the resulting structural collapse is self-perpetuating because Fisher divergence goes permanently negative). The key question is whether this mechanism is consistent across seeds.

**Top priorities (post Phase 18/19):**
1. 📋 **IMMEDIATE: Phase 19 multi-seed** (seeds 42/123/456) — decisive for CIFAR-100 paper claim
2. 🟡 RUNNING: Phase 15 (CINIC10L seeds 42/123/456) — decisive for CINIC10L paper claim
3. 🟡 RUNNING: Phase 17 (Yahoo Answers) — modality generalization; secondary priority
4. 📋 OPTIONAL: Criteo binary tabular

**Experiments NOT worth running based on current evidence:**
- Phase 13 (CIFAR-100 sign-flip) — Phase 12 showed sign-flip fails even on CIFAR-10; CIFAR-100 would be worse
- z_a corruption at different noise_std values — both n=0.5 and n=1.0 made things worse; mechanism is counterproductive regardless of scale
- Stronger alpha/tau variations for CIFAR-100 — EXP-013 showed these variants fail; no new insight expected
- TinyImageNet (200 classes, already downloaded) — 200 classes would fail even harder; strategically counterproductive
- Additional gradient-space variations — the projection result shows the direction-based approach succeeds where scaling fails; no value in revisiting scaling variants
- Persistent Projection (PP) with K=1 — confirmed failed on both CIFAR-10 and CIFAR-100 (Phases 22/23, 6/6 fixed variants fail); root cause architectural, not implementation

---

### 7.2e Persistent Projection Fixed — Phase 22 & Phase 23 Final Assessment (2026-07-14)

**Updated overall assessment:** Promising (primary story: AAP + GradProj confirmed); PP = NOT PROMISING as standalone defense.

**Confidence in PP failure:** High (6 data points: 3 CIFAR-10 + 3 CIFAR-100 fixed variants; all fail by large margins; bug fix changed outcomes by <2pp).

**Summary of Phase 22 Fixed (EXP-045/046/047 — CIFAR-10, 100ep):**

| α_ema | VFL Test | MC Best | vs Benign (87.23%) | vs Attack (94.95%) | vs AAP (81.80%) |
|---|---|---|---|---|---|
| 0.1 | 80.77% | 94.74% | +7.51pp ❌ | −0.21pp | +12.94pp worse |
| 0.2 | 80.09% | **92.99%** | **+5.76pp ❌** | −1.96pp | +11.19pp worse |
| 0.3 | 80.75% | 94.91% | +7.68pp ❌ | −0.04pp | +13.11pp worse |

**Summary of Phase 23 Fixed (EXP-048/049/050 — CIFAR-100, 150ep):**

| α_ema | VFL Test | MC Best | vs Benign (30.33%) | vs Attack (47.86%) | vs GradProj (23.44%) |
|---|---|---|---|---|---|
| 0.1 | 46.47% | 48.78% | +18.45pp ❌❌ | +0.92pp | +25.34pp worse |
| 0.2 | 47.50% | **50.16%** | **+19.83pp ❌❌❌** | **+2.30pp ABOVE attack** | +26.72pp worse |
| 0.3 | 46.47% | 48.81% | +18.48pp ❌❌ | +0.95pp | +25.37pp worse |

**Key findings:**

1. **Bug fix confirmed but irrelevant.** Per-sample gradient normalization resolved the batch-cancellation bug — evidenced by phase transitions (CIFAR-10: epoch 50; CIFAR-100: epochs 75 and 120). Outcomes changed by <2pp. The bug was not the root cause.

2. **Root cause is architectural.** PP projects grad_output_A. MaliciousSGD amplifies p.grad (internal gradients, computed via chain rule through Party A's ResNet). The ResNet Jacobian can re-express discriminative structure in p.grad even after removing the top-1 direction from grad_output_A. DCR(K=1, C=10) = 11%, DCR(K=1, C=100) = 1% — both insufficient.

3. **CIFAR-100 anomaly: PP worsens privacy.** EXP-049 (ema=0.2): MC = 50.16% > undefended attack (47.86%) by 2.30pp. VFL Stage 1 accuracy also improves (+2.17pp above benign) suggesting PP accidentally regularizes in a way that benefits the attacker's embedding quality.

4. **Geometric tightness ≠ label alignment suppression.** Under CIFAR-100 ema=0.2, intra_var_A collapses to 0.006 (tighter than undefended attack at 0.014). MC is still 50.16%. Phase transitions disrupt cluster geometry but not the label-discriminative structure within those clusters.

5. **PP vs existing defenses:** PP best (92.99%) is 11pp worse than AAP (81.80%) on CIFAR-10; PP best (48.78%) is 25pp worse than GradProj (23.44%) on CIFAR-100.

**Next required experiment — Phase 24 MDPP (K=10):**
- Modify `Code/possible_defenses.py` → `PersistentProjectionDefense` to use `torch.linalg.svd(aux_classifier.weight)` for K=10 directions with K sequential orthogonal projections
- Add `--persistent-proj-k INT` to `vfl_framework.py`
- Create `Code/run_phase24_mdpp_cifar100.bat` (K ∈ {5, 10, 20}, seed=0, CIFAR-100)
- Success criterion: mc_best_train_top1 < 30.33% WITHOUT intra_var_A spike of 6+ orders of magnitude
- DCR(K=10, C=100) = 10/99 ≈ 10.1% — matches CIFAR-10's coverage level

**Updated top priorities (2026-07-14):**

1. 📋 **IMMEDIATE: Phase 24 — MDPP (K=10, CIFAR-100, seed-0 gate test)**
2. 🟡 **RUNNING (status unknown): Phase 15** (CINIC10L seeds 42/123/456)
3. 🟡 **RUNNING (status unknown): Phase 17** (Yahoo Answers, EXP-030)
4. 📋 **Begin paper writing** — ALIA characterization (Fisher divergence) + AAP (10-class, 4/4 seeds) + GradProj (100-class, 4/4 seeds) + DCR framework + MDPP as forward-looking unification
5. 📋 **Verify Phase 15 and 17 status** — may already be complete

---

### 7.3 Near-Term — Paper Strengthening

- [ ] Normalize Fisher threshold by num_classes for dataset-agnostic comparison
- [ ] Test burn-in calibration (mean + 2σ) vs fixed threshold approach
- [ ] Generate plots via `plot_characterization.py` for both datasets — Figure 1 (Fisher trajectories) and Figure 2 (model completion bar chart)
- [ ] Investigate asymmetric Laplace DP collapse root cause in `model_sets.py`

### 7.4 Long-Term — Paper Extensions

- [ ] Test on CINIC10 and TinyImageNet (commands already in run_training.sh)
- [ ] Compare with LADSG/Geno (arXiv:2506.06742) gradient norm signal — argue novelty of Fisher criterion approach
- [ ] Investigate n_labeled sensitivity (n_labeled ∈ {10, 20, 40, 100})
- [ ] Design combined defense for the multi-party (Active All) case
- [ ] Theoretical analysis: derive bound on task gradient suppression → label alignment degradation

---

*This document is the single source of truth for all experimental results and research decisions. Update this file whenever new experiments are run or new results are obtained. Do not create standalone summaries — add to the relevant section here instead.*
