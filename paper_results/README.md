# paper_results/ — Master Results Directory

**Last Updated:** 2026-07-11  
**Status:** Partial — Phase 20 (CIFAR-100 grad proj seeds 42/123/456), Phase 21 (CIFAR-100 attack seeds 42/123/456), Phase 15 (CINIC10L seeds 42/123/456), and Phase 17 (Yahoo Answers) are RUNNING. All completed experiment results through EXP-032 are populated.

This directory is the **single source of truth** for all quantitative results used in the paper. Raw experiment `.txt` output files live in `Code/saved_experiment_results/`; this directory contains clean, paper-ready CSVs derived from those files and the `research_log.md`.

**Do not modify or move files in `Code/saved_experiment_results/`.** This directory contains only clean summary tables.

---

## Files

| File | Purpose | Experiments Covered | Status |
|---|---|---|---|
| [cifar10_results.csv](cifar10_results.csv) | All CIFAR-10 Stage 1 + Stage 2 results | EXP-001/003/008–011/014–017/023 + seed sweep | ✅ Complete |
| [cifar100_results.csv](cifar100_results.csv) | All CIFAR-100 Stage 1 + Stage 2 results | EXP-006/008–009/012–014/018–019/024–025/029/031–038 | ✅ Partial (seeds 42/123/456 pending) |
| [cinic10l_results.csv](cinic10l_results.csv) | All CINIC10L Stage 1 + Stage 2 results | EXP-020–022/026–028 | ✅ Partial (seeds 42/123/456 pending) |
| [yahoo_answers_results.csv](yahoo_answers_results.csv) | Yahoo Answers text modality test | EXP-030 | 🟡 Running |
| [fisher_divergence_summary.csv](fisher_divergence_summary.csv) | Per-epoch Fisher divergence trajectories | EXP-001/006/007/011/032 | ✅ Complete |
| [ablation_results.csv](ablation_results.csv) | Alpha/tau hyperparameter ablation at 100ep (CIFAR-10) and 150ep (CIFAR-100) | EXP-010/011/012/013/015 | ✅ Complete |

---

## Column Dictionary

### Core columns (all per-dataset CSVs)

| Column | Meaning |
|---|---|
| `phase` | Research phase name (e.g., "Phase 4 Three-Way (100ep)") |
| `exp_id` | Experiment ID from research_log.md (e.g., EXP-011) |
| `dataset` | Dataset name: CIFAR10, CIFAR100, CINIC10L, YahooAnswers |
| `num_classes` | Number of classes (10 or 100) |
| `stage1_epochs` | Number of VFL Stage 1 training epochs |
| `stage2_epochs` | Number of Stage 2 MixMatch SSL epochs (always 25) |
| `condition` | Experimental condition: benign / active_party_a / active_all_parties / active+defense / benign+defense |
| `attack_type` | Attack mechanism: none / MaliciousSGD / MaliciousSGD-all / MaliciousSGD+GC / MaliciousSGD+Laplace |
| `defense_name` | Defense applied: none / AsymmetricAdaptivePerturbation / GradientProjection / AdversarialAuxiliary / LaplaceDP_scale0.001 / GradientCompression_75pct / AsymmetricAdaptivePerturbation+GradNoise / AsymmetricAdaptivePerturbation+zaNoise |
| `defense_alpha` | Suppression strength hyperparameter (α) |
| `defense_tau` | Detection threshold (τ) |
| `defense_burn_in` | Epochs before defense can activate (burn-in) |
| `manual_seed` | Random seed (0 / 42 / 123 / 456); "all" for mean rows |
| `stage1_train_top1` | VFL Stage 1 training set top-1 accuracy at final epoch |
| `stage1_test_top1` | VFL Stage 1 test set top-1 accuracy at final epoch |
| `stage1_test_topk` | VFL Stage 1 test top-k accuracy (k=4 for 10-class; k=5 for 100-class) |
| `mc_best_train_top1` | **PRIMARY METRIC** — Stage 2 MixMatch best training set top-1 accuracy across 25 epochs. This is the label inference accuracy. |
| `mc_final_test_top1` | Stage 2 final epoch (25) test set top-1 accuracy. Secondary metric — see note. |
| `mc_vs_benign_pp` | mc_best_train_top1 minus the benign reference (in percentage points). Negative = defense below benign = defense succeeds. |
| `mc_vs_attack_pp` | mc_best_train_top1 minus the undefended attack (in pp). Negative = defense reduces attack. |
| `vfl_utility_cost_pp` | stage1_test_top1 (defended) minus stage1_test_top1 (benign). Negative = defense costs VFL utility. |
| `n_labeled` | Number of labeled examples for MixMatch Stage 2 (40 for 10-class; 400 for CIFAR-100) |
| `notes` | Interpretation, caveats, cross-references |

### Additional columns (CIFAR-100 only)

| Column | Meaning |
|---|---|
| `defense_variant` | Specific variant within a defense type (e.g., n=0.5 for noise_std, lambda=0.5 for adv_aux, za=0.5 for z_a corruption) |
| `stage1_test_top5` | VFL Stage 1 test top-5 accuracy (only relevant for 100-class) |
| `mc_vs_benign_mean_pp` | mc_best_train_top1 vs 4-seed benign mean 29.56% (preferred over single-seed seed-0 30.33%) |

### Fisher divergence CSV columns

| Column | Meaning |
|---|---|
| `epoch` | Training epoch index (0-indexed) |
| `fisher_A` | Fisher criterion for Party A: inter_var_A / intra_var_A |
| `fisher_B` | Fisher criterion for Party B: inter_var_B / intra_var_B |
| `fisher_divergence` | fisher_A − fisher_B. Positive = Party A more discriminative = attack detected |
| `intra_var_A` | Intra-class variance of Party A's embeddings. Spike indicates discriminative collapse. |
| `grad_norm_A` | L2 norm of grad_output_bottom_model_a (server→Party A communication gradient) |
| `defense_scale` | Scale factor applied to grad_output_A: max(0, 1 − α × (divergence − τ)). 1.0 = no suppression; 0.0 = complete block |
| `defense_fired` | Whether the defense was active this epoch |

---

## Which Paper Tables and Figures Use Which CSVs

| Future paper table/figure | Source CSV | Key columns | Missing data |
|---|---|---|---|
| **Table 1: Main Results — 3-way comparison** (benign/attack/defended) | cifar10_results.csv + cifar100_results.csv | condition, mc_best_train_top1, mc_vs_benign_pp, mc_vs_attack_pp | CIFAR-100 defended seeds 42/123/456 (Phase 20 running) |
| **Table 2: Multi-seed CIFAR-10 results** (4 seeds) | cifar10_results.csv | manual_seed, mc_best_train_top1, mc_vs_benign_pp, mean±std rows | Complete ✅ |
| **Table 3: Multi-seed CIFAR-100 comparison** (3×4 benign/attack/defended) | cifar100_results.csv | manual_seed, mc_best_train_top1, full 3-condition rows | Attack seeds 42/123/456 (Phase 21 running); Defended seeds 42/123/456 (Phase 20 running) |
| **Table 4: Hyperparameter ablation** (alpha and tau sweep) | ablation_results.csv | defense_alpha, defense_tau, mc_best_train_top1, passed_criterion | Complete for CIFAR-10 ✅; CIFAR-100 ablation only has Stage 1 VFL (no MC) |
| **Table 5: Competitor comparison** (GC, Laplace DP, our defense) | cifar10_results.csv + cifar100_results.csv | defense_name, mc_best_train_top1, vfl_utility_cost_pp | All 30ep. GC/Laplace VFL task data only — no Stage 2 for their defense variants. |
| **Table 6: VFL utility cost** (task accuracy tradeoff) | cifar10_results.csv + cifar100_results.csv + cinic10l_results.csv | condition, stage1_test_top1, vfl_utility_cost_pp | Complete for 100/150ep primary conditions ✅ |
| **Table 7: Defense failure exploration** (CIFAR-100 failed variants) | cifar100_results.csv | defense_name, defense_variant, mc_best_train_top1, mc_vs_benign_mean_pp | Complete through EXP-032 ✅ |
| **Table 8: Cross-dataset generalization** (CIFAR-10 / CINIC10L) | cifar10_results.csv + cinic10l_results.csv | dataset, mc_best_train_top1, mc_vs_benign_pp | CINIC10L multi-seed pending |
| **Table 9: Yahoo Answers modality generalization** | yahoo_answers_results.csv | mc_best_train_top1, mc_vs_benign_pp | All pending (Phase 17 running) |
| **Fig 1: Fisher divergence trajectory** (CIFAR-10, 100ep) | fisher_divergence_summary.csv | epoch, fisher_divergence, defense_scale (attack vs attack+defense vs benign) | Complete ✅ |
| **Fig 2: Fisher divergence trajectory** (CIFAR-100 grad proj, 150ep) | fisher_divergence_summary.csv | epoch, intra_var_A, fisher_divergence (Phase 19 collapse) | Complete ✅ |
| **Fig 3: t-SNE visualization** (semantic misalignment) | Raw CSV files in CIFAR10_csv_files/ | Per-sample embeddings | NOT YET CREATED — t-SNE script pending |

---

## Primary Metric Note

**`mc_best_train_top1` is the primary label inference metric** (not `mc_final_test_top1`).

The threat model is: Party A is trying to recover labels for the 49,960 unlabeled **training** examples. Training-set inference accuracy measures success directly. Test-set accuracy is a secondary indicator of generalization but can be misleading:
- Under the Gradient Projection defense (EXP-032), test MC is higher than training MC because structured embeddings generalize well even when semantic alignment is weak.
- The paper should lead with training-set accuracy; test-set accuracy can be reported in supplementary.

---

## Key Numbers Summary (Quick Reference)

### CIFAR-10 (100 epochs, primary setting α=1.0, τ=0.10, burn_in=8)

| Condition | MC Best Train | 4-seed mean ± std |
|---|---|---|
| Benign | 87.23% (seed-0) | 83.11 ± 2.84% |
| Attack (MaliciousSGD) | 95.42% (seed-0) | 94.95 ± 0.52% |
| **Attack + Defense** | **84.27% (seed-0)** | **81.80 ± 1.85%** |
| Attack advantage over benign | +8.19pp | +11.84pp |
| Defense margin below benign | −2.96pp | −1.32pp |
| Defense reduction from attack | −11.15pp | −13.15pp |
| VFL test accuracy cost | −1.93pp | — |
| **Result** | **✅ ALL 4 SEEDS DEFENDED < BENIGN** | |

### CIFAR-100 (150 epochs, Gradient Projection defense)

| Condition | MC Best Train | Seed-0 only |
|---|---|---|
| Benign (4-seed mean ± std) | 29.56 ± 2.93% | 30.33% |
| Attack (MaliciousSGD, seed-0) | 47.86% | — |
| **Gradient Projection defense (seed-0)** | **26.97%** | **−2.59pp below benign mean ✅** |
| VFL cost | −0.01pp | — |
| **CRITICAL CAVEAT** | **Seeds 42/123/456 (Phase 20) STILL RUNNING** | |

### CINIC10L (100 epochs, same defense params as CIFAR-10)

| Condition | MC Best Train | Seed-0 only |
|---|---|---|
| Benign | 65.70% | — |
| Attack | 86.59% (+20.89pp) | — |
| **Attack + Defense** | **62.43% (−3.26pp below benign ✅)** | — |
| VFL cost | +0.63pp (negligible) | — |
| **CRITICAL CAVEAT** | **Seeds 42/123/456 (Phase 15) STILL RUNNING** | |

---

## Missing Data (as of 2026-07-11)

1. **CIFAR-100 gradient projection seeds 42/123/456** (EXP-033/034/035, Phase 20): The decisive multi-seed test for the CIFAR-100 claim. Running. Per-seed targets: seed-42 < 26.19%, seed-123 < 28.56%, seed-456 < 33.14%. ETA ~18h total from Phase 20 start.

2. **CIFAR-100 attack baseline seeds 42/123/456** (EXP-036/037/038, Phase 21): Needed to fill the attack column of the 3×4 comparison table. Running. ETA ~12h total from Phase 21 start.

3. **CINIC10L seeds 42/123/456** (EXP-026/027/028, Phase 15): Multi-seed validation of CINIC10L defense. Running. ~40h total. Seed-42 Stage 1 benign and attack .txt files exist on disk (Phase 15 in progress).

4. **Yahoo Answers** (EXP-030, Phase 17): Stage 1 benign running. Then attack and defense Stage 1 + all 3 Stage 2 MCs to follow. Slowest dataset (~10-20× slower per epoch than CIFAR-10 due to BERT).

5. **Hyperparameter ablation Stage 2 for CIFAR-100**: Phase 3b (EXP-010) only ran Stage 1 VFL for CIFAR-100 ablation variants (a=0.5/2.0, t=0.05/0.15). No MC data exists for these. Would need new runs to fill ablation_results.csv CIFAR-100 MC columns — NOT currently planned.

6. **t-SNE plots**: Raw per-sample embedding data exists in CSV files under `Code/saved_experiment_results/csv_files/`. A t-SNE plotting script (`Code/plot_characterization.py`) exists but has not been run to generate visualization artifacts for the paper.

7. **Criteo binary tabular** dataset: Code infrastructure exists in `Code/datasets/criteo_preprocess.py` but the dataset files need to be downloaded. Not currently planned.

---

## How to Update After Experiments Complete

When Phase 20/21/15/17 finish:

1. Read the relevant `.txt` output files from `Code/saved_experiment_results/saved_models/`.
2. Extract `mc_best_train_top1` (search for "Best accuracy:" or the highest accuracy line in the training table).
3. Add rows to the appropriate CSV (cifar100_results.csv for Phase 20/21; cinic10l_results.csv for Phase 15; yahoo_answers_results.csv for Phase 17).
4. Fill in the `⏳` entries in `research_log.md` EXP-033–038 and EXP-026–028 tables.
5. Once Phase 20 + Phase 21 complete, assemble the 3×4 CIFAR-100 comparison table and add a `cifar100_3x4_comparison.csv` file here.
6. Compute mean ± std once all 4 seeds are in for each dataset/condition.
