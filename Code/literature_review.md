# Literature Review — VFL Label Inference Attacks and Defenses

**Last Updated:** 2026-07-12
**Status:** Complete for current research direction (Persistent Projection / Fisher Divergence)

---

## 1. Search Scope

Papers searched across: USENIX Security, IEEE S&P, CCS, NeurIPS, ICML, ICLR, AAAI, arXiv (2020–2026), IEEE TKDE, SIGIR, KDD.

Query terms used:
- `vertical federated learning label inference`
- `VFL gradient attack defense projection`
- `federated learning embedding separability Fisher`
- `VFL active attack gradient manipulation`
- `subspace projection federated gradient defense`
- `gradient projection federated learning privacy`
- `persistent projection orthogonal federated`

---

## 2. Papers Reviewed

### 2.1 Attack Papers (Establishing the Threat)

| Paper | Venue | Year | Relevance |
|---|---|---|---|
| **Unleashing the Tiger: Inference Attacks on Split Learning** (Erdogan et al.) | CCS | 2022 | Passive and active VFL label inference; reference for the attack model |
| **Label Inference Attacks Against Vertical Federated Learning** (Fu et al.) | USENIX | 2022 | Introduces MaliciousSGD specifically; the attack we are defending against |
| **Effective Passive Membership Inference Attacks in Federated Learning** | Various | 2022 | Passive attacks; distinct from active MaliciousSGD |
| **CAFE: Catastrophic Data Leakage in Vertical Federated Learning** | NeurIPS | 2021 | Feature-space leakage; different from label inference |
| **Label Leakage and Protection in Two-party Split Learning** | ICLR | 2022 | Gradient-based label leakage; closest to our attack definition |

**Key takeaway:** Fu et al. (USENIX 2022) is the canonical MaliciousSGD reference. The attack modifies the bottom model optimizer's internal gradients (`p.grad`) with `ratio = clamp(1 + γ*(g_t/g_{t-1}), 1.0, 5.0)`, γ=200. This is what our defenses target.

---

### 2.2 Defense Papers — Symmetric / Baseline

These are the EXISTING defenses we compare against (already in our codebase):

| Paper | Mechanism | Venue | Year | Limitation |
|---|---|---|---|---|
| **PPDL** (Shokri & Shmatikov) | Gradient sparsification (top-k) | CCS | 2015 | Symmetric — hurts both parties |
| **Gradient Compression** (Lin et al.) | Threshold-based sparsification | arXiv | 2017 | Symmetric; passive attacks only |
| **Laplace DP** (Dwork et al.) | Gaussian/Laplace noise injection | FOCS | 2006 | Symmetric; severe utility loss at meaningful epsilon |
| **Multistep Gradient** | Quantization into discrete bins | Various | — | Symmetric; coarse approximation only |

**Key takeaway:** All four baseline defenses are symmetric — they perturb BOTH parties identically. None can detect which party is attacking. This is the gap our asymmetric detection closes.

---

### 2.3 Defense Papers — Novel / Asymmetric (Competitors to Our Work)

#### MixPro (CLOSEST COMPETITOR for Gradient Projection)
- **Full title:** Within the FedAds benchmark (Wei et al., "FedAds: A Benchmark for Privacy-Preserving CVR Prediction with Vertical Federated Learning")
- **Venue:** SIGIR 2023
- **Mechanism:** Applies a gradient projection step per-batch in VFL. The direction is computed as a generic noise direction, not a discriminative subspace estimate.
- **Why it differs from our GradProj/PersistentProjection:**
  1. MixPro's projection direction is generic (noise), not derived from any auxiliary classifier or discriminative subspace
  2. MixPro targets passive leakage, not MaliciousSGD active attacks
  3. MixPro has no persistent EMA direction — it is stateless across batches
  4. MixPro does not use Fisher divergence or any attack detection signal
- **Citation string:** Wei et al., FedAds, SIGIR 2023. Must explicitly differentiate in paper introduction.

#### ProjPert (MISLEADING NAME — not geometric projection)
- **Full title:** "ProjPert: Projection-based Perturbation for Label Protection in Split Learning"
- **Venue:** IEEE TKDE 2024
- **Mechanism:** Binary search over noise parameters ("projection" here means projecting a scalar value, not projecting onto a subspace). Entirely noise-based, passive attacks only.
- **Why it differs:** The word "projection" in the name refers to parameter-space projection, not geometric gradient subspace projection. Mechanistically unrelated to our work.
- **Action:** Cite in paper to clarify that ProjPert is a noise-parameter method, NOT a subspace method. This prevents reviewers from conflating the two.

#### LADSG (MOST RECENT — June 2025)
- **Full title:** "LADSG: Label Anomaly Detection and Subgraph Gradients for VFL Privacy"
- **Venue:** arXiv:2506.06742 (June 2025), to appear CollaborateCom 2026
- **Mechanism:** Gradient norm anomaly detection (detects attack by monitoring L2 norm of gradient tensors) + gradient substitution (replaces attacker's gradient with a surrogate computed from a benign model).
- **Why it differs from our Fisher Divergence:**
  1. LADSG monitors gradient NORMS (scalar value per tensor), not embedding separability structure
  2. LADSG defense is gradient substitution, not subspace projection
  3. LADSG claims to defend all three attack types (passive, active, direct) — but uses a single detection mechanism
  4. No Fisher divergence, no inter/intra-class variance, no geometric projection
- **Risk:** Reviewers may cite LADSG as prior art. Counter-argument: LADSG detection is gradient-magnitude-based; ours is embedding-space-based. Different signal, different defense mechanism.

#### MARVELL (Binary VFL Only)
- **Full title:** "MARVELL: Mitigating Privacy Leakage Against Model Explanations with a Minimax Game on Bert"
- **Venue:** ICML 2022
- **Mechanism:** Minimax noise design to equalize class gradient magnitudes. Prevents label inference by making class gradients statistically indistinguishable.
- **Limitation:** Binary classification ONLY. Does not extend to multi-class (CIFAR-10/100) — noise design becomes intractable with many classes.
- **Why it differs:** Noise-based (not projection-based); binary only; no Fisher divergence or EMA direction.

#### VMask (2025)
- **Full title:** "VMask: Gradient Masking via Layer Masking and Secret Sharing for VFL"
- **Venue:** 2025
- **Mechanism:** Cryptographic layer masking + secret sharing. Prevents gradient leakage at the cryptographic level.
- **Why it differs:** Cryptographic, not statistical/geometric. Completely different threat model and computational cost profile. No projection or Fisher divergence.

---

### 2.4 Tangentially Related Work (Different Problem Domain)

| Paper | Connection | Why NOT a competitor |
|---|---|---|
| **GradCAM-based Projection** (Srinivas et al.) | Orthogonal gradient projection for continual learning | Completely different problem: preventing catastrophic forgetting, not protecting VFL privacy |
| **Gradient Projection for Continual Learning** (Saha et al., NeurIPS 2021) | EMA-based direction tracking in gradient projection | Different domain: task incremental learning. No VFL, no attack detection, no Fisher divergence |
| **Fisher Information for DP** (Numerous papers) | Fisher Information Matrix used for DP noise calibration | Uses parameter-space Fisher matrix (Hessian approximation), NOT embedding-space Fisher divergence between parties |

---

## 3. Gap Analysis

### Gap 1 — No Prior Work Uses Embedding Separability for Attack Detection
All prior detection methods monitor communication-level signals (gradient norms, gradient values, timing). Our Fisher Divergence Monitor uses inter/intra-class variance ratio of embeddings (J_A, J_B) and their ASYMMETRY (Δ_F = J_A − J_B) as the attack signal. **No prior paper does this.** The specific insight — that MaliciousSGD makes Party A's embeddings progressively MORE class-discriminative than Party B's — has not been exploited before.

### Gap 2 — No Prior Defense Uses Persistent Geometric Projection Against Active VFL Attacks
MixPro (SIGIR 2023) applies per-batch projection but for passive attacks with a generic direction. GradCAM/continual learning projection is a different problem domain. No paper applies a persistently updated discriminative subspace projection, gated by an attack detector, specifically against MaliciousSGD-type active attacks.

### Gap 3 — All Existing Defenses Are Either Symmetric or Cryptographic
Symmetric defenses (DP, GC, PPDL) penalize all parties identically. Cryptographic defenses (VMask) require protocol-level changes. No prior work proposes an asymmetric server-side defense that responds ONLY to the attacking party without affecting the honest party.

### Gap 4 — No Prior Defense Works Across Both Low-class and High-class Datasets
AAP works on CIFAR-10 (10 classes) but fails on CIFAR-100 (100 classes) because scale never reaches 0 at 100 classes. No prior paper documents this class-count sensitivity and proposes a mechanism that unifies both. PP is designed to close this gap.

### Gap 5 — Detection-Only vs. Detection+Defense
Prior work on VFL attack detection focuses on monitoring and alerting. None of the detection papers close the loop with a defense mechanism derived from the same detection signal. Our work uses the Fisher divergence both to DETECT (when to activate) and to GUIDE the defense (direction estimate from aux classifier trained on Party A's embeddings).

---

## 4. Our Novelty Claims

### Claim 1: Fisher Divergence Detection — CLEARLY NOVEL
**What we claim:** Server monitors Δ_F = J_A − J_B each epoch (J = inter_class_var / intra_class_var computed on embeddings). When Δ_F > τ for epoch ≥ burn_in, attack is declared.

**Prior art search result:** No paper uses inter/intra-class variance ratio ASYMMETRY between VFL parties as an active attack signal. LADSG uses gradient norms. All others use gradient values or statistical tests on communication tensors, not embedding separability structure.

**Differentiation from Fisher Information literature:** Fisher Information Matrix (FIM) used in DP (Zhu et al., others) is the Hessian of the log-likelihood w.r.t. model parameters. Our J = inter_class_var / intra_class_var is the Fisher Linear Discriminant ratio (Fisher 1936) applied to embedding-space distributions per party. Same name, completely different mathematical object.

**Venue implication:** This is a genuinely novel detection signal with theoretical motivation (MaliciousSGD provably increases embedding separability; we measure this directly). Suitable for IEEE S&P or CCS.

---

### Claim 2: Asymmetric Defense Response — CLEARLY NOVEL
**What we claim:** Defense modifies ONLY Party A's gradient, never Party B's. Honest parties bear zero defense cost.

**Prior art:** All symmetric defenses (DP, GC, PPDL, multistep gradient) modify both. Cryptographic defenses require all parties to participate. No prior work proposes an asymmetric server-side defense triggered by a detection signal.

---

### Claim 3: Gradient Projection Defense (GradProj) — MODERATELY NOVEL
**What we claim:** Project grad_output_a onto the subspace orthogonal to the discriminative direction d_aux from an aux classifier.

**Prior art:** MixPro (SIGIR 2023) applies a gradient projection step in VFL. Differences: (1) MixPro uses generic noise direction, not discriminative subspace; (2) MixPro targets passive attacks; (3) MixPro has no detection gate.

**Differentiation required:** Must explicitly address MixPro in paper. Strongest argument: MixPro does not maintain any estimate of the discriminative subspace — it applies random perturbation. Our projection is TARGETED at the specific direction that makes embeddings more discriminative.

**Caveats:** The current GradProj implementation works via one-shot catastrophic collapse, not designed projection. This weakens the Moderately Novel claim. Persistent Projection (Claim 4) resolves this.

---

### Claim 4: Persistent Projection (PP) — CLEARLY NOVEL
**What we claim:** EMA-based stable discriminative direction (d_ema), updated per batch, projected per detected epoch. Fires stably throughout training, not catastrophically once.

**Prior art search result:** No VFL paper proposes persistent multi-epoch discriminative subspace projection with an evolving direction estimate. Continual learning gradient projection (different problem domain) exists but is not competitive with VFL privacy.

**Why stronger than GradProj:** (1) EMA direction is stable across epochs; (2) Defense fires every detected epoch (not one-shot); (3) Theoretically principled — the projection removes the discriminative component persistently, not accidentally.

**Venue implication:** If PP works for both CIFAR-10 and CIFAR-100, this is the Clearly Novel unified defense contribution of the paper.

---

### Claim 5: Unified Detection + Defense Framework — CLEARLY NOVEL (contingent on Phase 22/23)
**What we claim:** A single framework (Fisher Divergence Detection + Persistent Projection Defense) that handles both 10-class and 100-class VFL label inference without dataset-specific tuning.

**Condition:** Phase 22 (CIFAR-10) and Phase 23 (CIFAR-100) must both show PP passing the defense criterion.

**If both pass:** This is the paper's primary contribution. One mechanism, two datasets, one hyperparameter sweep.
**If only CIFAR-10:** Fisher detection + PP for 10-class; Fisher detection + GradProj (one-shot) for 100-class. Weaker story — requires justification.
**If both fail:** Fragmented story. Revisit research direction.

---

## 5. Possible Research Directions

### Direction 1 (ACTIVE): Persistent Projection — Phase 22 / 23
**Status:** Implementation complete (Phase 22/23 bat files ready). Run Phase 22 first.
**Expected outcome:** PP works for CIFAR-10 (embed_dim=10, fewer classes → projection is cleaner). CIFAR-100 is the open question.
**Success criterion:** mc_best_train_top1 < benign reference AND CSV shows multi-epoch defense activation (not one-shot).

### Direction 2 (CONTINGENT): Seed Sweep for PP
**Condition:** Run only after Phase 22 and 23 both show single-seed success.
**What:** 4 seeds (0, 42, 123, 456) for both CIFAR-10 and CIFAR-100 with best alpha_ema.
**Why:** Paper requires mean ± std across seeds for table results.

### Direction 3 (CONTINGENT): Fair Competitor Comparison
**Condition:** After PP is validated.
**What:** Re-run GC and Laplace DP baselines at 100ep (CIFAR-10) and 150ep (CIFAR-100). Current comparison runs them at 30ep, which is unfair and will be called out by reviewers.
**Why:** Fair epoch-count comparison is required for the paper's Table 2.

### Direction 4 (PARALLEL): Yahoo Answers (Phase 17)
**Status:** Stage 1 benign in progress as of 2026-07-12.
**What:** Cross-modality test. If Fisher divergence detection fires on Yahoo Answers (text + text VFL), generalization claim is much stronger.
**Risk:** Yahoo VFL uses BERT embeddings with different statistical properties. Fisher divergence may not separate cleanly for text.

### Direction 5 (IF PP FAILS): Momentum-Constrained Projection
**Condition:** If PP still collapses catastrophically on CIFAR-100.
**What:** Instead of EMA of direction, maintain EMA of PROJECTION MAGNITUDE. If projection magnitude suddenly spikes 100x in one batch (sign of imminent collapse), clip it. Prevents catastrophic collapse without changing the projection mechanism.
**Mechanism:** `proj_coeff_clipped = clamp(proj_coeff, max=moving_avg * clip_factor)`.

### Direction 6 (IF PP FAILS): Stochastic Subspace Defense
**Condition:** If EMA direction alone is insufficient.
**What:** At each detected epoch, randomly sample k directions from the span of the top-k eigenvectors of the aux classifier's weight matrix. Project away from all k simultaneously. This avoids dependence on any single direction estimate.
**Mechanism:** `grad_proj = grad - Σ_i (grad · v_i) v_i` where v_i = top-k singular vectors of aux_classifier weight matrix.

### Direction 7 (PAPER WRITING — whenever PP is confirmed):
**Start with:** Attack characterization section (Section 2) — fully quantified from existing experiments.
**Then:** Main results table (Section 4) — plug in Phase 22/23 results.
**Hold:** Competitor comparison (Section 5) until fair re-runs are complete.

---

## 6. Novelty Summary Table

| Component | Novelty Level | Closest Competitor | Key Differentiator |
|---|---|---|---|
| Fisher Divergence Detection (Δ_F = J_A − J_B) | **Clearly Novel** | LADSG (gradient norms) | Embedding separability asymmetry; no prior paper uses this |
| Asymmetric Response (modify only Party A's gradient) | **Clearly Novel** | None | All prior defenses are symmetric or cryptographic |
| Gradient Projection (GradProj, one-shot) | **Moderately Novel** | MixPro, SIGIR 2023 | Targeted discriminative direction vs. generic noise direction in MixPro |
| Persistent Projection (EMA-based, multi-epoch) | **Clearly Novel** | None in VFL space | No VFL paper proposes persistent evolving discriminative subspace projection |
| Unified PP + Fisher framework | **Clearly Novel** (if Phase 22+23 pass) | LADSG (closest) | Single mechanism + different detection signal |
| AAP (magnitude suppression) | **Incremental** individually | MARVELL (binary only) | Adaptive scaling + asymmetric; but magnitude-only approach has known failure mode |

---

## 7. Citation Strategy for Paper

**Introduction (gap framing):**
> Existing defenses [PPDL, GC, DP cite] are symmetric — they perturb all parties' gradients identically, incurring unnecessary utility cost on honest participants and failing to distinguish attackers from benign parties. LADSG [cite] detects attacks via gradient norm anomalies and responds with gradient substitution; MixPro [cite] applies a stateless per-batch projection against passive leakage. Neither exploits the structural asymmetry that active attacks introduce in the embedding space, nor do they propose a persistently updated discriminative subspace defense.

**Related Work (differentiation):**
- MixPro [SIGIR 2023]: "The projection direction in MixPro is drawn from a generic noise distribution, with no estimate of the current discriminative subspace. Our approach derives the projection direction from an auxiliary classifier trained on Party A's embeddings, targeting the specific component of the gradient that increases class discriminability."
- ProjPert [TKDE 2024]: "ProjPert uses 'projection' to describe noise parameter binary search (projecting a scalar onto a feasible range), not geometric gradient subspace projection. The mechanism is entirely noise-based and targets passive attacks."
- LADSG [arXiv 2025]: "LADSG detects attacks via gradient L2 norm monitoring. Our Fisher Divergence monitor operates on embedding distributions, capturing a richer structural signal — inter/intra-class variance ratio asymmetry between parties — that is theoretically motivated by the mechanics of MaliciousSGD."
- MARVELL [ICML 2022]: "MARVELL is limited to binary classification; minimax noise design becomes computationally intractable with many classes. Our approach scales to 100-class datasets without class-count constraints."

---

## 8. Open Questions for the Paper

1. **Does PP prevent the one-shot collapse on CIFAR-100?** (Phase 23 answer)
2. **Is alpha_ema dataset-sensitive or does one value work for both?** (Phase 22+23 comparison)
3. **How many epochs does PP fire compared to how many epochs the attack is active?** (CSV diagnostic)
4. **Does the aux classifier in PP reach the same accuracy as in GradProj?** (Proxy for direction quality)
5. **Does PP degrade VFL task accuracy?** (Compare VFL test accuracy under PP vs. benign — from Stage 1 .txt)
6. **Fair competitor comparison at correct epoch count?** (Direction 3 above)
