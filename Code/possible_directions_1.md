# Possible Defense Directions Against Active Label Inference in VFL
## Part 1 of 3: Overview, Current Defense Analysis, Gradient & Embedding Defenses

**Status:** Living research document — expand without restructuring.
**Last major update:** 2026-07-02
**Cross-references:** See `possible_directions_2.md` (Sections 5–8) and `possible_directions_3.md` (Sections 9–15).
**Research log:** `research_log.md`

---

## 0. How to Use This Document

This document is a long-term research notebook, not a one-time brainstorm. Each defense direction has a consistent structure:

- **Core intuition** — the idea in plain language
- **Mathematical formulation** — precise definition
- **Implementation location** — exactly where in the VFL pipeline code to insert it
- **Required code modifications** — which files and functions to change
- **Strengths / Weaknesses / Risks**
- **Novelty estimate** — `Low | Medium | High | Very High`
- **Publication potential** — `Workshop | Mid-tier (USENIX/CCS/NDSS) | Top-tier (S&P/CCS) | Not standalone`
- **Recommended experiments** — what to run to validate or falsify

Sections are numbered globally (1–15) across the three files. File 1 covers Sections 1–4, File 2 covers Sections 5–8, File 3 covers Sections 9–15.

---

## 1. Defense Taxonomy and Master Comparison

### 1.1 Classification of All Defenses

```
VFL Defense Space
├── A. Gradient-Based (operate on ∂L/∂θ or grad_output)
│   ├── A1. Clipping / Norm Bounding
│   ├── A2. Noise Injection (DP)
│   ├── A3. Compression / Sparsification
│   └── A4. Asymmetric Suppression ← CURRENT DEFENSE
│
├── B. Embedding-Level (operate on z_a before or after top model)
│   ├── B1. Additive Noise on z_a
│   ├── B2. Projection / Dimensionality Reduction
│   ├── B3. Adversarial Perturbation
│   └── B4. Dropout / Masking
│
├── C. Fisher / Separability-Aware
│   ├── C1. Fisher Equalization (Fisher_A → Fisher_B)
│   ├── C2. Anti-Fisher Regularization
│   └── C3. Within-Class Variance Inflation
│
├── D. Information-Theoretic
│   ├── D1. Mutual Information Minimization (MINE)
│   ├── D2. Information Bottleneck
│   └── D3. Maximum Entropy Regularization
│
├── E. Adversarial Training
│   ├── E1. GAN-Based Obfuscation
│   ├── E2. Domain Adversarial Training
│   └── E3. Minimax Gradient Games
│
├── F. Optimization-Based (regularization in loss)
│   ├── F1. Orthogonality Constraints
│   ├── F2. Wasserstein Regularization
│   └── F3. Invariant Risk Minimization
│
├── G. Server-Side Architectural
│   ├── G1. Label Permutation / Blinding
│   ├── G2. Gradient Splitting
│   └── G3. Top Model Architecture Changes
│
├── H. Detection-Only (trigger action externally)
│   ├── H1. Fisher Divergence ← CURRENT DETECTION (Phase 1)
│   ├── H2. Centroid Drift
│   └── H3. Gradient Norm Anomaly
│
└── I. Hybrid (combine families)
    ├── I1. Detection + Suppression ← CURRENT FULL SYSTEM
    ├── I2. Detection + Embedding Noise ← RECOMMENDED NEXT
    └── I3. Fisher Equalization + DP Noise
```

### 1.2 Master Comparison Table

| Defense | Utility Cost | Privacy Gain | Novelty | Implementation Effort | Publication Potential |
|---|---|---|---|---|---|
| GC (gradient compression) | Low-Med | Low | Low | None (exists) | Not standalone |
| Laplace DP on grad | Low-Med | Med | Low | None (exists) | Not standalone |
| Current (Asym. suppression) | Low | Low-Med (30ep) / ? (100ep) | Medium | Done | Workshop if 100ep works |
| **Option B: Embedding noise** | Med | Med-High | Medium-High | Low | Mid-tier if tuned well |
| Fisher Equalization | Low | High | High | Medium | Mid-tier |
| MI Minimization (MINE) | High | Very High | High | High | Top-tier |
| Information Bottleneck | Med-High | High | Medium-High | High | Top-tier |
| Adversarial Autoencoder | High | Very High | Medium | Very High | Top-tier |
| Orthogonality Constraints | Low-Med | Med | Medium | Low | Workshop-mid |
| MaliciousSGD-Aware Disruption | Low | High | Very High | Low | Mid-tier |

### 1.3 Priority Decision Tree

```
Start: Phase 4 100-epoch results available?
       │
       ├─ YES: Defense shows >30pp reduction?
       │         ├─ YES → Polish current defense, add seeds, submit
       │         └─ NO  → Implement Option B (Section 2.4.2 / Section 4.1)
       │
       └─ NO  → Run Phase 4 first (run_phase4_cifar10_100ep.bat)
```

---

## 2. Current Defense: Deep Analysis and Immediate Improvements

### 2.1 Architecture Summary

The current defense is `AsymmetricAdaptivePerturbation` in `possible_defenses.py`. It operates
as follows:

**Detection:** Per-epoch Fisher divergence monitoring via `SeparabilityMonitor` in
`characterization_monitor.py`.

```
Fisher criterion (per party):
    J_P = Tr(S_B^P) / Tr(S_W^P)

where:
    S_W^P = sum_{c} sum_{z in class c} (z - μ_c)(z - μ_c)^T   (within-class scatter)
    S_B^P = sum_{c} n_c * (μ_c - μ)(μ_c - μ)^T               (between-class scatter)
    μ_c = mean of Party P embeddings for class c
    μ  = global mean of Party P embeddings

Fisher divergence:
    Δ_F = J_A - J_B
```

**Suppression:** When Δ_F > tau AND epoch > burn_in:

```
scale = max(0.0, 1.0 - alpha * (Δ_F - tau))
grad_output_A ← scale * grad_output_A
```

This suppresses the gradient signal the server sends back to Party A. Party A then computes
p.grad via chain rule on the (reduced) grad_output, and MaliciousSGD amplifies p.grad.

### 2.2 The Tug-of-War: Why It Fails at 30 Epochs

The fundamental conflict is that the defense and the attack target different tensors at
different points in the computation graph:

```
Backward pass timeline:
  [Server computes ∂L/∂z]
       │
       ▼
  [DEFENSE INTERCEPTS: scale grad_output_A by s ∈ [0,1]]
       │  ← defense operates here
       ▼
  [Party A receives: s * ∂L/∂z_a]
       │
       ▼
  [Chain rule: p.grad = (s * ∂L/∂z_a) * ∂z_a/∂θ_A]
       │
       ▼
  [MALICIOUSSGD INTERCEPTS: amplify p.grad by r ∈ [1,5]]
       │  ← attack operates here
       ▼
  [Effective update: θ_A ← θ_A - lr * r * p.grad]
       │
       └── Net effect on θ_A: s * r * (∂L/∂z_a) * (∂z_a/∂θ_A)
```

At 30 epochs, MaliciousSGD's gradient amplification is actively destabilizing early training
(the attack has NOT converged: active inference 23.45% < benign 47.98%). The defense
accidentally stabilizes the divergent gradients → Party A's bottom model converges faster
→ inference goes UP (23.45% → 52.28%).

At 100 epochs, this stabilization effect does not apply: the attack has already converged.
The question becomes whether 92 epochs of reduced task gradient (s ≈ 0.35–0.65 per epoch)
causes enough semantic misalignment to reduce model completion accuracy in Stage 2.

### 2.3 Evidence of Semantic Misalignment Mechanism (From Phase 2)

A critical observation from EXP-007:

```
Fisher divergence UNDER defense: 0.522 (vs 0.444 without defense)
Fisher divergence INCREASED, yet model completion accuracy DROPPED
```

This is the semantic misalignment signature: embeddings become MORE geometrically separated
(clusters are more distinct in z-space) but the clusters no longer correspond to ground-truth
labels. MixMatch SSL fails because pseudo-labels assigned to unlabeled data based on cluster
membership are wrong — high confidence, wrong class.

This mechanism is real. The question at 100 epochs is whether 92 epochs of accumulated
semantic misalignment overcomes MaliciousSGD's 5× re-amplification of the task gradient.

### 2.4 Immediate Improvements (Priority Order)

#### 2.4.1 Option A: Hard Clip at Ceiling

**Core intuition:** The current linear decay `1 - alpha*(Δ_F - tau)` is soft — at high
divergence values, scale is still > 0. Adding a hard clip at zero above a threshold forces
complete gradient suppression for severe attacks.

**Mathematical formulation:**

```
scale =
    1.0                              if Δ_F ≤ tau
    max(0, 1 - alpha*(Δ_F - tau))   if tau < Δ_F ≤ tau + 1/alpha
    0.0                              if Δ_F > tau + 1/alpha
```

**Implementation location:** `possible_defenses.py`, `compute_scale()` method of
`AsymmetricAdaptivePerturbation`.

**Required code change:** 3-line addition to the scale computation method.

**Strengths:**
- Completely suppresses gradient at high divergence values (Δ_F > 0.20 for alpha=1.0, tau=0.10)
- No new hyperparameters (the ceiling is implicit from existing alpha, tau)
- Takes 15 minutes to implement

**Weaknesses:**
- Still operates on grad_output, not p.grad; MaliciousSGD can still amplify the residual signal
- At ceiling, Party A receives zero task gradient from server — may destabilize top model training
- Hard cutoff may cause training instability (abrupt transitions in gradient scale)

**Risks:** If burn_in=8 is too early and Party A receives zero gradient in early training,
the top model may fail to learn a useful representation, degrading VFL task accuracy.

**Novelty:** Low (incremental tweak)
**Publication potential:** Not standalone; this is a hyperparameter experiment, not a contribution

**Recommended experiments:**
1. Run Option A with the Phase 4 100-epoch bat (add 3 lines, rerun)
2. Measure: (a) VFL task accuracy drop, (b) model completion accuracy with hard clip vs soft

---

#### 2.4.2 Option B: Embedding-Level Noise Injection ← RECOMMENDED NEXT STEP

**Core intuition:** Instead of suppressing the gradient AFTER it reaches Party A (where
MaliciousSGD can re-amplify it), inject calibrated Gaussian noise into z_a BEFORE the top
model's forward pass. This corrupts the embedding that gets saved to the checkpoint and
loaded in Stage 2. Party A's bottom model receives gradients computed on noisy embeddings,
nudging it toward representations with higher within-class variance. Since z_a is outside
Party A's computational control at this point, MaliciousSGD has no compensation mechanism.

**Mathematical formulation:**

Let ε ~ N(0, σ²I) where σ = β * max(0, Δ_F - tau).

Modified forward pass:
```
z_a_perturbed = z_a + ε                       (additive calibrated noise)
z = concat(z_a_perturbed, z_b)
ŷ = f_server(z; θ_server)
L = CE(ŷ, y)
```

Backward pass: ∂L/∂z_a is computed w.r.t. z_a_perturbed, not z_a.
The gradient reaching Party A is: (∂L/∂z_a_perturbed) evaluated at z_a + ε.

Over T epochs, Party A's bottom model accumulates updates computed on T different noise
realizations → the bottom model converges to a saddle region where embeddings have
high within-class variance (to be robust to the server's noise), disrupting label alignment.

**Key advantage over Option A:**
```
Option A path:      suppress grad_output → p.grad reduced → MaliciousSGD re-amplifies
Option B path:      perturb z_a → top model trains on noisy z_a → gradient to Party A is
                    evaluated at z_a + ε → Party A's model learns noisy representations
                    → saved .pth checkpoint has semantically degraded embeddings
                    → Stage 2 MixMatch gets noisy z_a → pseudo-labels wrong → inference fails
```

MaliciousSGD cannot compensate for Option B because z_a perturbation happens BEFORE the
top model's forward pass, in the server's computation, not in Party A's computation graph.

**Implementation location:** `vfl_framework.py`, function `simulate_train_round_per_batch()`,
immediately after `output_tensor_bottom_model_a = bottom_model_a(data_a)`.

**Required code changes:**

In `vfl_framework.py`, inside `simulate_train_round_per_batch()`:
```python
# After: output_tensor_bottom_model_a = bottom_model_a(data_a)
# Insert:
if defense is not None and defense.is_active(current_epoch):
    fisher_div = monitor.get_last_divergence()  # from SeparabilityMonitor
    if fisher_div > defense.tau:
        noise_sigma = defense.beta * (fisher_div - defense.tau)
        noise = torch.randn_like(output_tensor_bottom_model_a) * noise_sigma
        output_tensor_bottom_model_a = output_tensor_bottom_model_a + noise
        # Note: do NOT detach — gradient must still flow to Party A's bottom model
```

New hyperparameter: `beta` (noise intensity scale, analogous to `alpha` in Option A).
Reasonable starting range: beta ∈ {0.1, 0.5, 1.0, 2.0}.

**Strengths:**
- Directly disrupts the tensor that gets persisted in the .pth checkpoint
- No compensation mechanism available to MaliciousSGD (operates on server side)
- Gradient still flows to Party A (no complete suppression) — better utility preservation
- Accumulates over 100 epochs — cumulative effect is much stronger than Option A

**Weaknesses:**
- Beta hyperparameter is untested; needs ablation study
- Large noise may degrade VFL task accuracy if top model cannot compensate
- Stochastic effect (different noise each epoch) makes results less reproducible without seeding

**Risks:**
- If beta is too large, the top model trains on very noisy z_a and test accuracy drops.
  Mitigation: start beta=0.1 and increase gradually. Monitor VFL task accuracy.
- If noise is added to BOTH parties equally, the defense signal disappears. Must inject noise
  only into z_a (Party A's adversarial embeddings), not z_b.

**Novelty:** Medium-High — embedding-level noise in response to a detection signal is new
in the VFL defense context (most existing work uses gradient-level DP noise).

**Publication potential:** Mid-tier (USENIX Security / CCS) if combined with Phase 1
detection and ablation study showing meaningful reduction at 100 epochs.

**Recommended experiments:**
1. Add Option B implementation and add new flag `--embedding-noise-beta`
2. Run at beta ∈ {0.1, 0.5, 1.0} with the Phase 4 100-epoch setup (3 seeds each)
3. Measure: (a) VFL top-1 task accuracy, (b) model completion accuracy, (c) Fisher divergence
   under defense (does it still detect even while defending?)
4. Compare to Option A at same epoch count

---

#### 2.4.3 Option C: Centroid Drift Signal (Longer-Term)

**Core intuition:** Fisher divergence is a single-epoch snapshot. Centroid drift measures
how much the class centroids of Party A's embeddings MOVE between consecutive epochs. Under
MaliciousSGD, the optimizer's momentum ratio amplification causes directional centroid drift
toward class-discriminative regions. Under normal training, centroids drift less erratically.

**Mathematical formulation:**

Let μ_c^(t) = mean of Party A embeddings for class c at epoch t.

Centroid drift signal:
```
D_t = (1/C) * sum_c ||μ_c^(t) - μ_c^(t-1)||_2
```

Cross-epoch correlation:
```
ρ_t = corr(D_t, D_{t-1})    (consecutive epoch drift correlation)
```

Under MaliciousSGD: D_t is large and ρ_t is high (consistent directional drift).
Under benign: D_t is smaller and ρ_t is lower (random walk convergence).

Combined detection signal:
```
Δ_combined = w_1 * Δ_F + w_2 * D_t + w_3 * ρ_t
```

**Implementation location:** `characterization_monitor.py` — add centroid tracking alongside
existing Fisher computation. `vfl_framework.py` — pass additional metrics to defense.

**Strengths:**
- More robust than Fisher alone (harder to evade)
- Provides a temporal signal that Fisher misses
- Could enable earlier detection (before Fisher divergence stabilizes)

**Weaknesses:**
- Requires storing centroid history across epochs
- Two additional hyperparameters (w_2, w_3)
- Adds complexity to the already multi-parameter defense

**Novelty:** High — temporal centroid drift as VFL attack signal is not in the literature.
**Publication potential:** Mid-tier if shown to catch attacks that Fisher misses.

**Recommended experiments:**
1. Log centroid drift in existing Phase 3/4 runs (zero code change to training, just logging)
2. Compare drift profiles across benign/active/active+defense conditions
3. If drift clearly separates conditions → implement as detection signal

---

## 3. Gradient-Based Defenses

### 3.1 Gradient Compression (GC)

**Status:** Already implemented in the codebase. Referenced in the literature (Fu et al. 2022
already test this as a baseline defense).

**Core intuition:** Transmit only the top-K% of gradient values (by magnitude) to the other
party. Low-magnitude components carry less information about the label structure.

**Mathematical formulation:**

Let g = grad_output_A (the gradient server sends to Party A).

```
mask = top-k(|g|, k=K%*dim)     (boolean mask, True for top-K% by magnitude)
g_compressed = g * mask
```

**Failure mode against MaliciousSGD:** The attack operates AFTER GC's compression — Party A
receives the compressed gradient and then applies internal p.grad amplification. GC does not
reduce the information content of the gradient relative to MaliciousSGD's amplification.

**Publication potential:** Not standalone — already a baseline in Fu et al. 2022.

---

### 3.2 Gradient Clipping (L2 Norm Bounding)

**Core intuition:** Clip the gradient norm before sending to Party A. Prevents large-magnitude
gradients from carrying strong label information.

**Mathematical formulation:**

```
g_clipped = g * min(1, C / ||g||_2)
```

where C is the clipping threshold (hyperparameter).

**Implementation location:** `vfl_framework.py`, `simulate_train_round_per_batch()`, after
computing ∂L/∂z_a, before sending to Party A.

**Why this is insufficient alone:**
After clipping, MaliciousSGD still amplifies p.grad by up to 5×. If C is set aggressively
enough to hurt the attack, it will also hurt Party B's useful gradient signal (since the
top model sends one combined ∂L/∂z, and clipping affects both parties).

**Novelty:** Low (well-known technique, appears in standard DP literature)
**Publication potential:** Only useful as a baseline/competitor in a comparison table.

**Recommended experiment:** Run GC and gradient clipping at 100 epochs to complete the
defense comparison table in the paper. These are expected to perform worse than our defense.

---

### 3.3 Differential Privacy — Gaussian Mechanism on Gradients

**Core intuition:** Add calibrated Gaussian noise to gradients before sending to Party A,
providing (ε, δ)-DP guarantees. The privacy budget determines the noise scale.

**Mathematical formulation:**

```
g_DP = g + N(0, σ²I)
where σ = C * sqrt(2 * ln(1.25/δ)) / ε    (Gaussian mechanism)
C = gradient clipping threshold (sensitivity)
```

**DP guarantee:** The server's gradient transmission satisfies (ε, δ)-differential privacy
with respect to the label y (assuming label information flows only through the gradient).

**Fundamental problem:** Formal DP for VFL requires careful definition of what is being
protected. If the adversary (Party A) has full access to its own data x_A and can use
auxiliary information, the effective privacy leakage can exceed the formal DP bound.

**Implementation location:** `vfl_framework.py` or new class in `possible_defenses.py`.

**Strengths:**
- Formal privacy guarantee (unlike our current defense)
- Well-understood theory and analysis
- Works against any attack that relies on gradient information

**Weaknesses:**
- Large noise needed for strong privacy → major utility degradation
- Formal DP bound may not directly translate to reduced model completion accuracy
- Does not address Party A's INTERNAL p.grad amplification

**Novelty:** Low — gradient DP in FL is extremely well-studied. Laplace version already exists
in codebase.

**Publication potential:** Not standalone. Useful as a competing baseline.

---

### 3.4 Laplace Differential Privacy on Gradients

**Status:** Already implemented in the codebase. Use as a baseline competitor.

**Mathematical formulation:**

```
g_Lap = g + Lap(0, Δf/ε)
where Δf = L1 sensitivity of the gradient = max ||∂L/∂z_a||_1
```

**Publication potential:** Already a baseline in Fu et al. 2022. Use for comparison table only.

---

### 3.5 Adaptive Gradient Noise (Scale to Gradient Magnitude)

**Core intuition:** Add noise proportional to the gradient magnitude — large gradient = more
noise. This targets the high-magnitude gradients that carry the most label information,
without uniformly degrading small-magnitude updates.

**Mathematical formulation:**

```
σ_t = η * ||g_t||_2 / sqrt(d)      (noise scale proportional to gradient norm)
g_adaptive = g_t + N(0, σ_t² * I)
```

**Why this might be better than fixed-σ DP:**
Fixed σ DP hurts all gradient directions equally. Adaptive noise hurts large-magnitude
directions more, which are exactly the directions MaliciousSGD exploits.

**Novelty:** Low-Medium (appears in the adaptive clipping DP literature)
**Publication potential:** Baseline-level. Worth including in comparison table.

---

### 3.6 Gradient Sparsification with Label-Informed Pruning

**Core intuition:** Instead of keeping top-K% by magnitude, prune gradient dimensions that
correlate most strongly with label information. This requires the server to maintain a
running estimate of which gradient dimensions are label-informative.

**Mathematical formulation:**

Server tracks per-dimension label correlation:
```
ρ_i = corr(g_t[i], one_hot(y_t))    (correlation of gradient dim i with label)
```

Pruning mask (keep low-correlation dimensions):
```
mask_i = 1 if |ρ_i| < threshold else 0
g_pruned = g * mask
```

**Key challenge:** The server has access to labels (y_t), so this computation is feasible.
The gradient dimensions with high |ρ_i| are exactly the ones that leak label information.

**Novelty:** High — label-informed gradient pruning for privacy in VFL is not in the literature.
**Publication potential:** Mid-tier if implemented cleanly with formal analysis.

**Recommended experiments:**
1. Log per-dimension gradient-label correlation during training (zero overhead in Stage 1)
2. Identify whether there's a clear signal (some dimensions highly correlated with labels)
3. If yes, implement pruning mask and test at 100 epochs

---

## 4. Embedding-Level Defenses

These defenses operate directly on Party A's output embedding z_a, either before the top
model's forward pass or after. They are generally more principled than gradient-level defenses
because they target what actually gets saved to the .pth checkpoint and used in Stage 2.

### 4.1 Calibrated Gaussian Noise Injection (Option B, Detailed)

See Section 2.4.2 for full detail. This is the recommended immediate next step.

Key parameter relationships:
```
For a target reduction of R% in Fisher_A:
    sigma ≈ sqrt(Tr(S_W^A) / (C * n))    (approximate, empirically calibrated)
where C is the target Fisher_A reduction factor
```

---

### 4.2 Smooth Sensitivity-Based Noise (Formal DP on Embeddings)

**Core intuition:** Apply the smooth sensitivity framework to add the minimal noise necessary
to hide the label information in z_a, with formal DP guarantees on the embedding output.

**Mathematical formulation:**

Define the embedding function f_A as a mechanism M(D) = z_a.
Smooth sensitivity at scale β:
```
S_β(f_A) = max_{d(x,x')=1} max_S ||f_A(S∪{x}) - f_A(S∪{x'})||_2 * e^{-β*dist(x,x')}
```

For Lipschitz bottom models (which bottom_model_a typically is):
```
||z_a(x) - z_a(x')||_2 ≤ L * ||x - x'||_2
```

where L is the Lipschitz constant of f_A (bounded by spectral norm product of all layers).

**Implementation challenge:** Computing smooth sensitivity exactly requires database sensitivity
analysis, which is expensive. Approximation via clipping is more practical.

**Novelty:** High (smooth sensitivity for VFL embeddings is not studied)
**Publication potential:** Top-tier if formalized properly with privacy-utility tradeoff theorems.
**Estimated implementation effort:** Very High (requires theoretical development).

---

### 4.3 Random Orthogonal Projection of Embeddings

**Core intuition:** Before sending z_a to the top model (conceptually — the top model receives
a projection), apply a random orthogonal transformation R to z_a. If R changes each batch or
each epoch, Party A cannot learn which directions to optimize for label discrimination.

**Mathematical formulation:**

At each training step, sample R ~ Haar(d) (uniform random orthogonal matrix):
```
z_a_proj = R * z_a        (orthogonal transformation preserves norms)
z = concat(z_a_proj, z_b)
ŷ = f_server(z; θ_server)
```

The gradient flowing back is:
```
∂L/∂z_a = R^T * (∂L/∂z_a_proj)    (because R is orthogonal, R^T = R^{-1})
```

Party A receives a rotated gradient — the gradient magnitude is preserved but direction is
scrambled by a random rotation known only to the server.

**Key insight:** MaliciousSGD amplifies p.grad by a scalar ratio (g_t/g_{t-1}). If the
gradient direction rotates randomly each step, the ratio g_t/g_{t-1} measures magnitude
change, not directional coherence. MaliciousSGD's ability to guide embeddings toward
class-discriminative directions is disrupted.

**Critical issue:** For this to be a defense, R must be freshly sampled each step (not fixed).
A fixed R is just a change of basis that Party A can learn to undo.

**Strengths:**
- Norm-preserving (minimal utility impact on top model)
- Computationally cheap (d×d matrix multiply, d=embedding dimension ≈ 64–256)
- No gradient information is destroyed — just directionally scrambled

**Weaknesses:**
- If Party A can observe the pattern of rotations (e.g., through a side channel), it can
  attempt to invert R
- The top model must learn from differently-rotated embeddings each step — may slow convergence
- Effectiveness depends on whether MaliciousSGD's ratio tracking can adapt to direction changes

**Novelty:** High — random rotation as a VFL defense against gradient attacks is not studied.
**Publication potential:** Mid-tier if formally analyzed with rotation group theory.

**Recommended experiments:**
1. Implement R as a fresh random orthogonal matrix per batch (use `torch.linalg.qr` on a
   random Gaussian matrix)
2. Run at 100 epochs; measure VFL task accuracy and model completion accuracy
3. Check: does Fisher divergence still detect the attack under rotation? (It should, since
   Fisher is computed on the rotated embeddings from Party A's perspective)

---

### 4.4 Embedding Dimension Dropout

**Core intuition:** Randomly zero out individual dimensions of z_a before passing to the top
model. Party A cannot know which dimensions will survive → cannot concentrate class information
into surviving dimensions.

**Mathematical formulation:**

```
mask ~ Bernoulli(1-p)^d     (dropout mask, p is dropout probability)
z_a_dropped = z_a * mask / (1-p)    (scaled dropout to preserve expected norm)
```

For defense purposes, p should be higher during attack epochs (triggered by Fisher divergence):
```
p_t = p_base + p_extra * max(0, Δ_F_t - tau)
```

**Key insight:** Unlike training dropout (which averages over masks at test time), the defense
dropout is applied at inference time to the saved embeddings loaded in Stage 2. If the .pth
checkpoint is loaded in `model_completion.py`, z_a is computed by running Party A's bottom
model on test data — the dropout is NOT applied at this stage (the model was just trained
with dropout, not architecturally changed).

**This is a limitation:** The embedding noise defense (Option B) directly corrupts z_a
during training, affecting the bottom model's learned representations. Dropout during training
does not corrupt saved embeddings — it only forces the model to be more redundant.

**Novelty:** Low (dropout is well-understood; this application is not novel)
**Publication potential:** Not standalone.

---

### 4.5 Adversarial Embedding Perturbation (AEP)

**Core intuition:** At each step, solve a small adversarial optimization to find the noise
perturbation δ that maximally reduces the Fisher criterion of Party A's embeddings, then
add that perturbation to z_a before the top model's forward pass. This is a targeted attack
on class separability, not random noise.

**Mathematical formulation:**

Inner optimization (server-side, per batch):
```
δ* = argmin_{||δ||_2 ≤ ε} J(z_a + δ)

where J(z) = Tr(S_B(z)) / Tr(S_W(z))    (Fisher criterion of the batch)
```

This can be approximated by one step of projected gradient descent:
```
δ ← ε * sign(-∇_δ J(z_a + δ))    (one-step adversarial perturbation)
z_a_adv = z_a - δ*              (reduce Fisher = reduce class separability)
```

**Key advantage:** This is a TARGETED perturbation — it specifically minimizes the metric
we care about (Fisher separability), unlike random noise which may or may not reduce J.

**Strengths:**
- Maximally effective per unit of perturbation (compared to random noise)
- Can be combined with Option B (use adversarial perturbation direction, scale by β * Δ_F)

**Weaknesses:**
- Computationally expensive: requires computing Fisher criterion per batch, gradient w.r.t. δ
- Fisher computation requires class labels y_t — the server has these, so feasible
- One-step approximation may not find the true minimum

**Novelty:** High — adversarial embedding perturbation for privacy in VFL is novel.
**Publication potential:** Mid-tier (combines adversarial robustness ideas with VFL defense).

**Recommended experiments:**
1. Implement one-step AEP using PyTorch's autograd on a per-batch Fisher approximation
2. Compare AEP vs random noise (Option B) at the same perturbation budget ||δ|| = σ
3. Measure: which achieves larger Fisher reduction per unit of utility cost?

---

*Continue reading in `possible_directions_2.md` (Sections 5–8: Fisher-Based, Information-Theoretic, Adversarial Training, and Optimization-Based defenses)*
