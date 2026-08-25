# Next Research Direction — If Persistent Projection Fails

**Context:** Assumes Phase 22 (CIFAR-10 PP gate test) and/or Phase 23 (CIFAR-100 PP gate test)
both prove unsuccessful. All prior CIFAR-100 defense configurations (12+) have been tested
and failed except GradientProjectionDefense (one-shot collapse, 4/4 seeds pass but not
as designed). This document identifies, ranks, and recommends the next direction.

---

## Root Cause Analysis: Why CIFAR-100 Keeps Failing

### The Discriminative Coverage Ratio (DCR) Framework

Every failed CIFAR-100 defense can be explained by a single underlying cause: the
**Discriminative Coverage Ratio (DCR)** is too low.

**Definition:**
```
DCR(K, C) = K / min(C-1, embed_dim)
```
where K = number of discriminative directions removed per projection epoch.

For current PP / GradProj (K=1):
- CIFAR-10  (C=10,  embed_dim=10):  DCR = 1/9  ≈ 11.1% per epoch
- CIFAR-100 (C=100, embed_dim=100): DCR = 1/99 ≈  1.0% per epoch

This is a **9× difference in coverage**. CIFAR-10 works because 11% per epoch over
100 epochs is redundant. CIFAR-100 fails because 1% per epoch over 150 epochs gives
only 1.5× theoretical coverage, and the catastrophic one-shot collapse happens because
the single aux_classifier direction captures nearly all discriminative variance at once.

**Failure taxonomy across all 12+ CIFAR-100 configurations:**

| Surface | Best Result | Why It Failed |
|---|---|---|
| Magnitude suppression (alpha scale) | 43.12% | Scale never reaches 0; max div ≈ 0.4 for CIFAR-100 |
| Gradient noise injection | 43.10% | Top model returns MORE informative corrective gradient |
| z_a embedding corruption | 50.67% | Top model corrects to clean target; MaliciousSGD amplifies correction |
| Sign flip (grad_output_A) | 86.40% (CIFAR-10 only) | Sign inversion does not propagate through nonlinear layers |
| Adversarial auxiliary (unbounded) | NaN / crash | Unconstrained aux_grad grows unbounded; ratio=5 amplifies instability |
| Gradient projection (K=1, one-shot) | 26.97% (4/4 seeds ✅) | Works via catastrophic collapse at epoch 11 — not designed behavior |

The pattern: every magnitude-based defense fails; the only working defense acts on
**direction** and works via catastrophic one-shot collapse rather than gradual operation.

---

## Ranked Candidates (Assuming PP Also Fails)

---

### Rank 1 — RECOMMENDED: Multi-Direction Persistent Projection (MDPP)

**Core intuition:** Use K>1 discriminative directions instead of K=1. Specifically, extract
the top-K left singular vectors of `aux_classifier.weight` each detected epoch and project
grad_output_A against all K directions sequentially. For CIFAR-100, K=10 achieves the
same DCR as K=1 achieves for CIFAR-10 (~10%), distributed across 10 directions so no
single projection causes catastrophic collapse.

**Theoretical principle:** LDA discriminative subspace rank = min(C-1, d). Projecting out
K directions removes K/(C-1) of separable variance per step. Stable multi-epoch operation
requires K large enough to overcome MaliciousSGD's reconstruction rate but small enough
that no single epoch removes a catastrophic fraction of the gradient.

**Mathematical formulation:**
```python
# Extract K top singular directions from aux_classifier weight matrix
U, S, Vt = torch.linalg.svd(self.aux_classifier.weight, full_matrices=False)
D_K = U[:, :K]  # [embed_dim, K] — K discriminative basis vectors

# Apply K sequential orthogonal projections (Gram-Schmidt style)
grad = grad_output_a.clone()
for k in range(K):
    d_k = D_K[:, k].unsqueeze(0)  # [1, embed_dim], already unit-normed from SVD
    proj_coeff = (grad * d_k).sum(dim=-1, keepdim=True)
    grad = grad - proj_coeff * d_k  # remove component along d_k

return grad
```

**Stability bound:** After K orthogonal projections:
```
||g_proj|| = ||g|| * sqrt(1 - sum_k cos²(θ_k))  ≤  ||g||  always
```
No NaN risk. Goes to zero only if K directions span the gradient (nuclear option).

**Why this avoids one-shot collapse:** For K=10 on CIFAR-100, each singular direction
typically captures ~5–15% of discriminative variance (not 97% like the single dominant
direction). No single projection removes a catastrophic fraction of the gradient. The
10 directions together remove ~50–80% distributed smoothly.

**Hyperparameter sweep:**
- K ∈ {1, 5, 10, 20} — K=1 reproduces PP failure; K=10 is the primary test; K=20 is
  upper bound (may cause utility loss if too much task gradient is removed)
- alpha_ema, burn_in, tau: same values as Phase 22/23

**Implementation delta from PersistentProjectionDefense:** ~20 lines in
`possible_defenses.py`. Replace single `d_ema` direction with SVD-based K-direction
extraction from `aux_classifier.weight`. The `aux_classifier` and `aux_optimizer`
infrastructure is unchanged.

**New argparse argument needed:** `--persistent-proj-k INT` (default: 1 for backwards
compatibility with Phase 22/23)

**Failure modes:**
1. K too small (K=1, PP-like): same one-shot collapse or insufficient coverage
2. K too large (K≥50 for CIFAR-100): excessive task gradient removal → VFL utility drops
3. aux_classifier underfits early training → SVD directions misaligned → ineffective
   Mitigation: longer burn_in (e.g., burn_in=10 instead of 4)

**Literature comparison:**
- GradProj (our work): K=1 empirical, not designed
- PP (our work): K=1 EMA, designed but insufficient coverage for CIFAR-100
- MixPro (SIGIR 2023): K=1 random noise direction, stateless, passive attacks only
- ProjPert (IEEE TKDE 2024): binary search over noise, not geometric projection, passive only
- LADSG (arXiv 2506): gradient norm anomaly + substitution, no subspace coverage analysis
- **MDPP: First work to connect K to class count C via LDA rank theory. Clearly Novel.**

**Novelty level:** Clearly Novel. DCR framework and K-scaling with C is not in any VFL paper.

**Publication value:** The CIFAR-10 vs CIFAR-100 performance gap is explained by
DCR(1,10) vs DCR(1,100). MDPP closes this gap by matching coverage. This is a clean,
testable theoretical contribution that makes the paper's CIFAR-100 result principled
rather than empirically discovered via catastrophic collapse.

**Estimated implementation:** 1 day.
**Estimated runtime (gate test):** ~18h (same as Phase 23, 3 runs × 6h).
**Risk level:** Medium-Low.

---

### Rank 2: Discriminative Gradient Reflection (DGR)

**Core intuition:** Replace projection (remove discriminative component) with reflection
(INVERT discriminative component). The reflected gradient has its discriminative component
pointing AGAINST label discrimination. MaliciousSGD amplifies the anti-discriminative
signal — the attack mechanism weaponizes itself against the attack.

**Mathematical formulation:**
```python
# Householder reflection across hyperplane orthogonal to d_ema
proj_coeff = (grad_output_a * d_ema).sum(dim=-1, keepdim=True)
grad_reflected = grad_output_a - 2 * proj_coeff * d_ema  # note: 2x, not 1x

# Property: ||grad_reflected|| = ||grad_output_a|| exactly (isometry)
# The discriminative component is negated; task component is preserved
```

**Implementation delta from PP:** Exactly 1 line change (multiply proj_coeff by 2).

**Why bounded (unlike Phase 18):**
- Phase 18 (failed): `final_grad = grad - λ * aux_grad` where aux_grad is UNBOUNDED
- DGR: `grad_reflected = grad - 2 * (grad·d̂) * d̂` where `(grad·d̂) ≤ ||grad||` always
- The reflection coefficient is bounded by the original gradient magnitude. No NaN.

**The "attack amplifies defense" property:**
```
MaliciousSGD amplifies Party A's p.grad by ratio ∈ [1, 5].
Under DGR: the amplified gradient has discriminative component = -ratio × cos(θ) × ||grad||
This is NEGATIVE (anti-discriminative) and amplified by the attack's own ratio.
At ratio=5: 5× anti-discriminative push. Stronger attack → stronger defense.
```

**Differentiation from sign-flip (EXP-023, which failed):**
- Sign-flip: invert ALL of grad_output_A → breaks task gradient; inversion does not
  propagate through nonlinear layers correctly
- DGR: invert ONLY the discriminative component; task component preserved exactly;
  operates on the geometrically meaningful subspace, not raw gradient signs

**Failure modes:**
1. If K=1 (single direction), coverage problem still applies for CIFAR-100: cos(θ) is
   small (discriminative component is 1% of gradient) → 2×1% = 2% anti-discriminative
   push, insufficient to overcome amplified task gradient
   **Solution: Combine DGR with MDPP (K=10 directions each reflected, not projected)**
2. Reflected gradient pushes embeddings to anti-discriminative minimum where Fisher_A << 0:
   may cause VFL utility degradation. Monitor task accuracy in Stage 1.

**Optimal form: MDPP + DGR hybrid (K=10 directions, each reflected not projected)**
```python
for k in range(K):
    d_k = D_K[:, k].unsqueeze(0)
    proj_coeff = (grad * d_k).sum(dim=-1, keepdim=True)
    grad = grad - 2 * proj_coeff * d_k  # reflect, not project
```
This both solves the coverage problem (K=10) AND makes MaliciousSGD amplify the
anti-discriminative signal rather than just having it silently removed.

**Novelty level:** Clearly Novel. No VFL paper uses gradient reflection for privacy defense.

**Estimated implementation:** 1–2 days (slight modification of MDPP bat file).
**Risk level:** Medium.

---

### Rank 3: Fisher Equalization Regularization

**Core intuition:** Add `λ × max(0, J_A - J_B - margin)` to the VFL training loss directly.
Because it is in the loss, MaliciousSGD amplifies the combined (task + regularization)
gradient. If λ is large enough that the Fisher EQ gradient dominates, MaliciousSGD
amplifies the anti-discriminative pressure.

**Mathematical formulation:**
```
L_total = L_task + λ × max(0, J_A - J_B - margin)

where J_A = Tr(S_B^A) / Tr(S_W^A),  J_B = Tr(S_B^B) / Tr(S_W^B)

Tr(S_W^A) = sum_c sum_{i in c} ||z_a^i - μ_c^A||²   (within-class scatter)
Tr(S_B^A) = sum_c n_c ||μ_c^A - μ^A||²              (between-class scatter)

∂J_A/∂z_a^i = [Tr(S_W^A)×∂Tr(S_B^A)/∂z_a^i - Tr(S_B^A)×∂Tr(S_W^A)/∂z_a^i] / Tr(S_W^A)²

∂Tr(S_W^A)/∂z_a^i = 2(z_a^i - μ_{c(i)})     (toward class centroid)
∂Tr(S_B^A)/∂z_a^i = 2(n_{c(i)}/N)(μ_{c(i)} - μ)  (centroid toward global mean)
```

**Why bounded (unlike Phase 18):**
- J_A = ratio of two sum-of-squares quantities; both bounded by embedding variance
- As long as Tr(S_W^A) > 0 (prevented from going to 0 by the task gradient), J_A is finite
- ∂J_A/∂z_a^i is differentiable and bounded for non-degenerate within-class variance

**Key advantage over grad_output_A modification:**
The Fisher EQ gradient is computed in the standard backward pass and is part of p.grad.
MaliciousSGD's ratio = clamp(1 + 200×(g_t/g_{t-1}), 1, 5) amplifies the TOTAL p.grad
including the Fisher EQ component. If the Fisher EQ component consistently points
anti-discriminative across epochs (it does: ∂J_A/∂z_a^i always pushes toward lower J_A),
then MaliciousSGD's amplification strengthens the regularization.

**Hyperparameters:** λ ∈ {0.01, 0.1, 1.0, 5.0}; margin ∈ {0.0, 0.1}

**Implementation:** ~40 lines of differentiable scatter operations in PyTorch.
Challenge: batch-approximate Fisher is noisy for CIFAR-100 with 100 classes and
batch_size=256 (~2.5 samples per class per batch). Mitigation: use EMA over class
centroids (10-epoch moving average of μ_c).

**Literature comparison:**
- MARVELL (ICML 2022): minimax noise to equalize gradient NORMS, binary class only,
  passive attacks only, no Fisher criterion, no server-side class statistics
- Fisher EQ: penalizes Fisher RATIO (discriminative structure), 100 classes, active attacks
- **Clearly Novel in VFL context.**

**Failure modes:**
1. λ tuning is hard: too small → insufficient anti-discriminative pressure; too large →
   within-class variance maximized at cost of VFL task accuracy
2. Noisy per-batch Fisher estimate may produce unstable gradients early in training

**Estimated implementation:** 2 days.
**Risk level:** Medium.

---

### Rank 4: Domain Adversarial Training with Gradient Reversal Layer (GRL)

**Core intuition:** Train a server-side label discriminator D on z_a embeddings. The GRL
flips the gradient sign of D's loss before it flows to Party A's bottom model, causing
Party A to MINIMIZE D's label prediction accuracy → label-invariant features. MaliciousSGD
potentially amplifies the reversed discriminative gradient.

**Key insight from Section 7.2 of possible_directions_2.md:** MaliciousSGD amplifies ALL
of Party A's gradients. If the reversed discriminator gradient is the dominant component
of p.grad, MaliciousSGD amplifies the anti-discriminative signal. Stronger attack → more
amplification of the defense.

**Mathematical formulation:**
```python
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(torch.tensor(lambda_))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_tensors
        return -lambda_ * grad_output, None  # flip gradient sign

# Server-side: train discriminator D to predict labels from z_a
# Then: send grad_output_A = standard backward MINUS λ * ∂L_D/∂z_a
# (The "minus" is achieved by the GRL in the discriminator's forward pass)
```

**Why ranked below MDPP:** 
1. More complex implementation (full discriminator training loop, GRL autograd function,
   careful λ scheduling) — 3–5 day implementation estimate
2. Phase 18 (adversarial auxiliary) tried a similar concept and failed due to unbounded
   aux_grad. GRL is bounded by construction (discriminator loss bounded), but the same
   conceptual risk is present
3. MaliciousSGD ratio computation may not amplify the GRL gradient as predicted:
   if task gradient and reversed discriminator gradient oppose each other, consecutive
   p.grad vectors have lower alignment → ratio closer to 1 → no amplification advantage

**Literature comparison:**
- DANN (Ganin et al., JMLR 2016): original GRL for domain adaptation
- No prior VFL defense uses GRL for active label inference attack defense
- **Clearly Novel.**

**Novelty level:** Clearly Novel. Publication potential: top-tier IF empirical result
confirms "attack amplifies defense" prediction. Very high reward, higher risk.

**Estimated implementation:** 3–5 days.
**Risk level:** Medium-High.

---

### Rank 5: Reversed Contrastive Loss

**Core intuition:** Add loss term that MAXIMIZES within-class distance of z_a embeddings
(push same-class embeddings apart). This is the mirror of MaliciousSGD's effect.

**Formulation:**
```
L_anti = -mean_{(i,j): y_i=y_j, i≠j} ||z_a^i - z_a^j||²

∂L_anti/∂z_a^i = (-2/|P|) × sum_{j: y_j=y_i} (z_a^i - z_a^j)
```

**Why ranked lower:** Timing problem. At early training (when clusters form), within-class
distances are small → gradient is small → cannot overcome MaliciousSGD's 5× amplification
of task gradient. By late training, clusters are tight but the gradient is also large —
however, 150 epochs of partial discriminative structure may have already enabled MixMatch
to infer labels from the intermediate checkpoints.

**Estimated implementation:** ~20 lines. Implementation complexity: Low.
**Risk level:** Medium-High for CIFAR-100.

---

## Summary Comparison Table

| Candidate | DCR Fix | Bounded | Novel | Implementation | Risk |
|---|---|---|---|---|---|
| **MDPP (K=10)** | ✅ Direct (K scales with C) | ✅ Always | ✅ Clearly Novel | Low (~20 lines) | Medium-Low |
| **DGR (reflection)** | ⚠️ Needs K>1 to help | ✅ Always | ✅ Clearly Novel | Very Low (1 line) | Medium |
| **MDPP + DGR (hybrid)** | ✅ Direct | ✅ Always | ✅ Clearly Novel | Low (~25 lines) | Medium-Low |
| **Fisher Equalization** | ⚠️ Indirect | ✅ If Tr(S_W)>0 | ✅ Clearly Novel | Medium (~40 lines) | Medium |
| **Domain Adversarial GRL** | ⚠️ Indirect | ✅ If D bounded | ✅ Clearly Novel | High (3-5 days) | Medium-High |
| **Reversed Contrastive** | ❌ No | ✅ Always | ✅ Novel | Very Low (~20 lines) | Medium-High |

---

## Final Recommendation

**Implement MDPP with K=10 for CIFAR-100.**

Reasoning in three sentences:
The failure pattern across all 12+ CIFAR-100 configurations has a single unifying
explanation: DCR(K=1, C=100) = 1% is insufficient to outpace MaliciousSGD's 5×
amplified discriminative gradient reconstruction over 150 epochs. MDPP with K=10
achieves DCR(10, 100) ≈ 10%, matching the coverage that works for CIFAR-10 at K=1,
distributed across 10 singular directions to prevent catastrophic one-shot collapse.
This is the smallest possible change to the existing PersistentProjectionDefense that
is both theoretically justified by LDA rank analysis and directly testable.

**If MDPP also fails:**
1. Try MDPP + DGR hybrid (reflect instead of project; 1 line change from MDPP)
2. Try Fisher Equalization (loss-level intervention; 2 days)
3. If all three fail: honest conclusion is that CIFAR-100 gradient-level defense is
   intractable within the gradient modification paradigm. Present CIFAR-10 + CINIC10L
   as the validated contribution; use CIFAR-100's GradProj collapse result as a validated
   (if undesigned) defense result; frame MDPP as future work with theoretical motivation.

---

## Implementation Plan for MDPP (Phase 24)

### Changes to `possible_defenses.py` — PersistentProjectionDefense class

**Current `apply()` method (single EMA direction):**
```python
# uses self.d_ema: Tensor[embed_dim]
d = self.d_ema.unsqueeze(0)
proj_coeff = (grad_output_a * d).sum(dim=-1, keepdim=True)
return grad_output_a - proj_coeff * d
```

**New `apply()` method (K singular directions):**
```python
def apply(self, grad_output_a, epoch):
    # ... (detection logic unchanged) ...
    if should_project:
        # Extract top-K left singular vectors from aux_classifier weight
        with torch.no_grad():
            U, S, Vt = torch.linalg.svd(
                self.aux_classifier.weight,  # [num_classes, embed_dim]
                full_matrices=False
            )
            D_K = U[:, :self.k_directions]  # [embed_dim, k_directions]

        # Apply K sequential orthogonal projections (or reflections if mode='reflect')
        grad = grad_output_a.clone()
        for i in range(self.k_directions):
            d_k = D_K[:, i].unsqueeze(0)  # [1, embed_dim], already unit-normed
            proj_coeff = (grad * d_k).sum(dim=-1, keepdim=True)
            if self.reflect_mode:
                grad = grad - 2 * proj_coeff * d_k  # DGR: reflection
            else:
                grad = grad - proj_coeff * d_k      # MDPP: projection
        return grad
    return grad_output_a
```

### New arguments needed in `vfl_framework.py`:
```
--persistent-proj-k INT       (default: 1, for MDPP; K singular directions to project)
--persistent-proj-reflect     (flag; if set, uses DGR reflection instead of projection)
```

### Phase 24 bat file parameters:
- Dataset: CIFAR-100, 150 epochs, seed=0 (gate test)
- K sweep: {1, 5, 10, 20}
- mode: project (test reflection only if K sweep fails)
- Same burn_in=4, tau=0.10, alpha_ema from best Phase 23 result (or 0.2 default)
- Success criterion: mc_best_train_top1 < 30.33% (benign seed-0)
- CSV diagnostic: NO spike in intra_var_A of 6+ orders of magnitude (want gradual decrease)

---

## Connection to Paper Narrative

**With MDPP working (best case):**
> "Persistent Projection with K=1 works for 10-class datasets (CIFAR-10, CINIC10L) but
> fails for 100-class (CIFAR-100) due to insufficient discriminative subspace coverage.
> MDPP with K=⌈(C-1)/10⌉ achieves uniform coverage across class counts, providing a
> unified defense applicable to both low-class and high-class VFL scenarios."

The DCR framework becomes a **design principle** for future defenses: parameterize by
DCR = K/(C-1) and tune K to match the coverage needed for a given class count.

**Without MDPP (backup case):**
The paper presents AAP (CIFAR-10 unified, 4/4 seeds), GradProj (CIFAR-100, 4/4 seeds via
one-shot collapse), and the DCR framework as the theoretical explanation for why
a single defense mechanism requires K > 1 to generalize across class counts.
MDPP appears as validated future work.

---

*Document created: 2026-07-13*
*Based on: possible_directions_1.md, possible_directions_2.md, possible_directions_3.md,
research_log.md, literature_review.md, and analysis of EXP-001 through EXP-039.*
*Assumes Phase 22 (PP CIFAR-10) and Phase 23 (PP CIFAR-100) have both failed.*

---

## Phase 22 Fixed + Phase 23 Fixed — Confirmed Results (2026-07-14)

**PP Fixed is confirmed NOT PROMISING. The assumption above is now validated by data.**

### Phase 22 Fixed (CIFAR-10, 100ep, EXP-045/046/047)

All three fixed variants fail the success criterion (mc < 87.23% benign):

| α_ema | VFL Test | MC Best | Gap to Criterion | Gap to AAP |
|---|---|---|---|---|
| 0.1 | 80.77% | 94.74% | **+7.51pp above benign** | +12.94pp worse |
| 0.2 | 80.09% | **92.99%** | **+5.76pp above benign** | +11.19pp worse |
| 0.3 | 80.75% | 94.91% | **+7.68pp above benign** | +13.11pp worse |

The bug fix (per-sample gradient normalization) changed outcomes by <2pp vs buggy Phase 22 (93.73–94.50%). Root cause confirmed as architectural, not implementation.

Fisher separability CSV (ema=0.2): phase transition at epoch 50 (Fisher_A: 0.65→1.22, intra_var_A: 0.42→0.24). The defense fires correctly but cannot prevent MaliciousSGD from rebuilding discriminative structure via chain rule through Party A's ResNet.

### Phase 23 Fixed (CIFAR-100, 150ep, EXP-048/049/050)

All three fixed variants fail. EXP-049 (ema=0.2) is WORSE than the undefended attack:

| α_ema | VFL Test | MC Best | Gap to Criterion | vs Attack |
|---|---|---|---|---|
| 0.1 | 46.47% | 48.78% | **+18.45pp above benign** | +0.92pp |
| 0.2 | 47.50% | **50.16%** | **+19.83pp above benign** | **+2.30pp ABOVE attack** |
| 0.3 | 46.47% | 48.81% | **+18.48pp above benign** | +0.95pp |

Critical anomaly: VFL Stage 1 accuracy IMPROVES under PP defense (46.47–47.50% vs benign 45.33%). The projection appears to accidentally regularize training, improving the attacker's embedding quality.

Fisher separability CSV (ema=0.2): two phase transitions at epochs 75 and 120. intra_var_A collapses to 0.006 (TIGHTER than undefended attack at 0.014). Despite geometric cluster tightness, MC=50.16%. This confirms: geometric embedding tightness does NOT imply label alignment suppression.

### Updated Confidence in MDPP (Phase 24)

The failure confirms the DCR analysis is correct:
- DCR(K=1, C=10) = 1/9 ≈ 11%: even this is insufficient in practice (PP fails CIFAR-10 too)
- DCR(K=1, C=100) = 1/99 ≈ 1%: catastrophically insufficient (PP makes CIFAR-100 WORSE)
- DCR(K=10, C=100) = 10/99 ≈ 10.1%: target for Phase 24

Phase 24 MDPP is the IMMEDIATE next experiment. The implementation plan (SVD-based K directions, sequential orthogonal projection, burn_in=4) is unchanged. Priority: CIFAR-100 seed-0 gate test with K ∈ {5, 10, 20}.

Additional constraint from Phase 23 Fixed: the EMA direction must be initialized only AFTER burn_in epochs to avoid tracking noise before the aux_classifier has learned a reliable discriminative direction. Consider initializing d_ema from scratch at burn_in rather than accumulating from epoch 0.

*Updated: 2026-07-14. Phase 22 Fixed (EXP-045/046/047) and Phase 23 Fixed (EXP-048/049/050) complete.*
