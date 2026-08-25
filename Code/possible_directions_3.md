# Possible Defense Directions Against Active Label Inference in VFL
## Part 3 of 3: Server-Side, Detection-Only, Hybrid Defenses, Novel Hypotheses, Comparison, and Roadmap

**Status:** Living research document — expand without restructuring.
**Last major update:** 2026-07-09
**Cross-references:** `possible_directions_1.md` (Sections 1–4) | `possible_directions_2.md` (Sections 5–8)

---

## 9. Server-Side Architectural Defenses

These defenses modify how the server participates in training — they do not require changes
to Party A's or Party B's code, making them transparent and deployable without attacker
cooperation.

### 9.1 Label Permutation / Label Blinding

**Core intuition:** The server can choose to train with PERMUTED labels on Party A's side.
If Party A receives gradients computed on (x_A, π(y)) instead of (x_A, y), the embeddings
it learns will correlate with the permuted labels — which are uninformative about the true y.

**Mathematical formulation:**

Let π: Y → Y be a permutation of the label space. The server computes:
```
L_blinded = CE(ŷ(z_a, z_b), π(y))
```

Party A receives gradients that guide it toward encoding π(y) rather than y. Party B
receives gradients that guide it toward encoding y (to the extent it can distinguish from π(y)).

**Problem:** If the server trains on π(y) consistently, the TOP MODEL also learns to predict
π(y), destroying task utility. This cannot be a permanent defense.

**Viable variant — Randomized Label Masking:**
Randomly permute labels for a subset of batches (e.g., 30% of batches) and train normally
on the rest. The average gradient Party A sees is:
```
E[∂L/∂z_a] = (1-p) * ∂L_true/∂z_a + p * ∂L_perm/∂z_a
```

where p is the permutation probability. This adds systematic noise to Party A's gradient
signal that is label-structure-destroying, unlike random DP noise which is label-neutral.

**Another variant — Cyclic Label Rotation:**
Every K epochs, rotate the label mapping for Party A's gradient only. The top model is
trained with true labels, but the gradient backpropagated to Party A reflects a rotated
label space.

**Implementation challenge:** The gradient to z_a cannot be cleanly separated from the gradient
to z_b inside the top model — they both flow from the same loss. However, since the top model
computes ∂L/∂z (where z = concat(z_a, z_b)), the server can modify z_a's share of the gradient:

```python
# Server-side gradient modification:
z.requires_grad_(True)
y_hat = top_model(z)
loss = CE(y_hat, y)
loss.backward()

# Split gradient
grad_z_a = z.grad[:, :half]
grad_z_b = z.grad[:, half:]

# Permute ONLY the label-driven information in grad_z_a
# (This requires knowing which gradient directions correspond to label information)
```

**Limitation:** The gradient to z_a from a loss computed on TRUE y still carries correct
label information. Simply permuting y for the backward pass requires computing a separate
backward pass — more expensive but feasible.

**Novelty:** Medium (label blinding has been discussed in FL but not formalized for VFL)
**Publication potential:** Workshop — more of an ablation study topic.

---

### 9.2 Asymmetric Top Model Architecture

**Core intuition:** Design the top model such that z_a's contribution is passed through a
NOISE LAYER before the classification head, while z_b passes through directly. The noise
layer adds structured perturbation to z_a's pathway specifically.

**Modified top model:**

```
Standard:    [z_a | z_b] → MLP → softmax
Modified:    [NoiseLayer(z_a) | z_b] → MLP → softmax

NoiseLayer(z_a) = z_a + σ * ε,   ε ~ N(0, I),   σ detected from Fisher divergence
```

**Why this is different from Option B (Section 2.4.2):**
- Option B: add noise before top model's forward pass (outside the model)
- Section 9.2: embed the noise INSIDE the top model architecture as a fixed layer

The architectural integration has an advantage: the top model's weights are trained to be
robust to the noise (they compensate for z_a noise using z_b). This allows larger σ with less
task degradation. However, the noise is fixed at training time and the model adapts to it.

**Implementation:** Add a `NoisyInputLayer` before the top model's first linear layer that
applies only to the first `half` dimensions (Party A's embedding dimensions).

**Novelty:** Low-Medium (noisy input layers are a standard regularization trick)
**Publication potential:** Not standalone.

---

### 9.3 Top Model Gradient Splitting (Asymmetric Backpropagation)

**Core intuition:** The server can apply DIFFERENT backward passes for Party A and Party B.
It can send Party A a gradient that was computed on a MODIFIED version of z_a (e.g., z_a
replaced by random noise) while keeping Party B's gradient correct.

**Precise formulation:**

Standard backward:
```
grad_output_A = ∂L(top_model(z_a, z_b), y) / ∂z_a
```

Modified backward (split computation):
```
grad_output_A_modified = ∂L(top_model(z_a_proxy, z_b), y) / ∂z_a_proxy

where z_a_proxy = some modified version of z_a (e.g., random noise, benign reference, or
                  a projection that removes label-discriminative directions)
```

This gives Party A a gradient that was computed relative to a different z_a, making it
effectively a "confused" gradient that doesn't faithfully represent the loss surface.

**Key advantage:** This is invisible to Party A — it still receives a gradient tensor,
just one that was computed on a different manifold. MaliciousSGD amplifies this confused
gradient, amplifying the confusion.

**Variants:**
1. z_a_proxy = N(0, I) (pure random noise)
2. z_a_proxy = z_a projected onto a label-blind subspace
3. z_a_proxy = weighted average of z_a with a random sample from a benign reference distribution

**Novelty:** High — asymmetric gradient splitting in VFL is not published.
**Publication potential:** Mid-tier if the proxy choice is formalized.

---

### 9.4 Secure Aggregation with Homomorphic Encryption (Long-Term)

**Core intuition:** Use HE to allow the server to compute the top model's loss on ENCRYPTED
embeddings, such that it cannot read z_a in plaintext. Without plaintext access to z_a, the
server cannot compute the Fisher divergence detection signal.

**Trade-off:** This prevents the server from DETECTING the attack (it can't see z_a), but
the attack also can't use plaintext z_a. The question is whether MaliciousSGD can function
on encrypted embeddings — likely not, since it requires the actual gradient tensor.

**Why this is long-term:**
- HE adds 100–10000× overhead on computation
- CKKS (leveled HE for approximate arithmetic) makes neural network forward passes possible
  but slow
- Gradient computation through HE for deep models is not yet practical (2026)

**Novelty:** Low (HE for FL is a well-studied area; practical application is the challenge)
**Publication potential:** Not suitable for this project's timeline. Note for future work.

---

## 10. Detection-Only Approaches

The Phase 1 detection component is already solid. These are extensions and alternatives.

### 10.1 Current Fisher Divergence Detection (Reference)

**Already implemented:** `characterization_monitor.py` computes Fisher criterion per epoch
for both parties. Fisher divergence Δ_F = J_A - J_B.

**Phase 1 results:**
```
CIFAR10:  Δ_F(benign)=0.120, Δ_F(passive)=0.175, Δ_F(active)=0.564 → gap=0.444
CIFAR100: Δ_F(benign)=0.050, Δ_F(passive)=0.070, Δ_F(active)=0.181 → gap=0.131
```

**Limitation:** Single-epoch snapshot; no temporal context; requires access to ground truth
labels y at detection time (server has labels, so feasible).

---

### 10.2 Temporal Centroid Drift Detection

**Core intuition:** Track how class centroids of z_a move between consecutive epochs.
Under MaliciousSGD, centroids drift rapidly toward discriminative positions. Under benign
training, centroids drift more slowly and less directionally.

**Signal definition:**

```
Centroid at epoch t: μ_c^(t) = mean of z_a for class c at epoch t
Drift at epoch t:    δ_c^(t) = ||μ_c^(t) - μ_c^(t-1)||_2
Total drift:         D^(t) = mean_c δ_c^(t)
Drift persistence:   ρ^(t) = corr(δ_c^(t), δ_c^(t-1))    (across classes)
```

**Expected behavior:**
- Benign: D^(t) decreases over epochs (convergence), ρ^(t) ≈ 0 (random walk)
- Active: D^(t) is large and maintains non-zero value; ρ^(t) > 0 (consistent direction)

**Combined detection signal:**
```
Score^(t) = w_1 * Δ_F^(t) + w_2 * D^(t) + w_3 * ρ^(t)
Attack detected if Score^(t) > threshold (Bonferroni corrected for multiple testing)
```

**Implementation:** Add centroid tracking to `characterization_monitor.py`.
Store `self.prev_centroids` dictionary; compute drift on each `monitor()` call.

**Novelty:** High — temporal centroid drift as a VFL attack detection signal is not published.
**Publication potential:** Standalone contribution in a detection paper, or complement to
the Fisher divergence detection for a richer detection system.

**Recommended experiments:**
1. Add centroid tracking to existing runs (no retraining needed — add to characterization scripts)
2. Run Phase 4 with centroid drift logging
3. Test: can centroid drift detect the attack EARLIER than Fisher divergence?
   (Early detection = fewer epochs of attack damage before defense kicks in)

---

### 10.3 Gradient Norm Ratio Anomaly Detection

**Core intuition:** The party-gradient norm ratio ||∂L/∂z_a||_2 / ||∂L/∂z_b||_2 should
remain roughly constant under benign training (both gradients come from the same loss).
Under MaliciousSGD, Party A's INTERNAL gradient amplification causes its weights to update
much more aggressively, which manifests as a different gradient norm pattern on the SERVER'S
next round of z_a (since Party A's weights converged faster).

**Note:** The server cannot observe p.grad (which MaliciousSGD modifies) — only its own
∂L/∂z_a. But the EFFECT of MaliciousSGD manifests in z_a's evolution: after Party A's
weights update more aggressively, the next epoch's z_a will be different from what it would
be without MaliciousSGD.

**Indirect detection signal:**
```
Ratio^(t) = ||∂L/∂z_a^(t) - ∂L/∂z_a^(t-1)||_2 / ||∂L/∂z_b^(t) - ∂L/∂z_b^(t-1)||_2
```

Under MaliciousSGD: Party A's gradient changes more between epochs (due to aggressive weight
updates). Under benign: ratio ≈ 1.

**Novelty:** Medium-High
**Publication potential:** Detection-only contribution; useful in combination with Fisher.

---

### 10.4 Silhouette Score Monitoring

**Core intuition:** The silhouette score measures cluster quality for each embedding sample.
Under MaliciousSGD, silhouette scores for Party A's embeddings should increase rapidly
(tight, well-separated clusters). Under benign training, increase is slower.

**Already partially tracked:** The characterization monitor CSV files already track silhouette
scores. Re-analyze Phase 1 CSVs to verify this signal.

**Signal:**
```
Silhouette_diff^(t) = SC(z_a^(t)) - SC(z_b^(t))
```

Under active attack: Silhouette_diff increases rapidly; under benign: small and stable.

**Action:** Re-read the Phase 1 CSV files and check silhouette trajectories before implementing
any new code.

---

### 10.5 Embedding Alignment Test (Server-Side Statistical Test)

**Core intuition:** Fit a linear SVM on z_a at each epoch (server has labels). If the SVM
accuracy improves rapidly, the embeddings are becoming linearly separable — attack detected.

**Signal:**
```
SVM_acc^(t) = accuracy of linear SVM trained on (z_a^(t), y) on held-out validation set
Rate = SVM_acc^(t) - SVM_acc^(t-1)    (rate of improvement)
```

Under MaliciousSGD: SVM_acc improves faster than under benign training.
Detection: `Rate > rate_threshold OR SVM_acc^(t) > absolute_threshold`

**Advantage over Fisher:** SVM accuracy is a direct measure of linear separability, which
is exactly what model completion exploits. Fisher criterion is a proxy; SVM accuracy is
the direct target.

**Disadvantage:** Training a linear SVM per epoch adds compute overhead (O(n²) for SVM).
For n=50000 and d=64, this may be acceptable with scikit-learn's SGD-based LinearSVC.

**Novelty:** Medium
**Publication potential:** Useful as a stronger detection baseline; not standalone novel.

---

## 11. Hybrid Defense Architectures

Hybrid architectures combine a DETECTION component with a SUPPRESSION or DISRUPTION component.
The current defense is already hybrid (Fisher detection + gradient suppression). These are
alternatives.

### 11.1 Current System: Fisher Detection + Asymmetric Gradient Suppression

**Status:** Implemented, not validated at 100 epochs.
**Mechanism:** Detect high Δ_F → suppress grad_output_A by scale factor.
**Known failure mode:** At 30 epochs, MaliciousSGD re-amplifies the suppressed gradient by 5×.
**Potential:** May work at 100 epochs if cumulative semantic misalignment overcomes re-amplification.
**Next step:** Run Phase 4 to determine.

---

### 11.2 Fisher Detection + Embedding Noise (TESTED — FAILED FOR CIFAR-100)

**Combines:** Phase 1 Fisher divergence detection + Option B embedding noise injection.

**Pipeline:**
```
Epoch t:
1. SeparabilityMonitor computes Δ_F^(t) = J_A^(t) - J_B^(t)
2. If epoch > burn_in AND Δ_F^(t) > tau:
   a. σ_t = beta * (Δ_F^(t) - tau)
   b. z_a_noisy = z_a + N(0, σ_t² * I)
   c. top_model forward pass uses z_a_noisy instead of z_a
   d. backward pass computes gradient at z_a_noisy → Party A receives noisy gradient signal
3. Party A's bottom model learns to produce embeddings with high within-class variance
   (to be robust to the server's noise from its perspective)
4. Over 100 epochs, embeddings become semantically misaligned despite geometric separation
5. Stage 2 model completion: MixMatch pseudo-labels unreliable → low inference accuracy
```

**Why this is better than 11.1:**
- MaliciousSGD cannot compensate: perturbation happens before the gradient computation
- The .pth checkpoint contains a bottom model that learned under consistent noise → its
  embeddings are inherently noisy when evaluated even WITHOUT the noise (the model adapted)
- The defense accumulates its effect through model weight updates, not just gradient modification

**Implementation effort:** Low (add ~15 lines to `vfl_framework.py`)
**New hyperparameter:** `--embedding-noise-beta` (range: 0.1 to 2.0)

**Experimental design:**

```
Conditions (CIFAR10, 100 epochs, 3 seeds each):
  [1] Benign, no defense
  [2] Active (MaliciousSGD), no defense
  [3] Active + current defense (grad suppression, alpha=1.0, tau=0.10)
  [4] Active + Option B (embedding noise, beta=0.5, tau=0.10)
  [5] Active + Option B (embedding noise, beta=1.0, tau=0.10)

Metrics: VFL task top-1, model completion accuracy, Fisher divergence trajectory
PAPER CLAIM: [2] >> [4] and [4] ≈ [1]
```

**Novelty:** Medium-High — the specific combination of Fisher divergence detection with
calibrated embedding noise injection is not in the VFL privacy literature.
**Publication potential:** Mid-tier (USENIX Security / CCS) — with Phase 1 detection results
plus 100-epoch defense results showing meaningful reduction and an ablation study.

**⚠️ EMPIRICAL STATUS (2026-07-09): FAILED FOR CIFAR-100.**

EXP-024 (std=0.5): 50.67% — ABOVE undefended attack (47.86%). EXP-025 (std=1.0): 51.64% — WORST RESULT IN PROJECT HISTORY. The mechanism analysis above (Section 11.2) was incorrect in assuming the top model would send "confused" gradients. Instead, corrupting z_a causes the top model to send MORE class-informative corrective gradients, which MaliciousSGD amplifies. Both increasing and decreasing noise_std worsens results. The z_a corruption surface is exhausted.

For CIFAR-10 and CINIC10L (10-class datasets), the standard gradient suppression defense is already sufficient (scale reaches 0.0 around epoch 74) — z_a corruption is unnecessary for 10-class problems. The z_a corruption was motivated solely by CIFAR-100's persistent failure; that motivation is now moot since CIFAR-100 remains unsolved even with this approach.

---

### 11.3 Fisher Equalization + DP Embedding Noise (Layered Defense)

**Combines:** Section 5.1 (Fisher Equalization regularization) + calibrated Gaussian noise.

**Rationale:** Fisher equalization provides a training-time signal that directly opposes class
discriminability. DP embedding noise provides a per-sample perturbation. Together:
- Fisher EQ reduces the EXPECTED Fisher criterion of z_a over training
- DP noise adds variance to individual samples within each class
- Combined effect: lower Fisher_A + higher within-class variance = much harder inference

**Formal defense claim:**
```
Defense provides (ε, δ)-DP on z_a subject to:
  ε = 2 * sigma_clip / sigma_DP    (composed sensitivity from clipping and noise)
  Additionally: Fisher_A ≤ Fisher_B + margin    (structural guarantee from EQ regularization)
```

**Novelty:** High — combining structural regularization with formal DP guarantees for VFL defense.
**Publication potential:** Top-tier potential if the composition theorem is formalized correctly.
**Implementation effort:** High (requires formal DP accounting + Fisher EQ implementation).

---

### 11.4 Adversarial Autoencoder + Centroid Drift Early Detection

**Combines:** Section 7.1 (adversarial autoencoder) + Section 10.2 (centroid drift).

**Pipeline:**
```
Phase 1 (Detection, epochs 1–burn_in):
  Monitor centroid drift D^(t) and Δ_F^(t)
  If EITHER exceeds threshold → activate defense

Phase 2 (Defense, after detection):
  Apply adversarial autoencoder obfuscation to z_a
  Continue centroid drift monitoring as defense quality signal
```

**Why early detection matters:** The earlier the defense activates, the more cumulative
epochs of disruption the bottom model experiences. Centroid drift may detect the attack
2–5 epochs earlier than Fisher divergence (hypothesis — needs testing).

**Novelty:** High (layered detection-defense with temporal and static signals)
**Publication potential:** Mid-tier to top-tier depending on empirical results.

---

## 12. Novel Research Hypotheses

These are speculative but potentially high-value directions. Clearly labeled as hypotheses
requiring experimental validation before inclusion in a paper.

### 12.1 MaliciousSGD-Aware Momentum Disruption (Exploit the Attack's Mechanism)

**Hypothesis:** MaliciousSGD requires gradient CONSISTENCY across consecutive steps to compute
a stable ratio g_t / g_{t-1}. If the server injects inconsistency into the gradient it sends
to Party A, the ratio g_t/g_{t-1} becomes noisy and the amplification ratio r ∈ [1,5] is
applied to the wrong direction.

**Core insight:** MaliciousSGD's ratio is:
```
ratio = clamp(1.0 + gamma * (g_t / g_{t-1}), 1.0, 5.0)
```

This ratio is large when g_t and g_{t-1} point in the same direction (same sign, large
magnitude). If the server FLIPS the sign of grad_output_A every other step:
```
g_t (sent to Party A) = (-1)^t * ∂L/∂z_a
```

Then at Party A:
```
p.grad^(t) = (-1)^t * ∂L/∂θ_A
ratio = clamp(1 + gamma * ((-1)^t ∂L/∂θ_A) / ((-1)^{t-1} ∂L/∂θ_A), 1, 5)
      = clamp(1 + gamma * (-1), 1, 5)
      = clamp(1 - gamma, 1, 5)
      = 1.0    (since 1 - gamma < 1 for gamma > 0)
```

If this analysis is correct, alternating sign injection causes MaliciousSGD to always apply
ratio=1.0 (no amplification), reducing the attack to standard SGD.

**Mathematical verification needed:** Does `clamp(1-gamma, 1, 5) = 1.0` for all reasonable gamma?
For gamma > 0 (typically gamma = 200 in the codebase): 1 - 200 = -199 → clamp to 1.0. Yes.

**If the hypothesis holds:** This would be an elegant and extremely novel result.
MaliciousSGD's own detection mechanism (comparing consecutive gradients) becomes its weakness:
the server can neutralize the amplification with a simple sign alternation.

**Implementation:**
```python
# In simulate_train_round_per_batch(), after computing grad_output_A:
if step % 2 == 0:
    grad_output_A = -grad_output_A    # flip sign on odd steps
```

**Risks:**
- Sign-flipped gradients will cause Party A's model to take steps in the WRONG direction
  on alternating steps → potentially degrading VFL task accuracy (Party A's contribution
  to the top model will be trained inconsistently)
- The top model trains on z_a that was produced by alternating-gradient bottom model →
  may be less task-accurate

**Experiment to validate hypothesis:**
1. Add sign-flip code (2 lines in `vfl_framework.py`)
2. Run at 100 epochs; check if model completion accuracy drops
3. Check VFL top-1 task accuracy — does it also drop?
4. Log MaliciousSGD's actual ratio values during training (instrument `my_optimizers.py`)
   to verify ratio=1.0 under sign flipping

**Novelty:** Very High — exploiting the attack mechanism's ratio computation to neutralize it.
**Publication potential:** Top-tier if the theoretical analysis holds and empirical results confirm.
This is the kind of elegant result that S&P/CCS reviewers love.

**Status (2026-07-09): ❌ FAILED — DEMOTED TO TIER 5.**

EXP-023 (CIFAR-10, 100 epochs, sign-flip=True): Best MC = 86.40% vs benign 83.11% → defense criterion NOT met. Sign-flip is **worse** than standard suppression (86.40% vs 81.80% mean defended across 4 seeds). Phase 13 (CIFAR-100 sign-flip) should NOT be run.

**Post-mortem:** The theoretical prediction (ratio=1.0) did not hold because: (1) MaliciousSGD's ratio is computed on `p.grad` (internal parameter gradients), which is derived from `grad_output_A` via chain rule through a nonlinear bottom model — sign inversion of `grad_output_A` does not guarantee sign inversion of `p.grad`; (2) even if ratio=1.0 were achieved, standard SGD over 100 epochs is sufficient to build partially discriminative embeddings in a 10-class problem. The standard suppression defense outperforms sign-flip because it achieves scale=0 around epoch 74, providing 26+ epochs of zero gradient versus sign-flip's 100 epochs of ratio=1.0 SGD.

---

### 12.2 Synthetic Gradient Injection (Noise Gradient Matching Attack Distribution)

**Hypothesis:** If the server knows the distribution of gradients under a BENIGN scenario,
it can REPLACE Party A's gradient with a sample from the benign distribution. Party A's
bottom model then trains as if no attack was occurring, but without knowing it.

**Mathematical formulation:**

During characterization (Phase 1), the server records:
```
G_benign = {grad_output_A^(t) : t = 1,...,T, under benign condition}
```

Under active attack, when Δ_F > tau:
```
grad_output_A_replaced = sample from G_benign    (or Gaussian with benign statistics)
```

Party A receives a gradient from the benign distribution → MaliciousSGD amplifies a benign
gradient → the ratio g_t/g_{t-1} is now computed on benign-distribution gradients.

**Why this might work:** MaliciousSGD uses CONSECUTIVE gradient alignment to determine when
to amplify. Benign gradients have lower consecutive alignment (convergence property: gradients
become smaller and less aligned as training progresses). Thus, MaliciousSGD would apply lower
amplification ratios even though the attack is active.

**Risks:**
- The replaced gradient does not encode the true loss signal for the current batch
- Party A's bottom model cannot contribute meaningfully to the task if it receives random gradients
- The top model degrades because Party A's contribution becomes noise from the task perspective

**A compromise variant:** Weight-mix the true gradient with a benign sample:
```
grad_output_A_mixed = (1-α) * grad_output_A_true + α * grad_output_A_benign_sample
```

As α → 1, the defense becomes stronger but task utility degrades.

**Novelty:** High — synthetic gradient substitution for VFL defense is not published.
**Publication potential:** Mid-tier as an ablation point; not standalone.

---

### 12.3 Label Clustering Confusion via Inter-Class Centroid Attraction

**Hypothesis:** Instead of destroying all class structure in z_a, target specific PAIRS of
classes that are most discriminable by the attacker and MERGE their centroids.

**Rationale:** Model completion (Stage 2 MixMatch) fails when pseudo-label assignment is
wrong. The easiest pseudo-label mistakes are confusing classes with NEARBY centroids.
If the defense merges centroids of the K most-separated class pairs, MixMatch's pseudo-labels
will be wrong exactly for the most attack-advantaged class boundaries.

**Implementation:**
1. Server computes pairwise centroid distances at each epoch (already has centroids from Fisher computation)
2. Identify top-K most separated pairs: (c1*, c2*) = argmax ||μ_{c1} - μ_{c2}||_2
3. Add an attraction term for those pairs:
```
R_targeted = sum_{(c1,c2) in top-K} ||μ_c1 - μ_c2||_2²    (MINIMIZE this)
```

**Advantage over global centroid compression (Section 5.4):** Targeted attraction of the
most separated pairs preserves most of the class structure (utility for top model) while
disrupting the specific boundaries that the attack exploits most.

**Novelty:** High — targeted centroid confusion based on attack detection is novel.
**Publication potential:** Mid-tier if combined with Fisher detection.

---

### 12.4 Adaptive Defense Intensity: Proportional to Attack Confidence

**Hypothesis:** The defense should scale its intensity proportionally to how confident the
detection is (not just on/off). At moderate Δ_F (borderline detection), apply weak defense.
At high Δ_F (certain attack), apply strong defense.

**This is already partially implemented** in the current defense with `alpha * (Δ_F - tau)`.
However, the CHOICE of defense action (gradient suppression vs embedding noise vs centroid
attraction) could also be adaptive:

```python
if delta_F < tau:
    # No action
elif tau <= delta_F < tau + mild_threshold:
    # Mild: apply Option A (soft gradient suppression)
elif tau + mild_threshold <= delta_F < tau + severe_threshold:
    # Moderate: apply Option B (embedding noise, moderate beta)
else:
    # Severe: apply Option B with large beta + reverse contrastive loss
```

**Novelty:** Medium (adaptive thresholding is standard)
**Publication potential:** Ablation study content; supports the paper's hyperparameter analysis.

---

### 12.5 Cross-Epoch Embedding Consistency Constraint

**Hypothesis:** Under benign training, a data point x_a should produce similar embeddings
in consecutive epochs (embeddings converge). Under MaliciousSGD, the aggressive gradient
amplification causes large embedding changes between consecutive epochs for the same x_a.

**Signal:** Per-sample embedding velocity:
```
v_i^(t) = ||z_a^(t)(x_i) - z_a^(t-1)(x_i)||_2    (for sample i)
Mean_velocity^(t) = mean_i v_i^(t)
```

Under MaliciousSGD: Mean_velocity is large; under benign: small.

**Defense option:** Penalize large embedding velocity:
```
R_velocity = mean_i ||z_a^(t)(x_i) - z_a_stored^(t-1)(x_i)||_2²
```

This requires storing z_a from the previous epoch for all training samples (memory cost: n * d).
For n=50000, d=64: 50000 * 64 * 4 bytes = 12.8 MB — acceptable.

The regularization slows down embedding changes → MaliciousSGD cannot rapidly update z_a
toward discriminative positions → delayed convergence of the attack.

**Novel aspect:** The velocity constraint is an EWC (Elastic Weight Consolidation) analog
applied to EMBEDDINGS rather than WEIGHTS. It's an embedding-space version of continual
learning regularization, applied to prevent rapid discriminability increase.

**Novelty:** High
**Publication potential:** Mid-tier if empirically validated.

---

## 13. Comparative Analysis and Tradeoffs

### 13.1 Utility-Privacy Tradeoff Summary

```
HIGH UTILITY COST (>10% VFL task accuracy drop expected):
  - MINE-based MI minimization (Section 6.1)
  - IRM (Section 8.4)
  - Full adversarial autoencoder (Section 7.1)

MEDIUM UTILITY COST (2-10% drop):
  - IB regularization (Section 6.2)
  - Fisher equalization (Section 5.1)
  - Reversed contrastive loss (Section 8.2)
  - Embedding noise Option B (Section 2.4.2) at high beta

LOW UTILITY COST (<2% drop):
  - Current gradient suppression (exists, not validated at 100ep)
  - Gradient clipping (Section 3.2)
  - Orthogonality constraints (Section 8.1) at low lambda
  - Sign-flip momentum disruption (Section 12.1) — *if hypothesis holds*
  - Domain adversarial training with GRL (Section 7.2) at low lambda
```

### 13.2 Implementation Effort vs. Expected Privacy Gain

```
LOW EFFORT / MEDIUM GAIN (implement first):
  - Option B: Embedding noise (Section 2.4.2)             ~1 day to implement
  - Option A: Hard clip (Section 2.4.1)                   ~1 hour
  - Sign-flip momentum disruption (Section 12.1)          ~1 hour
  - Orthogonality constraints (Section 8.1)               ~1 day

MEDIUM EFFORT / MEDIUM-HIGH GAIN:
  - Domain adversarial training w/ GRL (Section 7.2)      ~3 days
  - Fisher equalization regularization (Section 5.1)      ~2 days
  - Reversed contrastive loss (Section 8.2)               ~1 day
  - MMD class alignment (Section 6.3)                     ~2 days

HIGH EFFORT / HIGH GAIN:
  - MINE-based MI minimization (Section 6.1)              ~1-2 weeks
  - Variational IB (Section 6.2)                          ~1 week
  - Adversarial autoencoder (Section 7.1)                 ~1 week
  - Fisher EQ + formal DP composition (Section 11.3)      ~2 weeks
```

### 13.3 Defense Taxonomy: Attack-Aware vs. Attack-Agnostic

```
ATTACK-AWARE (specifically target MaliciousSGD's mechanism):
  - Sign-flip momentum disruption (12.1)  — exploits ratio computation
  - Synthetic gradient injection (12.2)   — matches benign gradient distribution
  - Cross-epoch velocity penalty (12.5)   — slows embedding velocity

ATTACK-AGNOSTIC (work against any label inference strategy):
  - MINE MI minimization (6.1)           — bounds MI(z_a; y) regardless of attack
  - IB regularization (6.2)              — compresses label information structurally
  - MMD class alignment (6.3)            — makes class distributions indistinguishable
  - Adversarial autoencoder (7.1)        — trained to fool any classifier

DETECTION-TRIGGERED (only activate when attack is detected):
  - Current defense (2.1)                — Fisher detection + gradient suppression
  - Option B (2.4.2)                     — Fisher detection + embedding noise
  - All hybrid designs in Section 11
```

### 13.4 Coverage of Attack Surface

```
MaliciousSGD attack surface:
  [A] Gradient signal from server (grad_output_A)
      → Current defense, Option A, Gradient clipping, DP noise
  [B] Internal gradient amplification (p.grad)
      → Sign-flip (12.1) indirectly; Domain adversarial GRL (7.2) via amplification reversal
  [C] Embedding representation quality (z_a)
      → Option B, IB, MINE, Fisher EQ, Orthogonality, Reversed contrastive
  [D] Checkpoint (saved .pth)
      → Option B (corrupts bottom model's learned representations)
      → All representation-level defenses if applied consistently over 100 epochs
  [E] Stage 2 inference (MixMatch on z_a)
      → Option B (noisy z_a), IB (stochastic z_a), Adversarial autoencoder (obfuscated z_a)
```

The most complete defenses cover [C], [D], AND [E]. The current defense covers only [A].

---

## 14. Prioritized Research Roadmap

### Status Update — 2026-07-09

**Completed since last update:**
- ✅ Phase 4 (100-epoch CIFAR10): attack 95.42%, defended 84.27% — defense succeeds (Tier 1A done)
- ✅ Phase 8 (100-epoch CIFAR10 ablation): 4/5 variants succeed; a=2.0 gives best suppression (Tier 1B done)
- ✅ Phase 6B (CIFAR-10 seed sweep, 4 seeds): 4/4 seeds confirmed; mean ± std computed (Tier 2D done)
- ✅ Phase 7 (competitor comparison GC, Laplace DP): MaliciousSGD dominates on both datasets
- ✅ Phase 9 (Option B gradient noise injection CIFAR-100): ALL FAILED — n=0.5 → 48.36%, n=1.0 → 49.64%, n=2.0 → 43.10%. Gradient-space exhausted.
- ✅ EXP-017 (CIFAR-10 benign+defense 100ep): 84.35% ≈ benign mean — asymmetry confirmed
- ✅ EXP-018 (CIFAR-100 benign+defense 150ep): 34.74% — dormant, run variance explained the +4.41pp gap
- ✅ **Phase 10 (CINIC10L generalization): COMPLETE — DEFENSE SUCCEEDS (EXP-022: 62.43% < 65.70% benign ✅)**
- ✅ **Phase 12 (sign-flip CIFAR-10): COMPLETE — ❌ FAILED (EXP-023: 86.40% > 83.11% benign)**
- ✅ **Phase 14A (z_a corruption CIFAR-100, std=0.5): COMPLETE — ❌❌ FAILED BADLY (EXP-024: 50.67% > 47.86% attack)**
- ✅ **Phase 14B (z_a corruption CIFAR-100, std=1.0): COMPLETE — ❌❌❌ WORST RESULT EVER (EXP-025: 51.64%)**

**Critical assessment — 2026-07-09:**
Both sign-flip and z_a embedding corruption have now been tested and both failed. All defense attack surfaces for CIFAR-100 have been exhausted: gradient suppression (3 variants), gradient noise injection (3 variants), embedding-space z_a corruption (2 variants), sign-flip (1 variant, CIFAR-10 only). None of the 9 total variants bring CIFAR-100 inference below benign (30.33%). The CIFAR-100 problem is an open research problem. The paper's best path is a two-dataset (CIFAR-10 + CINIC10L) 10-class contribution with an honest limitation section on CIFAR-100.

### Tier 1: IMMEDIATE — Highest Priority (Updated 2026-07-09)

| Priority | Task | Code files | Why | Status |
|---|---|---|---|---|
| **1A-NOW** | **Multi-seed CINIC10L validation (3 seeds minimum)** | New bat file | EXP-022 result (62.43%) is single-seed; multi-seed required for any paper claim | ❌ NOT DONE — HIGHEST PRIORITY |
| **1B-NOW** | **Multi-seed CIFAR-100 benign baseline (3 seeds)** | New bat file | Single-seed benign 30.33% is insufficient reference point for CIFAR-100 comparisons | ❌ NOT DONE |
| 1C-DONE | ~~Implement embedding-space z_a noise (Section 11.2)~~ | — | ~~DONE — EXP-024/025: BOTH FAILED~~ | ❌ FAILED — see EXP-024/025 |
| 1D-DONE | ~~Sign-flip momentum disruption (Section 12.1)~~ | — | ~~DONE — EXP-023: FAILED~~ | ❌ FAILED — see EXP-023 |
| 1E-DONE | ~~Run Phase 4 (100-epoch CIFAR10)~~ | — | ~~Critical missing experiment~~ | ✅ DONE — EXP-011 |
| 1F-DONE | ~~Run Phase 3 model completion ablation~~ | — | ~~Fills in ablation table~~ | ✅ DONE — EXP-015 |

### Tier 2: Short-Term (Multi-Seed Validation + Paper Scope Decision)

| Priority | Task | Code files | Why | Status |
|---|---|---|---|---|
| **2A-NOW** | **CINIC10L multi-seed (3 seeds minimum)** | New bat file | EXP-022 is single-seed (62.43%); paper claim requires mean ± std | ❌ NOT DONE |
| **2B-NOW** | **CIFAR-100 benign baseline multi-seed (3 seeds)** | New bat file | Single-seed 30.33% is unstable reference; EXP-018 showed +4.41pp run variance | ❌ NOT DONE |
| 2C-DONE | ~~Await CINIC10L (Phase 10) results~~ | — | ~~DONE — EXP-022: 62.43% < 65.70% benign ✅~~ | ✅ DONE |
| 2D-DONE | ~~Add `--manual-seed` to scripts~~ | — | ~~DONE~~ | ✅ DONE |
| 2E | Domain adversarial training with GRL for CIFAR-100 (Section 3A / Tier 3) | `model_sets.py`, `vfl_framework.py` | Only remaining untested fundamentally-different approach for CIFAR-100 | ❌ NOT DONE |

### Tier 3: Medium-Term (Paper Strengthening)

| Priority | Task | Code files | Why |
|---|---|---|---|
| 3A | Domain adversarial training with GRL (Section 7.2) | `model_sets.py`, `vfl_framework.py` | Potentially top-tier insight (MaliciousSGD amplifies adversarial gradient) |
| 3B | Fisher equalization regularization (Section 5.1) | `vfl_framework.py` (~50 lines) | Direct attack on the detection signal |
| 3C | Reversed contrastive loss (Section 8.2) | `vfl_framework.py` (~30 lines) | Novel use of standard technique |
| 3D | Centroid drift detection logging | `characterization_monitor.py` | Strengthens detection component |

### Tier 4: Long-Term / Paper Follow-Up

| Priority | Task | Why |
|---|---|---|
| 4A | MINE MI minimization (Section 6.1) | Formal MI bound → top-tier paper potential |
| 4B | Variational IB (Section 6.2) | Stochastic bottom model → uncertainty quantification |
| 4C | Fisher EQ + formal DP composition (Section 11.3) | Formal privacy guarantees |
| 4D | CIFAR100 150-epoch equivalents of all experiments | Full second dataset results |
| 4E | Competitor defenses (GC, Laplace DP) at 100 epochs | Complete comparison table |

### Tier 5: Do Not Pursue (Low ROI)

- Homomorphic encryption (Section 9.4) — impractical for this timeline
- IRM (Section 8.4) — poorly defined in VFL context; theory not settled
- Label permutation (Section 9.1) — too aggressive on utility; publishable only as baseline

---

## 15. Experimental Design Templates

### 15.1 Standard Experiment Checklist

For every new defense experiment, ensure:

```
Pre-experiment:
□ Fixed seed specified (--manual-seed, use seeds {42, 123, 456})
□ Checkpoint naming will not overwrite existing results (check filename)
□ Baseline conditions (benign, active-no-defense) included in the same run
□ VFL task accuracy will be logged (the .txt files in saved_models/)
□ Fisher divergence will be logged (--monitor-separability True if applicable)
□ Batch file created and saved with run conditions documented in comments

Post-experiment:
□ Model completion run on all new checkpoints
□ Results logged in research_log.md with EXPXXX label
□ Comparison table updated
□ If unexpected result: mechanism hypothesis added to research_log.md
□ If confirms hypothesis: update this possible_directions.md with result status
```

### 15.2 Minimum Results Table for Publication

A paper-ready results table requires at minimum:

```
| Condition | CIFAR10 Inference Acc. | CIFAR100 Inference Acc. |
|---|---|---|
| Benign (no attack, no defense)  | mean ± std (3 seeds) | mean ± std (3 seeds) |
| Active attack (no defense)      | mean ± std (3 seeds) | mean ± std (3 seeds) |
| Existing defense 1: GC          | mean ± std           | mean ± std           |
| Existing defense 2: Laplace DP  | mean ± std           | mean ± std           |
| Our defense: [chosen method]    | mean ± std (3 seeds) | mean ± std (3 seeds) |
```

All conditions MUST use the SAME epoch count (100 for CIFAR10, 150 for CIFAR100).
All conditions MUST use the SAME seeds for the random number generators.

### 15.3 Ablation Study Design

For any defense with hyperparameter(s), the ablation must cover:

```
Single hyperparameter ablation (fix all others):
  Parameter: alpha (or beta, or lambda)
  Values: [very small, small, medium, large]
  Example for Option B: beta ∈ {0.1, 0.5, 1.0, 2.0}
  Fixed: tau=0.10, burn_in=8, seed=42

Multi-parameter grid (if budget allows):
  (tau, beta): {0.05, 0.10, 0.15} × {0.5, 1.0}
  Run at seed=42 only for grid; run best configuration at 3 seeds

Report format:
  Two tables: (a) ablation on CIFAR10, (b) ablation on CIFAR100
  Metric: model completion top-1 accuracy AND VFL task top-1 accuracy
  Highlight: the configuration that maximizes (attack - defense) while minimizing utility cost
```

### 15.4 Statistical Validity Requirements

```
Minimum 3 seeds for any number that goes in a paper claim.
Report: mean ± standard deviation.
Significance test: paired t-test (attack vs. defense at same seeds).
Null hypothesis: defense has no effect (attack_acc ≈ defense_acc).
Rejection at p < 0.05 required.

For 3 seeds, degrees of freedom = 2 → t-statistic table:
  t_crit(df=2, p=0.05) = 2.920 (two-tailed), 2.353 (one-tailed)
  
To reject H0: |t| > 2.920 (two-tailed) where:
  t = (mean_attack - mean_defense) / (std_diff / sqrt(3))
  
This requires the difference (attack - defense) to be consistent in sign across all 3 seeds
AND large in magnitude. If all 3 seeds show >15pp reduction, the t-test will likely pass.
```

### 15.5 Publication Narrative Templates

**If Phase 4 + current defense shows meaningful reduction:**
```
Narrative: "Asymmetric Server-Side Defense Against Active Label Inference in VFL"
Contribution 1: Fisher divergence detection system (Phase 1) — existing
Contribution 2: Asymmetric Adaptive Perturbation defense (Phase 2/4) — new
Contribution 3: 100-epoch ablation study showing defense effectiveness
Venue: USENIX Security / CCS workshop / IEEE S&P poster
```

**If Phase 4 defense fails + Option B works:**
```
Narrative: "Detection-Triggered Embedding-Level Defense Against MaliciousSGD in VFL"
Contribution 1: Fisher divergence detection system (Phase 1) — existing
Contribution 2: Embedding-level noise injection defense triggered by Fisher detection
Contribution 3: Ablation and comparison vs. gradient-level defenses
Venue: USENIX Security / NDSS (mid-tier if well-executed)
```

**If Domain Adversarial GRL works + MaliciousSGD amplifies adversarial gradient:**
```
Narrative: "Turning the Attack Against Itself: Exploiting MaliciousSGD's Amplification for Privacy Defense"
Contribution 1: Theoretical analysis of how GRL interacts with gradient amplification
Contribution 2: Empirical demonstration that attack strength → defense strength
Contribution 3: Privacy-utility tradeoff analysis
Venue: IEEE S&P / ACM CCS (top-tier potential)
```

---

---

## 16. Persistent Projection Defense — Primary Research Direction (2026-07-12)

**Status as of 2026-07-12:** 🔵 ACTIVE RESEARCH DIRECTION — literature review confirms Clearly Novel; implementation planned for Phase 22 (CIFAR-10) + Phase 23 (CIFAR-100).

**Last major update:** 2026-07-12

---

### 16.1 Motivation and Background

The current defense story is fragmented:
- **CIFAR-10 / CINIC10L (10-class):** AsymmetricAdaptivePerturbation (AAP) works. Mechanism: gradient scale reaches 0.0 by epoch ~74, blocking grad_output_A for 25+ epochs. Theory is sound. Novelty is incremental (adaptive gradient perturbation paradigm exists in FL; novel only in Fisher divergence gating).
- **CIFAR-100 (100-class):** GradientProjection works empirically (4/4 seeds). BUT: mechanism is catastrophic single-activation collapse (epoch 11–12), not designed behavior. Defense fires ONCE, destroys intra_var_A by 6 orders of magnitude, never fires again. Not theoretically principled.

A fragmented paper (different defenses for different datasets) will draw reviewer objections. The preferred narrative is a **single unified mechanism** that:
1. Is theoretically motivated (not empirically discovered)
2. Works across all dataset complexities
3. Operates persistently throughout training (not via one-shot collapse)
4. Builds on the Fisher Divergence Detection component (shared detection layer)

**Literature review conclusion (EXP-039, 2026-07-12):** Persistent Projection for VFL label inference is **Clearly Novel**. The closest competitor (MixPro, SIGIR 2023) uses a generic projection step without any discriminative direction targeting, without active attack focus, and without a persistent evolving basis. No prior VFL paper proposes multi-epoch stable subspace projection as a defense.

---

### 16.2 Mechanism Design

**Core idea:** At every epoch where Fisher divergence exceeds tau, project grad_output_A onto the orthogonal complement of the running discriminative direction. Use an exponential moving average (EMA) of the auxiliary classifier's gradient direction to maintain a stable, slowly-evolving estimate of the discriminative subspace.

**Key difference from current GradientProjection (Phase 19/EXP-032):**

| Aspect | Current GradientProjection | Persistent Projection |
|---|---|---|
| Firing | Once at epoch 11 (then forever silent) | Every detected epoch |
| Direction estimate | Instantaneous (exactly aux_classifier at activation epoch) | EMA across epochs: stable and gradual |
| Collapse risk | Catastrophic (projects out ~97% of gradient in one shot) | Bounded per-epoch removal |
| Mechanism | Accidental self-terminating collapse | Designed progressive discriminative suppression |
| Theory | No formal bound | ‖grad_proj‖ ≤ ‖grad‖ always; EMA smoothness preserves task gradient signal |

**Mathematical formulation:**

At each epoch t where divergence > tau:
```
d_inst_t = d(L_aux)/d(z_a) / ‖d(L_aux)/d(z_a)‖    [unit-norm instantaneous direction]

d_ema_t = normalize((1 − α_ema) × d_ema_{t-1} + α_ema × d_inst_t)    [EMA update]

grad_proj_t = grad_output_a − α_proj × (grad_output_a · d_ema_t) × d_ema_t    [projection]
```

**Hyperparameters:**
- `alpha_ema`: EMA decay rate for direction update (0.1 = slow update, 0.5 = fast update). Too high → instability; too low → stale direction. Sweep: {0.1, 0.2, 0.3}.
- `alpha_proj`: Projection intensity (0 = no projection, 1 = full orthogonal projection). Start with 1.0.
- `burn_in`: Epochs before projection starts. Shorter than current (2–4 vs 8) to prevent discriminative structure accumulation before defense activates.
- `tau`: Fisher divergence threshold. Same as AAP (0.10 for CIFAR-10, possibly 0.07 for CIFAR-100).

**Stability properties:**
- ‖grad_proj‖ = ‖grad‖ × sin(θ) where θ is angle between grad and d_ema. Since sin(θ) ∈ [0,1], grad_proj ≤ grad in magnitude — no growth, no NaN risk (same as EXP-032 GradProj).
- EMA smoothing prevents abrupt directional jumps that caused the epoch-11 catastrophe.
- Firing every detected epoch provides cumulative effect over many epochs, reducing the per-epoch required removal and distributing the semantic disruption across the training trajectory.

---

### 16.3 Theoretical Framing

**Why progressive removal is sufficient:**

MaliciousSGD requires repeated gradient signals aligned with the class-discriminative direction to build and maintain class-separable embeddings. If the server persistently removes the component of grad_output_A aligned with d_aux at each epoch, Party A's bottom model receives a gradient from which the discriminative signal is surgically absent. Despite MaliciousSGD amplifying grad_proj by up to 5×, the amplified gradient is still ‖grad‖ × sin(θ) — orthogonal to the discriminative direction. Over 150 epochs of receiving orthogonally-constrained gradients, the bottom model cannot build class-separable embeddings in the discriminative direction.

**Connection to Fisher Divergence:**

The projection direction d_aux is derived from the gradient of a cross-entropy loss on the server's auxiliary classifier over z_a. This direction points toward maximizing inter-class separation in z_a — exactly what MaliciousSGD is trying to amplify. The defense removes the component of grad_output_A that is aligned with this direction, preventing the alignment-amplification loop that drives the attack.

**Why AAP (magnitude suppression) fails for CIFAR-100 but PP (direction removal) succeeds:**

AAP scales down the ENTIRE grad_output_A by a factor proportional to (divergence − tau). At CIFAR-100's divergence levels (max ~0.4), the scale reaches ~0.7 at best — never zero. MaliciousSGD amplifies the remaining 70% gradient by ratio ∈ [1,5], partially compensating. With 100 classes and 150 epochs, even 70% of a well-aligned gradient over 150 epochs builds sufficient discriminative structure.

PP removes the discriminative component DIRECTIONALLY — the orthogonal gradient that remains is maximally uninformative about class membership (it maximizes intra-class variance rather than inter-class distance). Regardless of how much MaliciousSGD amplifies this orthogonal gradient, the amplification cannot create class discriminability because the class-relevant direction is absent.

---

### 16.4 Implementation Plan

**Code changes required:**

1. **`Code/possible_defenses.py`** — Add `PersistentProjectionDefense` class:
   - `__init__`: Initialize EMA state, aux_classifier, hyperparameters
   - `update_direction(z_a, y)`: Compute aux_classifier gradient, update d_ema
   - `apply(grad_output_a, divergence, epoch)`: Project if detected, return grad_proj
   - EMA direction update must be called BEFORE apply (direction updated on clean z_a before gradient modification)

2. **`Code/vfl_framework.py`** — Add `--persistent-projection` flag:
   - Similar to `--asymmetric-defense` wiring
   - Call `defense.update_direction(z_a.detach(), labels)` during each training batch
   - Call `defense.apply(grad_output_a, divergence, epoch)` before gradient is sent to Party A
   - Write discriminative direction EMA trajectory to CSV alongside Fisher divergence

3. **Batch file:** `Code/run_phase22_pp_cifar10.bat` — CIFAR-10, 100 epochs, alpha_ema sweep {0.1, 0.2, 0.3}, seeds 0 and 42

**Critical implementation detail — direction update timing:**

The direction update must use z_a BEFORE the top model's backward pass, and the projection must use the UPDATED direction. Order in each batch:
```
1. Forward pass: z_a = bottom_model_a(x_a)
2. Update discriminative direction: defense.update_direction(z_a.detach(), y)
3. Top model forward + loss backward
4. grad_output_a = z_a.grad  [server computed]
5. grad_output_a = defense.apply(grad_output_a, divergence, epoch)  [project]
6. Send projected grad_output_a to Party A
```

If the direction is updated AFTER the projection, the direction lags by one batch. Either ordering is defensible but must be consistent.

---

### 16.5 Experimental Design

**Phase 22 — CIFAR-10 Persistent Projection:**

```
Conditions:
  [1] Benign (reference, use EXP-016 results: 83.11 ± 2.84%)
  [2] Attack, no defense (reference, use EXP-016 results: 94.95 ± 0.52%)
  [3] Attack + PP, alpha_ema=0.1, alpha_proj=1.0, burn_in=4
  [4] Attack + PP, alpha_ema=0.2, alpha_proj=1.0, burn_in=4
  [5] Attack + PP, alpha_ema=0.3, alpha_proj=1.0, burn_in=4
  [6] Attack + PP, alpha_ema=0.2, alpha_proj=1.0, burn_in=2  [very early activation]

Seeds: {0, 42} initially. Add {123, 456} if seed-0 or seed-42 succeeds.
Success criterion: Mean defended MC < benign mean (83.11%).
Comparison: vs AAP defended 81.80 ± 1.85% — PP should be similar or better.

Diagnostic logged per run:
  - Fisher divergence CSV (same as current)
  - Direction stability: ‖d_ema_t − d_ema_{t-1}‖ per epoch (should be small after convergence)
  - Projection magnitude: (grad · d_ema) / ‖grad‖ per batch (angle cos; should be large for active, small for benign)
  - Number of epochs where defense fires (should be many, not 1)
```

**Phase 23 — CIFAR-100 Persistent Projection:**

```
Conditions:
  [1] Benign (reference, use EXP-029: 29.56 ± 2.93%)
  [2] Attack, no defense (reference, use EXP-036/037/038: 49.87 ± 1.17%)
  [3] Attack + PP, best alpha_ema from Phase 22, alpha_proj=1.0, burn_in=4
  [4] Attack + PP, same but burn_in=2

Seeds: {0} initially. Multi-seed if seed-0 shows MC < benign.
Success criterion: MC < 29.56% (benign mean); defense fires more than once (no catastrophic collapse).
Key diagnostic: intra_var_A should NOT spike by 6 orders of magnitude — if it does, EMA is insufficient.

Expected behavior (if PP is working as designed):
  - Defense fires from epoch 4 or 5 onward (every epoch where divergence > tau)
  - Fisher divergence slowly decreases (gradual rather than abrupt negative swing)
  - intra_var_A does NOT spike catastrophically
  - After 150 epochs, Party A's embedding is weakly class-discriminative but not structurally shattered
```

---

### 16.6 Research Strategy — Unified vs. Fragmented Defense

**Question:** Should we replace AAP and one-shot GradProj with Persistent Projection, or present them as parallel defenses?

**If Persistent Projection works for BOTH CIFAR-10 and CIFAR-100:**
- Replace entirely. Single unified defense mechanism.
- Paper narrative: "Fisher Divergence Detection + Discriminative Subspace Projection" is the contribution.
- AAP and one-shot GradProj become negative ablation results: AAP insufficiently strong for 100-class; one-shot GradProj works but lacks theory; PP provides the principled, persistent, dataset-agnostic solution.
- This is the STRONGEST narrative and the target.

**If Persistent Projection works only for CIFAR-10 (not CIFAR-100):**
- Reconsider alpha_proj < 1.0 for CIFAR-100 (partial projection to avoid collapse)
- Or accept fragmented story with theoretical justification: 100-class problems require one-shot collapse because the signal-to-noise ratio for progressive removal is too low over 150 epochs
- Present as: "PP for CIFAR-10 (progressive), one-shot GradProj for CIFAR-100 (structural disruption)" — two operating modes of the same framework

**If Persistent Projection fails for both datasets:**
- Revert to fragmented story: AAP + GradProj with honest theory limitations
- Frame GradProj's one-shot collapse as a discovered mechanism, not a design choice
- Venue implications: this is a workshop / PETS level paper, not IEEE S&P / CCS

**Reviewer risk for presenting different defenses per dataset:**
- Will be flagged. Common phrasing: "It seems the authors tried many approaches and report the best per dataset. Is this principled?"
- Mitigation if stuck with fragmentation: provide a theoretical decision criterion — "use PP for C (num_classes) ≤ K; use one-shot GradProj for C > K" where K is calibrated from Fisher signal-to-noise ratio analysis.

**Recommendation:** Implement and test PP first. The additional experimental cost (Phase 22 + 23) is 4–6 GPU runs. The potential benefit (unified story, stronger novelty claim, cleaner paper) is high. Run Phase 22 before making the final paper architecture decision.

---

### 16.7 Novelty Assessment (From EXP-039 Literature Review)

| Component | Novelty | Closest Competitor | Key Differentiator |
|---|---|---|---|
| Fisher Divergence Detection | Clearly Novel | None found | No prior VFL paper monitors J_A − J_B |
| Aux-classifier-directed projection (one-shot) | Moderately Novel | MixPro (SIGIR 2023) | Our direction is discriminative-targeted; we address active attacks; we have persistent design |
| Persistent Projection (EMA-based, multi-epoch) | Clearly Novel | None found | No VFL defense paper proposes this mechanism |
| Unified PP + Fisher framework | Clearly Novel | None found | No paper combines Fisher detection with persistent discriminative projection |

**Recommended citation strategy:**
1. Cite Fu et al. (USENIX 2022) as the primary attack paper we defend against
2. Cite MixPro as the closest mechanistic competitor, then differentiate in a comparison table
3. Cite ProjPert as a name-similar but mechanistically different work (clarify the difference upfront to preempt reviewer confusion)
4. Cite LADSG (arXiv 2506.06742, June 2025) as the most recent concurrent work claiming active attack defense — compare our detection mechanism vs. their gradient norm anomaly detection
5. Cite MARVELL as the prior work on gradient distribution equalization — position Fisher detection as the detection-theoretic advance over MARVELL's empirical norm-balancing

---

*End of possible_directions.md (Parts 1–3)*
*For questions or to add a new direction: update the relevant section in-place and update the last-updated date at the top of each file.*
*Cross-references:*
  *- `research_log.md` — all experimental results*
  *- `possible_defenses.py` — implemented defenses*
  *- `characterization_monitor.py` — Fisher/silhouette monitoring*
  *- `vfl_framework.py` — main training loop (primary modification target)*
