# Possible Defense Directions Against Active Label Inference in VFL
## Part 2 of 3: Fisher-Based, Information-Theoretic, Adversarial Training, and Optimization-Based Defenses

**Status:** Living research document — expand without restructuring.
**Last major update:** 2026-07-02
**Cross-references:** `possible_directions_1.md` (Sections 1–4) | `possible_directions_3.md` (Sections 9–15)

---

## 5. Fisher-Based and Separability-Aware Defenses

This family of defenses directly targets the Fisher criterion — the exact metric used in
Phase 1 to detect the attack. Rather than detecting the attack and suppressing gradients,
these defenses actively manipulate the separability of Party A's embeddings.

### 5.1 Fisher Equalization: Make Fisher_A ≈ Fisher_B

**Core intuition:** The attack's detection signal is Fisher_A >> Fisher_B. If the defense
can force Fisher_A toward Fisher_B (not by improving Fisher_B, but by degrading Fisher_A),
the attack detection signal disappears AND the embeddings lose label discrimination.

**Why this is different from the current defense:**
- Current defense: detect high Fisher_A → suppress gradient
- Fisher equalization: detect high Fisher_A → add a training regularization term that
  directly penalizes Fisher_A being large relative to Fisher_B

**Mathematical formulation:**

Add a regularization term to the VFL training loss:

```
L_total = L_task(ŷ, y) + λ * R_Fisher(z_a, z_b, y)

where:
    R_Fisher = max(0, J_A - J_B - margin)

J_A = Tr(S_B^A) / Tr(S_W^A)    (Fisher criterion of Party A's embeddings)
J_B = Tr(S_B^B) / Tr(S_W^B)    (Fisher criterion of Party B's embeddings)
margin = 0.0 (or small positive constant for slack)
```

This loss term penalizes configurations where Party A's embeddings are more discriminative
than Party B's — exactly the condition the attack creates.

**Gradient of R_Fisher:**

```
∂R_Fisher/∂z_a = ∂/∂z_a [Tr(S_B^A) / Tr(S_W^A)]

= [Tr(S_W^A) * ∂Tr(S_B^A)/∂z_a - Tr(S_B^A) * ∂Tr(S_W^A)/∂z_a] / Tr(S_W^A)²
```

Each of ∂Tr(S_B)/∂z_a and ∂Tr(S_W)/∂z_a can be computed analytically:
- ∂Tr(S_W)/∂z_i = 2(z_i - μ_{c(i)})   (where c(i) is the class of sample i)
- ∂Tr(S_B)/∂z_i = 2n_{c(i)}(μ_{c(i)} - μ) / N   (weighted centroid displacement)

The total gradient is differentiable and can be backpropagated through PyTorch's autograd if
S_B and S_W are computed via tensor operations.

**Implementation location:**
- `vfl_framework.py`: add R_Fisher computation in the training loop
- Server computes J_A and J_B after each batch (or each epoch for efficiency)
- The regularization gradient is combined with the task gradient before backprop

**Required code changes:**
1. Compute J_A and J_B using differentiable PyTorch scatter operations (batch approximation)
2. Add `--fisher-lambda` argument to `vfl_framework.py`
3. Add R_Fisher to the total loss before `loss.backward()`

**Strengths:**
- Principled: directly targets the metric the attack exploits
- The regularization gradient provides a directed force against class separability in z_a
- Server controls λ — Party A does not know this regularization is being applied
- Works even if Party A uses a different attack strategy (any strategy increasing J_A is penalized)

**Weaknesses:**
- Computing Fisher criterion per batch is noisy — per-epoch computation introduces a 1-epoch lag
- Large λ may degrade VFL task accuracy (the top model relies on z_a being somewhat informative)
- Differentiable Fisher computation via scatter ops is non-trivial to implement correctly
- The optimization landscape may have saddle points: J_A = J_B can be achieved by degrading
  both (bad for task) or by specifically degrading z_a's label alignment (good for defense)

**Risks:**
- If the top model's loss gradient already tends to reduce Fisher_A (which it would NOT under
  MaliciousSGD), the regularization adds conflicting signal
- The batch Fisher approximation may be too noisy for stable training with small batch sizes

**Novelty:** High — Fisher equalization as a VFL defense regularization is not in the literature.
**Publication potential:** Mid-tier (USENIX Security / CCS) — the connection between Fisher
criterion, attack detection, and defense regularization is elegant and novel.

**Recommended experiments:**
1. Implement differentiable Fisher computation as a utility function, validate analytically
2. Run λ ∈ {0.01, 0.1, 1.0} at 100 epochs with seeds {42, 123, 456}
3. Log Fisher_A and Fisher_B trajectories during training under this defense
4. Compare model completion accuracy: (i) no defense, (ii) current defense, (iii) Fisher EQ

---

### 5.2 Anti-Fisher Regularization (Maximize Within-Class Variance of z_a)

**Core intuition:** A simpler version of Fisher equalization. Instead of equalizing Fisher_A
to Fisher_B, directly add a regularization term that MAXIMIZES within-class scatter of z_a
and/or MINIMIZES between-class scatter.

**Mathematical formulation:**

```
R_anti = -Tr(S_W^A) + Tr(S_B^A)    (note the signs — we want the OPPOSITE of Fisher)

L_total = L_task + λ_1 * (-Tr(S_W^A)) + λ_2 * Tr(S_B^A)
```

The first term (-Tr(S_W^A)) is minimized by maximizing within-class variance.
The second term (Tr(S_B^A)) is minimized by reducing between-class variance.
Together they push against the Fisher criterion directly.

**Simpler alternative:** Just minimize between-class scatter:
```
R_simple = Tr(S_B^A) = sum_c n_c * ||μ_c^A - μ^A||_2²
```

**Why this might work at 100 epochs:** If the server applies this regularization for 90+
epochs (after burn_in=8), the cumulative gradient toward reducing S_B^A competes with
MaliciousSGD's amplified gradient toward increasing S_B^A. The tug-of-war now happens in
the SAME tensor (z_a's Fisher component), not between different tensors as in the current defense.

**Implementation location:** Same as 5.1 — add to training loss in `vfl_framework.py`.

**Advantage over Section 5.1:** Simpler gradient computation (no ratio derivative).
**Disadvantage:** No equalization property — λ must be tuned more carefully.

**Novelty:** Medium-High
**Publication potential:** Workshop to mid-tier.

**Recommended experiment:** Run alongside Fisher Equalization (5.1) to compare.

---

### 5.3 Within-Class Variance Inflation via KL Divergence Target

**Core intuition:** Define a target distribution for z_a within each class: an isotropic
Gaussian N(μ_c, σ²_target * I). Use KL divergence between the empirical within-class
distribution and the target to add a regularization term.

**Mathematical formulation:**

For each class c, the empirical within-class distribution of z_a is approximated as
N(μ_c, Σ_c) where Σ_c is the within-class covariance.

Target distribution: N(μ_c, σ²_target * I) with large σ_target.

```
R_KL = sum_c KL(N(μ_c, Σ_c) || N(μ_c, σ²_target * I))
     = (1/2) * sum_c [Tr(Σ_c / σ²_target) + (d - Tr(log(Σ_c/σ²_target))) - d]
     ≈ (1/2σ²_target) * Tr(S_W^A / n_c)    (simplified for large σ_target)
```

This simplification shows R_KL ~ -Tr(S_W^A) (up to constants), consistent with Section 5.2.

**Strengths:** Provides a probabilistic interpretation and connects to Variational Inference.
**Weaknesses:** Computing per-class covariance Σ_c is expensive for 100 classes (CIFAR100).

**Novelty:** Medium (VAE-style regularization for VFL defense)
**Publication potential:** Mid-tier with formal analysis.

---

### 5.4 Between-Class Centroid Compression

**Core intuition:** Force the class centroids of z_a to be as close together as possible,
making it hard for MixMatch to distinguish classes even when the embeddings have clean
geometric structure.

**Mathematical formulation:**

```
R_centroid = sum_{c1 < c2} ||μ_c1^A - μ_c2^A||_2²    (sum of pairwise centroid distances)
```

Wait — to MINIMIZE separability, we want small centroid distances, so we MINIMIZE R_centroid.
Adding a term `-λ * sum_{c1 < c2} ||μ_c1^A - μ_c2^A||_2²` to the loss (negative, so
gradient descent REDUCES centroid distances):

```
L_total = L_task - λ * sum_{c1 < c2} ||μ_c1^A - μ_c2^A||_2²
```

This is a centroid attraction term — the server's regularization pulls class centroids
of Party A's embeddings closer together in embedding space.

**Critical warning:** This may cause dimensional collapse — all embeddings may converge to
a single point to minimize the centroid distance term, destroying all task utility.
Must pair with a term that prevents dimensional collapse (e.g., a reconstruction loss or
a constraint that ||z_a|| ≥ ε for all samples).

**Novelty:** Medium
**Publication potential:** Workshop level. Too aggressive by itself.

---

### 5.5 Cross-Party Separability Matching (Full Fisher Equalization)

**Core intuition:** Rather than just suppressing Fisher_A, force Fisher_A = Fisher_B
by simultaneously encouraging J_B while discouraging J_A. This is the bidirectional version
of Section 5.1.

**Mathematical formulation:**

```
L_total = L_task + λ * (J_A - J_B)²    (squared difference — symmetric penalty)
```

The gradient w.r.t. z_a is:
```
∂(J_A - J_B)²/∂z_a = 2(J_A - J_B) * ∂J_A/∂z_a
```

The gradient w.r.t. z_b is:
```
∂(J_A - J_B)²/∂z_b = -2(J_A - J_B) * ∂J_B/∂z_b
```

This simultaneously: (i) reduces J_A when J_A > J_B, and (ii) increases J_B when J_A > J_B.
The equilibrium is J_A = J_B.

**Potential issue:** The server controls the loss for the TOP MODEL. The regularization gradient
for z_a flows back to Party A, and for z_b flows back to Party B. But Party B is benign —
its J_B increasing may or may not be beneficial (we don't want Party B to also develop overly
class-discriminative embeddings, since that might leak information too, just not to the server
through MaliciousSGD).

**Novelty:** High
**Publication potential:** Mid-tier if formalized and the equilibrium properties are analyzed.

---

## 6. Information-Theoretic Defenses

These defenses provide the most formal and theoretically grounded privacy guarantees. They
are also the most computationally expensive and hardest to implement well.

### 6.1 Mutual Information Minimization via MINE

**Core intuition:** Directly minimize the mutual information I(z_a; y) — the information
that Party A's embeddings contain about the label. If I(z_a; y) ≈ 0, no inference attack
can succeed, regardless of the attack strategy.

**Why this is the holy grail:** All other defenses target specific attack mechanisms.
Minimizing mutual information provides a mechanism-agnostic defense.

**Mathematical formulation:**

Mutual information:
```
I(z_a; y) = E_{p(z_a, y)} [log p(z_a, y) / (p(z_a) * p(y))]
           = H(y) - H(y | z_a)
```

For the defense, we want to minimize I(z_a; y) while keeping I(z; y) (with both parties'
embeddings) high for task performance.

MINE (Mutual Information Neural Estimation, Belghazi et al. 2018):

```
I(z_a; y) ≥ sup_{T: Z×Y → R} [E_{p(z_a,y)}[T(z_a,y)] - log(E_{p(z_a)p(y)}[e^{T(z_a,y)}])]
```

where T is a statistics network (discriminator) parameterized by a neural network.

Training procedure:
1. Server trains a statistics network T_φ to maximize the MINE lower bound on I(z_a; y)
2. Server adds -λ * MINE(z_a; y) to the VFL training loss
3. The gradient of -MINE(z_a; y) w.r.t. z_a flows back to Party A as additional gradient

```
L_total = L_task(ŷ, y) - λ * MINE_estimate(z_a, y)
```

The negative sign: maximizing MINE gives an estimate of MI; including -λ*MINE in the
minimization loss discourages high MI between z_a and y.

**Full training loop with MINE defense:**

```python
# Outer loop (per batch):
# Step 1: Update T_φ (statistics network) to better estimate I(z_a; y)
for _ in range(n_critic_steps):
    z_a = bottom_model_a(data_a).detach()    # detached - we're training T_φ not θ_A
    z_a_shuffled = z_a[torch.randperm(len(z_a))]    # marginal approximation
    mi_estimate = T_φ(z_a, y).mean() - torch.log(torch.exp(T_φ(z_a_shuffled, y)).mean())
    loss_T = -mi_estimate    # maximize MINE lower bound
    optimizer_T.zero_grad()
    loss_T.backward()
    optimizer_T.step()

# Step 2: Update VFL parameters with MI penalty
z_a = bottom_model_a(data_a)    # not detached
z = concat(z_a, z_b)
y_hat = top_model(z)
L_task = cross_entropy(y_hat, y)
mi_estimate = T_φ(z_a, y).mean() - torch.log(torch.exp(T_φ(z_a_shuffled, y)).mean())
L_total = L_task - lambda_MI * mi_estimate    # minimize task loss, minimize MI
L_total.backward()
```

**Implementation location:**
- New class `MINEDefense` in `possible_defenses.py`
- `vfl_framework.py`: integrate MINE training loop into `simulate_train_round_per_batch()`
- New argument: `--mine-lambda`, `--mine-n-critic-steps`, `--mine-hidden-dim`

**Strengths:**
- Mechanism-agnostic: works against MaliciousSGD and any other label inference strategy
- Formal interpretation: bounds the label information in Party A's embeddings
- Gradient flows directly to z_a → Party A's bottom model learns MI-minimizing representations

**Weaknesses:**
- MINE is notoriously unstable to train — variance of the gradient is very high
- Requires training a separate statistics network T_φ with its own optimizer, lr schedule, etc.
- n_critic_steps inner loop increases training time by O(n_critic_steps)
- The MINE estimate can be poor for high-dimensional z_a (d > 64)
- λ controls the utility-privacy tradeoff but is hard to set without task degradation

**Risks:**
- Mode collapse of T_φ: statistics network always outputs constant → MI estimate is always 0
  → no gradient signal → defense inactive
- Gradient explosion in the critic training step (clip critic gradients, use spectral norm)
- MINE underestimates MI → may give false confidence that I(z_a; y) is low

**Novelty:** Medium-High (MINE has been used in other privacy contexts; VFL application is
less common but not entirely unstudied).
**Publication potential:** Top-tier (S&P / CCS / NeurIPS) if implementation is clean, stable,
and comparisons to other defenses are thorough. This is the right level of contribution for
a strong VFL privacy paper.

**Estimated implementation effort:** Very High (1-2 weeks of careful implementation and tuning).

**Recommended experiments:**
1. Start with a simple 2-layer statistics network T_φ (z_a and y as inputs, MLP output)
2. Validate MINE implementation on synthetic data (verify MI estimate matches analytic value)
3. Run n_critic_steps ∈ {1, 5, 10} — more critic steps = better MI estimate but slower
4. Run λ ∈ {0.01, 0.1, 1.0, 10.0} at 100 epochs
5. Check: does VFL task accuracy remain acceptable (>50% top-1 for CIFAR10)?

---

### 6.2 Information Bottleneck (IB) Regularization

**Core intuition:** The Information Bottleneck principle (Tishby & Zaslavsky, 2015) says
a good representation z_a should maximize I(z_a; ŷ) while minimizing I(z_a; x_a) and
I(z_a; y). For VFL defense, we want z_a to contain only the information needed for the
TASK (predict ŷ together with z_b) but not standalone label information I(z_a; y).

**Mathematical formulation:**

Standard IB objective:
```
min I(z_a; x_a) - β * I(z_a; ŷ)    (compress while preserving task-relevant information)
```

For VFL, we need a VERTICAL IB that distinguishes task information from label information:

```
L_VIB = -I(z_a, z_b; y) + β_A * I(z_a; x_a) + β_priv * I(z_a; y|z_b)
```

The last term `I(z_a; y|z_b)` is the conditional mutual information — the information in
z_a about y that CANNOT be explained by z_b alone. This is exactly what the label inference
attacker exploits when Party B is unavailable in Stage 2.

**Variational IB implementation:**

Parameterize the encoder as a stochastic mapping (as in Alemi et al., 2017):
```
z_a ~ q(z_a | x_a) = N(μ_A(x_a), σ_A²(x_a))    (reparameterization trick)
```

The IB objective becomes (ELBO):
```
L_VIB = E[L_task] + λ * KL[q(z_a|x_a) || r(z_a)]

where r(z_a) = N(0, I) is the marginal prior
```

The KL term encourages z_a to be close to the prior (a standard Gaussian), which destroys
label-specific structure in the embedding — exactly what the defense needs.

**Implementation location:**
- Modify `model_sets.py`: make `bottom_model_a` output (μ, log σ) instead of z
- `vfl_framework.py`: sample z_a = μ + ε*σ (reparameterization), add KL loss

**Key code change in `model_sets.py`:**

```python
class StochasticBottomModel(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = ...    # existing architecture
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

# In vfl_framework.py:
mu_a, logvar_a = bottom_model_a(data_a)
eps = torch.randn_like(mu_a)
z_a = mu_a + eps * torch.exp(0.5 * logvar_a)    # reparameterization
kl_a = -0.5 * torch.mean(1 + logvar_a - mu_a**2 - logvar_a.exp())
L_total = L_task + lambda_IB * kl_a
```

**Strengths:**
- Principled: the KL term has a clear interpretation as "how much information about x_a"
- The prior N(0,I) discourages class-discriminative structure in z_a
- Party A's model is now a variational encoder — each inference run produces a different z_a
  → Stage 2 model completion gets stochastic embeddings → inference unstable

**Weaknesses:**
- Makes the entire bottom model stochastic — this is a significant architectural change
- The beta hyperparameter (λ_IB) controls utility-privacy tradeoff; must be tuned carefully
- Stage 2 model completion's behavior with stochastic embeddings is unpredictable
- The KL term penalizes ALL information in z_a, not just label-specific information

**Risks:**
- The bottom model may collapse to outputting z_a ≈ N(0,I) for all inputs if λ_IB is too large
- High λ_IB → poor VFL task performance; low λ_IB → insufficient privacy

**Novelty:** Medium-High (VIB has been studied for FL; VFL-specific application with attack
detection trigger is novel).
**Publication potential:** Mid-tier to top-tier depending on the formalism depth.

**Recommended experiments:**
1. Start with λ_IB = 0.001 (very small) to verify architecture works
2. Progressively increase λ_IB; plot utility-privacy curve
3. Measure: (a) VFL task accuracy, (b) KL divergence from prior, (c) model completion accuracy
4. Compare to Option B (non-stochastic embedding noise) at matched utility levels

---

### 6.3 Maximum Mean Discrepancy (MMD) for Class Distribution Alignment

**Core intuition:** Instead of minimizing MI directly, use MMD to force the class-conditional
distributions of z_a to be indistinguishable. If p(z_a | y=c₁) ≈ p(z_a | y=c₂) for all
classes c₁, c₂, then z_a carries no discriminative information about y.

**Mathematical formulation:**

MMD with kernel k (typically RBF):
```
MMD²(P, Q) = E_{x~P, x'~P}[k(x,x')] - 2*E_{x~P, y~Q}[k(x,y)] + E_{y~Q, y'~Q}[k(y,y')]
```

For VFL defense, compute MMD between class-conditional distributions of z_a:
```
R_MMD = sum_{c1 < c2} MMD²(p(z_a|y=c1), p(z_a|y=c2))
```

This penalizes when class-conditional distributions are distinguishable.

```
L_total = L_task + λ * R_MMD
```

**Differentiable approximation (per batch):**
```python
def mmd_class_alignment(z_a, labels, kernel='rbf', gamma=1.0):
    classes = labels.unique()
    total_mmd = 0
    for i, c1 in enumerate(classes):
        for c2 in classes[i+1:]:
            z_c1 = z_a[labels == c1]
            z_c2 = z_a[labels == c2]
            total_mmd += compute_mmd(z_c1, z_c2, kernel, gamma)
    return total_mmd / (len(classes) * (len(classes)-1) / 2)
```

**Key advantage over MI minimization:** MMD is fully differentiable, does not require a
separate statistics network (unlike MINE), and has a closed form for RBF kernels.

**Weaknesses:**
- Computing O(C²) pairwise MMD terms is expensive for CIFAR100 (C=100 → 4950 terms per batch)
- Requires choosing the kernel bandwidth γ — needs tuning
- MMD alignment does not imply MI = 0, only that marginals are matched

**Novelty:** High (MMD for VFL label privacy is not published to our knowledge)
**Publication potential:** Mid-tier with formal analysis of the MMD-to-privacy guarantee chain.

**Computational shortcut for CIFAR100:** Instead of all C(C-1)/2 pairs, compute:
```
R_MMD_approx = MMD²(z_a, z_a_shuffled_label)    (shuffle labels → approximate marginal)
```
This approximates the total variation distance between the joint and marginal distributions.

---

### 6.4 Entropy Maximization of Embeddings

**Core intuition:** Maximize the entropy of z_a's distribution (across all samples, ignoring
labels). High entropy means z_a takes diverse values, which makes it harder to form the
tight, well-separated clusters that MixMatch needs for pseudo-label assignment.

**Mathematical formulation:**

Differential entropy of z_a:
```
H(z_a) = -E[log p(z_a)]
```

For a Gaussian approximation N(μ, Σ):
```
H(z_a) = (d/2)(1 + log(2π)) + (1/2)log|Σ|    ∝ log|Σ|
```

Maximizing H(z_a) ≈ maximizing the determinant of the empirical covariance of z_a.

Differentiable approximation:
```
R_entropy = -log|Σ_total|    (minimize this to maximize entropy)
           ≈ -sum of log singular values of z_a matrix
```

**Implementation:**
```python
z_a_centered = z_a - z_a.mean(0)
U, S, V = torch.linalg.svd(z_a_centered)
log_det = torch.log(S + 1e-8).sum()
R_entropy = -log_det    # minimize to maximize entropy (maximize log det)
```

**Relationship to other defenses:**
- High total entropy does NOT mean low within-class variance (can have many tight, spread-out clusters)
- Need CONDITIONAL entropy: H(z_a | y) high, H(z_a) can be anything
- Pure entropy maximization is insufficient; need to pair with a class-conditioning term

**Novelty:** Low as standalone. Worth exploring as a component of a hybrid defense.

---

### 6.5 Conditional Entropy Maximization H(z_a | y)

**Core intuition:** Maximize within-class entropy of z_a, i.e., ensure that within any
given class, z_a takes diverse, spread-out values. This directly attacks class separability.

**Mathematical formulation:**

```
H(z_a | y) = -sum_c p(y=c) * E_{z_a|y=c}[log p(z_a|y=c)]

For Gaussian class conditionals:
H(z_a|y) ≈ (1/C) * sum_c (1/2) * log|Σ_c| + constant
```

Maximizing H(z_a|y) ≈ maximizing average within-class covariance determinant.

Differentiable approximation (per class, per batch):
```
L_entropy = -(1/C) * sum_c log|Σ_c^batch|
           ≈ -(1/C) * sum_c sum log sv(Z_c)

where Z_c = matrix of z_a samples for class c in the batch
```

This is closely related to Section 5.2 (Anti-Fisher) but in the entropy formulation:
- Anti-Fisher: maximize Tr(S_W^A) [within-class variance, trace]
- Conditional entropy max: maximize log|Σ_c| [within-class variance, log-determinant]

The log-determinant version is stronger: it requires the variance to be SPREAD ACROSS ALL
DIMENSIONS, not concentrated in a few directions.

**Novelty:** Medium-High as a VFL defense regularization term.
**Publication potential:** Mid-tier in combination with Fisher equalization.

---

## 7. Adversarial Training Defenses

These methods treat the privacy defense as a minimax game between a defender (server) and
the potential attacker's inference strategy.

### 7.1 Adversarial Autoencoder for Embedding Obfuscation

**Core intuition:** Train an auxiliary network on the server side that learns to produce
obfuscated versions of z_a that: (a) preserve task-relevant information for the top model,
and (b) minimize label inference accuracy of any inference model.

**Architecture:**

```
Party A → bottom_model_a → z_a → [SERVER OBFUSCATOR: g_φ] → z_a_obf
                                        ↓
                               top_model(concat(z_a_obf, z_b)) → ŷ
```

The obfuscator g_φ is trained simultaneously with the VFL system:
- Against the task loss: g_φ should preserve task-useful information
- Against an adversarial classifier: g_φ should fool a label classifier on z_a_obf

**Minimax objective:**

```
min_{θ_A, θ_server, φ} max_{θ_adv}  L_task - λ * L_adv

where:
    L_task = CE(top_model(g_φ(z_a), z_b), y)    (task performance)
    L_adv  = CE(classifier_adv(z_a_obf), y)     (adversary's label inference)
```

The adversarial classifier `classifier_adv` is updated to MAXIMIZE label prediction accuracy
from z_a_obf. The obfuscator g_φ is updated to MINIMIZE the adversary's accuracy.

**Training alternation:**

```
# Adversary step:
z_a_obf = g_φ(z_a).detach()
loss_adv = CE(classifier_adv(z_a_obf), y)
loss_adv.backward(); optimizer_adv.step()

# Obfuscator + VFL step:
z_a_obf = g_φ(z_a)
z = concat(z_a_obf, z_b)
y_hat = top_model(z)
loss_task = CE(y_hat, y)
loss_priv = CE(classifier_adv(z_a_obf), y)
loss_total = loss_task - lambda_adv * loss_priv
loss_total.backward(); optimizer_vfl.step()
```

**Important note:** In this framework, Party A's bottom model is an UNWILLING participant
in the obfuscation. The server applies g_φ to z_a before the top model, and Party A's
gradient flows through g_φ first. If MaliciousSGD amplifies the gradient BEFORE it reaches
Party A, it amplifies (∂L/∂z_a_obf * ∂z_a_obf/∂z_a), which is modulated by the Jacobian
of g_φ — effectively, the obfuscator changes the gradient landscape that MaliciousSGD operates in.

**Strengths:**
- The obfuscator is trained to be robust against ANY label classifier, not just a specific attack
- The adversary training provides a direct optimization signal against label leakage
- Does not require a separate detection signal (Fisher divergence) — can run unconditionally

**Weaknesses:**
- Mode collapse: the adversary collapses to a trivial strategy; the obfuscator overfits to it
- Training instability (common to all minimax games)
- The server-side obfuscator g_φ must be non-trivial (if g_φ is too weak, Party A can use the
  obfuscated gradient to reconstruct the original label structure)
- CIFAR100 with C=100 classes requires a capable adversary (more complex architecture)

**Risks:**
- If the adversary is too weak, the defense gives false confidence (easy to fool a weak adversary
  but a real MixMatch-based inference can still succeed)
- If the adversary is too strong, the obfuscator may sacrifice too much task utility

**Novelty:** Medium (adversarial representation learning for FL privacy is studied in HFL;
less studied in VFL with bottom model gradient access)
**Publication potential:** Mid-tier to top-tier depending on the analysis depth.

**Recommended experiments:**
1. Start with a 2-layer MLP obfuscator g_φ (same dimension as z_a)
2. Start with a 2-layer MLP adversary classifier_adv
3. Run with fixed MaliciousSGD (so we know what the "attack" looks like) to verify defense works
4. Then test generalization: does the defense work against a stronger inference method than the one used in training?

---

### 7.2 Domain Adversarial Training for Label-Invariant Features

**Core intuition:** Use DANN (Domain-Adversarial Neural Networks, Ganin et al. 2016) to
train Party A's bottom model to produce features that are INVARIANT with respect to the
label. Treat the label y as the "domain" and train a gradient reversal layer to confuse a
label discriminator.

**Mathematical formulation:**

Modified training objective for Party A:
```
L_A = L_task(ŷ, y) - λ * L_domain(D(z_a), y)

where D is a label discriminator (tries to predict y from z_a)
      L_domain = CE(D(z_a), y)
```

The gradient reversal layer (GRL): during the forward pass, acts as identity. During
backward pass, negates the gradient. This causes the bottom model to MAXIMIZE label confusion
while the discriminator minimizes label prediction error.

In VFL:
- Party B produces z_b → NOT subject to domain adversarial training
- Party A's bottom model is trained with reversed gradient from label discriminator
- MaliciousSGD still amplifies ALL of Party A's gradients, including the reversed signal

**Critical interaction with MaliciousSGD:**
MaliciousSGD amplifies p.grad by ratio r. The GRL provides gradient -∂L_domain/∂z_a → 
-∂L_domain/∂θ_A. After MaliciousSGD: -r * ∂L_domain/∂θ_A. This means MaliciousSGD
accidentally AMPLIFIES the domain confusion gradient — a fortuitous effect. The attack
that amplifies gradients also amplifies the defense's adversarial gradient.

**This is a potentially powerful insight:** If the domain adversarial gradient dominates
the task gradient for Party A at 100 epochs, MaliciousSGD's amplification makes the defense
STRONGER rather than weaker — the opposite of the current defense's failure mode.

**Implementation location:**
- `model_sets.py`: add `GradientReversalLayer` class
- `vfl_framework.py`: add discriminator D training loop; modify Party A's backward pass

**Novelty:** High — the MaliciousSGD interaction with GRL (adversarial amplification of the
domain confusion gradient) is a novel and potentially publishable insight.
**Publication potential:** Top-tier potential if the interaction with MaliciousSGD is formally
analyzed and empirically verified.

**Recommended experiments:**
1. Implement GRL using the standard `torch.autograd.Function` approach
2. Run λ ∈ {0.1, 1.0, 10.0}
3. Specifically test: does higher MaliciousSGD gamma → STRONGER defense (because amplification
   of the adversarial gradient is proportional to gamma)?
4. This would be a novel result: the attack's own amplification mechanism is weaponized against it

---

### 7.3 Minimax Label Confusion via Online Adversary

**Core intuition:** Unlike the full adversarial autoencoder (7.1), this approach maintains
a lightweight ONLINE adversary that is always trying to infer labels from the most recent
z_a, providing a real-time signal for how much label information z_a currently contains.

**Implementation difference from 7.1:**
- No obfuscator network g_φ
- The adversary's loss is negated and directly backpropagated into the VFL task loss
- Simpler to implement than a full minimax game

```
L_total = L_task - λ * L_adv_online

where L_adv_online = CE(linear_head(z_a), y)   (a single linear layer adversary)
```

Using a LINEAR adversary has an information-theoretic interpretation: minimizing linear
adversary accuracy minimizes the label information that is LINEARLY ACCESSIBLE from z_a.
Non-linear inference (like MixMatch) can still potentially succeed if there is non-linear
label structure.

**For stronger defense:** Use a 2-layer MLP adversary to capture non-linear structure.

**Strengths:** Simple to implement, direct gradient signal, no separate training loop needed.
**Weaknesses:** May not generalize to the specific MixMatch inference method used in Stage 2.

**Novelty:** Medium
**Publication potential:** Workshop to mid-tier.

---

### 7.4 Feature Distribution Matching (Maximizing Alignment with Reference Distribution)

**Core intuition:** Train Party A's embeddings to be indistinguishable from a reference
distribution that carries no label information (e.g., embeddings from a benign Party A that
was never exposed to MaliciousSGD). The server maintains a reference embedding distribution
from a pretrained benign checkpoint and uses discrepancy minimization to align z_a.

**Mathematical formulation:**

Let z_a_ref ~ p_benign(z_a) be embeddings from a benign Party A.
Let z_a_att ~ p_attack(z_a) be embeddings from Party A under MaliciousSGD.

Server adds a discrepancy term:
```
L_total = L_task + λ * D(p(z_a), p_ref)

where D is a distributional distance (Wasserstein, MMD, or KL divergence)
```

**Key insight:** If z_a is indistinguishable from the benign reference, then no attack that
relies on the extra structure created by MaliciousSGD can succeed.

**Practical implementation:** Store a small reference distribution (e.g., 1000 samples from
a pre-run benign Party A at epoch T) as a fixed reference set. Use MMD or maximum likelihood
to measure discrepancy per batch.

**Novelty:** Medium-High (reference distribution matching for VFL privacy is not published)
**Publication potential:** Mid-tier.

---

## 8. Optimization-Based Defenses (Regularization in Loss)

### 8.1 Orthogonality Constraints on Party A's Embeddings

**Core intuition:** Force Party A's embedding to be orthogonal to a linear probe for the
label y. If ⟨z_a, w*⟩ ≈ 0 for all label directions w*, z_a cannot support label inference.

**Mathematical formulation:**

Compute a label direction estimate (per epoch):
```
w_c = μ_c^A - μ^A    (class centroid deviation for class c)
```

Orthogonality constraint:
```
R_ortho = sum_c ||z_a * w_c||_2²   (project z_a onto label directions, penalize)
        = ||W * z_a||_F²
```

where W = matrix of class centroid deviations.

```
L_total = L_task + λ * R_ortho
```

**Gradient:** ∂R_ortho/∂z_a = 2 * W^T * (W * z_a) — pushes z_a to be orthogonal to
all class centroid directions.

**Strengths:**
- Interpretable: explicitly removes label-discriminative directions from z_a
- Computationally efficient: orthogonal projection is a linear operation
- Can be applied selectively: only to the dimensions most correlated with label directions

**Weaknesses:**
- Class centroid directions w_c are computed from the current batch — noisy estimate
- MaliciousSGD is trying to ALIGN z_a with w_c; the orthogonality constraint is a direct
  counterpressure. The tug-of-war plays out in the optimization: which gradient is larger?
- Effective only if the orthogonality gradient magnitude exceeds MaliciousSGD's amplification.
  The defense gradient is λ * 2 * W^T * W * z_a; the attack amplifies up to 5× task gradient.
  Need λ large enough to compete.

**The tug-of-war equation:**
```
Update to θ_A for task gradient: -lr * r * ∂L_task/∂θ_A    (MaliciousSGD, r up to 5)
Update to θ_A for ortho gradient: -lr * λ * ∂R_ortho/∂θ_A
```

For the defense to win: λ * ||∂R_ortho/∂θ_A|| > r * ||∂L_task/∂θ_A||
With r=5: λ > 5 * ||∂L_task/∂θ_A|| / ||∂R_ortho/∂θ_A||

This is a computable quantity — we can measure the required λ empirically.

**Novelty:** Medium (orthogonality constraints appear in disentangled representation learning;
VFL defense application is novel)
**Publication potential:** Mid-tier if combined with theoretical analysis of the tug-of-war condition.

**Recommended experiments:**
1. Log ||∂L_task/∂θ_A|| and ||∂R_ortho/∂θ_A|| per epoch (no changes to training)
2. Compute the required λ to overcome MaliciousSGD's amplification
3. Run with that λ at 100 epochs; verify the balance condition holds

---

### 8.2 Contrastive Label Confusion (Attracting Same-Class Embeddings, Repelling Same-Label)

**Core intuition:** Use contrastive learning in REVERSE: instead of pushing same-class samples
together (which MaliciousSGD does), PUSH THEM APART. Within-class distances should be large;
between-class distances should be small.

**Mathematical formulation (Reversed SimCLR):**

```
L_confusion = -(1/N) * sum_{i,j: y_i=y_j, i≠j} log [ exp(-||z_a^i - z_a^j||² / τ) /
                                                       sum_k exp(-||z_a^i - z_a^k||² / τ) ]
```

Note the negative sign — this is the NEGATIVE of the InfoNCE loss. Minimizing L_confusion
pushes same-class embeddings APART (reversed from standard contrastive learning).

Simultaneously, we can attract different-class embeddings:
```
L_mix = (1/N) * sum_{i,j: y_i≠y_j} ||z_a^i - z_a^j||²    (minimize inter-class distance)
```

**The combined reversed contrastive loss:**
```
L_total = L_task + λ_1 * L_confusion + λ_2 * L_mix
```

**Novelty:** High — reversed contrastive learning for VFL privacy is novel.
**Publication potential:** Mid-tier to top-tier. The reversal of a standard technique for
privacy purposes is an elegant and publishable idea.

**Risk:** If L_confusion is too strong, the task loss cannot compensate → embeddings become
random → poor VFL accuracy. The λ hyperparameters require careful tuning.

**Recommended experiments:**
1. Implement with a single λ (set λ_1 = λ, λ_2 = 0) to test confusion alone
2. Add λ_2 term and check if inter-class attraction provides additional benefit
3. Run at τ ∈ {0.1, 0.5, 1.0}, λ ∈ {0.01, 0.1, 1.0}

---

### 8.3 Wasserstein Regularization for Marginal Distribution Alignment

**Core intuition:** Use the Wasserstein distance (Earth Mover's Distance) to measure how
different z_a's distribution is from a label-neutral reference, and minimize this distance
as a regularizer.

**Mathematical formulation:**

Let P = empirical distribution of z_a under the current model.
Let Q = reference distribution (e.g., standard Gaussian, or benign Party A embeddings).

```
L_Wasserstein = W_1(P, Q)    (1-Wasserstein distance)
L_total = L_task + λ * W_1(P, Q)
```

For efficient computation, use the Sinkhorn algorithm (entropic regularized OT):
```
W_ε(P, Q) = min_{T ∈ Π(P,Q)} [sum_{i,j} T_{ij} * c(z_a^i, q^j) - ε * H(T)]
```

where c is a cost function (typically squared L2), Π(P,Q) is the set of valid transport
plans, and ε is an entropic regularization parameter.

**Python implementation:** Available via the `geomloss` library (POT: Python Optimal Transport).

**Strengths:**
- The Wasserstein distance is geometrically meaningful (takes into account the structure of
  the space, unlike total variation distance)
- Regularizing toward a standard Gaussian reference destroys class-specific structure

**Weaknesses:**
- Wasserstein distance computation is expensive (O(n² log n) for exact; O(n²) for Sinkhorn)
- Requires choosing a reference distribution Q that is label-neutral
- Gradient through Sinkhorn requires careful implementation

**Novelty:** Medium-High for VFL context
**Publication potential:** Mid-tier.

---

### 8.4 Invariant Risk Minimization (IRM) for Cross-Party Feature Independence

**Core intuition:** IRM (Arjovsky et al., 2019) finds representations that are invariant
across different "environments." In VFL, we can define the two parties' data distributions
as two environments and seek features that generalize across them — this naturally prevents
either party from developing environment-specific (and thus label-leaking) representations.

**Mathematical formulation:**

IRM objective:
```
min_{Φ, w} sum_e [R^e(w ∘ Φ)] + λ * ||∇_{w|w=1} R^e(w ∘ Φ)||²
```

For VFL:
- Environments: E = {condition that z_a was produced, condition that z_b was produced}
- Representation: Φ = concat(z_a, z_b)
- Penalty: the IRM penalty forces the linear classifier on top of Φ to be the same optimal
  linear classifier in both "environments"

**Why this is relevant to VFL privacy:** If z_a alone cannot support an optimal linear
classifier (because the full representation requires both z_a and z_b), then any attack
on z_a alone will fail.

**Novelty:** Medium (IRM is a standard technique; VFL application for privacy is less studied)
**Publication potential:** Workshop to mid-tier.

**Note:** IRM is primarily designed for out-of-distribution generalization; adapting it for
privacy in VFL requires a creative redefinition of "environments." This is research-level work.

---

*Continue reading in `possible_directions_3.md` (Sections 9–15: Server-Side, Detection-Only, Hybrid, Novel Hypotheses, Comparison Table, and Roadmap)*
