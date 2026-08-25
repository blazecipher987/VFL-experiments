import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


# Privacy Preserving Deep Learning
def bound(grad, gamma):
    if grad < -gamma:
        return -gamma
    elif grad > gamma:
        return gamma
    else:
        return grad


def generate_lap_noise(beta):
    # beta = sensitivity / epsilon
    u1 = np.random.random()
    u2 = np.random.random()
    if u1 <= 0.5:
        n_value = -beta * np.log(1. - u2)
    else:
        n_value = beta * np.log(u2)
    # print(n_value)
    return n_value


def sigma(x, c, sensitivity):
    x = 2. * c * sensitivity / x
    return x


def get_grad_num(layer_grad_list):
    num_grad = 0
    num_grad_per_layer = []
    for grad_tensor in layer_grad_list:
        num_grad_this_layer = 0
        if len(grad_tensor.shape) == 1:
            num_grad_this_layer = grad_tensor.shape[0]
        elif len(grad_tensor.shape) == 2:
            num_grad_this_layer = grad_tensor.shape[0] * grad_tensor.shape[1]
        num_grad += num_grad_this_layer
        num_grad_per_layer.append(num_grad_this_layer)
    return num_grad, num_grad_per_layer


def get_grad_layer_id_by_grad_id(num_grad_per_layer, id):
    id_layer = 0
    id_temp = id
    for num_grad_this_layer in num_grad_per_layer:
        id_temp -= num_grad_this_layer
        if id_temp >= 0:
            id_layer += 1
        else:
            id_temp += num_grad_this_layer
            break
    return id_layer, id_temp


def get_one_grad_by_grad_id(layer_grad_list, num_grad_per_layer, id):
    id_layer, id_in_this_layer = get_grad_layer_id_by_grad_id(num_grad_per_layer, id)
    grad_this_layer = layer_grad_list[id_layer]
    if len(grad_this_layer.shape) == 1:
        the_grad = grad_this_layer[id_in_this_layer]
    else:
        the_grad = grad_this_layer[id_in_this_layer // grad_this_layer.shape[1]][
            id_in_this_layer % grad_this_layer.shape[1]]
    return the_grad


def set_one_grad_by_grad_id(layer_grad_list, num_grad_per_layer, id, set_value):
    id_layer, id_in_this_layer = get_grad_layer_id_by_grad_id(num_grad_per_layer, id)
    grad_this_layer = layer_grad_list[id_layer]
    if len(grad_this_layer.shape) == 1:
        layer_grad_list[id_layer][id_in_this_layer] = set_value
    else:
        layer_grad_list[id_layer][id_in_this_layer // grad_this_layer.shape[1]][
            id_in_this_layer % grad_this_layer.shape[1]] = set_value


def dp_gc_ppdl(epsilon, sensitivity, layer_grad_list, theta_u, gamma, tau):
    grad_num, num_grad_per_layer = get_grad_num(layer_grad_list)
    c = int(theta_u * grad_num)
    # print("c:", c)
    # exit()
    epsilon1 = 8. / 9 * epsilon
    epsilon2 = 2. / 9 * epsilon
    used_grad_ids = []
    really_useful_grad_ids = []
    done_grad_count = 0
    while 1:
        r_tau = generate_lap_noise(sigma(epsilon1, c, sensitivity))
        while 1:
            while 1:
                grad_id = random.randint(0, grad_num - 1)
                if grad_id not in used_grad_ids:
                    used_grad_ids.append(grad_id)
                    break
                if len(used_grad_ids) == grad_num:
                    return
            grad = get_one_grad_by_grad_id(layer_grad_list, num_grad_per_layer, grad_id)
            r_w = generate_lap_noise(2 * sigma(epsilon1, c, sensitivity))
            if abs(bound(grad, gamma)) + r_w >= tau + r_tau:
                r_w_ = generate_lap_noise(sigma(epsilon2, c, sensitivity))
                set_one_grad_by_grad_id(layer_grad_list, num_grad_per_layer, grad_id, bound((grad + r_w_), gamma))
                really_useful_grad_ids.append(grad_id)
                done_grad_count += 1
                if done_grad_count >= c:
                    for id in range(0, grad_num):
                        if id not in really_useful_grad_ids:
                            set_one_grad_by_grad_id(layer_grad_list, num_grad_per_layer, id, 0.)
                    # print("really_useful_grad_ids:", really_useful_grad_ids)
                    # print("len really_useful_grad_ids:", len(really_useful_grad_ids))
                    # exit()
                    return
                else:
                    break


# Multistep gradient
def multistep_gradient(tensor, bound_abs, bins_num=12):
    # Criteo 1e-3
    max_min = 2 * bound_abs
    interval = max_min / bins_num
    tensor_ratio_interval = torch.div(tensor, interval)
    tensor_ratio_interval_rounded = torch.round(tensor_ratio_interval)
    tensor_multistep = tensor_ratio_interval_rounded * interval
    return tensor_multistep


# Gradient Compression
class TensorPruner:
    def __init__(self, zip_percent):
        self.thresh_hold = 0.
        self.zip_percent = zip_percent

    def update_thresh_hold(self, tensor):
        tensor_copy = tensor.clone().detach()
        tensor_copy = torch.abs(tensor_copy)
        survivial_values = torch.topk(tensor_copy.reshape(1, -1),
                                      int(tensor_copy.reshape(1, -1).shape[1] * self.zip_percent))
        self.thresh_hold = survivial_values[0][0][-1]

    def prune_tensor(self, tensor):
        # whether the tensor to process is on cuda devices
        background_tensor = torch.zeros(tensor.shape).to(torch.float)
        if 'cuda' in str(tensor.device):
            background_tensor = background_tensor.cuda()
        # print("background_tensor", background_tensor)
        tensor = torch.where(abs(tensor) > self.thresh_hold, tensor, background_tensor)
        # print("tensor:", tensor)
        return tensor


# Differential Privacy(Noisy Gradients)
class DPLaplacianNoiseApplyer():
    def __init__(self, beta):
        self.beta = beta

    def noisy_count(self):
        # beta = sensitivity / epsilon
        beta = self.beta
        u1 = np.random.random()
        u2 = np.random.random()
        if u1 <= 0.5:
            n_value = -beta * np.log(1. - u2)
        else:
            n_value = beta * np.log(u2)
        n_value = torch.tensor(n_value)
        # print(n_value)
        return n_value

    def laplace_mech(self, tensor):
        # generate noisy mask
        # whether the tensor to process is on cuda devices
        noisy_mask = torch.zeros(tensor.shape).to(torch.float)
        if 'cuda' in str(tensor.device):
            noisy_mask = noisy_mask.cuda()
        noisy_mask = noisy_mask.flatten()
        for i in range(noisy_mask.shape[0]):
            noisy_mask[i] = self.noisy_count()
        noisy_mask = noisy_mask.reshape(tensor.shape)
        # print("noisy_tensor:", noisy_mask)
        tensor = tensor + noisy_mask
        return tensor


# ── Novel Defense: Asymmetric Adaptive Perturbation ────────────────────────────
#
# NOVELTY
# -------
# Every existing defense above (PPDL, GC, Laplace DP, Multistep Gradient)
# applies *identical* perturbation to BOTH parties' gradients.  They are
# statistically blind — they cannot distinguish an attacker from an honest
# party, so they penalise everyone and hurt overall VFL utility.
#
# This defense breaks that symmetry:
#
#   1. DETECTION — The server tracks the Fisher divergence
#      (Fisher_A − Fisher_B) via the SeparabilityMonitor.
#      Under MaliciousSGD, Party A's internal gradient amplification makes
#      its embedding space progressively more class-discriminative than
#      Party B's, so the divergence grows strongly positive.
#      Under benign training the divergence stays near zero.
#
#   2. ASYMMETRIC RESPONSE — Only Party A's gradient is suppressed when the
#      divergence exceeds a threshold.  Party B's gradient is never touched,
#      so honest parties bear zero utility cost.
#
#   3. ADAPTIVE SCALING — Suppression is proportional to divergence severity.
#      Mild anomaly → mild attenuation.  Strong attack → strong attenuation.
#      No binary on/off switch means the defense degrades gracefully.
#
# MECHANISM (per batch, after the backward pass on the top model)
# ---------------------------------------------------------------
#   divergence d = Fisher_A − Fisher_B   (refreshed once per epoch)
#
#   if epoch >= burn_in  AND  d > tau:
#       scale = max(0.0,  1.0 − alpha * (d − tau))
#       grad_output_bottom_model_a  *=  scale       # Party A suppressed
#       grad_output_bottom_model_b     unchanged     # Party B untouched
#
# HYPERPARAMETERS
# ---------------
#   alpha    aggressiveness of suppression.
#            With tau=0.10, alpha=1.0: scale hits 0 when d = 1.10.
#            Increase alpha to be more aggressive; decrease to be more lenient.
#            Default: 1.0
#
#   tau      detection threshold.  Must be set above the benign divergence
#            ceiling.  Our experiments show benign ≈ 0.0 on CIFAR10/100,
#            so tau=0.10 gives a comfortable margin.
#            Default: 0.10
#
#   burn_in  epochs to wait before activating.  Random initialisation creates
#            spurious divergence for ~8 epochs; the defense would fire on noise
#            without this guard.
#            Default: 8
# ──────────────────────────────────────────────────────────────────────────────
class AsymmetricAdaptivePerturbation:

    def __init__(self, alpha: float = 1.0, tau: float = 0.10, burn_in: int = 8,
                 gradient_noise_std: float = 0.0, sign_flip: bool = False,
                 za_noise_std: float = 0.0):
        self.alpha = alpha
        self.tau = tau
        self.burn_in = burn_in
        self.gradient_noise_std = gradient_noise_std
        self.sign_flip = sign_flip
        self.za_noise_std = za_noise_std

        # Stores the Fisher divergence from the most recently completed epoch.
        # Starts at 0.0 so the defense is fully passive during the first epoch,
        # before any monitor data exists.
        self.current_divergence = 0.0
        # Toggled each batch when sign_flip is active and the defense fires.
        self._flip_toggle = False

    def get_scale_factor(self) -> float:
        """
        Returns the multiplier for Party A's gradient.

        Returns 1.0 (no change) when divergence is at or below tau.
        Linearly decreases toward 0.0 as divergence exceeds tau.
        Clipped at 0.0 so gradients are never reversed.

          scale = max(0.0,  1.0 − alpha * (divergence − tau))
        """
        d = self.current_divergence
        if d <= self.tau:
            # divergence is within normal range; no intervention needed
            return 1.0
        return max(0.0, 1.0 - self.alpha * (d - self.tau))

    def apply(self, grad_output_a: torch.Tensor, epoch: int):
        """
        Attenuates grad_output_a and optionally injects gradient noise.

        Called every batch inside simulate_train_round_per_batch(), AFTER the
        symmetric defenses have already been applied.

        When gradient_noise_std > 0, calibrated Gaussian noise is injected
        alongside suppression.  Noise magnitude is referenced to the pre-suppression
        gradient so it remains meaningful even when scale -> 0 (preventing the
        MaliciousSGD zero-gradient vacuum that caused Phase 6A to fail on CIFAR-100).

        Args:
            grad_output_a      : gradient tensor the server is about to send to Party A
                                 (shape: [batch_size, embedding_dim])
            epoch              : current training epoch, used for the burn-in guard

        Returns:
            (grad_output_a, scale)
              grad_output_a — the (possibly scaled + noised) gradient tensor
              scale         — the suppression scale factor (1.0 = no change)
        """
        if epoch < self.burn_in:
            # During burn-in, embeddings are not yet stable; divergence readings
            # are unreliable, so we skip suppression entirely.
            return grad_output_a, 1.0

        scale = self.get_scale_factor()
        if scale < 1.0:
            if self.sign_flip:
                # Alternate sign every batch when defense fires.
                # Consecutive opposite-sign gradients force MaliciousSGD ratio =
                # clamp(1 + gamma*(−g/g), 1, 5) = 1.0 — amplification collapses
                # to standard SGD and weight updates average to zero over pairs.
                self._flip_toggle = not self._flip_toggle
                sign = -1.0 if self._flip_toggle else 1.0
                grad_output_a = grad_output_a * (scale * sign)
            elif self.gradient_noise_std > 0.0:
                # Calibrate noise BEFORE suppression so the reference magnitude
                # is non-zero even when scale -> 0.
                # noise_mag = std * (1 - scale) * E[|grad|]
                # At scale=0: output is pure noise  -> MaliciousSGD amplifies random directions
                # At scale=0.5: output = 0.5*signal + 0.5*noise -> partial disruption
                noise_mag = self.gradient_noise_std * (1.0 - scale) * grad_output_a.abs().mean()
                grad_output_a = grad_output_a * scale + torch.randn_like(grad_output_a) * noise_mag
            else:
                # Suppression only (original behaviour).
                grad_output_a = grad_output_a * scale
        return grad_output_a, scale

    def apply_za_corruption(self, output_tensor_a: torch.Tensor, epoch: int) -> torch.Tensor:
        """
        Returns a noisy copy of z_a for the top model's forward pass.
        The caller should assign only the .data of the returned tensor to
        input_tensor_top_model_a; output_tensor_a itself is left untouched so the
        separability monitor and the bottom model's backward path see clean embeddings.
        Noise magnitude scales with suppression intensity: zero when scale=1 (no attack
        detected), maximal when scale=0 (full suppression).
        """
        if epoch < self.burn_in or self.za_noise_std <= 0.0:
            return output_tensor_a
        scale = self.get_scale_factor()
        if scale < 1.0:
            noise = self.za_noise_std * (1.0 - scale) * torch.randn_like(output_tensor_a)
            return output_tensor_a + noise
        return output_tensor_a

    def update_divergence(self, sep_monitor) -> None:
        """
        Refreshes current_divergence from the monitor's latest epoch results.

        Called once per epoch in main(), immediately after
        sep_monitor.compute_epoch_metrics().  The updated divergence will be
        used for all batches in the *next* epoch (1-epoch lag is intentional —
        the Fisher criterion needs a full epoch of data to be reliable).

        Args:
            sep_monitor : SeparabilityMonitor instance (or None if monitoring
                          is disabled, in which case this is a no-op)
        """
        if sep_monitor is not None and len(sep_monitor.results) > 0:
            # results[-1] is the dict produced by the most recent compute_epoch_metrics()
            self.current_divergence = sep_monitor.results[-1]['fisher_divergence']


# ── Novel Defense: Adversarial Auxiliary Classifier (GRL-based) ───────────────
#
# MOTIVATION
# ----------
# AsymmetricAdaptivePerturbation (above) works on 10-class datasets because
# scale eventually hits 0 — providing 26+ epochs of zero gradient.  For CIFAR-100,
# scale never reaches 0 (Fisher divergence peaks at ~0.45; would need alpha > 2.86).
# Increasing alpha beyond 2.0 makes things WORSE: zeroing the gradient too early
# removes the corrective task signal, and MaliciousSGD amplifies unchecked internal
# gradients, producing MORE discriminative embeddings (confirmed EXP-013, Phase 6A).
#
# MECHANISM
# ---------
# The server maintains a small auxiliary classifier A_aux: z_a → labels.
# Each batch:
#   1. A_aux is updated to MAXIMISE classification accuracy from z_a alone
#      (standard cross-entropy, server trains it with server-owned labels).
#   2. The GRADIENT OF A_aux's loss w.r.t. z_a is computed and REVERSED.
#   3. This reversed gradient is added to the normal top-model gradient sent to Party A:
#        final_grad = grad_from_top  −  lambda_adv × (dL_aux / dz_a)
#
# WHY THIS WORKS WHERE SUPPRESSION FAILS
# ----------------------------------------
# dL_aux / dz_a points in the direction that makes z_a MORE class-discriminative.
# Subtracting it (with reversal) pushes z_a AWAY from discriminativeness — actively,
# not just passively (suppression).  Crucially, MaliciousSGD amplifies whatever
# gradient it receives, INCLUDING the adversarial correction.  Higher amplification
# → stronger anti-discriminative push.  The defense is self-reinforcing:
#   stronger attack → larger Fisher divergence → larger aux gradient → stronger defense.
# This is the opposite of suppression, where amplifying a zero gradient does nothing.
#
# ACTIVATION
# ----------
# Same Fisher-divergence detection as AsymmetricAdaptivePerturbation:
# defense fires when epoch >= burn_in AND divergence > tau.
# Can be used standalone OR combined with the existing asymmetric defense
# (if combined, asymmetric suppression runs first, then adversarial correction is added).
#
# HYPERPARAMETERS
# ---------------
#   embedding_dim  output dimension of Party A's bottom model (size_bottom_out).
#                  CIFAR-100: 100.  CIFAR-10/CINIC10L: 10.
#   num_classes    number of label classes (auxiliary classifier output dim).
#   lambda_adv     scale of the reversed gradient.  1.0 = equal weight to top-model
#                  gradient.  Increase to make the adversarial correction dominate.
#   burn_in        same purpose as AsymmetricAdaptivePerturbation burn_in.
#   tau            same purpose as AsymmetricAdaptivePerturbation tau.
#   aux_lr         learning rate for the auxiliary classifier (Adam).
# ──────────────────────────────────────────────────────────────────────────────
class AdversarialAuxiliaryDefense:

    def __init__(self, embedding_dim: int = 100, num_classes: int = 100,
                 lambda_adv: float = 1.0, burn_in: int = 8, tau: float = 0.10,
                 aux_lr: float = 1e-3):
        self.lambda_adv = lambda_adv
        self.burn_in = burn_in
        self.tau = tau
        self.current_divergence = 0.0

        # Server-side auxiliary classifier: z_a → class logits
        # Trained to be accurate on z_a; its REVERSED gradient is sent to Party A.
        self.aux_classifier = nn.Linear(embedding_dim, num_classes).cuda()
        self.aux_optimizer = optim.Adam(self.aux_classifier.parameters(), lr=aux_lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def update_divergence(self, sep_monitor) -> None:
        """Same interface as AsymmetricAdaptivePerturbation.update_divergence."""
        if sep_monitor is not None and len(sep_monitor.results) > 0:
            self.current_divergence = sep_monitor.results[-1]['fisher_divergence']

    def apply(self, grad_output_a: torch.Tensor, z_a: torch.Tensor,
              target: torch.Tensor, epoch: int) -> torch.Tensor:
        """
        Computes and injects the adversarial auxiliary gradient.

        Args:
            grad_output_a : [batch, embedding_dim] gradient from top model to z_a
            z_a           : [batch, embedding_dim] Party A's embedding (detached)
            target        : [batch] integer class labels (server has these)
            epoch         : current training epoch

        Returns:
            Modified gradient tensor to be sent to Party A's bottom model.
        """
        if epoch < self.burn_in or self.current_divergence <= self.tau:
            return grad_output_a

        # Attach gradient tracking to z_a for the auxiliary forward pass.
        # clone() ensures the original embedding tensor is not modified.
        z_a_aux = z_a.clone().requires_grad_(True)

        # Forward through auxiliary classifier: server learns to classify from z_a
        aux_logits = self.aux_classifier(z_a_aux)
        aux_loss = self.loss_fn(aux_logits, target)

        # Backprop: populates both aux_classifier.parameters().grad AND z_a_aux.grad
        self.aux_optimizer.zero_grad()
        aux_loss.backward()

        # Capture d(L_aux)/d(z_a): direction that makes z_a MORE discriminative
        aux_grad = z_a_aux.grad.detach()

        # Update auxiliary classifier so it stays accurate on z_a (server trains it)
        self.aux_optimizer.step()

        # Reverse the auxiliary gradient: subtract it from the top-model gradient.
        # Party A receives a signal that actively pushes z_a AWAY from discriminativeness.
        # MaliciousSGD amplifying this reversed signal makes the defense stronger.
        return grad_output_a - self.lambda_adv * aux_grad


# ── Novel Defense: Gradient Projection (Phase 19) ─────────────────────────────
#
# MOTIVATION
# ----------
# AdversarialAuxiliaryDefense (Phase 18) failed because it computes
#   final_grad = grad_output_a − lambda * aux_grad
# aux_grad = d(L_aux)/d(z_a) is UNBOUNDED: the unconstrained nn.Linear
# aux_classifier grows large weights over 150 epochs; combined with MaliciousSGD
# amplifying z_a, logits overflow → NaN cascade → model destruction.
#
# MECHANISM
# ---------
# Instead of SUBTRACTING a scaled aux_grad (which can explode), we PROJECT
# grad_output_a onto the subspace ORTHOGONAL to aux_grad:
#
#   d_aux      = d(L_aux)/d(z_a)          (discriminative direction, per sample)
#   d_aux_norm = d_aux / ||d_aux||         (unit vector, avoids magnitude dependency)
#   proj_coeff = grad_output_a · d_aux_norm  (scalar alignment per sample)
#   grad_proj  = grad_output_a − proj_coeff * d_aux_norm
#
# grad_proj removes only the component of grad_output_a that would push z_a
# toward class-discriminativeness.  The remaining gradient is unmodified.
#
# NUMERICAL SAFETY
# ----------------
# By Cauchy-Schwarz: ||grad_proj|| ≤ ||grad_output_a|| ALWAYS.
# We never ADD anything to the gradient — we only remove a component.
# MaliciousSGD amplifying grad_proj can only reach at most the original magnitude.
# d_aux_norm uses a +1e-8 denominator guard against zero-norm edge cases.
# → Zero NaN risk from the projection itself.
#
# AUX CLASSIFIER UPDATE
# ---------------------
# The same nn.Linear(embedding_dim, num_classes) is trained continuously each
# batch (not just when the projection fires).  This keeps it accurate so that
# d_aux tracks the current discriminative direction.
# The projection is gated by burn_in + divergence threshold — same Fisher
# detection logic as all prior defenses — to avoid perturbing benign training.
# ──────────────────────────────────────────────────────────────────────────────
class GradientProjectionDefense:

    def __init__(self, embedding_dim: int = 100, num_classes: int = 100,
                 burn_in: int = 8, tau: float = 0.10, aux_lr: float = 1e-3):
        self.burn_in = burn_in
        self.tau = tau
        self.current_divergence = 0.0

        self.aux_classifier = nn.Linear(embedding_dim, num_classes).cuda()
        self.aux_optimizer = optim.Adam(self.aux_classifier.parameters(), lr=aux_lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def update_divergence(self, sep_monitor) -> None:
        if sep_monitor is not None and len(sep_monitor.results) > 0:
            self.current_divergence = sep_monitor.results[-1]['fisher_divergence']

    def apply(self, grad_output_a: torch.Tensor, z_a: torch.Tensor,
              target: torch.Tensor, epoch: int) -> torch.Tensor:
        """
        Removes from grad_output_a the component aligned with the discriminative direction.

        The aux classifier is updated every batch regardless of whether the projection
        fires, so it stays accurate and d_aux tracks the current discriminative direction.
        The projection itself is gated by burn_in and Fisher divergence threshold.

        Args:
            grad_output_a : [batch, embedding_dim] gradient from top model to z_a
            z_a           : [batch, embedding_dim] Party A's embedding (detached)
            target        : [batch] integer class labels (server has these)
            epoch         : current training epoch

        Returns:
            grad_output_a with its discriminative component removed (when defense active),
            or grad_output_a unchanged (during burn-in or when no attack detected).
        """
        # Step 1: train aux classifier on current z_a (always, keeps it current)
        z_a_aux = z_a.clone().requires_grad_(True)
        self.aux_optimizer.zero_grad()
        aux_logits = self.aux_classifier(z_a_aux)
        aux_loss = self.loss_fn(aux_logits, target)
        aux_loss.backward()
        d_aux = z_a_aux.grad.detach()  # [batch, embedding_dim]
        self.aux_optimizer.step()

        # Step 2: apply projection only when attack is detected
        if epoch < self.burn_in or self.current_divergence <= self.tau:
            return grad_output_a

        # Normalize per sample so projection magnitude is independent of aux_grad scale
        d_aux_norm = d_aux / (d_aux.norm(dim=-1, keepdim=True) + 1e-8)
        # Scalar projection of grad_output_a onto discriminative direction
        proj_coeff = (grad_output_a * d_aux_norm).sum(dim=-1, keepdim=True)
        # Remove the discriminative component; result is orthogonal to d_aux_norm
        grad_proj = grad_output_a - proj_coeff * d_aux_norm
        return grad_proj


# ── Novel Defense: Persistent Projection (Phase 22/23) ────────────────────────
#
# MOTIVATION
# ----------
# GradientProjectionDefense (Phase 19) works on CIFAR-100 but via an
# undesired mechanism: catastrophic single-activation collapse at epoch 11.
# The defense fires ONCE (when Fisher divergence barely crosses tau after
# burn_in), the aux_classifier direction aligns perfectly with grad_output_a
# after 11 epochs of training, projection removes ~97% of gradient, intra_var_A
# spikes 6 orders of magnitude, and Fisher divergence goes permanently negative
# for the remaining 138 epochs.  The defense self-terminates.
#
# This is theoretically unsound: the result depends on the accident of when
# Fisher divergence first crosses tau, and the defense does no further work.
#
# MECHANISM
# ---------
# Replace the per-batch instantaneous direction with an exponential moving
# average (EMA) of the discriminative direction:
#
#   d_mean   = mean(d(L_aux)/d(z_a))  over the batch     [embedding_dim]
#   d_mean_n = d_mean / ||d_mean||     unit vector
#   d_ema_t  = normalize((1-α)*d_ema_{t-1} + α*d_mean_n)  EMA update
#
#   Projection (every detected epoch, not just the first):
#   proj_coeff = grad_output_a · d_ema          [batch, 1]
#   grad_proj  = grad_output_a − proj_coeff * d_ema
#
# KEY DIFFERENCES FROM GradientProjectionDefense
# -----------------------------------------------
#   GradientProjectionDefense:
#     - Uses raw per-sample d_aux from the current batch (unstable, direction
#       changes every batch).
#     - One-shot collapse: fires catastrophically once, then self-terminates.
#     - burn_in=8 (needs 8 epochs before aux classifier direction is meaningful)
#
#   PersistentProjectionDefense:
#     - Uses d_ema: smoothed, epoch-stable unit vector updated every batch.
#     - Fires every epoch the Fisher divergence exceeds tau (persistent).
#     - Shorter default burn_in=4 (EMA stabilizes faster than single-batch estimates).
#     - Direction naturally adapts as z_a distribution evolves over training.
#
# NUMERICAL GUARANTEES
# --------------------
#   - d_ema is always unit-normed (renormalized after each EMA update).
#   - ||grad_proj|| ≤ ||grad_output_a|| by Cauchy-Schwarz (projection only removes).
#   - No NaN risk: all divisions guarded by +1e-8.
# ──────────────────────────────────────────────────────────────────────────────
class PersistentProjectionDefense:

    def __init__(self, embedding_dim: int = 100, num_classes: int = 100,
                 alpha_ema: float = 0.2, burn_in: int = 4, tau: float = 0.10,
                 aux_lr: float = 1e-3):
        self.alpha_ema = alpha_ema
        self.burn_in = burn_in
        self.tau = tau
        self.current_divergence = 0.0
        self.d_ema = None  # [embedding_dim], always unit-normed after first update

        self.aux_classifier = nn.Linear(embedding_dim, num_classes).cuda()
        self.aux_optimizer = optim.Adam(self.aux_classifier.parameters(), lr=aux_lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def update_divergence(self, sep_monitor) -> None:
        if sep_monitor is not None and len(sep_monitor.results) > 0:
            self.current_divergence = sep_monitor.results[-1]['fisher_divergence']

    def apply(self, grad_output_a: torch.Tensor, z_a: torch.Tensor,
              target: torch.Tensor, epoch: int) -> torch.Tensor:
        """
        Updates d_ema every batch; projects grad_output_a every detected epoch.

        Step 1 (always): Train aux classifier → compute d_inst per batch →
                         update d_ema with EMA → normalize to unit vector.
        Step 2 (gated):  If epoch >= burn_in AND divergence > tau, remove the
                         d_ema component from grad_output_a.
        """
        # Step 1: train aux classifier (always — keeps direction estimate current)
        z_a_aux = z_a.clone().requires_grad_(True)
        self.aux_optimizer.zero_grad()
        aux_logits = self.aux_classifier(z_a_aux)
        aux_loss = self.loss_fn(aux_logits, target)
        aux_loss.backward()
        d_inst = z_a_aux.grad.detach()  # [batch, embedding_dim]
        self.aux_optimizer.step()

        # Update EMA direction: normalize per-sample FIRST, then average.
        # Rationale: raw mean of cross-entropy grads ≈ 0 for balanced batches
        # (prediction errors cancel across classes), so normalizing the raw mean
        # produces a random direction. Per-sample unit vectors do not cancel.
        d_inst_norm = d_inst / (d_inst.norm(dim=-1, keepdim=True) + 1e-8)  # [batch, embed_dim]
        d_mean = d_inst_norm.mean(dim=0)                     # mean of unit vectors [embedding_dim]
        d_mean_norm = d_mean / (d_mean.norm() + 1e-8)        # unit vector
        if self.d_ema is None:
            self.d_ema = d_mean_norm
        else:
            self.d_ema = (1.0 - self.alpha_ema) * self.d_ema + self.alpha_ema * d_mean_norm
            self.d_ema = self.d_ema / (self.d_ema.norm() + 1e-8)

        # Step 2: project every detected epoch (not just the first)
        if epoch < self.burn_in or self.current_divergence <= self.tau:
            return grad_output_a

        # Broadcast d_ema [embedding_dim] → [1, embedding_dim] for batch projection
        d = self.d_ema.unsqueeze(0)
        proj_coeff = (grad_output_a * d).sum(dim=-1, keepdim=True)
        return grad_output_a - proj_coeff * d


# ── Novel Defense: Multi-Direction Persistent Projection (Phase 24) ───────────
#
# MOTIVATION
# ----------
# PersistentProjectionDefense removes ONE EMA-smoothed direction from
# grad_output_a.  For C=100 classes the discriminative subspace has
# dimension C-1=99; projecting one direction removes <1% of it:
#   DCR(K=1, C=100) = 1 / min(99, embed_dim) ≈ 1%
# MaliciousSGD can immediately re-route its attack signal through
# the remaining 98 untouched directions.
#
# This class generalises PersistentProjectionDefense to K directions,
# extracted from the aux classifier's weight matrix W_aux via SVD.
# The top-K right singular vectors span the subspace of W_aux most
# used to discriminate between classes.
#   DCR(K=10, C=100) = 10 / 99 ≈ 10%
#   DCR(K=20, C=100) = 20 / 99 ≈ 20%
#
# MECHANISM
# ---------
# Every batch:
#   1. Train W_aux on current z_a (keeps the prototype matrix current).
#   2. SVD of W_aux [num_classes, embed_dim]:
#        W = U @ diag(S) @ Vh   →   D_inst = Vh[:K]  [K, embed_dim]
#      Rows of Vh are right singular vectors; already unit-normed by SVD.
#   3. EMA + QR re-orthogonalise:
#        D_mixed = (1-α)*D_ema + α*D_inst          [K, embed_dim]
#        Q, _ = qr(D_mixed.T)  →  D_ema = Q.T      [K, embed_dim]
#      QR guarantees D_ema remains orthonormal even after blending.
#
# When detected (epoch ≥ burn_in AND Δ_F > τ):
#   proj_coeffs = grad_output_a @ D_ema.T   [batch, K]
#   grad_proj   = grad_output_a - proj_coeffs @ D_ema   [batch, embed_dim]
#
# NUMERICAL GUARANTEES
# --------------------
#   - D_ema is always orthonormal (QR renormalises after each update).
#   - By Cauchy-Schwarz: ||grad_proj|| ≤ ||grad_output_a|| always.
#   - SVD fallback: RuntimeError (degenerate W) → return grad unchanged.
# ──────────────────────────────────────────────────────────────────────────────
class MultiDirectionPersistentProjection:

    def __init__(self, embedding_dim: int = 100, num_classes: int = 100,
                 k_directions: int = 10, alpha_ema: float = 0.2,
                 burn_in: int = 4, tau: float = 0.10, aux_lr: float = 1e-3):
        self.k_directions = min(k_directions, embedding_dim, num_classes)
        self.alpha_ema = alpha_ema
        self.burn_in = burn_in
        self.tau = tau
        self.current_divergence = 0.0
        self.D_ema = None  # [K, embed_dim], orthonormal rows

        # bias=False: W_aux is a clean class-prototype matrix, not shifted
        self.aux_classifier = nn.Linear(embedding_dim, num_classes, bias=False).cuda()
        self.aux_optimizer = optim.Adam(self.aux_classifier.parameters(), lr=aux_lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def update_divergence(self, sep_monitor) -> None:
        if sep_monitor is not None and len(sep_monitor.results) > 0:
            self.current_divergence = sep_monitor.results[-1]['fisher_divergence']

    def apply(self, grad_output_a: torch.Tensor, z_a: torch.Tensor,
              target: torch.Tensor, epoch: int) -> torch.Tensor:
        """
        Step 1 (always): train W_aux → SVD → update D_ema.
        Step 2 (gated):  if epoch >= burn_in AND divergence > tau,
                         project out all K directions simultaneously.
        """
        # Step 1: train aux classifier to keep W_aux tracking current z_a
        z_a_aux = z_a.clone().requires_grad_(True)
        self.aux_optimizer.zero_grad()
        aux_logits = self.aux_classifier(z_a_aux)
        aux_loss = self.loss_fn(aux_logits, target)
        aux_loss.backward()
        self.aux_optimizer.step()

        # Step 2: extract top-K directions from W_aux via SVD
        with torch.no_grad():
            W = self.aux_classifier.weight.data  # [num_classes, embed_dim]
            try:
                _, _, Vh = torch.linalg.svd(W, full_matrices=False)
                D_inst = Vh[:self.k_directions]  # [K, embed_dim], orthonormal rows
            except RuntimeError:
                return grad_output_a  # degenerate W; skip update this batch

            if self.D_ema is None:
                self.D_ema = D_inst
            else:
                D_mixed = (1.0 - self.alpha_ema) * self.D_ema + self.alpha_ema * D_inst
                # QR on transpose gives orthonormal columns; transpose back to rows
                Q, _ = torch.linalg.qr(D_mixed.T)  # [embed_dim, K]
                self.D_ema = Q.T                     # [K, embed_dim], orthonormal rows

        if epoch < self.burn_in or self.current_divergence <= self.tau:
            return grad_output_a

        # Project out all K directions in one matrix multiply
        D = self.D_ema                           # [K, embed_dim]
        proj_coeffs = grad_output_a @ D.T        # [batch, K]
        return grad_output_a - proj_coeffs @ D   # [batch, embed_dim]
