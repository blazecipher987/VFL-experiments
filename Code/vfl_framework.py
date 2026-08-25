import argparse
import ast
import os
import time
import dill
import random
import numpy as np
from time import time
import sys
sys.path.insert(0, "./")

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from datasets import get_dataset
from my_utils import utils
from models import model_sets
import my_optimizers
import possible_defenses
import characterization_monitor

plt.switch_backend('agg')

D_ = 2 ** 13
BATCH_SIZE = 1000


def split_data(data):
    if args.dataset == 'Yahoo':
        x_b = data[1]
        x_a = data[0]
    elif args.dataset in ['CIFAR10', 'CIFAR100', 'CINIC10L']:
        x_a = data[:, :, :, 0:args.half]
        x_b = data[:, :, :, args.half:32]
    elif args.dataset == 'TinyImageNet':
        x_a = data[:, :, :, 0:args.half]
        x_b = data[:, :, :, args.half:64]
    elif args.dataset == 'Criteo':
        x_b = data[:, args.half:D_]
        x_a = data[:, 0:args.half]
    elif args.dataset == 'BCW':
        x_b = data[:, args.half:28]
        x_a = data[:, 0:args.half]
    else:
        raise Exception('Unknown dataset name!')
    if args.test_upper_bound:
        x_b = torch.zeros_like(x_b)
    return x_a, x_b


class VflFramework(nn.Module):

    def __init__(self):
        super(VflFramework, self).__init__()
        # counter for direct label inference attack
        self.inferred_correct = 0
        self.inferred_wrong = 0
        # bottom model a can collect output_a for label inference attack
        self.collect_outputs_a = False
        self.outputs_a = torch.tensor([]).cuda()
        # In order to evaluate attack performance, we need to collect label sequence of training dataset
        self.labels_training_dataset = torch.tensor([], dtype=torch.long).cuda()
        # In order to evaluate attack performance, we need to collect label sequence of training dataset
        self.if_collect_training_dataset_labels = False

        # adversarial options — symmetric defenses (applied to both parties)
        self.defense_ppdl = args.ppdl
        self.defense_gc = args.gc
        self.defense_lap_noise = args.lap_noise
        self.defense_multistep_grad = args.multistep_grad
        # self.defense_ss = args.ss

        # asymmetric adaptive perturbation (novel Phase 2 defense)
        # flag comes from argparse; the actual defense instance is assigned in
        # main() after the SeparabilityMonitor has been created, because the
        # defense depends on the monitor's Fisher divergence readings.
        self.defense_asymmetric = args.asymmetric_defense
        self.asymmetric_defense = None  # set to AsymmetricAdaptivePerturbation in main()
        self.adversarial_aux_defense = None  # set to AdversarialAuxiliaryDefense in main()
        self.gradient_projection_defense = None  # set to GradientProjectionDefense in main()
        self.persistent_projection_defense = None  # set to PersistentProjectionDefense in main()
        self.mdpp_defense = None  # set to MultiDirectionPersistentProjection in main()

        # current training epoch — updated at the start of each epoch in main()
        # so that batch-level methods can access it for the burn-in guard.
        self.current_epoch = 0

        # indicates whether to conduct the direct label inference attack
        self.direct_attack_on = False

        # separability monitor (set to a SeparabilityMonitor instance to enable)
        self.sep_monitor = None

        # loss funcs
        self.loss_func_top_model = nn.CrossEntropyLoss()
        self.loss_func_bottom_model = utils.keep_predict_loss

        # bottom model A
        self.malicious_bottom_model_a = model_sets.BottomModel(dataset_name=args.dataset).get_model(
            half=args.half,
            is_adversary=True
        )
        # bottom model B
        self.benign_bottom_model_b = model_sets.BottomModel(dataset_name=args.dataset).get_model(
            half=args.half,
            is_adversary=False
        )
        # top model
        self.top_model = model_sets.TopModel(dataset_name=args.dataset).get_model()

        # This setting is for adversarial experiments except sign SGD
        if args.use_mal_optim_top:
            self.optimizer_top_model = my_optimizers.MaliciousSGD(self.top_model.parameters(),
                                                                  lr=args.lr,
                                                                  momentum=args.momentum,
                                                                  weight_decay=args.weight_decay)
        else:
            self.optimizer_top_model = optim.SGD(self.top_model.parameters(),
                                                 lr=args.lr,
                                                 momentum=args.momentum,
                                                 weight_decay=args.weight_decay)
        if args.dataset != 'Yahoo':
            if args.use_mal_optim:
                self.optimizer_malicious_bottom_model_a = my_optimizers.MaliciousSGD(
                    self.malicious_bottom_model_a.parameters(),
                    lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)
            else:
                self.optimizer_malicious_bottom_model_a = optim.SGD(
                    self.malicious_bottom_model_a.parameters(),
                    lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)
            if args.use_mal_optim_all:
                self.optimizer_benign_bottom_model_b = my_optimizers.MaliciousSGD(
                    self.benign_bottom_model_b.parameters(),
                    lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)
            else:
                self.optimizer_benign_bottom_model_b = optim.SGD(self.benign_bottom_model_b.parameters(),
                                                                 lr=args.lr,
                                                                 momentum=args.momentum,
                                                                 weight_decay=args.weight_decay)
        else:
            if args.use_mal_optim:
                self.optimizer_malicious_bottom_model_a = my_optimizers.MaliciousSGD(
                    [
                        {"params": self.malicious_bottom_model_a.mixtext_model.bert.parameters(), "lr": 5e-6},
                        {"params": self.malicious_bottom_model_a.mixtext_model.linear.parameters(), "lr": 5e-4},
                    ],
                    lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)
            else:
                self.optimizer_malicious_bottom_model_a = optim.SGD(
                    [
                        {"params": self.malicious_bottom_model_a.mixtext_model.bert.parameters(), "lr": 5e-6},
                        {"params": self.malicious_bottom_model_a.mixtext_model.linear.parameters(), "lr": 5e-4},
                    ],
                    lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)
            if args.use_mal_optim_all:
                self.optimizer_benign_bottom_model_b = my_optimizers.MaliciousSGD(
                    [
                        {"params": self.benign_bottom_model_b.mixtext_model.bert.parameters(), "lr": 5e-6},
                        {"params": self.benign_bottom_model_b.mixtext_model.linear.parameters(), "lr": 5e-4},
                    ],
                    lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)
            else:
                self.optimizer_benign_bottom_model_b = optim.SGD([
                    {"params": self.benign_bottom_model_b.mixtext_model.bert.parameters(), "lr": 5e-6},
                    {"params": self.benign_bottom_model_b.mixtext_model.linear.parameters(), "lr": 5e-4},
                ],
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay)

    def forward(self, x):
        # in vertical federated setting, each party has non-lapping features of the same sample
        x_a, x_b = split_data(x)
        out_a = self.malicious_bottom_model_a(x_a)
        out_b = self.benign_bottom_model_b(x_b)
        if args.use_top_model:
            out = self.top_model(out_a, out_b)
        else:
            out = out_a + out_b
        return out

    def simulate_train_round_per_batch(self, data, target):
        timer_mal = 0
        timer_benign = 0
        # simulate: bottom models forward, top model forward, top model backward and update, bottom backward and update

        # In order to evaluate attack performance, we need to collect label sequence of training dataset
        if self.if_collect_training_dataset_labels:
            self.labels_training_dataset = torch.cat((self.labels_training_dataset, target), dim=0)
        # store grad of input of top model/outputs of bottom models
        input_tensor_top_model_a = torch.tensor([], requires_grad=True)
        input_tensor_top_model_b = torch.tensor([], requires_grad=True)

        # --bottom models forward--
        x_a, x_b = split_data(data)

        # make x_b random noise
        # x_b = torch.rand_like(x_b)

        # -bottom model A-
        self.malicious_bottom_model_a.train(mode=True)
        start = time()
        output_tensor_bottom_model_a = self.malicious_bottom_model_a(x_a)
        end = time()
        time_cost = end - start
        timer_mal += time_cost
        # bottom model a can collect output_a for label inference attack
        if self.collect_outputs_a:
            self.outputs_a = torch.cat((self.outputs_a, output_tensor_bottom_model_a.data))
        # -bottom model B-
        self.benign_bottom_model_b.train(mode=True)
        start2 = time()
        output_tensor_bottom_model_b = self.benign_bottom_model_b(x_b)
        end2 = time()
        time_cost2 = end2 - start2
        timer_benign += time_cost2
        # -top model-
        # (we omit interactive layer for it doesn't effect our attack or possible defenses)
        # by concatenating output of bottom a/b(dim=10+10=20), we get input of top model
        # z_a embedding corruption: feed noisy z_a to the top model while keeping
        # output_tensor_bottom_model_a clean for (a) the separability monitor and
        # (b) the bottom model's backward path.  Detection logic still reads the
        # clean Fisher divergence from the previous epoch.
        if (self.defense_asymmetric and self.asymmetric_defense is not None
                and self.asymmetric_defense.za_noise_std > 0.0):
            input_tensor_top_model_a.data = self.asymmetric_defense.apply_za_corruption(
                output_tensor_bottom_model_a, self.current_epoch
            ).data
        else:
            input_tensor_top_model_a.data = output_tensor_bottom_model_a.data
        input_tensor_top_model_b.data = output_tensor_bottom_model_b.data

        if args.use_top_model:
            self.top_model.train(mode=True)
            output_framework = self.top_model(input_tensor_top_model_a, input_tensor_top_model_b)
            # --top model backward/update--
            loss_framework = model_sets.update_top_model_one_batch(optimizer=self.optimizer_top_model,
                                                                   model=self.top_model,
                                                                   output=output_framework,
                                                                   batch_target=target,
                                                                   loss_func=self.loss_func_top_model)
        else:
            output_framework = input_tensor_top_model_a + input_tensor_top_model_b
            loss_framework = self.loss_func_top_model(output_framework, target)
            loss_framework.backward()

        # read grad of: input of top model(also output of bottom models), which will be used as bottom model's target
        grad_output_bottom_model_a = input_tensor_top_model_a.grad
        grad_output_bottom_model_b = input_tensor_top_model_b.grad

        # collect raw (pre-defense) embeddings and gradient norms for separability monitor
        if self.sep_monitor is not None:
            self.sep_monitor.collect_batch(
                output_tensor_bottom_model_a,
                output_tensor_bottom_model_b,
                target,
                grad_output_bottom_model_a,
                grad_output_bottom_model_b,
            )

        # defenses here: the server(who controls top model) can defend against label inference attack by protecting
        # print("before defense, grad_output_bottom_model_a:", grad_output_bottom_model_a)
        # gradients sent to bottom models
        model_all_layers_grads_list = [grad_output_bottom_model_a, grad_output_bottom_model_b]
        # privacy preserving deep learning
        if self.defense_ppdl:
            possible_defenses.dp_gc_ppdl(epsilon=1.8, sensitivity=1, layer_grad_list=[grad_output_bottom_model_a],
                                         theta_u=args.ppdl_theta_u, gamma=0.001, tau=0.0001)
            possible_defenses.dp_gc_ppdl(epsilon=1.8, sensitivity=1, layer_grad_list=[grad_output_bottom_model_b],
                                         theta_u=args.ppdl_theta_u, gamma=0.001, tau=0.0001)
        # gradient compression
        if self.defense_gc:
            tensor_pruner = possible_defenses.TensorPruner(zip_percent=args.gc_preserved_percent)
            for tensor_id in range(len(model_all_layers_grads_list)):
                tensor_pruner.update_thresh_hold(model_all_layers_grads_list[tensor_id])
                # print("tensor_pruner.thresh_hold:", tensor_pruner.thresh_hold)
                model_all_layers_grads_list[tensor_id] = tensor_pruner.prune_tensor(
                    model_all_layers_grads_list[tensor_id])
        # differential privacy
        if self.defense_lap_noise:
            dp = possible_defenses.DPLaplacianNoiseApplyer(beta=args.noise_scale)
            for tensor_id in range(len(model_all_layers_grads_list)):
                model_all_layers_grads_list[tensor_id] = dp.laplace_mech(model_all_layers_grads_list[tensor_id])
        # multistep gradient
        if self.defense_multistep_grad:
            for tensor_id in range(len(model_all_layers_grads_list)):
                model_all_layers_grads_list[tensor_id] = possible_defenses.multistep_gradient(
                    model_all_layers_grads_list[tensor_id], bins_num=args.multistep_grad_bins,
                    bound_abs=args.multistep_grad_bound_abs)
        # sign SGD
        # if self.defense_ss:
        #     for tensor in model_all_layers_grads_list:
        #         torch.sign(tensor, out=tensor)
        grad_output_bottom_model_a, grad_output_bottom_model_b = tuple(model_all_layers_grads_list)
        # print("after defense, grad_output_bottom_model_a:", grad_output_bottom_model_a)

        # asymmetric adaptive perturbation defense —
        # runs AFTER all symmetric defenses so it is the final gate on Party A's gradient.
        # Party B's gradient (grad_output_bottom_model_b) is intentionally never modified here.
        if self.defense_asymmetric and self.asymmetric_defense is not None:
            grad_output_bottom_model_a, _ = self.asymmetric_defense.apply(
                grad_output_bottom_model_a, self.current_epoch
            )

        # adversarial auxiliary classifier defense —
        # server maintains a classifier on z_a and sends its REVERSED gradient to Party A,
        # actively pushing z_a toward class-indiscriminativeness.
        # MaliciousSGD amplifying a reversed gradient strengthens the defense (self-reinforcing).
        # Can be combined with asymmetric_defense: suppression runs first, then adversarial correction.
        if self.adversarial_aux_defense is not None:
            grad_output_bottom_model_a = self.adversarial_aux_defense.apply(
                grad_output_bottom_model_a,
                output_tensor_bottom_model_a.detach(),
                target,
                self.current_epoch,
            )

        # gradient projection defense —
        # removes from grad_output_a the component aligned with the discriminative direction.
        # ||grad_proj|| <= ||grad_output_a|| always — no NaN amplification risk.
        if self.gradient_projection_defense is not None:
            grad_output_bottom_model_a = self.gradient_projection_defense.apply(
                grad_output_bottom_model_a,
                output_tensor_bottom_model_a.detach(),
                target,
                self.current_epoch,
            )

        # persistent projection defense —
        # EMA-based discriminative direction projected every detected epoch.
        # Fixes the one-shot collapse in GradientProjectionDefense.
        if self.persistent_projection_defense is not None:
            grad_output_bottom_model_a = self.persistent_projection_defense.apply(
                grad_output_bottom_model_a,
                output_tensor_bottom_model_a.detach(),
                target,
                self.current_epoch,
            )

        # multi-direction persistent projection defense —
        # Projects K SVD-derived directions from W_aux simultaneously.
        # DCR(K, C) = K / min(C-1, embed_dim); CIFAR-100 K=10 → ~10% coverage.
        if self.mdpp_defense is not None:
            grad_output_bottom_model_a = self.mdpp_defense.apply(
                grad_output_bottom_model_a,
                output_tensor_bottom_model_a.detach(),
                target,
                self.current_epoch,
            )

        # server sends back output_tensor_server_a.grad to the adversary (participant a), so
        # the adversary can use this gradient to perform direct label inference attack.
        if self.direct_attack_on:
            for sample_id in range(len(grad_output_bottom_model_a)):
                grad_per_sample = grad_output_bottom_model_a[sample_id]
                for logit_id in range(len(grad_per_sample)):
                    if grad_per_sample[logit_id] < 0:
                        inferred_label = logit_id
                        if inferred_label == target[sample_id]:
                            self.inferred_correct += 1
                        else:
                            self.inferred_wrong += 1
                        break

        # --bottom models backward/update--
        # -bottom model a: backward/update-
        # print("malicious_bottom_model_a")
        start = time()
        model_sets.update_bottom_model_one_batch(optimizer=self.optimizer_malicious_bottom_model_a,
                                                 model=self.malicious_bottom_model_a,
                                                 output=output_tensor_bottom_model_a,
                                                 batch_target=grad_output_bottom_model_a,
                                                 loss_func=self.loss_func_bottom_model)
        end = time()
        time_cost = end - start
        timer_mal += time_cost
        # -bottom model b: backward/update-
        # print("benign_bottom_model_b")
        model_sets.update_bottom_model_one_batch(optimizer=self.optimizer_benign_bottom_model_b,
                                                 model=self.benign_bottom_model_b,
                                                 output=output_tensor_bottom_model_b,
                                                 batch_target=grad_output_bottom_model_b,
                                                 loss_func=self.loss_func_bottom_model)
        end2 = time()
        time_cost2 = end2 - end
        timer_benign += time_cost2
        timer_on = False
        if timer_on:
            print("timer_mal:", timer_mal)
            print("timer_benign:", timer_benign)

        return loss_framework


def correct_counter(output, target, topk=(1, 5)):
    correct_counts = []
    for k in topk:
        _, pred = output.topk(k, 1, True, True)
        correct_k = torch.eq(pred, target.view(-1, 1)).sum().float().item()
        correct_counts.append(correct_k)
    return correct_counts


def test_per_epoch(test_loader, framework, k=5, loss_func_top_model=None):
    test_loss = 0
    correct_top1 = 0
    correct_topk = 0
    count = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.dataset == 'Yahoo':
                for i in range(len(data)):
                    data[i] = data[i].long().cuda()
                target = target[0].long().cuda()
            else:
                data = data.float().cuda()
                target = target.long().cuda()
            # set all sub-models to eval mode.
            framework.malicious_bottom_model_a.eval()
            framework.benign_bottom_model_b.eval()
            framework.top_model.eval()
            # run forward process of the whole framework
            x_a, x_b = split_data(data)
            output_tensor_bottom_model_a = framework.malicious_bottom_model_a(x_a)
            output_tensor_bottom_model_b = framework.benign_bottom_model_b(x_b)

            if args.use_top_model:
                output_framework = framework.top_model(output_tensor_bottom_model_a, output_tensor_bottom_model_b)
            else:
                output_framework = output_tensor_bottom_model_a + output_tensor_bottom_model_b

            correct_top1_batch, correct_topk_batch = correct_counter(output_framework, target, (1, k))

            # sum up batch loss
            test_loss += loss_func_top_model(output_framework, target).data.item()

            correct_top1 += correct_top1_batch
            correct_topk += correct_topk_batch
            # print("one batch done")
            count += 1
            if int(0.1 * len(test_loader)) > 0:
                count_percent_10 = count // int(0.1 * len(test_loader))
                if count_percent_10 <= 10 and count % int(0.1 * len(test_loader)) == 0 and\
                        count // int(0.1 * len(test_loader)) > 0:
                    print(f'{count // int(0.1 * len(test_loader))}0 % completed...')
                # print(count)

            if args.dataset == 'Criteo' and count == test_loader.train_batches_num:
                break

        if args.dataset == 'Criteo':
            num_samples = len(test_loader) * BATCH_SIZE
        else:
            num_samples = len(test_loader.dataset)
        test_loss /= num_samples
        print('Loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%), Top {} Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss,
            correct_top1, num_samples, 100.00 * float(correct_top1) / num_samples,
            k,
            correct_topk, num_samples, 100.00 * float(correct_topk) / num_samples
        ))


def set_loaders():
    dataset_setup = get_dataset.get_dataset_setup_by_name(args.dataset)
    train_dataset = dataset_setup.get_transformed_dataset(args.path_dataset, None, True)
    test_dataset = dataset_setup.get_transformed_dataset(args.path_dataset, None, False)
    if args.dataset == 'Criteo':
        train_loader = train_dataset
        test_loader = test_dataset
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size, shuffle=True,
            # num_workers=args.workers
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            # num_workers=args.workers
        )
    # check size_bottom_out and num_classes
    if args.use_top_model is False:
        if dataset_setup.size_bottom_out != dataset_setup.num_classes:
            raise Exception('If no top model is used,'
                            ' output tensor of the bottom model must equal to number of classes.')
    return train_loader, test_loader


def main():
    # write experiment setting into file name
    setting_str = ""
    setting_str += "_"
    setting_str += "lr="
    setting_str += str(args.lr)
    if args.use_mal_optim:
        setting_str += "_"
        setting_str += "mal"
        if args.use_mal_optim_all:
            setting_str += "-all"
        if args.use_mal_optim_top:
            setting_str += "-top"
    else:
        setting_str += "_"
        setting_str += "normal"
    if args.ppdl:
        setting_str += "_"
        setting_str += "ppdl-theta_u="
        setting_str += str(args.ppdl_theta_u)
    if args.gc:
        setting_str += "_"
        setting_str += "gc-preserved_percent="
        setting_str += str(args.gc_preserved_percent)
    if args.lap_noise:
        setting_str += "_"
        setting_str += "lap_noise-scale="
        setting_str += str(args.noise_scale)
    if args.multistep_grad:
        setting_str += "_"
        setting_str += "multistep_grad_bins="
        setting_str += str(args.multistep_grad_bins)
    if args.test_upper_bound:
        setting_str += "_upperbound"
    if args.asymmetric_defense:
        # encode defense hyperparameters in the filename so results are self-documenting
        setting_str += f"_asym_def-a={args.asymmetric_alpha}-t={args.asymmetric_tau}-b={args.asymmetric_burn_in}"
        if args.asymmetric_noise_std > 0.0:
            setting_str += f"-n={args.asymmetric_noise_std}"
        if args.asymmetric_sign_flip:
            setting_str += f"-sf"
        if args.asymmetric_za_noise_std > 0.0:
            setting_str += f"-za={args.asymmetric_za_noise_std}"
    if args.adversarial_aux_defense:
        setting_str += f"_adv_aux-l={args.adversarial_aux_lambda}"
    if args.gradient_projection_defense:
        setting_str += "_grad_proj"
    if args.persistent_projection:
        setting_str += f"_pers_proj-ema={args.persistent_proj_alpha_ema}"
    if args.mdpp:
        setting_str += f"_mdpp-k={args.mdpp_k_directions}-ema={args.mdpp_alpha_ema}"
    setting_str += "_"
    if args.dataset != 'Yahoo':
        setting_str += "half="
        setting_str += str(args.half)
    if not args.use_top_model:
        setting_str += '_NoTopModel'
    print("settings:", setting_str)

    model = VflFramework()
    model = model.cuda()
    cudnn.benchmark = True

    if args.monitor_separability:
        csv_dir = args.save_dir + f"/csv_files/{args.dataset}_csv_files"
        model.sep_monitor = characterization_monitor.SeparabilityMonitor(
            save_dir=csv_dir,
            setting_str=setting_str,
            dataset_name=args.dataset,
        )

    if args.adversarial_aux_defense:
        if model.sep_monitor is None:
            print("[WARNING] --adversarial-aux-defense requires --monitor-separability True. "
                  "Defense is DISABLED because the monitor is not active.")
        else:
            model.adversarial_aux_defense = possible_defenses.AdversarialAuxiliaryDefense(
                embedding_dim=args.adversarial_aux_embedding_dim,
                num_classes=args.adversarial_aux_num_classes,
                lambda_adv=args.adversarial_aux_lambda,
                burn_in=args.asymmetric_burn_in,
                tau=args.asymmetric_tau,
                aux_lr=args.adversarial_aux_lr,
            )
            print(f"[Defense] AdversarialAuxiliaryDefense active: "
                  f"embedding_dim={args.adversarial_aux_embedding_dim}, "
                  f"num_classes={args.adversarial_aux_num_classes}, "
                  f"lambda={args.adversarial_aux_lambda}, "
                  f"burn_in={args.asymmetric_burn_in}, tau={args.asymmetric_tau}, "
                  f"aux_lr={args.adversarial_aux_lr}")

    if args.gradient_projection_defense:
        if model.sep_monitor is None:
            print("[WARNING] --gradient-projection-defense requires --monitor-separability True. "
                  "Defense is DISABLED because the monitor is not active.")
        else:
            model.gradient_projection_defense = possible_defenses.GradientProjectionDefense(
                embedding_dim=args.gradient_proj_embedding_dim,
                num_classes=args.gradient_proj_num_classes,
                burn_in=args.asymmetric_burn_in,
                tau=args.asymmetric_tau,
                aux_lr=args.gradient_proj_lr,
            )
            print(f"[Defense] GradientProjectionDefense active: "
                  f"embedding_dim={args.gradient_proj_embedding_dim}, "
                  f"num_classes={args.gradient_proj_num_classes}, "
                  f"burn_in={args.asymmetric_burn_in}, tau={args.asymmetric_tau}, "
                  f"aux_lr={args.gradient_proj_lr}")

    if args.persistent_projection:
        if model.sep_monitor is None:
            print("[WARNING] --persistent-projection requires --monitor-separability True. "
                  "Defense is DISABLED because the monitor is not active.")
        else:
            model.persistent_projection_defense = possible_defenses.PersistentProjectionDefense(
                embedding_dim=args.gradient_proj_embedding_dim,
                num_classes=args.gradient_proj_num_classes,
                alpha_ema=args.persistent_proj_alpha_ema,
                burn_in=args.persistent_proj_burn_in,
                tau=args.asymmetric_tau,
                aux_lr=args.gradient_proj_lr,
            )
            print(f"[Defense] PersistentProjectionDefense active: "
                  f"embedding_dim={args.gradient_proj_embedding_dim}, "
                  f"num_classes={args.gradient_proj_num_classes}, "
                  f"alpha_ema={args.persistent_proj_alpha_ema}, "
                  f"burn_in={args.persistent_proj_burn_in}, tau={args.asymmetric_tau}, "
                  f"aux_lr={args.gradient_proj_lr}")

    if args.mdpp:
        if model.sep_monitor is None:
            print("[WARNING] --mdpp requires --monitor-separability True. "
                  "Defense is DISABLED because the monitor is not active.")
        else:
            model.mdpp_defense = possible_defenses.MultiDirectionPersistentProjection(
                embedding_dim=args.gradient_proj_embedding_dim,
                num_classes=args.gradient_proj_num_classes,
                k_directions=args.mdpp_k_directions,
                alpha_ema=args.mdpp_alpha_ema,
                burn_in=args.mdpp_burn_in,
                tau=args.asymmetric_tau,
                aux_lr=args.gradient_proj_lr,
            )
            print(f"[Defense] MultiDirectionPersistentProjection active: "
                  f"k={args.mdpp_k_directions}, alpha_ema={args.mdpp_alpha_ema}, "
                  f"burn_in={args.mdpp_burn_in}, tau={args.asymmetric_tau}, "
                  f"embedding_dim={args.gradient_proj_embedding_dim}, "
                  f"num_classes={args.gradient_proj_num_classes}")

    if args.asymmetric_defense:
        if model.sep_monitor is None:
            # the defense reads Fisher divergence from the monitor each epoch;
            # without it the defense cannot function — warn loudly rather than silently no-op.
            print("[WARNING] --asymmetric-defense requires --monitor-separability True. "
                  "Defense is DISABLED because the monitor is not active.")
        else:
            model.asymmetric_defense = possible_defenses.AsymmetricAdaptivePerturbation(
                alpha=args.asymmetric_alpha,
                tau=args.asymmetric_tau,
                burn_in=args.asymmetric_burn_in,
                gradient_noise_std=args.asymmetric_noise_std,
                sign_flip=args.asymmetric_sign_flip,
                za_noise_std=args.asymmetric_za_noise_std,
            )
            print(f"[Defense] AsymmetricAdaptivePerturbation active: "
                  f"alpha={args.asymmetric_alpha}, tau={args.asymmetric_tau}, "
                  f"burn_in={args.asymmetric_burn_in}, "
                  f"gradient_noise_std={args.asymmetric_noise_std}, "
                  f"sign_flip={args.asymmetric_sign_flip}, "
                  f"za_noise_std={args.asymmetric_za_noise_std}")

    stone1 = args.stone1  # 50 int(args.epochs * 0.5)
    stone2 = args.stone2  # 85 int(args.epochs * 0.8)
    lr_scheduler_top_model = torch.optim.lr_scheduler.MultiStepLR(model.optimizer_top_model,
                                                                  milestones=[stone1, stone2], gamma=args.step_gamma)
    lr_scheduler_m_a = torch.optim.lr_scheduler.MultiStepLR(model.optimizer_malicious_bottom_model_a,
                                                            milestones=[stone1, stone2], gamma=args.step_gamma)
    lr_scheduler_b_b = torch.optim.lr_scheduler.MultiStepLR(model.optimizer_benign_bottom_model_b,
                                                            milestones=[stone1, stone2], gamma=args.step_gamma)

    train_loader, val_loader = set_loaders()

    dir_save_model = args.save_dir + f"/saved_models/{args.dataset}_saved_models"
    if not os.path.exists(dir_save_model):
        os.makedirs(dir_save_model)

    # start training. do evaluation every epoch.
    # print('Test the initialized model:')
    # print('Evaluation on the training dataset:')
    # test_per_epoch(test_loader=train_loader, framework=model, k=args.k, loss_func_top_model=model.loss_func_top_model)
    # print('Evaluation on the testing dataset:')
    # test_per_epoch(test_loader=val_loader, framework=model, k=args.k, loss_func_top_model=model.loss_func_top_model)
    for epoch in range(args.epochs):
        # make epoch visible to batch-level methods (e.g. defense burn-in guard)
        model.current_epoch = epoch

        print('model.optimizer_top_model current lr {:.5e}'.format(model.optimizer_top_model.param_groups[0]['lr']))
        print('model.optimizer_malicious_bottom_model_a current lr {:.5e}'.format(
            model.optimizer_malicious_bottom_model_a.param_groups[0]['lr']))
        print('model.optimizer_benign_bottom_model_b current lr {:.5e}'.format(
            model.optimizer_benign_bottom_model_b.param_groups[0]['lr']))

        if epoch == 0:
            model.direct_attack_on = True
        else:
            model.direct_attack_on = False

        if epoch == args.epochs - 1 and args.if_cluster_outputsA:
            model.collect_outputs_a = True
            model.if_collect_training_dataset_labels = True
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.dataset == 'Yahoo':
                for i in range(len(data)):
                    data[i] = data[i].long().cuda()
                target = target[0].long().cuda()
            else:
                data = data.float().cuda()
                target = target.long().cuda()
            loss_framework = model.simulate_train_round_per_batch(data, target)
            if batch_idx % 25 == 0:
                if args.dataset == 'Criteo':
                    num_samples = len(train_loader) * BATCH_SIZE
                else:
                    num_samples = len(train_loader.dataset)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), num_samples,
                           100. * batch_idx / len(train_loader), loss_framework.data.item()))
        lr_scheduler_top_model.step()
        lr_scheduler_m_a.step()
        lr_scheduler_b_b.step()

        if model.sep_monitor is not None:
            model.sep_monitor.compute_epoch_metrics(epoch)
            # refresh the Fisher divergence used by both defenses for the next epoch
            if model.asymmetric_defense is not None:
                model.asymmetric_defense.update_divergence(model.sep_monitor)
            if model.adversarial_aux_defense is not None:
                model.adversarial_aux_defense.update_divergence(model.sep_monitor)
            if model.gradient_projection_defense is not None:
                model.gradient_projection_defense.update_divergence(model.sep_monitor)
            if model.persistent_projection_defense is not None:
                model.persistent_projection_defense.update_divergence(model.sep_monitor)
            if model.mdpp_defense is not None:
                model.mdpp_defense.update_divergence(model.sep_monitor)

        if epoch == args.epochs - 1:
            txt_name = f"{args.dataset}_saved_framework{setting_str}"
            savedStdout = sys.stdout
            with open(dir_save_model + '/' + txt_name + '.txt', 'w+') as file:
                sys.stdout = file
                print('Evaluation on the training dataset:')
                test_per_epoch(test_loader=train_loader, framework=model, k=args.k,
                               loss_func_top_model=model.loss_func_top_model)
                print('Evaluation on the testing dataset:')
                test_per_epoch(test_loader=val_loader, framework=model, k=args.k,
                               loss_func_top_model=model.loss_func_top_model)

                if not args.use_top_model:
                    # performance of the direct label inference attack
                    print("inferred correctly:", model.inferred_correct)
                    if args.dataset == 'Criteo':
                        num_samples = len(train_loader) * BATCH_SIZE
                    else:
                        num_samples = len(train_loader.dataset)
                    num_all_train_samples = num_samples
                    print("all:", num_all_train_samples)
                    print("Direct label inference accuracy:", model.inferred_correct / num_all_train_samples)
                    print("Direct label inference attack evaluated...")

                sys.stdout = savedStdout
            print('Last epoch evaluation saved to txt!')

        print('Evaluation on the training dataset:')
        test_per_epoch(test_loader=train_loader, framework=model, k=args.k,
                       loss_func_top_model=model.loss_func_top_model)
        print('Evaluation on the testing dataset:')
        test_per_epoch(test_loader=val_loader, framework=model, k=args.k, loss_func_top_model=model.loss_func_top_model)

    if model.sep_monitor is not None:
        model.sep_monitor.save_to_csv()

    # save model
    torch.save(model, os.path.join(dir_save_model, f"{args.dataset}_saved_framework{setting_str}.pth"),
               pickle_module=dill)

    if args.if_cluster_outputsA:
        outputsA_list = model.outputs_a.detach().clone().cpu().numpy().tolist()
        labels_list = model.labels_training_dataset.detach().clone().cpu().numpy().tolist()
        # plot TSNE cluster result
        outputsA_pca_tsne = TSNE()
        outputsA_pca_tsne.fit_transform(outputsA_list)
        df_outputsA_pca_tsne = pd.DataFrame(outputsA_pca_tsne.embedding_, index=labels_list)
        # plot the TSNE result
        colors = ['k', 'r', 'y', 'g', 'c', 'b', 'm', 'grey', 'orange', 'pink']
        # get num_classes
        dataset_setup = get_dataset.get_dataset_setup_by_name(args.dataset)
        num_classes = dataset_setup.num_classes
        for i in range(num_classes):
            plt.scatter(df_outputsA_pca_tsne.loc[i][0], df_outputsA_pca_tsne.loc[i][1], color=colors[i], marker='.')
        plt.title('VFL OutputsA TSNE' + setting_str)
        # plt.show()
        dir_save_tsne_pic = args.save_dir + f"/csv_files/{args.dataset}_csv_files"
        if not os.path.exists(dir_save_tsne_pic):
            os.makedirs(dir_save_tsne_pic)
        df_outputsA_pca_tsne.to_csv(
            dir_save_tsne_pic + f"/{args.dataset}_outputs_a_tsne{setting_str}.csv")
        plt.savefig(os.path.join(dir_save_tsne_pic, f"{args.dataset}_Resnet_VFL_OutputsA_TSNE{setting_str}.png"))
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='vfl framework training')
    # dataset paras
    parser.add_argument('-d', '--dataset', default='Criteo', type=str,
                        help='name of dataset',
                        choices=['CIFAR10', 'CIFAR100', 'TinyImageNet', 'CINIC10L', 'Yahoo', 'Criteo', 'BCW'])
    parser.add_argument('--path-dataset', help='path_dataset',
                        type=str, default='D:/Datasets/yahoo_answers_csv/')
    '''
    'D:/Datasets/CIFAR10'
    'D:/Datasets/CIFAR100'
    'D:/Datasets/TinyImageNet'
    'D:/Datasets/CINIC10L'
    'D:/Datasets/BC_IDC'
    'D:/Datasets/Criteo/criteo1e?.csv'
    'D:/Datasets/yahoo_answers_csv/'
    'D:/Datasets/BreastCancerWisconsin/wisconsin.csv'
    '''
    # framework paras
    parser.add_argument('--use-top-model', help='vfl framework has top model or not. If no top model'
                                                'is used, automatically turn on direct label inference attack,'
                                                'and report label inference accuracy on the training dataset',
                        type=ast.literal_eval, default=True)
    parser.add_argument('--test-upper-bound', help='if set to True, test the upper bound of our attack: if all the'
                                                   'adversary\'s samples are labeled, how accurate is the adversary\'s '
                                                   'label inference ability?',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--half', help='half number of features, generally seen as the adversary\'s feature num. '
                                       'You can change this para (lower that party_num) to evaluate the sensitivity '
                                       'of our attack -- pls make sure that the model to be resumed is '
                                       'correspondingly trained.',
                        type=int,
                        default=16)  # choices=[16, 14, 32, 1->party_num]. CIFAR10-16, Liver-14, TinyImageNet-32
    # evaluation & visualization paras
    parser.add_argument('--k', help='top k accuracy',
                        type=int, default=5)
    parser.add_argument('--if-cluster-outputsA', help='if_cluster_outputsA',
                        type=ast.literal_eval, default=True)
    # attack paras
    parser.add_argument('--monitor-separability',
                        help='log per-epoch embedding separability metrics (Fisher, silhouette, etc.) '
                             'for Phase 1 characterization study',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--use-mal-optim',
                        help='whether the attacker uses the malicious optimizer',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--use-mal-optim-all',
                        help='whether all participants use the malicious optimizer. If set to '
                             'True, use_mal_optim will be automatically set to True.',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--use-mal-optim-top',
                        help='whether the server(top model) uses the malicious optimizer',
                        type=ast.literal_eval, default=False)
    # saving path paras
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained models and csv files',
                        default='./saved_experiment_results', type=str)
    # possible defenses on/off paras
    parser.add_argument('--ppdl', help='turn_on_privacy_preserving_deep_learning',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--gc', help='turn_on_gradient_compression',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--lap-noise', help='turn_on_lap_noise',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--multistep_grad', help='turn on multistep-grad',
                        type=ast.literal_eval, default=False)
    # paras about possible defenses
    parser.add_argument('--ppdl-theta-u', help='theta-u parameter for defense privacy-preserving deep learning',
                        type=float, default=0.75)
    parser.add_argument('--gc-preserved-percent', help='preserved-percent parameter for defense gradient compression',
                        type=float, default=0.75)
    parser.add_argument('--noise-scale', help='noise-scale parameter for defense noisy gradients',
                        type=float, default=1e-3)
    parser.add_argument('--multistep_grad_bins', help='number of bins in multistep-grad',
                        type=int, default=6)
    parser.add_argument('--multistep_grad_bound_abs', help='bound of multistep-grad',
                        type=float, default=3e-2)
    # asymmetric adaptive perturbation defense (novel Phase 2 defense)
    # requires --monitor-separability True to function
    parser.add_argument('--asymmetric-defense',
                        help='turn on AsymmetricAdaptivePerturbation: server suppresses only '
                             'Party A\'s gradient when Fisher divergence exceeds tau. '
                             'Requires --monitor-separability True.',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--asymmetric-alpha',
                        help='suppression aggressiveness for asymmetric defense. '
                             'scale = max(0, 1 - alpha*(divergence - tau)). '
                             'Higher = faster suppression per unit of divergence.',
                        type=float, default=1.0)
    parser.add_argument('--asymmetric-tau',
                        help='Fisher divergence threshold to trigger suppression. '
                             'Set above benign-training ceiling (~0.0). '
                             'Recommended: 0.10 for CIFAR10, 0.05 for CIFAR100.',
                        type=float, default=0.10)
    parser.add_argument('--asymmetric-burn-in',
                        help='epochs before asymmetric defense activates. '
                             'Random-init embeddings produce spurious divergence '
                             'for the first ~8 epochs; this guard prevents false positives.',
                        type=int, default=8)
    parser.add_argument('--asymmetric-noise-std',
                        help='gradient noise injection for asymmetric defense. '
                             'When > 0, Gaussian noise is added after suppression: '
                             'grad = scale*grad + noise_std*(1-scale)*E[|grad|]*randn(). '
                             'Prevents MaliciousSGD zero-gradient vacuum at high alpha. '
                             '0.0 = suppression only (original behaviour).',
                        type=float, default=0.0)
    parser.add_argument('--asymmetric-sign-flip',
                        help='When defense fires, alternate the sign of Party A\'s gradient '
                             'every batch. Consecutive opposite-sign gradients force '
                             'MaliciousSGD ratio=1.0 on every batch, collapsing amplification '
                             'to standard SGD and causing weight updates to average to zero.',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--asymmetric-za-noise-std',
                        help='Embedding-space corruption: add Gaussian noise to z_a BEFORE '
                             'the top model forward pass. noise = std*(1-scale)*randn(z_a). '
                             'Monitor and bottom model backward still see clean z_a. '
                             '0.0 = disabled.',
                        type=float, default=0.0)
    # adversarial auxiliary classifier defense (Phase 18 — for high-class-count datasets)
    parser.add_argument('--adversarial-aux-defense',
                        help='Server maintains an auxiliary classifier on z_a and sends its '
                             'REVERSED gradient to Party A, actively pushing z_a toward '
                             'class-indiscriminativeness. Requires --monitor-separability True.',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--adversarial-aux-lambda',
                        help='Scale of the reversed auxiliary gradient (lambda_adv). '
                             '1.0 = equal weight to top-model gradient. '
                             'Increase to make adversarial correction dominate.',
                        type=float, default=1.0)
    parser.add_argument('--adversarial-aux-lr',
                        help='Learning rate for the server auxiliary classifier (Adam). '
                             'Default 1e-3 works well; lower if aux classifier over-fits early.',
                        type=float, default=1e-3)
    parser.add_argument('--adversarial-aux-embedding-dim',
                        help='Output dimension of Party A bottom model (size_bottom_out). '
                             'CIFAR-100: 100.  CIFAR-10/CINIC10L: 10.',
                        type=int, default=100)
    parser.add_argument('--adversarial-aux-num-classes',
                        help='Number of label classes (output dim of auxiliary classifier). '
                             'CIFAR-100: 100.  CIFAR-10/CINIC10L: 10.',
                        type=int, default=100)
    # gradient projection defense (Phase 19 — numerically safe alternative to adv aux)
    # projects grad_output_a onto the subspace orthogonal to the discriminative direction.
    # ||grad_proj|| <= ||grad_output_a|| always — no NaN amplification risk.
    # requires --monitor-separability True to function.
    parser.add_argument('--gradient-projection-defense',
                        help='Remove the discriminative component from grad_output_a via '
                             'orthogonal projection. Safe by construction: projection can '
                             'only shrink the gradient, never amplify it. '
                             'Requires --monitor-separability True.',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--gradient-proj-embedding-dim',
                        help='Output dimension of Party A bottom model (size_bottom_out). '
                             'CIFAR-100: 100.  CIFAR-10/CINIC10L: 10.',
                        type=int, default=100)
    parser.add_argument('--gradient-proj-num-classes',
                        help='Number of label classes. CIFAR-100: 100.  CIFAR-10/CINIC10L: 10.',
                        type=int, default=100)
    parser.add_argument('--gradient-proj-lr',
                        help='Learning rate for the auxiliary classifier used to identify '
                             'the discriminative direction (Adam). Default 1e-3.',
                        type=float, default=1e-3)
    # persistent projection defense (Phase 22/23 — EMA-based stable projection)
    # shares --gradient-proj-embedding-dim, --gradient-proj-num-classes, --gradient-proj-lr,
    # and --asymmetric-tau from the args above.
    parser.add_argument('--persistent-projection',
                        help='EMA-based persistent discriminative subspace projection. '
                             'Maintains a smoothed direction estimate and projects every detected '
                             'epoch — fixes the one-shot collapse in --gradient-projection-defense. '
                             'Requires --monitor-separability True.',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--persistent-proj-alpha-ema',
                        help='EMA smoothing coefficient for the discriminative direction. '
                             '0.1=slow adaptation, 0.3=fast. Default 0.2.',
                        type=float, default=0.2)
    parser.add_argument('--persistent-proj-burn-in',
                        help='Epochs before PersistentProjection activates. '
                             'Shorter than GradProj default (4 vs 8) since EMA stabilises faster.',
                        type=int, default=4)
    parser.add_argument('--mdpp',
                        help='Multi-Direction Persistent Projection: removes K SVD-derived '
                             'directions from W_aux simultaneously each detected epoch. '
                             'Covers DCR(K,C)=K/min(C-1,embed_dim) of the discriminative '
                             'subspace. Uses --gradient-proj-* and --asymmetric-tau. '
                             'Requires --monitor-separability True.',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--mdpp-k-directions',
                        help='Number of SVD directions to project out. '
                             'CIFAR-100 (C=100, embed=100): k=10 → DCR≈10%%, k=20 → DCR≈20%%.',
                        type=int, default=10)
    parser.add_argument('--mdpp-alpha-ema',
                        help='EMA coefficient blending current SVD directions into D_ema. '
                             'Same semantics as --persistent-proj-alpha-ema. Default 0.2.',
                        type=float, default=0.2)
    parser.add_argument('--mdpp-burn-in',
                        help='Epochs before MDPP activates. Default 4.',
                        type=int, default=4)
    # training paras
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of datasets loading workers (default: 4)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                        metavar='LR', help='initial learning rate')  # TinyImageNet=5e-2, Yahoo=1e-3
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--step-gamma', default=0.1, type=float, metavar='S',
                        help='gamma for step scheduler')
    parser.add_argument('--stone1', default=50, type=int, metavar='s1',
                        help='stone1 for step scheduler')
    parser.add_argument('--stone2', default=85, type=int, metavar='s2',
                        help='stone2 for step scheduler')
    parser.add_argument('--manual-seed', dest='manual_seed', type=int, default=0,
                        help='manual random seed for reproducibility across runs')
    args = parser.parse_args()
    if args.use_mal_optim_all:
        args.use_mal_optim = True
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.manual_seed)
    main()
