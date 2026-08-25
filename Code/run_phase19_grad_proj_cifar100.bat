@echo off
:: ============================================================
:: Phase 19 -- Gradient Projection Defense, CIFAR-100, 150 epochs
::
:: WHY THIS BAT EXISTS:
::   Phase 18 (AdversarialAuxiliaryDefense) failed catastrophically on all 3
::   lambda values (0.5, 1.0, 2.0) due to a NaN cascade:
::     - aux_grad = d(L_aux)/d(z_a) is unbounded (nn.Linear grows large weights)
::     - final_grad = grad_output_a - lambda * aux_grad
::     - MaliciousSGD amplifies the NaN-containing gradient by up to 5x
::     - Result: model destroyed (lambda=0.5 produced 1.001% MC; 1.0/2.0 crashed)
::
::   Root fix: instead of SUBTRACTING a potentially large aux_grad, PROJECT
::   grad_output_a onto the subspace ORTHOGONAL to the discriminative direction.
::
::   Projection formula (GradientProjectionDefense in possible_defenses.py):
::     d_aux      = d(L_aux)/d(z_a)
::     d_aux_norm = d_aux / ||d_aux||        (unit vector — magnitude doesn't matter)
::     proj_coeff = grad_output_a . d_aux_norm
::     grad_proj  = grad_output_a - proj_coeff * d_aux_norm
::
::   KEY NUMERICAL SAFETY:
::     ||grad_proj|| <= ||grad_output_a|| always (Cauchy-Schwarz)
::     MaliciousSGD amplifying grad_proj can only reach the ORIGINAL magnitude.
::     Zero NaN risk from the projection itself.
::     Normalization means aux_classifier weight growth doesn't matter.
::
::   DETECTION: Same Fisher divergence gate as all prior defenses.
::     Aux classifier trains every batch (to stay current on z_a).
::     Projection fires only when epoch >= burn_in AND divergence > tau.
::
:: CHECKPOINT NAMING:
::   vfl_framework.py saves:
::     CIFAR100_saved_framework_lr=0.1_mal_grad_proj_half=16.pth
::   Renamed to _ep150.pth for consistency with prior phases.
::
:: SUCCESS CRITERIA:
::   Below 43.10%  (best prior CIFAR-100 attempt, Phase 9 noise n=2.0)
::   Below 29.56%  (4-seed benign mean, EXP-029) = CIFAR-100 story complete
::   Note: compare against 29.56 +/- 2.93% (4-seed std), not single-seed 30.33%.
:: ============================================================

set DATASET=CIFAR100
set DATAPATH=.\data\CIFAR100
set EPOCHS=150
set HALF=16
set K=5

set TAU=0.10
set BURNIN=8
set PROJ_LR=1e-3
set EMBED_DIM=100
set NUM_CLASSES=100

set MODELSDIR=.\saved_experiment_results\saved_models\CIFAR100_saved_models
set RESUMEDIR=.\saved_experiment_results\saved_models\

echo.
echo ============================================================
echo  PHASE 19 -- CIFAR-100 Gradient Projection Defense, 150 Epochs
echo ============================================================
echo.

:: ----------------------------------------------------------
:: STAGE 1: MaliciousSGD attack + gradient projection defense, 150 epochs
::
:: --gradient-projection-defense True: removes discriminative component from
::   grad_output_a via orthogonal projection; ||grad_proj|| <= ||grad_output_a||
::   always, so MaliciousSGD cannot amplify past original magnitude.
:: --monitor-separability True: required for Fisher divergence detection.
:: --use-mal-optim True: Party A uses MaliciousSGD (standard attack setup).
:: --use-mal-optim-top False: server uses standard SGD (matches EXP-014/015).
:: ----------------------------------------------------------
echo ===== [1/2] Stage 1 -- Attack + Gradient Projection, 150 epochs =====
echo Saves to: mal_grad_proj_half=16.pth
echo Renames to: mal_grad_proj_half=16_ep150.pth
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --gradient-projection-defense True ^
  --gradient-proj-embedding-dim %EMBED_DIM% ^
  --gradient-proj-num-classes %NUM_CLASSES% ^
  --gradient-proj-lr %PROJ_LR% ^
  --asymmetric-burn-in %BURNIN% ^
  --asymmetric-tau %TAU% ^
  --if-cluster-outputsA True

echo.
echo Renaming checkpoint to _ep150 variant...
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_grad_proj_half=16.pth" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_grad_proj_half=16_ep150.pth"
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_grad_proj_half=16.txt" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_grad_proj_half=16_ep150.txt"
echo [1/2] Stage 1 done.
echo.

echo ============================================================
echo  Stage 1 complete. Starting Stage 2 model completion...
echo ============================================================
echo.

:: ----------------------------------------------------------
:: STAGE 2: Model completion on gradient projection checkpoint
:: ----------------------------------------------------------
echo ===== [2/2] Stage 2 -- Model completion on grad_proj checkpoint =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% ^
  --dataset-path %DATAPATH% ^
  --n-labeled 400 ^
  --party-num 2 ^
  --half %HALF% ^
  --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_grad_proj_half=16_ep150.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.
echo ============================================================
echo  Phase 19 COMPLETE -- Results in:
echo  %MODELSDIR%\
echo.
echo  Stage 1: CIFAR100_saved_framework_lr=0.1_mal_grad_proj_half=16_ep150.txt
echo  Stage 2: model_completion_..._mal_grad_proj_half=16_ep150.pth_..._nlabeled=400.txt
echo.
echo  BASELINES (CIFAR-100, 4-seed anchored):
echo    Benign mean (4-seed):         29.56%% +/- 2.93%% (EXP-029, Phase 16)
echo    No defense (attack):          47.86%% (EXP-014, single-seed)
echo    Best prior defense (Phase 9): 43.10%% (z_a noise n=2.0)
echo    Phase 18 adv_aux lam=0.5:      1.001%% (NaN collapse -- model destroyed)
echo.
echo  TARGETS:
echo    Below 43.10%% = improvement over all prior CIFAR-100 defense attempts
echo    Below 29.56%% = below benign mean = CIFAR-100 defense story complete
echo    Below 26.63%% = below benign 1-sigma lower bound
echo ============================================================
echo.
pause
