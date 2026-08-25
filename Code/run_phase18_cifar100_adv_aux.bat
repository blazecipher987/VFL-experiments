@echo off
:: ============================================================
:: Phase 18 -- CIFAR100: Adversarial Auxiliary Classifier Defense
::
:: WHY THIS EXISTS:
::   All gradient-space and embedding-space approaches for CIFAR-100
::   have failed (8/8 configurations). The fundamental problem:
::
::   - Too little suppression (alpha=1.0): scale stays at ~0.65,
::     MaliciousSGD builds discriminative structure over 150 epochs.
::   - Too much suppression (alpha=2.0): zeroing gradient early lets
::     MaliciousSGD amplify INTERNAL gradients unchecked -> WORSE.
::   This is a catch-22 in the gradient suppression design space.
::
:: NEW MECHANISM:
::   Instead of suppressing gradients, the server REVERSES them.
::   The server trains a small auxiliary classifier A_aux on z_a and
::   sends the REVERSED gradient of A_aux's loss to Party A:
::
::     final_grad = grad_from_top - lambda * d(L_aux)/d(z_a)
::
::   d(L_aux)/d(z_a) points toward MORE discriminativeness.
::   Subtracting it pushes z_a AWAY from discriminativeness.
::
::   Crucially: MaliciousSGD AMPLIFIES the reversed gradient.
::   Stronger attack -> larger ratio -> stronger anti-discriminative push.
::   The defense is SELF-REINFORCING -- opposite of the catch-22.
::
:: LAMBDA SWEEP:
::   lambda_adv controls the strength of the adversarial correction.
::   Run 3 values: 0.5 (mild), 1.0 (equal weight), 2.0 (dominant).
::   The benign VFL utility is maintained by the normal top-model
::   gradient component (grad_from_top remains, just corrected).
::
::   Target: defended MC < benign MC (30.33%% seed-0 reference).
::   Stretch: defended MC below the best prior attempt (43.10%%).
::
:: PARAMETERS:
::   EPOCHS    : 150 (same as CIFAR100 standard)
::   HALF      : 16
::   K         : 5  (top-5 for 100-class)
::   n_labeled : 400 (4 per class)
::   TAU       : 0.10 (Fisher detection threshold)
::   BURNIN    : 8
::   AUX_DIM   : 100 (z_a embedding dim for CIFAR100)
::   AUX_CLASSES: 100
::   AUX_LR    : 1e-3 (Adam for server auxiliary classifier)
::
:: CHECKPOINT NAMES:
::   mal_adv_aux-l=0.5_half=16_ep150.pth
::   mal_adv_aux-l=1.0_half=16_ep150.pth
::   mal_adv_aux-l=2.0_half=16_ep150.pth
::
:: PREREQUISITE:
::   Benign and attack baselines (normal_half=16.pth, mal_half=16.pth)
::   must already exist from Phase 5. Phase 18 only runs Stage 1 defense
::   + Stage 2 for each lambda variant. No need to re-run benign/attack.
::
:: RUNTIME ESTIMATE:
::   ~6 hrs per lambda variant (150ep CIFAR100 + MC).
::   Total sequential: ~18 hrs for all 3 lambdas.
:: ============================================================

set DATASET=CIFAR100
set DATAPATH=.\data\CIFAR100
set EPOCHS=150
set HALF=16
set K=5

set TAU=0.10
set BURNIN=8
set AUX_DIM=100
set AUX_CLASSES=100
set AUX_LR=0.001

set MODELSDIR=.\saved_experiment_results\saved_models\CIFAR100_saved_models
set RESUMEDIR=.\saved_experiment_results\saved_models\

echo.
echo ============================================================
echo  PHASE 18 -- CIFAR100 Adversarial Auxiliary Classifier Defense
echo  Lambda sweep: 0.5, 1.0, 2.0
echo  Baselines (benign + attack) reused from Phase 5.
echo ============================================================
echo.
echo  MECHANISM: final_grad = grad_from_top - lambda * d(L_aux)/d(z_a)
echo  MaliciousSGD amplifying the reversed gradient strengthens defense.
echo.

:: ============================================================
::  LAMBDA = 0.5  (mild adversarial correction)
:: ============================================================
echo ===================================================
echo  LAMBDA = 0.5  (mild -- adversarial correction at half weight)
echo ===================================================
echo.

echo ===== [L0.5-1/2] Stage 1 -- Active + Adv Aux Defense (lambda=0.5), 150ep =====
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --adversarial-aux-defense True ^
  --adversarial-aux-lambda 0.5 ^
  --adversarial-aux-lr %AUX_LR% ^
  --adversarial-aux-embedding-dim %AUX_DIM% ^
  --adversarial-aux-num-classes %AUX_CLASSES% ^
  --asymmetric-tau %TAU% ^
  --asymmetric-burn-in %BURNIN%
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_adv_aux-l=0.5_half=16.pth" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_adv_aux-l=0.5_half=16_ep150.pth"
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_adv_aux-l=0.5_half=16.txt" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_adv_aux-l=0.5_half=16_ep150.txt"
echo.

echo ===== [L0.5-2/2] Stage 2 -- MC on lambda=0.5 checkpoint =====
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 400 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_adv_aux-l=0.5_half=16_ep150.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1
echo.
echo LAMBDA=0.5 COMPLETE.
echo.

:: ============================================================
::  LAMBDA = 1.0  (equal weight: adversarial = top-model gradient)
:: ============================================================
echo ===================================================
echo  LAMBDA = 1.0  (equal weight -- adversarial correction at full weight)
echo ===================================================
echo.

echo ===== [L1.0-1/2] Stage 1 -- Active + Adv Aux Defense (lambda=1.0), 150ep =====
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --adversarial-aux-defense True ^
  --adversarial-aux-lambda 1.0 ^
  --adversarial-aux-lr %AUX_LR% ^
  --adversarial-aux-embedding-dim %AUX_DIM% ^
  --adversarial-aux-num-classes %AUX_CLASSES% ^
  --asymmetric-tau %TAU% ^
  --asymmetric-burn-in %BURNIN%
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_adv_aux-l=1.0_half=16.pth" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_adv_aux-l=1.0_half=16_ep150.pth"
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_adv_aux-l=1.0_half=16.txt" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_adv_aux-l=1.0_half=16_ep150.txt"
echo.

echo ===== [L1.0-2/2] Stage 2 -- MC on lambda=1.0 checkpoint =====
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 400 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_adv_aux-l=1.0_half=16_ep150.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1
echo.
echo LAMBDA=1.0 COMPLETE.
echo.

:: ============================================================
::  LAMBDA = 2.0  (adversarial correction dominates)
:: ============================================================
echo ===================================================
echo  LAMBDA = 2.0  (dominant -- adversarial correction at 2x weight)
echo ===================================================
echo.

echo ===== [L2.0-1/2] Stage 1 -- Active + Adv Aux Defense (lambda=2.0), 150ep =====
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --adversarial-aux-defense True ^
  --adversarial-aux-lambda 2.0 ^
  --adversarial-aux-lr %AUX_LR% ^
  --adversarial-aux-embedding-dim %AUX_DIM% ^
  --adversarial-aux-num-classes %AUX_CLASSES% ^
  --asymmetric-tau %TAU% ^
  --asymmetric-burn-in %BURNIN%
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_adv_aux-l=2.0_half=16.pth" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_adv_aux-l=2.0_half=16_ep150.pth"
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_adv_aux-l=2.0_half=16.txt" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_adv_aux-l=2.0_half=16_ep150.txt"
echo.

echo ===== [L2.0-2/2] Stage 2 -- MC on lambda=2.0 checkpoint =====
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 400 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_adv_aux-l=2.0_half=16_ep150.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1
echo.
echo LAMBDA=2.0 COMPLETE.
echo.

echo ============================================================
echo  Phase 18 COMPLETE -- CIFAR100 Adversarial Aux Defense Done.
echo.
echo  Baselines from Phase 5 (EXP-012):
echo    Benign MC  : 30.33%% (seed 0)
echo    Attack MC  : 47.86%%
echo    Best prior defense (alpha=1.0): 43.10%% -- still 12.77pp above benign
echo.
echo  New results in CIFAR100_saved_models/:
echo    model_completion_*_adv_aux-l=0.5_*_nlabeled=400.txt
echo    model_completion_*_adv_aux-l=1.0_*_nlabeled=400.txt
echo    model_completion_*_adv_aux-l=2.0_*_nlabeled=400.txt
echo.
echo  SUCCESS if any defended MC < 30.33%% (benign baseline).
echo  PARTIAL if any defended MC < 43.10%% (prior best).
echo  Check VFL accuracy .txt files for utility cost:
echo    expect some drop vs attack-only (adversarial correction hurts joint task).
echo.
echo  If lambda=2.0 overshoots (VFL utility collapses), try lambda=1.5 next.
echo  If lambda=0.5 shows progress but not success, combine with --asymmetric-defense.
echo ============================================================
echo.
pause
