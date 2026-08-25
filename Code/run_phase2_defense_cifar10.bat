@echo off
:: ============================================================
:: Phase 2 — Defense Evaluation (CIFAR10)
:: AsymmetricAdaptivePerturbation vs. Active Attack
::
:: Purpose: measure whether the defense reduces Party A's
:: embedding class-separability under MaliciousSGD, and
:: produce defended .pth checkpoints for Stage 2 model
:: completion evaluation.
::
:: Run this AFTER Phase 1 characterization is complete.
:: Run model_completion on the saved .pth outputs from here
:: to get the final defended vs. undefended accuracy comparison.
::
:: Defense hyperparameters (from Phase 1 findings):
::   tau      = 0.10   (benign CIFAR10 divergence ≈ -0.12, safe margin)
::   alpha    = 1.0    (scale hits 0 when divergence = tau + 1/alpha = 1.10)
::   burn_in  = 8      (spurious init divergence clears by epoch ~4 on CIFAR10)
:: ============================================================

set DATASET=CIFAR10
set DATAPATH=.\data\CIFAR10
set EPOCHS=30
set HALF=16
set K=4

set TAU=0.10
set ALPHA=1.0
set BURNIN=8

echo.
echo ===== CONDITION 1: Active attack  +  defense ON (key result) =====
echo   Expected: Fisher divergence should SHRINK vs. undefended active run.
echo   Fisher_divergence (Phase 1 undefended): +0.444 at epoch 29
echo   Target: divergence suppressed toward ~0 by defense.
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --asymmetric-defense True ^
  --asymmetric-alpha %ALPHA% ^
  --asymmetric-tau %TAU% ^
  --asymmetric-burn-in %BURNIN% ^
  --if-cluster-outputsA True

echo.
echo ===== CONDITION 2: Benign baseline  +  defense ON (false-positive check) =====
echo   Expected: defense should NOT trigger (benign divergence ≈ -0.12, below tau=0.10).
echo   VFL top-1 accuracy should match undefended benign baseline.
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --asymmetric-defense True ^
  --asymmetric-alpha %ALPHA% ^
  --asymmetric-tau %TAU% ^
  --asymmetric-burn-in %BURNIN% ^
  --if-cluster-outputsA True

echo.
echo ===== Phase 2 CIFAR10 complete =====
echo Next step: run model_completion on the defended .pth checkpoints
echo saved under: .\saved_experiment_results\saved_models\CIFAR10_saved_models\
echo Look for filenames containing "asym_def"
echo.
pause
