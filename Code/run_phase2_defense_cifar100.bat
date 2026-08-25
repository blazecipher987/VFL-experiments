@echo off
:: ============================================================
:: Phase 2 — Defense Evaluation (CIFAR100)
:: AsymmetricAdaptivePerturbation vs. Active Attack
::
:: Defense hyperparameters (from Phase 1 findings):
::   tau      = 0.10   (benign CIFAR100 divergence ≈ +0.002, safe margin)
::   alpha    = 1.0    (scale hits 0 when divergence = 1.10)
::   burn_in  = 8      (spurious init divergence clears by epoch ~8 on CIFAR100)
::
:: Note: the Fisher divergence gap is smaller on CIFAR100 (0.131) vs
:: CIFAR10 (0.444) because 100 classes compete for the same embedding
:: space. The defense should still trigger because 0.131 > tau=0.10.
:: If suppression is too weak, re-run with --asymmetric-tau 0.05.
:: ============================================================

set DATASET=CIFAR100
set DATAPATH=.\data\CIFAR100
set EPOCHS=30
set HALF=16
set K=5

set TAU=0.10
set ALPHA=1.0
set BURNIN=8

echo.
echo ===== CONDITION 1: Active attack  +  defense ON (key result) =====
echo   Expected: Fisher divergence shrinks vs. undefended active run.
echo   Fisher_divergence (Phase 1 undefended): +0.131 at epoch 29
echo   tau=0.10 means defense fires when divergence exceeds 0.10.
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
echo   Expected: defense should NOT trigger (benign divergence ≈ +0.002, below tau=0.10).
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
echo ===== Phase 2 CIFAR100 complete =====
echo Next step: run model_completion on the defended .pth checkpoints.
echo Look for filenames containing "asym_def" under:
echo   .\saved_experiment_results\saved_models\CIFAR100_saved_models\
echo.
pause
