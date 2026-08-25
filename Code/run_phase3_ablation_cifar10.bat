@echo off
:: ============================================================
:: Phase 3 — Ablation Study: CIFAR10
::
:: Tests sensitivity of AsymmetricAdaptivePerturbation to
:: the two key hyperparameters: alpha (suppression strength)
:: and tau (detection threshold).
::
:: Phase 2 baseline (already done): alpha=1.0, tau=0.10, burn_in=8
:: -> Active+defense: 52.28% train, 48.96% test (model completion)
:: -> Fisher divergence ep29: +0.522
::
:: This bat runs 6 NEW Stage 1 conditions:
::   alpha ablation (tau=0.10 fixed): alpha=0.5, alpha=2.0
::   tau ablation (alpha=1.0 fixed):  tau=0.05, tau=0.15
::   Benign FP check for each tau:    tau=0.05, tau=0.15 (benign)
::
:: After this bat completes, run run_phase3_model_completion_ablation.bat
:: to get model completion accuracy for each new checkpoint.
:: ============================================================

set DATASET=CIFAR10
set DATAPATH=.\data\CIFAR10
set EPOCHS=30
set HALF=16
set K=4
set BURNIN=8

echo.
echo ========================================================
echo  CIFAR10 Phase 3 Ablation — Stage 1 VFL Training
echo ========================================================

:: ---- ALPHA ABLATION (tau=0.10 fixed) ----

echo.
echo ----- [1/6] alpha=0.5, tau=0.10 (less aggressive suppression) -----
echo   Expect: higher attack accuracy than alpha=1.0 (less suppression)
echo   but same or better VFL utility
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --asymmetric-defense True ^
  --asymmetric-alpha 0.5 ^
  --asymmetric-tau 0.10 ^
  --asymmetric-burn-in %BURNIN% ^
  --if-cluster-outputsA True

echo.
echo ----- [2/6] alpha=2.0, tau=0.10 (more aggressive suppression) -----
echo   Expect: lower attack accuracy than alpha=1.0
echo   but larger VFL utility cost
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --asymmetric-defense True ^
  --asymmetric-alpha 2.0 ^
  --asymmetric-tau 0.10 ^
  --asymmetric-burn-in %BURNIN% ^
  --if-cluster-outputsA True

:: ---- TAU ABLATION (alpha=1.0 fixed) ----

echo.
echo ----- [3/6] alpha=1.0, tau=0.05 (lower threshold, defense fires earlier) -----
echo   CIFAR10 benign divergence is ~-0.12, so tau=0.05 is still safe.
echo   Expect: stronger suppression than tau=0.10 (fires from a lower divergence level)
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --asymmetric-defense True ^
  --asymmetric-alpha 1.0 ^
  --asymmetric-tau 0.05 ^
  --asymmetric-burn-in %BURNIN% ^
  --if-cluster-outputsA True

echo.
echo ----- [4/6] alpha=1.0, tau=0.15 (higher threshold, defense fires later) -----
echo   Expect: defense fires less aggressively, higher residual attack accuracy
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --asymmetric-defense True ^
  --asymmetric-alpha 1.0 ^
  --asymmetric-tau 0.15 ^
  --asymmetric-burn-in %BURNIN% ^
  --if-cluster-outputsA True

:: ---- BENIGN FALSE-POSITIVE CHECK for tau variants ----

echo.
echo ----- [5/6] Benign, tau=0.05 — FP check (defense should still NOT fire) -----
echo   CIFAR10 benign divergence ~-0.12 which is well below tau=0.05
echo   FP rate should be zero.
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --asymmetric-defense True ^
  --asymmetric-alpha 1.0 ^
  --asymmetric-tau 0.05 ^
  --asymmetric-burn-in %BURNIN% ^
  --if-cluster-outputsA True

echo.
echo ----- [6/6] Benign, tau=0.15 — FP check (higher threshold, also should NOT fire) -----
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --asymmetric-defense True ^
  --asymmetric-alpha 1.0 ^
  --asymmetric-tau 0.15 ^
  --asymmetric-burn-in %BURNIN% ^
  --if-cluster-outputsA True

echo.
echo ========================================================
echo  CIFAR10 Phase 3 Ablation Stage 1 COMPLETE
echo.
echo  6 new checkpoints saved to:
echo    .\saved_experiment_results\saved_models\CIFAR10_saved_models\
echo.
echo  CSV files saved to:
echo    .\saved_experiment_results\csv_files\CIFAR10_csv_files\
echo.
echo  Next: run run_phase3_model_completion_ablation.bat
echo ========================================================
echo.
pause
