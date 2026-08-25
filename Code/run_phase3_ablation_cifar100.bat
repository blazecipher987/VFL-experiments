@echo off
:: ============================================================
:: Phase 3 — Ablation Study: CIFAR100
::
:: Tests sensitivity of AsymmetricAdaptivePerturbation to
:: alpha and tau on CIFAR100 (100 classes, k=5, half=16).
::
:: Phase 2 baseline (already done): alpha=1.0, tau=0.10, burn_in=8
:: -> Active+defense: 21.35% train, 17.96% test (model completion)
:: -> Fisher divergence ep29: +0.167
:: -> Benign+defense: 11.27% train (high variance issue — single run)
::
:: This bat runs 8 NEW Stage 1 conditions:
::   alpha ablation (tau=0.10 fixed): alpha=0.5, alpha=2.0  [active]
::   tau ablation (alpha=1.0 fixed):  tau=0.05, tau=0.15    [active]
::   Benign runs for ALL 4 above + tau=0.10                  [benign x4]
::
:: The 4 benign runs produce independent model completion samples
:: for CIFAR100, which we average to get a stable benign baseline.
:: Since the defense does NOT fire on CIFAR100 benign (divergence
:: goes negative after burn-in), these benign checkpoints are
:: functionally equivalent to benign-no-defense.
:: ============================================================

set DATASET=CIFAR100
set DATAPATH=.\data\CIFAR100
set EPOCHS=30
set HALF=16
set K=5
set BURNIN=8

echo.
echo ========================================================
echo  CIFAR100 Phase 3 Ablation — Stage 1 VFL Training
echo ========================================================

:: ---- ALPHA ABLATION — ACTIVE ----

echo.
echo ----- [1/8] Active, alpha=0.5, tau=0.10 -----
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
echo ----- [2/8] Active, alpha=2.0, tau=0.10 -----
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

:: ---- TAU ABLATION — ACTIVE ----

echo.
echo ----- [3/8] Active, alpha=1.0, tau=0.05 -----
echo   Note: CIFAR100 active divergence peaks at ~0.13; tau=0.05 fires much earlier (epoch 1-2)
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
echo ----- [4/8] Active, alpha=1.0, tau=0.15 -----
echo   Note: CIFAR100 active divergence is only ~0.13 at peak; tau=0.15 barely fires.
echo   Expect: much weaker suppression (defense close to inactive)
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

:: ---- BENIGN RUNS — FOR STABLE BASELINE ----
:: These 4 benign runs (with different defense settings) produce
:: functionally equivalent checkpoints since the defense never fires
:: on CIFAR100 benign training. Running model_completion on all 4
:: gives us 4 independent samples to average for the benign baseline,
:: replacing the single unreliable 11.27% from Phase 2.

echo.
echo ----- [5/8] Benign, alpha=0.5, tau=0.10 — baseline sample 2 -----
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --asymmetric-defense True ^
  --asymmetric-alpha 0.5 ^
  --asymmetric-tau 0.10 ^
  --asymmetric-burn-in %BURNIN% ^
  --if-cluster-outputsA True

echo.
echo ----- [6/8] Benign, alpha=2.0, tau=0.10 — baseline sample 3 -----
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --asymmetric-defense True ^
  --asymmetric-alpha 2.0 ^
  --asymmetric-tau 0.10 ^
  --asymmetric-burn-in %BURNIN% ^
  --if-cluster-outputsA True

echo.
echo ----- [7/8] Benign, alpha=1.0, tau=0.05 — baseline sample 4 -----
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
echo ----- [8/8] Benign, alpha=1.0, tau=0.15 — baseline sample 5 -----
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
echo  CIFAR100 Phase 3 Ablation Stage 1 COMPLETE
echo.
echo  8 new checkpoints saved to:
echo    .\saved_experiment_results\saved_models\CIFAR100_saved_models\
echo.
echo  Benign baseline samples [5-8] above are used to stabilize
echo  the CIFAR100 benign model completion average.
echo  Since the defense does not fire on CIFAR100 benign (divergence
echo  goes negative by epoch 7, confirmed in Phase 2 CSV), these
echo  checkpoints are functionally equivalent to benign-no-defense.
echo.
echo  Next: run run_phase3_model_completion_ablation.bat
echo ========================================================
echo.
pause
