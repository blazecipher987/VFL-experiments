@echo off
:: ============================================================
:: Phase 2b — Establish 30-Epoch Undefended Baselines
::
:: CRITICAL: Run this before analysing Phase 2 results in the paper.
::
:: Problem we are solving:
::   Phase 2 defense checkpoints are 30-epoch trained.
::   The Phase 1 model completion baselines (94.99% CIFAR10,
::   43.35% CIFAR100) came from LONGER training (100-150 epochs).
::   Comparing a 30-epoch defended checkpoint against a 100-epoch
::   undefended one is unfair — training duration is a confound.
::
:: This bat runs Stage 2 model completion on the 30-epoch
:: UNDEFENDED active checkpoints that already exist from the
:: Phase 1 characterization runs (run_phase1_characterization.bat).
:: Those checkpoints are named:
::   CIFAR10_saved_framework_lr=0.1_mal_half=16.pth    (30 ep, active, no defense)
::   CIFAR100_saved_framework_lr=0.1_mal_half=16.pth   (30 ep, active, no defense)
::
:: After this bat completes you will have the FAIR comparison:
::
::   CIFAR10 (all 30 epochs):
::     Active, no defense  [this bat]  vs
::     Active + defense    [Phase 2]   vs
::     Benign              [Phase 2 benign+defense ≈ benign]
::
::   CIFAR100 (all 30 epochs):
::     Same structure.
::
:: The 30-epoch baselines replace the 94.99% / 43.35% numbers
:: in the main result table. They will be lower than those
:: (less training = less convergence) but the comparison is honest.
:: ============================================================

set RESUMEDIR=.\saved_experiment_results\saved_models\
set CIFAR10PATH=.\data\CIFAR10
set CIFAR100PATH=.\data\CIFAR100

echo.
echo ============================================================
echo  CIFAR10 — 30-Epoch Active Baseline (no defense)
echo ============================================================
echo  Loads: CIFAR10_saved_framework_lr=0.1_mal_half=16.pth
echo  (produced by Phase 1 characterization, 30 epochs, active Party A)
echo.
python model_completion.py ^
  --dataset-name CIFAR10 ^
  --dataset-path %CIFAR10PATH% ^
  --n-labeled 40 ^
  --party-num 2 ^
  --half 16 ^
  --k 4 ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_mal_half=16.pth ^
  --num-layer 1 ^
  --activation_func_type ReLU ^
  --use-bn True ^
  --epochs 25 ^
  --print-to-txt 1

echo.
echo ============================================================
echo  CIFAR10 — 30-Epoch Benign Baseline (no defense)
echo ============================================================
echo  Loads: CIFAR10_saved_framework_lr=0.1_normal_half=16.pth
echo  (produced by Phase 1 characterization, 30 epochs, benign)
echo.
python model_completion.py ^
  --dataset-name CIFAR10 ^
  --dataset-path %CIFAR10PATH% ^
  --n-labeled 40 ^
  --party-num 2 ^
  --half 16 ^
  --k 4 ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_normal_half=16.pth ^
  --num-layer 1 ^
  --activation_func_type ReLU ^
  --use-bn True ^
  --epochs 25 ^
  --print-to-txt 1

echo.
echo ============================================================
echo  CIFAR100 — 30-Epoch Active Baseline (no defense)
echo ============================================================
echo  Loads: CIFAR100_saved_framework_lr=0.1_mal_half=16.pth
echo  (produced by Phase 1 characterization, 30 epochs, active Party A)
echo.
python model_completion.py ^
  --dataset-name CIFAR100 ^
  --dataset-path %CIFAR100PATH% ^
  --n-labeled 400 ^
  --party-num 2 ^
  --half 16 ^
  --k 5 ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_half=16.pth ^
  --num-layer 1 ^
  --activation_func_type ReLU ^
  --use-bn True ^
  --epochs 25 ^
  --print-to-txt 1

echo.
echo ============================================================
echo  CIFAR100 — 30-Epoch Benign Baseline (no defense)
echo ============================================================
echo  Loads: CIFAR100_saved_framework_lr=0.1_normal_half=16.pth
echo  (produced by Phase 1 characterization, 30 epochs, benign)
echo.
python model_completion.py ^
  --dataset-name CIFAR100 ^
  --dataset-path %CIFAR100PATH% ^
  --n-labeled 400 ^
  --party-num 2 ^
  --half 16 ^
  --k 5 ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_normal_half=16.pth ^
  --num-layer 1 ^
  --activation_func_type ReLU ^
  --use-bn True ^
  --epochs 25 ^
  --print-to-txt 1

echo.
echo ============================================================
echo  Phase 2b COMPLETE
echo.
echo  Now you have the fair 3-way comparison (all 30 epochs):
echo.
echo  CIFAR10:
echo    Benign (no defense)  : normal_half=16.pth    [this bat]
echo    Active (no defense)  : mal_half=16.pth        [this bat]
echo    Active + defense     : mal_asym_def...pth     [Phase 2]
echo.
echo  CIFAR100:
echo    Same 3 conditions.
echo.
echo  The CORRECT result table for the paper uses only these
echo  30-epoch-consistent numbers. The 94.99%% / 43.35%% numbers
echo  from the original model_completion.bat came from longer
echo  training and should NOT be used as the primary comparison.
echo ============================================================
echo.
pause
