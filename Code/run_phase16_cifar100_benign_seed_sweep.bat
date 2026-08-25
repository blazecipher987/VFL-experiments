@echo off
:: ============================================================
:: Phase 16 -- CIFAR100 Benign Baseline 3-Seed Sweep
::
:: WHY THIS EXISTS:
::   The CIFAR100 benign baseline is currently a single run at
::   seed 0: 30.33% model completion accuracy (from Phase 5,
::   EXP-012). All 8 CIFAR100 defense configurations are compared
::   against this single-seed reference. Before any CIFAR100 claim
::   can be made (even a negative one -- "defense fails for CIFAR100"),
::   the benign baseline needs a variance estimate.
::
::   If the 30.33% seed-0 value is an outlier, the failure gap
::   (best defended 43.10% vs benign 30.33% = 12.77pp above benign)
::   changes. Three additional seeds give us mean +/- std for the
::   benign reference so any claim is properly calibrated.
::
::   This is BENIGN ONLY -- no attack, no defense. Fastest way to
::   get the variance estimate without running full three-way sweeps.
::
:: WHAT RUNS (per seed, 3 seeds):
::   Stage 1: Benign VFL training at 150 epochs
::   Stage 2: Model completion from the benign checkpoint (n=400)
::   Each checkpoint renamed with _seed{N} suffix.
::
:: PARAMETERS:
::   EPOCHS    : 150  (same as Phase 5 -- full convergence for CIFAR100)
::   HALF      : 16   (Party A gets left 16 pixels)
::   K         : 5    (top-5 for 100-class problem)
::   n_labeled : 400  (4 per class x 100 classes)
::
:: SEED 0 REFERENCE (EXP-012, Phase 5):
::   VFL Train=99.97%, Test=52.33%
::   Benign MC Best=30.33%
::
:: RUNTIME ESTIMATE:
::   ~5-6 hrs per seed on a single GPU (CIFAR100, 150 epochs).
::   Total sequential: ~15-18 hrs for all 3 seeds.
::   Combined with seed 0: report mean +/- std across {0, 42, 123, 456}.
:: ============================================================

set DATASET=CIFAR100
set DATAPATH=.\data\CIFAR100
set EPOCHS=150
set HALF=16
set K=5

set MODELSDIR=.\saved_experiment_results\saved_models\CIFAR100_saved_models
set RESUMEDIR=.\saved_experiment_results\saved_models\

echo.
echo ============================================================
echo  PHASE 16 -- CIFAR100 Benign Baseline Seed Sweep: 42, 123, 456
echo  (Seed 0 reference: 30.33%% benign MC from EXP-012)
echo ============================================================
echo.

:: ============================================================
::  SEED 42
:: ============================================================
echo ===================================================
echo  SEED 42
echo ===================================================
echo.

echo ===== [S42-1/2] Stage 1 -- Benign 150ep, seed=42 =====
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False ^
  --manual-seed 42
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_normal_half=16.pth" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_normal_half=16_seed42.pth"
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_normal_half=16.txt" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_normal_half=16_seed42.txt"
echo.

echo ===== [S42-2/2] Stage 2 -- Benign MC, seed=42 =====
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 400 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_normal_half=16_seed42.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed 42
echo.
echo SEED 42 COMPLETE.
echo.

:: ============================================================
::  SEED 123
:: ============================================================
echo ===================================================
echo  SEED 123
echo ===================================================
echo.

echo ===== [S123-1/2] Stage 1 -- Benign 150ep, seed=123 =====
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False ^
  --manual-seed 123
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_normal_half=16.pth" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_normal_half=16_seed123.pth"
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_normal_half=16.txt" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_normal_half=16_seed123.txt"
echo.

echo ===== [S123-2/2] Stage 2 -- Benign MC, seed=123 =====
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 400 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_normal_half=16_seed123.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed 123
echo.
echo SEED 123 COMPLETE.
echo.

:: ============================================================
::  SEED 456
:: ============================================================
echo ===================================================
echo  SEED 456
echo ===================================================
echo.

echo ===== [S456-1/2] Stage 1 -- Benign 150ep, seed=456 =====
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False ^
  --manual-seed 456
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_normal_half=16.pth" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_normal_half=16_seed456.pth"
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_normal_half=16.txt" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_normal_half=16_seed456.txt"
echo.

echo ===== [S456-2/2] Stage 2 -- Benign MC, seed=456 =====
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 400 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_normal_half=16_seed456.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed 456
echo.
echo SEED 456 COMPLETE.
echo.

echo ============================================================
echo  Phase 16 COMPLETE -- CIFAR100 Benign Seed Sweep Done.
echo.
echo  Seed 0 reference (EXP-012): 30.33%%
echo.
echo  New results in CIFAR100_saved_models/:
echo    normal_half=16_seed42.pth  / model_completion_*_seed42.pth_*_nlabeled=400.txt
echo    normal_half=16_seed123.pth / model_completion_*_seed123.pth_*_nlabeled=400.txt
echo    normal_half=16_seed456.pth / model_completion_*_seed456.pth_*_nlabeled=400.txt
echo.
echo  Report: mean +/- std across seeds {0, 42, 123, 456} for benign MC.
echo  If the mean deviates significantly from 30.33%%, revisit all
echo  CIFAR100 defense comparisons (EXP-012 through EXP-019, EXP-024/025).
echo ============================================================
echo.
pause
