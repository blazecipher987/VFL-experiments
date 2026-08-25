@echo off
:: ============================================================
:: Phase 6B — CIFAR10 100-Epoch Seed Sweep (Reproducibility)
::
:: WHY THIS EXISTS:
::   EXP-011 (Phase 4) ran the three-way CIFAR10 comparison at
::   100 epochs using manualSeed=0 only. A single run provides
::   no variance estimate. Before submitting a paper, results must
::   hold across multiple seeds. This bat runs the same three
::   conditions at seeds {42, 123, 456}.
::
::   Combined with seed 0 (EXP-011), the paper can report:
::     mean +/- std across 4 seeds for:
::       benign ASR, attack ASR, defended ASR, VFL cost
::
:: PREREQUISITE:
::   vfl_framework.py must support --manual-seed (added in Phase 6B
::   code change). model_completion.py already supported --manualSeed.
::
:: WHAT RUNS (per seed, 3 seeds total):
::   Stage 1: benign / active / active+defense at 100 epochs
::   Stage 2: model_completion on each
::   Each checkpoint renamed with _seed{N} suffix immediately after
::   Stage 1 training to prevent overwriting between seeds.
::
:: NAMING:
::   normal_half=16_seed{N}.pth      (benign)
::   mal_half=16_seed{N}.pth         (attack)
::   mal_asym_def-..._ep100_seed{N}.pth  (defense)
::
:: PARALLELISM:
::   Seeds are independent. To run in parallel on multiple machines,
::   split into three terminal windows, each running one seed block.
::   On a single GPU, run sequentially (full file).
::
:: RUNTIME ESTIMATE:
::   ~3 hrs per seed (Stage 1 only) on a single GPU.
::   Total sequential: ~9 hrs for all three seeds.
:: ============================================================

set DATASET=CIFAR10
set DATAPATH=.\data\CIFAR10
set EPOCHS=100
set HALF=16
set K=4

set TAU=0.10
set ALPHA=1.0
set BURNIN=8

set MODELSDIR=.\saved_experiment_results\saved_models\CIFAR10_saved_models
set RESUMEDIR=.\saved_experiment_results\saved_models\

echo.
echo ============================================================
echo  PHASE 6B -- CIFAR10 100-Epoch Seed Sweep: seeds 42, 123, 456
echo ============================================================
echo.

:: ============================================================
::  SEED 42
:: ============================================================
echo ===================================================
echo  SEED 42
echo ===================================================
echo.

echo ===== [S42-1/3] Benign 100ep, seed=42 =====
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False ^
  --manual-seed 42
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_normal_half=16.pth" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_normal_half=16_seed42.pth"
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_normal_half=16.txt" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_normal_half=16_seed42.txt"
echo.

echo ===== [S42-2/3] Active no defense 100ep, seed=42 =====
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --manual-seed 42
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_half=16.pth" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_half=16_seed42.pth"
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_half=16.txt" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_half=16_seed42.txt"
echo.

echo ===== [S42-3/3] Active + Defense 100ep, seed=42 =====
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --asymmetric-defense True ^
  --asymmetric-alpha %ALPHA% ^
  --asymmetric-tau %TAU% ^
  --asymmetric-burn-in %BURNIN% ^
  --if-cluster-outputsA True ^
  --manual-seed 42
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16.pth" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16_ep100_seed42.pth"
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16.txt" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16_ep100_seed42.txt"
echo.

echo ===== [S42 Stage 2] Model completion, seed=42 =====
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 40 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_normal_half=16_seed42.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed 42
echo.
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 40 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_mal_half=16_seed42.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed 42
echo.
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 40 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16_ep100_seed42.pth ^
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

echo ===== [S123-1/3] Benign 100ep, seed=123 =====
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False ^
  --manual-seed 123
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_normal_half=16.pth" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_normal_half=16_seed123.pth"
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_normal_half=16.txt" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_normal_half=16_seed123.txt"
echo.

echo ===== [S123-2/3] Active no defense 100ep, seed=123 =====
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --manual-seed 123
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_half=16.pth" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_half=16_seed123.pth"
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_half=16.txt" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_half=16_seed123.txt"
echo.

echo ===== [S123-3/3] Active + Defense 100ep, seed=123 =====
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --asymmetric-defense True ^
  --asymmetric-alpha %ALPHA% ^
  --asymmetric-tau %TAU% ^
  --asymmetric-burn-in %BURNIN% ^
  --if-cluster-outputsA True ^
  --manual-seed 123
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16.pth" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16_ep100_seed123.pth"
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16.txt" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16_ep100_seed123.txt"
echo.

echo ===== [S123 Stage 2] Model completion, seed=123 =====
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 40 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_normal_half=16_seed123.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed 123
echo.
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 40 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_mal_half=16_seed123.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed 123
echo.
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 40 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16_ep100_seed123.pth ^
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

echo ===== [S456-1/3] Benign 100ep, seed=456 =====
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False ^
  --manual-seed 456
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_normal_half=16.pth" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_normal_half=16_seed456.pth"
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_normal_half=16.txt" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_normal_half=16_seed456.txt"
echo.

echo ===== [S456-2/3] Active no defense 100ep, seed=456 =====
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --manual-seed 456
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_half=16.pth" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_half=16_seed456.pth"
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_half=16.txt" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_half=16_seed456.txt"
echo.

echo ===== [S456-3/3] Active + Defense 100ep, seed=456 =====
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --asymmetric-defense True ^
  --asymmetric-alpha %ALPHA% ^
  --asymmetric-tau %TAU% ^
  --asymmetric-burn-in %BURNIN% ^
  --if-cluster-outputsA True ^
  --manual-seed 456
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16.pth" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16_ep100_seed456.pth"
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16.txt" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16_ep100_seed456.txt"
echo.

echo ===== [S456 Stage 2] Model completion, seed=456 =====
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 40 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_normal_half=16_seed456.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed 456
echo.
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 40 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_mal_half=16_seed456.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed 456
echo.
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 40 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16_ep100_seed456.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed 456
echo.
echo SEED 456 COMPLETE.
echo.

echo ============================================================
echo  Phase 6B COMPLETE -- All 3 seeds done.
echo.
echo  Seed 0 results (EXP-011) already exist as:
echo    normal_half=16.pth / mal_half=16.pth / _ep100.pth
echo.
echo  New seed results in CIFAR10_saved_models/:
echo    *_seed42.pth / *_seed123.pth / *_seed456.pth
echo    model_completion_*_seed42.pth_..._nlabeled=40.txt
echo    model_completion_*_seed123.pth_..._nlabeled=40.txt
echo    model_completion_*_seed456.pth_..._nlabeled=40.txt
echo.
echo  Report: mean +/- std across seeds {0, 42, 123, 456} for
echo    benign ASR, attack ASR, defended ASR, VFL cost.
echo ============================================================
echo.
pause
