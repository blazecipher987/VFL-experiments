@echo off
:: ============================================================
:: Phase 15 -- CINIC10L 3-Seed Reproducibility Sweep
::
:: WHY THIS EXISTS:
::   EXP-022 (Phase 10) showed the defense succeeds on CINIC10L:
::     Benign=65.70%, Attack=86.59%, Defended=62.43% (seed 0).
::   That is a single-seed result. No paper claim can be made on a
::   single seed. This bat runs the same three conditions at seeds
::   {42, 123, 456} to produce mean +/- std across 4 total seeds
::   (including seed 0 from Phase 10).
::
::   Combined with Phase 10 (seed 0), the paper can report:
::     mean +/- std across 4 seeds for:
::       benign MC, attack MC, defended MC, VFL utility cost
::
::   SUCCESS CRITERION: defended mean < benign mean across all seeds
::   (replicating the 62.43% < 65.70% result from seed 0).
::
:: WHAT RUNS (per seed, 3 seeds total):
::   Stage 1: benign / active / active+defense at 100 epochs
::   Stage 2: model_completion on each checkpoint
::   Each checkpoint renamed with _seedNN suffix immediately after
::   Stage 1 to prevent overwrite between seeds.
::
:: SEED 0 REFERENCE (Phase 10, already done):
::   Benign MC:   65.70%  (model_completion_CINIC10L_..._normal_half=16.pth...)
::   Attack MC:   86.59%  (model_completion_CINIC10L_..._mal_half=16.pth...)
::   Defended MC: 62.43%  (model_completion_CINIC10L_..._mal_asym_def-a=1.0-t=0.1-b=8_half=16.pth...)
::
:: NAMING CONVENTION:
::   Stage 1 checkpoints renamed to:
::     *_normal_half=16_seed{N}.pth        (benign)
::     *_mal_half=16_seed{N}.pth           (attack)
::     *_mal_asym_def-..._ep100_seed{N}.pth (defense)
::
:: RUNTIME ESTIMATE:
::   CINIC10L train split: 90,000 images (~1.8x CIFAR-10 per epoch).
::   Stage 1: ~1.8x CIFAR-10 per seed = ~5.4 hrs per seed on a single GPU.
::   Total sequential for 3 seeds: ~16 hrs.
::   Stage 2: ~20 min per checkpoint = ~3 hrs for all 9.
:: ============================================================

set DATASET=CINIC10L
set DATAPATH=.\data\CINIC10L
set EPOCHS=100
set HALF=16
set K=4

set TAU=0.10
set ALPHA=1.0
set BURNIN=8

set MODELSDIR=.\saved_experiment_results\saved_models\CINIC10L_saved_models
set RESUMEDIR=.\saved_experiment_results\saved_models\

echo.
echo ============================================================
echo  PHASE 15 -- CINIC10L Seed Sweep: seeds 42, 123, 456
echo  Seed 0 already done in Phase 10 (EXP-020/021/022)
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
move /Y "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_normal_half=16.pth" ^
         "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_normal_half=16_seed42.pth"
move /Y "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_normal_half=16.txt" ^
         "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_normal_half=16_seed42.txt"
echo.

echo ===== [S42-2/3] Active no defense 100ep, seed=42 =====
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --manual-seed 42
move /Y "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_mal_half=16.pth" ^
         "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_mal_half=16_seed42.pth"
move /Y "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_mal_half=16.txt" ^
         "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_mal_half=16_seed42.txt"
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
move /Y "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16.pth" ^
         "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16_ep100_seed42.pth"
move /Y "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16.txt" ^
         "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16_ep100_seed42.txt"
echo.

echo ===== [S42 Stage 2] Model completion, seed=42 =====
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 40 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CINIC10L_saved_framework_lr=0.1_normal_half=16_seed42.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed 42
echo.
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 40 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CINIC10L_saved_framework_lr=0.1_mal_half=16_seed42.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed 42
echo.
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 40 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CINIC10L_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16_ep100_seed42.pth ^
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
move /Y "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_normal_half=16.pth" ^
         "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_normal_half=16_seed123.pth"
move /Y "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_normal_half=16.txt" ^
         "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_normal_half=16_seed123.txt"
echo.

echo ===== [S123-2/3] Active no defense 100ep, seed=123 =====
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --manual-seed 123
move /Y "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_mal_half=16.pth" ^
         "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_mal_half=16_seed123.pth"
move /Y "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_mal_half=16.txt" ^
         "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_mal_half=16_seed123.txt"
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
move /Y "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16.pth" ^
         "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16_ep100_seed123.pth"
move /Y "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16.txt" ^
         "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16_ep100_seed123.txt"
echo.

echo ===== [S123 Stage 2] Model completion, seed=123 =====
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 40 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CINIC10L_saved_framework_lr=0.1_normal_half=16_seed123.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed 123
echo.
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 40 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CINIC10L_saved_framework_lr=0.1_mal_half=16_seed123.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed 123
echo.
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 40 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CINIC10L_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16_ep100_seed123.pth ^
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
move /Y "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_normal_half=16.pth" ^
         "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_normal_half=16_seed456.pth"
move /Y "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_normal_half=16.txt" ^
         "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_normal_half=16_seed456.txt"
echo.

echo ===== [S456-2/3] Active no defense 100ep, seed=456 =====
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --manual-seed 456
move /Y "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_mal_half=16.pth" ^
         "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_mal_half=16_seed456.pth"
move /Y "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_mal_half=16.txt" ^
         "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_mal_half=16_seed456.txt"
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
move /Y "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16.pth" ^
         "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16_ep100_seed456.pth"
move /Y "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16.txt" ^
         "%MODELSDIR%\CINIC10L_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16_ep100_seed456.txt"
echo.

echo ===== [S456 Stage 2] Model completion, seed=456 =====
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 40 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CINIC10L_saved_framework_lr=0.1_normal_half=16_seed456.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed 456
echo.
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 40 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CINIC10L_saved_framework_lr=0.1_mal_half=16_seed456.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed 456
echo.
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 40 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CINIC10L_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16_ep100_seed456.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed 456
echo.
echo SEED 456 COMPLETE.
echo.

echo ============================================================
echo  Phase 15 COMPLETE -- All 3 new seeds done.
echo.
echo  Seed 0 reference (Phase 10, EXP-020/021/022):
echo    normal_half=16.pth         -> MC benign:   65.70%%
echo    mal_half=16.pth            -> MC attack:   86.59%%
echo    mal_asym_def-..._half=16   -> MC defended: 62.43%%
echo.
echo  New results in CINIC10L_saved_models/:
echo    model_completion_*_seed42.pth..._nlabeled=40.txt
echo    model_completion_*_seed123.pth..._nlabeled=40.txt
echo    model_completion_*_seed456.pth..._nlabeled=40.txt
echo.
echo  Report: mean +/- std across 4 seeds {0, 42, 123, 456} for:
echo    benign MC, attack MC, defended MC, VFL test accuracy
echo.
echo  PAPER CLAIM holds if defended mean < benign mean across
echo  all 4 seeds (replicating seed-0 result: 62.43%% < 65.70%%).
echo ============================================================
echo.
pause
