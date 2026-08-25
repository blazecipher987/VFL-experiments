@echo off
:: ============================================================
:: Phase 14b — z_a Embedding Corruption, CIFAR-100, noise_std=1.0, 150 epochs
::
:: WHY THIS BAT EXISTS:
::   Companion to Phase 14a (noise_std=0.5).  Tests a stronger corruption
::   level.  Run in parallel on Instance 3 while Phase 14a runs on Instance 2.
::
::   noise_std=1.0 means: when scale=0.6 (typical for CIFAR-100 with alpha=1.0),
::     noise amplitude = 1.0 * (1 - 0.6) * randn = 0.4 * randn
::   At full suppression (scale=0.0):
::     noise amplitude = 1.0 * 1.0 * randn = 1.0 * randn
::
::   Tradeoff: higher noise_std disrupts MaliciousSGD more strongly but may
::   also interfere with the top model's ability to use Party B's embeddings
::   (since the top model trains on noisy Party A contribution, it may not
::   learn to split attention cleanly across both parties).
::   This effect is less severe than Phase 9's gradient noise because Party B's
::   gradient is NEVER touched here.
::
:: CHECKPOINT NAMING:
::   vfl_framework.py saves:
::     CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-za=1.0_half=16.pth
::   Renamed to _ep150.pth for consistency.
:: ============================================================

set DATASET=CIFAR100
set DATAPATH=.\data\CIFAR100
set EPOCHS=150
set HALF=16
set K=5

set TAU=0.10
set ALPHA=1.0
set BURNIN=8
set ZA_STD=1.0

set MODELSDIR=.\saved_experiment_results\saved_models\CIFAR100_saved_models
set RESUMEDIR=.\saved_experiment_results\saved_models\

echo.
echo ============================================================
echo  PHASE 14b -- CIFAR-100 z_a Corruption (noise_std=1.0), 150 Epochs
echo ============================================================
echo.

:: ----------------------------------------------------------
:: STAGE 1: Attack + z_a corruption (noise_std=1.0), 150 epochs
:: ----------------------------------------------------------
echo ===== [1/2] Stage 1 -- Attack + z_a corruption (std=1.0), 150 epochs =====
echo Saves to: mal_asym_def-a=1.0-t=0.1-b=8-za=1.0_half=16.pth
echo Renames to: mal_asym_def-a=1.0-t=0.1-b=8-za=1.0_half=16_ep150.pth
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
  --asymmetric-za-noise-std %ZA_STD% ^
  --if-cluster-outputsA True

echo.
echo Renaming z_a checkpoint to _ep150 variant...
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-za=1.0_half=16.pth" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-za=1.0_half=16_ep150.pth"
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-za=1.0_half=16.txt" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-za=1.0_half=16_ep150.txt"
echo [1/2] Stage 1 done.
echo.

echo ============================================================
echo  Stage 1 complete. Starting Stage 2 model completion...
echo ============================================================
echo.

:: ----------------------------------------------------------
:: STAGE 2: Model completion on z_a=1.0 CIFAR-100 checkpoint
:: ----------------------------------------------------------
echo ===== [2/2] Stage 2 -- Model completion on z_a=1.0 checkpoint =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% ^
  --dataset-path %DATAPATH% ^
  --n-labeled 400 ^
  --party-num 2 ^
  --half %HALF% ^
  --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-za=1.0_half=16_ep150.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.
echo ============================================================
echo  Phase 14b COMPLETE -- Results in:
echo  %MODELSDIR%\
echo.
echo  model_completion_..._mal_asym_def-a=1.0-t=0.1-b=8-za=1.0_half=16_ep150.pth_..._nlabeled=400.txt
echo.
echo  BASELINES:
echo    Benign (seed-0):          30.33%% (EXP-012)
echo    No defense:               47.86%% (EXP-014)
echo    Standard defense:         43.12%% (EXP-015)
echo    Gradient noise best n=2.0: 43.10%% (EXP-019)
echo.
echo  Compare with Phase 14a (z_a std=0.5):
echo    If 14b < 14a: higher noise is better; try std=2.0 next
echo    If 14b > 14a: 0.5 is better; sweet spot is below 1.0
echo    If both > 43.10%%: z_a corruption also insufficient at this alpha/tau
echo ============================================================
echo.
pause
