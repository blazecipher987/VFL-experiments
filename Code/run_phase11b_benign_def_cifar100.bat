@echo off
:: ============================================================
:: Phase 11b -- Benign + Defense (CIFAR-100, 150 epochs)
::
:: PURPOSE:
::   Same asymmetry check as Phase 11a but for CIFAR-100.
::   With Option B defense (Phase 9), the defense adds noise
::   proportional to (1-scale). For benign training, scale=1.0
::   (divergence stays below tau), so noise term = 0 regardless
::   of noise_std. Result should match benign baseline.
::
::   Fills the 2x2 table for CIFAR-100 + Option B:
::     |           | Defense OFF | Defense ON  |
::     | Benign    | 30.33%%     | ~30.33%%    | <- this run
::     | Attack    | 47.86%%     | Phase 9     |
::
:: REFERENCE (30ep EXP-008):
::   Benign no defense: 12.76%% (30ep -- different epoch count)
::   Benign + defense:  11.27%% (-1.49pp -- defense nearly silent)
::   Full 150ep benign (EXP-010): 30.33%%
::
:: CHECKPOINT NAME (from vfl_framework.py naming logic):
::   normal (no mal-optim) + asym_def flags, no -n= suffix
::   -> CIFAR100_saved_framework_lr=0.1_normal_asym_def-a=1.0-t=0.1-b=8_half=16.pth
::   Rename to _ep150 for consistency with Phase 9 naming.
::
:: SUCCESS CRITERION:
::   MC(benign+defense) within 3pp of 30.33%% (benign baseline).
::
:: PARAMETERS (same as Phase 5 / Phase 9 for direct comparability):
::   EPOCHS   : 150
::   K        : 5
::   HALF     : 16
::   n_labeled: 400
::   TAU      : 0.10
::   ALPHA    : 1.0
::   BURNIN   : 8
:: ============================================================

set DATASET=CIFAR100
set DATAPATH=.\data\CIFAR100
set EPOCHS=150
set HALF=16
set K=5

set TAU=0.10
set ALPHA=1.0
set BURNIN=8

set MODELSDIR=.\saved_experiment_results\saved_models\CIFAR100_saved_models
set RESUMEDIR=.\saved_experiment_results\saved_models\

echo.
echo ============================================================
echo  PHASE 11b -- Benign + Defense (CIFAR-100, 150 epochs)
echo  alpha=%ALPHA%, tau=%TAU%, burn_in=%BURNIN%
echo ============================================================
echo.

:: ----------------------------------------------------------
:: STAGE 1: Benign training with defense enabled
:: Defense dormant expected: divergence stays below tau
:: (CIFAR-100 benign divergence goes negative after burn-in)
:: ----------------------------------------------------------
echo ===== [1/2] Stage 1 -- Benign + Defense, 150 epochs =====
echo Saves to: CIFAR100_saved_framework_lr=0.1_normal_asym_def-a=1.0-t=0.1-b=8_half=16.pth
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --asymmetric-defense True ^
  --asymmetric-alpha %ALPHA% ^
  --asymmetric-tau %TAU% ^
  --asymmetric-burn-in %BURNIN%

echo.
echo Renaming to _ep150 variant for consistency with Phase 9 naming...
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_normal_asym_def-a=1.0-t=0.1-b=8_half=16.pth" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_normal_asym_def-a=1.0-t=0.1-b=8_half=16_ep150.pth"
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_normal_asym_def-a=1.0-t=0.1-b=8_half=16.txt" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_normal_asym_def-a=1.0-t=0.1-b=8_half=16_ep150.txt"
echo [1/2] Stage 1 done.
echo.

:: ----------------------------------------------------------
:: STAGE 2: Model completion on benign+defense checkpoint
:: ----------------------------------------------------------
echo ===== [2/2] Stage 2 -- Model completion =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% ^
  --dataset-path %DATAPATH% ^
  --n-labeled 400 ^
  --party-num 2 ^
  --half %HALF% ^
  --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_normal_asym_def-a=1.0-t=0.1-b=8_half=16_ep150.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.
echo [2/2] done.
echo.

echo ============================================================
echo  Phase 11b COMPLETE
echo  Result: %MODELSDIR%\
echo    model_completion_...normal_asym_def-a=1.0-t=0.1-b=8..._nlabeled=400.txt
echo.
echo  Compare against EXP-010 benign baseline: 30.33%%
echo  Gap should be less than 3pp to confirm defense asymmetry.
echo ============================================================
echo.
pause
