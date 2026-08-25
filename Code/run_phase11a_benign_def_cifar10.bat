@echo off
:: ============================================================
:: Phase 11a -- Benign + Defense (CIFAR-10, 100 epochs)
::
:: PURPOSE:
::   Verify the defense is transparent to honest participants.
::   The defense gates on Fisher divergence > tau=0.10. During
::   benign training, divergence stays below tau (confirmed at
::   30ep in EXP-007, delta=-0.80pp). This run confirms the
::   same holds at full 100 epochs.
::
::   Fills the 2x2 table:
::     |           | Defense OFF | Defense ON  |
::     | Benign    | ~83-87%%    | ~83-87%%    | <- this run
::     | Attack    | ~94-95%%    | ~80-84%%    | <- Phase 6B
::
:: REFERENCE (30ep EXP-007):
::   Benign no defense: 47.98%%
::   Benign + defense:  47.18%% (-0.80pp -- defense nearly silent)
::
:: CHECKPOINT NAME (from vfl_framework.py naming logic):
::   normal (no mal-optim) + asym_def flags
::   -> CIFAR10_saved_framework_lr=0.1_normal_asym_def-a=1.0-t=0.1-b=8_half=16.pth
::
:: SUCCESS CRITERION:
::   MC(benign+defense) within 3pp of MC(benign no defense).
::   This proves defense asymmetry: dormant on honest gradients.
::
:: PARAMETERS (same as Phase 4 / Phase 6B):
::   EPOCHS   : 100
::   K        : 4
::   HALF     : 16
::   n_labeled: 40
::   TAU      : 0.10
::   ALPHA    : 1.0
::   BURNIN   : 8
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
echo  PHASE 11a -- Benign + Defense (CIFAR-10, 100 epochs)
echo  alpha=%ALPHA%, tau=%TAU%, burn_in=%BURNIN%
echo ============================================================
echo.

:: ----------------------------------------------------------
:: STAGE 1: Benign training with defense enabled
:: Defense dormant expected: scale=1.0 throughout (div < tau)
:: ----------------------------------------------------------
echo ===== [1/2] Stage 1 -- Benign + Defense, 100 epochs =====
echo Saves to: CIFAR10_saved_framework_lr=0.1_normal_asym_def-a=1.0-t=0.1-b=8_half=16.pth
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
  --n-labeled 40 ^
  --party-num 2 ^
  --half %HALF% ^
  --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_normal_asym_def-a=1.0-t=0.1-b=8_half=16.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.
echo [2/2] done.
echo.

echo ============================================================
echo  Phase 11a COMPLETE
echo  Result: %MODELSDIR%\
echo    model_completion_...normal_asym_def-a=1.0-t=0.1-b=8..._nlabeled=40.txt
echo.
echo  Compare against Phase 6B benign baseline (~83-87%%)
echo  Gap should be less than 3pp to confirm defense asymmetry.
echo ============================================================
echo.
pause
