@echo off
:: ============================================================
:: Phase 10 — CINIC10L: New Dataset Generalization Test
::
:: WHY THIS BAT EXISTS:
::   CINIC10L is a 10-class image dataset (32x32 RGB) that mixes
::   original CIFAR-10 images with visually similar ImageNet images,
::   making classification harder than CIFAR-10 but preserving the
::   same 10-class structure. The defense relies on semantic
::   misalignment via Fisher divergence detection, which works on
::   10-class problems. CINIC10L tests whether the defense
::   generalizes across datasets of the same class count but
::   different difficulty.
::
:: DATASET:
::   Place the CINIC10L dataset at .\data\CINIC10L\ with structure:
::     .\data\CINIC10L\train\{class_name}\*.png   (90,000 images)
::     .\data\CINIC10L\valid\{class_name}\*.png   (90,000 images)
::     .\data\CINIC10L\test\{class_name}\*.png    (90,000 images)
::   Classes (lowercase): airplane, automobile, bird, cat, deer,
::                         dog, frog, horse, ship, truck
::   Download: https://datashare.is.ed.ac.uk/handle/10283/3192
::
:: NO CODE CHANGES REQUIRED: CINIC10L is fully implemented in
::   datasets/cinic10.py, models/model_sets.py, vfl_framework.py,
::   and model_completion.py. This bat is the only new artifact.
::
:: WHAT RUNS (Stage 1 + Stage 2 for 3 conditions):
::   [1] Benign, 100 epochs            -> normal_half=16.pth
::   [2] Active (no defense), 100 ep   -> mal_half=16.pth
::   [3] Active + defense, 100 ep      -> mal_asym_def-..._half=16.pth
::   [4] Stage 2 benign
::   [5] Stage 2 attack
::   [6] Stage 2 defended (KEY RESULT)
::
:: PARAMETERS (same as CIFAR10 Phase 4 for direct comparability):
::   EPOCHS    : 100
::   HALF      : 16 (Party A gets left 16 pixels, Party B gets right 16)
::   K         : 4  (top-k, 10 classes)
::   n_labeled : 40 (4 labeled per class, same as CIFAR10)
::   TAU       : 0.10
::   ALPHA     : 1.0
::   BURNIN    : 8  (conservative, consistent with other experiments)
::
:: EXPECTED RESULTS (based on CIFAR10 Phase 4):
::   Benign MC   : lower than CIFAR10 (harder dataset) — estimate 70-85%
::   Attack MC   : lower than CIFAR10 (harder) — estimate 75-90%
::   Defended MC : should be below benign — estimate 5-15pp below benign
::
:: SUCCESS CRITERION: Defended MC < Benign MC
::   (same threshold as CIFAR10: defense brings MC below benign baseline)
::
:: TRAINING TIME NOTE:
::   CINIC10L train split has 90,000 images vs CIFAR10's 50,000.
::   Each epoch takes ~1.8x longer. 100 epochs ~ 1.8x CIFAR10 Phase 4.
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
echo  PHASE 10 -- CINIC10L Training + Model Completion
echo  Dataset: CINIC10L (10-class, 32x32, harder than CIFAR10)
echo ============================================================
echo.

:: ----------------------------------------------------------
:: STAGE 1, CONDITION 1: Benign, 100 epochs
:: ----------------------------------------------------------
echo ===== [1/3] Stage 1 -- Benign, 100 epochs =====
echo Saves to: CINIC10L_saved_framework_lr=0.1_normal_half=16.pth
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False

echo.
echo [1/3] done.
echo.

:: ----------------------------------------------------------
:: STAGE 1, CONDITION 2: Active attack, no defense, 100 epochs
:: ----------------------------------------------------------
echo ===== [2/3] Stage 1 -- Active (no defense), 100 epochs =====
echo Saves to: CINIC10L_saved_framework_lr=0.1_mal_half=16.pth
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True

echo.
echo [2/3] done.
echo.

:: ----------------------------------------------------------
:: STAGE 1, CONDITION 3: Active attack + defense, 100 epochs
:: ----------------------------------------------------------
echo ===== [3/3] Stage 1 -- Active + Defense, 100 epochs =====
echo Saves to: CINIC10L_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16.pth
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
echo [3/3] Stage 1 done.
echo.

echo ============================================================
echo  Stage 1 complete. Starting Stage 2 model completion...
echo ============================================================
echo.

:: ----------------------------------------------------------
:: STAGE 2, CONDITION 4: Benign -- model completion
:: ----------------------------------------------------------
echo ===== [4/6] Stage 2 -- Benign =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% ^
  --dataset-path %DATAPATH% ^
  --n-labeled 40 ^
  --party-num 2 ^
  --half %HALF% ^
  --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CINIC10L_saved_framework_lr=0.1_normal_half=16.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.
echo [4/6] done.
echo.

:: ----------------------------------------------------------
:: STAGE 2, CONDITION 5: Active (no defense) -- model completion
:: ----------------------------------------------------------
echo ===== [5/6] Stage 2 -- Active no defense (ATTACK BASELINE) =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% ^
  --dataset-path %DATAPATH% ^
  --n-labeled 40 ^
  --party-num 2 ^
  --half %HALF% ^
  --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CINIC10L_saved_framework_lr=0.1_mal_half=16.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.
echo [5/6] done.
echo.

:: ----------------------------------------------------------
:: STAGE 2, CONDITION 6: Active + defense -- model completion (KEY RESULT)
:: ----------------------------------------------------------
echo ===== [6/6] Stage 2 -- Active + Defense (KEY RESULT) =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% ^
  --dataset-path %DATAPATH% ^
  --n-labeled 40 ^
  --party-num 2 ^
  --half %HALF% ^
  --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CINIC10L_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.
echo [6/6] done.
echo.

echo ============================================================
echo  Phase 10 COMPLETE -- Results in:
echo  %MODELSDIR%\
echo.
echo  [4] model_completion_...normal_half=16.pth..._nlabeled=40.txt
echo      (benign baseline)
echo.
echo  [5] model_completion_...mal_half=16.pth..._nlabeled=40.txt
echo      (undefended attack baseline)
echo.
echo  [6] model_completion_...mal_asym_def-a=1.0-t=0.1-b=8_half=16.pth..._nlabeled=40.txt
echo      (defended -- KEY RESULT)
echo.
echo  PAPER CLAIM holds if [5] is substantially above [4],
echo  and [6] falls below [4] (benign baseline).
echo ============================================================
echo.
pause
