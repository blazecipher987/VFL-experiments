@echo off
:: ============================================================
:: Phase 5 — CIFAR100 150-Epoch Re-Run (Critical Missing Experiment)
::
:: WHY THIS BAT EXISTS:
::   Phase 4 (run_phase4_cifar10_100ep.bat) confirmed the defense
::   works on CIFAR10 at 100 epochs: attack 95.42% -> defended
::   84.27%, below benign 87.23%. Phase 5 is the exact CIFAR100
::   equivalent. Without it, the paper cannot claim the defense
::   generalises across datasets.
::
::   Prior 30-epoch CIFAR100 comparison (Phase 2, EXP-007/EXP-008)
::   was a null result (+1.09pp) because MaliciousSGD on CIFAR100
::   does work at 30 epochs (+7.5pp) but the defence was too weak
::   against a partially-converged attack. At 150 epochs the attack
::   is fully converged (43.35% in EXP-005) and the defence scale
::   will progressively approach 0 (same mechanism as Phase 4).
::
:: NAMING STRATEGY:
::   BASELINE files keep their STANDARD names (mal_half=16.pth,
::   normal_half=16.pth). The 30-epoch versions on disk are replaced;
::   their results are already in research_log.md (EXP-007/EXP-009).
::
::   DEFENSE files get an _ep150 suffix to distinguish them from the
::   30-epoch defended checkpoint already on disk (EXP-007). Without
::   renaming, model completion would overwrite the Phase 2 result.
::
:: WHAT RUNS:
::   Stage 1 (VFL training, 3 conditions):
::     [1] Benign, 150 epochs          -> normal_half=16.pth (overwrites 30ep)
::     [2] Active no defense, 150 ep   -> mal_half=16.pth    (overwrites 30ep)
::     [3] Active + defense, 150 ep    -> mal_asym_def-..._half=16.pth
::                                         -> renamed to ..._ep150.pth
::
::   Stage 2 (model completion, 3 conditions):
::     [4] Benign ep150                -> loads normal_half=16.pth
::     [5] Active no defense ep150     -> loads mal_half=16.pth
::     [6] Active + defense ep150      -> loads ..._asym_def-..._ep150.pth
::
::   VALID PAPER CLAIM requires: [5] >> [6]
::   If [6] is close to [4] (benign baseline), the defense works.
::
:: KEY PARAMETERS vs CIFAR10 Phase 4:
::   EPOCHS  : 150  (CIFAR10 used 100)
::   K       : 5    (CIFAR10 used 4)
::   n_labeled: 400 (CIFAR10 used 40; CIFAR100 has 100 classes)
::   TAU     : 0.10 (same as Phase 2 CIFAR100 for direct comparability)
::   ALPHA   : 1.0  (same)
::   BURNIN  : 8    (same; CIFAR100 needs burn-in due to noisy init divergence)
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
echo  PHASE 5 -- CIFAR100 150-Epoch Training + Model Completion
echo ============================================================
echo.

:: ----------------------------------------------------------
:: STAGE 1, CONDITION 1: Benign, 150 epochs
::
:: Overwrites normal_half=16.pth with the 150-epoch version.
:: Standard name kept -- all other bats still work.
:: ----------------------------------------------------------
echo ===== [1/3] Stage 1 -- Benign, 150 epochs =====
echo Saves to standard name: normal_half=16.pth
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False

echo.
echo [1/3] done.
echo.

:: ----------------------------------------------------------
:: STAGE 1, CONDITION 2: Active (no defense), 150 epochs
::
:: Overwrites mal_half=16.pth with the 150-epoch version.
:: Standard name kept.
:: Expected inference accuracy: ~40-50%% (reproducing EXP-005).
:: ----------------------------------------------------------
echo ===== [2/3] Stage 1 -- Active no defense, 150 epochs =====
echo Saves to standard name: mal_half=16.pth
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
:: STAGE 1, CONDITION 3: Active + Defense, 150 epochs
::
:: vfl_framework.py saves to:
::   mal_asym_def-a=1.0-t=0.1-b=8_half=16.pth
:: That name already exists (30-epoch version from EXP-007).
:: We rename immediately to _ep150.pth to preserve both results.
:: Defense params unchanged from Phase 2: alpha=1.0, tau=0.10, burn_in=8.
:: ----------------------------------------------------------
echo ===== [3/3] Stage 1 -- Active + Defense, 150 epochs =====
echo Saves initially to: mal_asym_def-a=1.0-t=0.1-b=8_half=16.pth
echo Renames to:         mal_asym_def-a=1.0-t=0.1-b=8_half=16_ep150.pth
echo (preserves the existing 30-epoch defended checkpoint)
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
echo Renaming defense checkpoint and vfl-accuracy txt to _ep150 variants...
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16.pth" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16_ep150.pth"
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16.txt" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16_ep150.txt"
echo [3/3] done.
echo.

echo ============================================================
echo  Stage 1 complete. Starting Stage 2 model completion...
echo ============================================================
echo.

:: ----------------------------------------------------------
:: STAGE 2, CONDITION 4: Benign 150ep -- model completion
:: ----------------------------------------------------------
echo ===== [4/6] Stage 2 -- Benign, 150 epochs =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% ^
  --dataset-path %DATAPATH% ^
  --n-labeled 400 ^
  --party-num 2 ^
  --half %HALF% ^
  --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_normal_half=16.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.

:: ----------------------------------------------------------
:: STAGE 2, CONDITION 5: Active no defense 150ep -- model completion
:: ----------------------------------------------------------
echo ===== [5/6] Stage 2 -- Active no defense, 150 epochs (ATTACK BASELINE) =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% ^
  --dataset-path %DATAPATH% ^
  --n-labeled 400 ^
  --party-num 2 ^
  --half %HALF% ^
  --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_half=16.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.

:: ----------------------------------------------------------
:: STAGE 2, CONDITION 6: Active+defense 150ep -- model completion
:: ----------------------------------------------------------
echo ===== [6/6] Stage 2 -- Active + Defense, 150 epochs (KEY RESULT) =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% ^
  --dataset-path %DATAPATH% ^
  --n-labeled 400 ^
  --party-num 2 ^
  --half %HALF% ^
  --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16_ep150.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.
echo ============================================================
echo  Phase 5 COMPLETE -- Results in:
echo  %MODELSDIR%\
echo.
echo  [4] model_completion_..._normal_half=16.pth_..._nlabeled=400.txt
echo      (benign 150-epoch baseline)
echo.
echo  [5] model_completion_..._mal_half=16.pth_..._nlabeled=400.txt
echo      (undefended attack at 150 epochs -- should be ~40-50%%)
echo.
echo  [6] model_completion_..._mal_asym_def-a=1.0-t=0.1-b=8_half=16_ep150.pth_..._nlabeled=400.txt
echo      (defended attack at 150 epochs -- KEY RESULT)
echo.
echo  PAPER CLAIM holds if [5] >> [6], and [6] is close to [4].
echo ============================================================
echo.
pause
