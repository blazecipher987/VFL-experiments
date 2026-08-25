@echo off
:: ============================================================
:: Phase 4 — CIFAR10 100-Epoch Re-Run (Critical Missing Experiment)
::
:: WHY THIS BAT EXISTS:
::   The original 100-epoch CIFAR10 active checkpoint (EXP-002,
::   `mal_half=16.pth`) was overwritten by the Phase 1
::   characterization bat, which saved to the same filename at
::   only 30 epochs. vfl_framework.py names checkpoints from
::   parameters only — epoch count is NOT in the filename.
::
:: NAMING STRATEGY:
::   BASELINE files keep their STANDARD names (mal_half=16.pth,
::   normal_half=16.pth). This preserves compatibility with all
::   other bat files in the project that reference these names.
::   The 30-epoch versions on disk are simply replaced — their
::   results are already recorded in research_log.md (EXP-009).
::
::   DEFENSE files (our novel contribution) get an _ep100 suffix
::   to distinguish them from the 30-epoch defended checkpoint
::   already on disk. Without renaming, model completion would
::   overwrite the 30-epoch defense result txt (52.28%, EXP-008).
::
:: WHAT RUNS:
::   Stage 1 (VFL training, 3 conditions):
::     [1] Benign, 100 epochs          → normal_half=16.pth (overwrites 30ep)
::     [2] Active no defense, 100 ep   → mal_half=16.pth    (overwrites 30ep)
::     [3] Active + defense, 100 ep    → mal_asym_def-..._half=16.pth
::                                        → renamed to ..._ep100.pth
::
::   Stage 2 (model completion, 3 conditions):
::     [4] Benign ep100                → loads normal_half=16.pth
::     [5] Active no defense ep100     → loads mal_half=16.pth
::     [6] Active + defense ep100      → loads ..._asym_def-..._ep100.pth
::
::   VALID PAPER CLAIM requires: [5] >> [6]
::   If [6] is close to [4] (benign baseline), the defense works.
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
echo  PHASE 4 — CIFAR10 100-Epoch Training + Model Completion
echo ============================================================
echo.

:: ----------------------------------------------------------
:: STAGE 1, CONDITION 1: Benign, 100 epochs
::
:: Overwrites normal_half=16.pth with the 100-epoch version.
:: Standard name kept — all other bats still work.
:: ----------------------------------------------------------
echo ===== [1/3] Stage 1 — Benign, 100 epochs =====
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
:: STAGE 1, CONDITION 2: Active (no defense), 100 epochs
::
:: Overwrites mal_half=16.pth with the 100-epoch version.
:: Standard name kept — all other bats still work.
:: Expected inference accuracy: ~90-95%% (reproducing EXP-003).
:: ----------------------------------------------------------
echo ===== [2/3] Stage 1 — Active no defense, 100 epochs =====
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
:: STAGE 1, CONDITION 3: Active + Defense, 100 epochs
::
:: vfl_framework.py saves to:
::   mal_asym_def-a=1.0-t=0.1-b=8_half=16.pth
:: That name already exists (30-epoch version from EXP-007).
:: We rename immediately to _ep100.pth to preserve both results.
:: Defense params unchanged from Phase 2: alpha=1.0, tau=0.10, burn_in=8.
:: ----------------------------------------------------------
echo ===== [3/3] Stage 1 — Active + Defense, 100 epochs =====
echo Saves initially to: mal_asym_def-a=1.0-t=0.1-b=8_half=16.pth
echo Renames to:         mal_asym_def-a=1.0-t=0.1-b=8_half=16_ep100.pth
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
echo Renaming defense checkpoint and vfl-accuracy txt to _ep100 variants...
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16.pth" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16_ep100.pth"
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16.txt" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16_ep100.txt"
echo [3/3] done.
echo.

echo ============================================================
echo  Stage 1 complete. Starting Stage 2 model completion...
echo ============================================================
echo.

:: ----------------------------------------------------------
:: STAGE 2, CONDITION 4: Benign 100ep — model completion
:: ----------------------------------------------------------
echo ===== [4/6] Stage 2 — Benign, 100 epochs =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% ^
  --dataset-path %DATAPATH% ^
  --n-labeled 40 ^
  --party-num 2 ^
  --half %HALF% ^
  --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_normal_half=16.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.

:: ----------------------------------------------------------
:: STAGE 2, CONDITION 5: Active no defense 100ep — model completion
:: ----------------------------------------------------------
echo ===== [5/6] Stage 2 — Active no defense, 100 epochs (ATTACK BASELINE) =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% ^
  --dataset-path %DATAPATH% ^
  --n-labeled 40 ^
  --party-num 2 ^
  --half %HALF% ^
  --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_mal_half=16.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.

:: ----------------------------------------------------------
:: STAGE 2, CONDITION 6: Active+defense 100ep — model completion
:: ----------------------------------------------------------
echo ===== [6/6] Stage 2 — Active + Defense, 100 epochs (KEY RESULT) =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% ^
  --dataset-path %DATAPATH% ^
  --n-labeled 40 ^
  --party-num 2 ^
  --half %HALF% ^
  --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16_ep100.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.
echo ============================================================
echo  Phase 4 COMPLETE — Results in:
echo  %MODELSDIR%\
echo.
echo  [4] model_completion_..._normal_half=16.pth_..._nlabeled=40.txt
echo      (benign 100-epoch baseline)
echo.
echo  [5] model_completion_..._mal_half=16.pth_..._nlabeled=40.txt
echo      (undefended attack at 100 epochs — should be ~90-95%%)
echo.
echo  [6] model_completion_..._mal_asym_def-a=1.0-t=0.1-b=8_half=16_ep100.pth_..._nlabeled=40.txt
echo      (defended attack at 100 epochs — KEY RESULT)
echo.
echo  PAPER CLAIM holds if [5] >> [6], and [6] is close to [4].
echo ============================================================
echo.
pause
