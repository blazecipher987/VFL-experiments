@echo off
:: ============================================================
:: Phase 8 — CIFAR-10 Ablation at 100 Epochs
::
:: WHAT THIS DOES:
::   Reruns the hyperparameter ablation for CIFAR-10 at 100 epochs.
::   Phase 3 ablation ran only 30 epochs, which is insufficient for
::   CIFAR-10 because MaliciousSGD needs ~100 epochs to converge
::   (EXP-011 showed 30ep attack gives ~52% vs 100ep gives ~95%).
::   These 100-epoch results are the authoritative ablation table.
::
:: CONDITIONS (4 defended, ablating alpha and tau):
::   Alpha ablation (tau=0.10 fixed):
::     [1] alpha=0.5, tau=0.10, burn_in=8
::     [2] alpha=2.0, tau=0.10, burn_in=8
::   Tau ablation (alpha=1.0 fixed):
::     [3] alpha=1.0, tau=0.05, burn_in=8
::     [4] alpha=1.0, tau=0.15, burn_in=8
::
::   Reference (already done in Phase 4 / EXP-011):
::     alpha=1.0, tau=0.10 → defended 84.27%% (use this as pivot)
::
:: CHECKPOINT NAMING:
::   Stage 1 saves to the standard ablation names (same as Phase 3).
::   These OVERWRITE the 30-epoch Phase 3 checkpoints — that is
::   intentional. The 30-epoch results are already in research_log.md
::   and are noted as unreliable. We rename each to _ep100 immediately
::   after training to keep the naming consistent with Phase 4/5/6.
::
:: WHAT TO COMPARE IN THE PAPER:
::   All 5 alpha/tau conditions at 100 epochs:
::     alpha=0.5: [1] result
::     alpha=1.0: 84.27%% (EXP-011, Phase 4) ← reference
::     alpha=2.0: [2] result
::     tau=0.05:  [3] result
::     tau=0.10:  84.27%% (EXP-011, Phase 4) ← reference
::     tau=0.15:  [4] result
::   Attack (no defense): 95.42%% (EXP-011)
::   Benign: 87.23%% (EXP-011)
::
:: RUNTIME: ~3 hrs per 100-ep Stage 1 run * 4 = ~12 hrs Stage 1
::          + ~1 hr for 4 Stage 2 runs = ~13 hrs total
:: ============================================================

set DATASET=CIFAR10
set DATAPATH=.\data\CIFAR10
set EPOCHS=100
set HALF=16
set K=4
set BURNIN=8

set MODELSDIR=.\saved_experiment_results\saved_models\CIFAR10_saved_models
set RESUMEDIR=.\saved_experiment_results\saved_models\

echo.
echo ============================================================
echo  PHASE 8 -- CIFAR-10 Ablation at 100 Epochs
echo ============================================================
echo.

:: ===========================================================
:: STAGE 1 — CONDITION 1: alpha=0.5, tau=0.10
:: ===========================================================
echo ===== [1/8] Stage 1 -- alpha=0.5, tau=0.10 (100 epochs) =====
echo Saves to: mal_asym_def-a=0.5-t=0.1-b=8_half=16.pth
echo Renames to: mal_asym_def-a=0.5-t=0.1-b=8_half=16_ep100.pth
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --asymmetric-defense True ^
  --asymmetric-alpha 0.5 ^
  --asymmetric-tau 0.10 ^
  --asymmetric-burn-in %BURNIN% ^
  --if-cluster-outputsA True

echo.
echo Renaming alpha=0.5 checkpoint to _ep100...
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=0.5-t=0.1-b=8_half=16.pth" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=0.5-t=0.1-b=8_half=16_ep100.pth"
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=0.5-t=0.1-b=8_half=16.txt" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=0.5-t=0.1-b=8_half=16_ep100.txt"
echo [1/8] done.
echo.

:: ===========================================================
:: STAGE 1 — CONDITION 2: alpha=2.0, tau=0.10
:: ===========================================================
echo ===== [2/8] Stage 1 -- alpha=2.0, tau=0.10 (100 epochs) =====
echo Saves to: mal_asym_def-a=2.0-t=0.1-b=8_half=16.pth
echo Renames to: mal_asym_def-a=2.0-t=0.1-b=8_half=16_ep100.pth
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --asymmetric-defense True ^
  --asymmetric-alpha 2.0 ^
  --asymmetric-tau 0.10 ^
  --asymmetric-burn-in %BURNIN% ^
  --if-cluster-outputsA True

echo.
echo Renaming alpha=2.0 checkpoint to _ep100...
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=2.0-t=0.1-b=8_half=16.pth" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=2.0-t=0.1-b=8_half=16_ep100.pth"
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=2.0-t=0.1-b=8_half=16.txt" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=2.0-t=0.1-b=8_half=16_ep100.txt"
echo [2/8] done.
echo.

:: ===========================================================
:: STAGE 1 — CONDITION 3: alpha=1.0, tau=0.05
:: ===========================================================
echo ===== [3/8] Stage 1 -- alpha=1.0, tau=0.05 (100 epochs) =====
echo Saves to: mal_asym_def-a=1.0-t=0.05-b=8_half=16.pth
echo Renames to: mal_asym_def-a=1.0-t=0.05-b=8_half=16_ep100.pth
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --asymmetric-defense True ^
  --asymmetric-alpha 1.0 ^
  --asymmetric-tau 0.05 ^
  --asymmetric-burn-in %BURNIN% ^
  --if-cluster-outputsA True

echo.
echo Renaming tau=0.05 checkpoint to _ep100...
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.05-b=8_half=16.pth" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.05-b=8_half=16_ep100.pth"
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.05-b=8_half=16.txt" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.05-b=8_half=16_ep100.txt"
echo [3/8] done.
echo.

:: ===========================================================
:: STAGE 1 — CONDITION 4: alpha=1.0, tau=0.15
:: ===========================================================
echo ===== [4/8] Stage 1 -- alpha=1.0, tau=0.15 (100 epochs) =====
echo Saves to: mal_asym_def-a=1.0-t=0.15-b=8_half=16.pth
echo Renames to: mal_asym_def-a=1.0-t=0.15-b=8_half=16_ep100.pth
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --asymmetric-defense True ^
  --asymmetric-alpha 1.0 ^
  --asymmetric-tau 0.15 ^
  --asymmetric-burn-in %BURNIN% ^
  --if-cluster-outputsA True

echo.
echo Renaming tau=0.15 checkpoint to _ep100...
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.15-b=8_half=16.pth" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.15-b=8_half=16_ep100.pth"
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.15-b=8_half=16.txt" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.15-b=8_half=16_ep100.txt"
echo [4/8] done.
echo.

echo ============================================================
echo  Stage 1 complete. Starting Stage 2 model completion...
echo ============================================================
echo.

:: ===========================================================
:: STAGE 2 — MODEL COMPLETION
:: All checkpoints are now _ep100.pth
:: ===========================================================

echo ===== [5/8] Stage 2 -- alpha=0.5, tau=0.10 =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% ^
  --dataset-path %DATAPATH% ^
  --n-labeled 40 ^
  --party-num 2 ^
  --half %HALF% ^
  --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=0.5-t=0.1-b=8_half=16_ep100.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.
echo [5/8] done.
echo.

echo ===== [6/8] Stage 2 -- alpha=2.0, tau=0.10 =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% ^
  --dataset-path %DATAPATH% ^
  --n-labeled 40 ^
  --party-num 2 ^
  --half %HALF% ^
  --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=2.0-t=0.1-b=8_half=16_ep100.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.
echo [6/8] done.
echo.

echo ===== [7/8] Stage 2 -- alpha=1.0, tau=0.05 =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% ^
  --dataset-path %DATAPATH% ^
  --n-labeled 40 ^
  --party-num 2 ^
  --half %HALF% ^
  --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.05-b=8_half=16_ep100.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.
echo [7/8] done.
echo.

echo ===== [8/8] Stage 2 -- alpha=1.0, tau=0.15 =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% ^
  --dataset-path %DATAPATH% ^
  --n-labeled 40 ^
  --party-num 2 ^
  --half %HALF% ^
  --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.15-b=8_half=16_ep100.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.
echo [8/8] done.
echo.

echo ============================================================
echo  Phase 8 COMPLETE
echo.
echo  CIFAR-10 Ablation Table (100 epochs):
echo.
echo  ALPHA ABLATION (tau=0.10 fixed):
echo    alpha=0.5 : model_completion_..._mal_asym_def-a=0.5-t=0.1-b=8_half=16_ep100.pth_..._nlabeled=40.txt
echo    alpha=1.0 : 84.27%% (EXP-011, Phase 4) [reference]
echo    alpha=2.0 : model_completion_..._mal_asym_def-a=2.0-t=0.1-b=8_half=16_ep100.pth_..._nlabeled=40.txt
echo.
echo  TAU ABLATION (alpha=1.0 fixed):
echo    tau=0.05  : model_completion_..._mal_asym_def-a=1.0-t=0.05-b=8_half=16_ep100.pth_..._nlabeled=40.txt
echo    tau=0.10  : 84.27%% (EXP-011, Phase 4) [reference]
echo    tau=0.15  : model_completion_..._mal_asym_def-a=1.0-t=0.15-b=8_half=16_ep100.pth_..._nlabeled=40.txt
echo.
echo  All TXT results in:
echo    .\saved_experiment_results\saved_models\CIFAR10_saved_models\
echo.
echo  EXPECTED PATTERN: alpha increasing -> lower ASR (stronger defense);
echo    tau decreasing -> fires earlier -> potentially lower or higher ASR
echo    depending on false-positive rate.
echo    alpha=2.0 should outperform (lower ASR) Phase 6B if Phase 6A pattern holds.
echo ============================================================
echo.
pause
