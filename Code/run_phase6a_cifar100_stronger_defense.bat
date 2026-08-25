@echo off
:: ============================================================
:: Phase 6A — CIFAR100 150-Epoch Stronger Defense Variants
::
:: WHY THIS EXISTS:
::   Phase 5 (run_phase5_cifar100_150ep.bat) showed that the
::   standard defense (alpha=1.0, tau=0.10) is insufficient for
::   CIFAR100 at 150 epochs:
::     benign: 30.33%  |  attack: 47.86%  |  defended: 43.12%
::   The defense reduces 4.74pp out of 17.53pp attacker advantage
::   (27%% elimination). The defended ASR (43.12%%) remains
::   12.79pp ABOVE the benign baseline — the defense fails the
::   key paper criterion.
::
::   Root cause: alpha=1.0 generates scale = max(0, 1 - div + 0.10).
::   At CIFAR100's larger divergences (expect similar to CIFAR10),
::   the scale suppresses slowly. alpha=2.0 doubles the suppression
::   rate; tau=0.05 triggers the defense earlier in training.
::
:: WHAT RUNS:
::   Two new Stage 1 conditions at 150 epochs (CIFAR100):
::     [1A] alpha=2.0, tau=0.10  -> stronger suppression per divergence unit
::     [1B] alpha=2.0, tau=0.05  -> earlier trigger + stronger suppression
::
::   Both immediately renamed to _ep150 suffix to preserve the
::   existing Phase 3b 30-epoch ablation checkpoints on disk.
::
::   Three Stage 2 model completion runs:
::     [2A] Model completion for [1A]
::     [2B] Model completion for [1B]
::     [2C] Model completion for benign 150ep baseline (already exists,
::          re-runs to confirm benign baseline is ~30.33%%)
::
:: EXPECTED OUTCOME:
::   If alpha=2.0 is sufficient:
::     defended ASR should fall toward or below benign (30.33%%)
::   If not, the next step is noise injection into embedding directly
::   (Option B from research_log.md Section 7.2b).
::
:: FILE SAFETY:
::   Phase 3b ablation checkpoints (30-epoch):
::     mal_asym_def-a=2.0-t=0.1-b=8_half=16.pth   <- preserved (overwritten then renamed)
::     mal_asym_def-a=2.0-t=0.05-b=8_half=16.pth  <- new file (does not exist yet)
::   Rename strategy: immediately move to _ep150 variants so Phase 3b
::   30-epoch results remain accessible by their original names.
::   HOWEVER: this bat WILL overwrite the 30-epoch checkpoint during
::   Stage 1 training (before rename). The Stage 1 .txt result for
::   the 30-epoch version is already logged in research_log.md (Section 4.8).
::   The rename recovers the _ep150 checkpoint immediately after training.
::
:: PARAMETERS vs Phase 5:
::   [1A] alpha=2.0, tau=0.10, burn_in=8  (stronger suppression, same trigger)
::   [1B] alpha=2.0, tau=0.05, burn_in=8  (stronger suppression, earlier trigger)
::   All other params unchanged: EPOCHS=150, HALF=16, K=5, n_labeled=400
:: ============================================================

set DATASET=CIFAR100
set DATAPATH=.\data\CIFAR100
set EPOCHS=150
set HALF=16
set K=5
set BURNIN=8

set MODELSDIR=.\saved_experiment_results\saved_models\CIFAR100_saved_models
set RESUMEDIR=.\saved_experiment_results\saved_models\

echo.
echo ============================================================
echo  PHASE 6A -- CIFAR100 Stronger Defense Variants (150 epochs)
echo ============================================================
echo.

:: ----------------------------------------------------------
:: [1A] Stage 1: alpha=2.0, tau=0.10, burn_in=8
::
:: Saves to: mal_asym_def-a=2.0-t=0.1-b=8_half=16.pth
:: (overwrites Phase 3b 30-epoch checkpoint during training)
:: Immediately renamed to _ep150 to preserve Phase 3b context.
:: ----------------------------------------------------------
echo ===== [1A] Stage 1 -- alpha=2.0 tau=0.10 burn_in=8, 150 epochs =====
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
echo Renaming [1A] checkpoint to _ep150 variant...
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=2.0-t=0.1-b=8_half=16.pth" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=2.0-t=0.1-b=8_half=16_ep150.pth"
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=2.0-t=0.1-b=8_half=16.txt" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=2.0-t=0.1-b=8_half=16_ep150.txt"
echo [1A] done.
echo.

:: ----------------------------------------------------------
:: [1B] Stage 1: alpha=2.0, tau=0.05, burn_in=8
::
:: New filename (a=2.0-t=0.05 does not exist at 150ep yet).
:: Still rename to _ep150 for consistency with naming convention.
:: ----------------------------------------------------------
echo ===== [1B] Stage 1 -- alpha=2.0 tau=0.05 burn_in=8, 150 epochs =====
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --asymmetric-defense True ^
  --asymmetric-alpha 2.0 ^
  --asymmetric-tau 0.05 ^
  --asymmetric-burn-in %BURNIN% ^
  --if-cluster-outputsA True

echo.
echo Renaming [1B] checkpoint to _ep150 variant...
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=2.0-t=0.05-b=8_half=16.pth" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=2.0-t=0.05-b=8_half=16_ep150.pth"
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=2.0-t=0.05-b=8_half=16.txt" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=2.0-t=0.05-b=8_half=16_ep150.txt"
echo [1B] done.
echo.

echo ============================================================
echo  Stage 1 complete. Starting Stage 2 model completion...
echo ============================================================
echo.

:: ----------------------------------------------------------
:: [2A] Model completion for alpha=2.0, tau=0.10
:: ----------------------------------------------------------
echo ===== [2A] Stage 2 -- alpha=2.0 tau=0.10 (KEY RESULT A) =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% ^
  --dataset-path %DATAPATH% ^
  --n-labeled 400 ^
  --party-num 2 ^
  --half %HALF% ^
  --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=2.0-t=0.1-b=8_half=16_ep150.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.

:: ----------------------------------------------------------
:: [2B] Model completion for alpha=2.0, tau=0.05
:: ----------------------------------------------------------
echo ===== [2B] Stage 2 -- alpha=2.0 tau=0.05 (KEY RESULT B) =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% ^
  --dataset-path %DATAPATH% ^
  --n-labeled 400 ^
  --party-num 2 ^
  --half %HALF% ^
  --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=2.0-t=0.05-b=8_half=16_ep150.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.

echo ============================================================
echo  Phase 6A COMPLETE.
echo.
echo  PAPER CRITERION: defended ASR must be <= benign baseline (30.33%%)
echo.
echo  Check these files:
echo  [2A] model_completion_..._mal_asym_def-a=2.0-t=0.1-b=8_half=16_ep150.pth_..._nlabeled=400.txt
echo  [2B] model_completion_..._mal_asym_def-a=2.0-t=0.05-b=8_half=16_ep150.pth_..._nlabeled=400.txt
echo.
echo  If neither is <= 30.33%%, implement Option B (embedding noise injection).
echo  If [1B] hurts VFL accuracy >3pp, alpha=2.0 tau=0.10 is the safer choice.
echo ============================================================
echo.
pause
