@echo off
:: ============================================================
:: Phase 13 — Sign-Flip Momentum Disruption, CIFAR-100, 150 epochs
::
:: WHY THIS BAT EXISTS:
::   Phase 12 tests sign-flip on CIFAR-10 (~3 hours).  If the CIFAR-10
::   result is promising (MC below benign 83.11%), this bat runs the
::   CIFAR-100 equivalent at 150 epochs.
::
::   CIFAR-100 context:
::     Benign baseline:  ~30.33%  (EXP-012, seed-0 — high run-to-run variance)
::     No defense:        47.86%  (EXP-014, Phase 5)
::     Standard defense:  43.12%  (EXP-015, Phase 5 — NOT below benign)
::     Gradient noise:   43-49%   (EXP-019, Phase 9 — ALL FAILED)
::
::   Sign-flip TARGET: MC < 43.12%% (any improvement over standard defense)
::   IDEAL target:     MC < 30.33%% (below benign — confirms semantic misalignment)
::
::   Theory: with alpha=1.0, tau=0.10, CIFAR-100 Fisher divergence peaks
::   around 0.4-0.5, giving scale ≈ 0.6.  At scale=0.6, alternating signs
::   give ratio = clamp(1 + 200*(−0.6g / 0.6g), 1, 5) = 1.0 on EVERY batch
::   (not just alternating).  MaliciousSGD amplification is completely
::   neutralised regardless of the specific scale value.
::
:: WHAT RUNS (Stage 1 only — new condition):
::   Benign (normal_half=16.pth) and undefended attack (mal_half=16.pth)
::   baselines already on disk from Phase 5.  Only attack+sign-flip runs here.
::
:: CHECKPOINT NAMING:
::   vfl_framework.py saves:
::     CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-sf_half=16.pth
::   Renamed to _ep150.pth for consistency with Phase 5/6/7 naming.
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
echo  PHASE 13 -- CIFAR-100 Sign-Flip Defense, 150 Epochs
echo ============================================================
echo.
echo  NOTE: Run Phase 12 (CIFAR-10) first to validate sign-flip theory.
echo  Only proceed here if Phase 12 MC is below benign baseline (83.11%%).
echo.

:: ----------------------------------------------------------
:: STAGE 1: Attack + Sign-Flip, 150 epochs
::
:: --asymmetric-sign-flip True: when Fisher divergence > tau, alternate sign
:: of grad_output_A every batch.  MaliciousSGD ratio forced to 1.0 every batch.
:: ----------------------------------------------------------
echo ===== [1/2] Stage 1 -- Attack + Sign-Flip, 150 epochs =====
echo Saves to: mal_asym_def-a=1.0-t=0.1-b=8-sf_half=16.pth
echo Renames to: mal_asym_def-a=1.0-t=0.1-b=8-sf_half=16_ep150.pth
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
  --asymmetric-sign-flip True ^
  --if-cluster-outputsA True

echo.
echo Renaming sign-flip checkpoint to _ep150 variant...
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-sf_half=16.pth" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-sf_half=16_ep150.pth"
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-sf_half=16.txt" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-sf_half=16_ep150.txt"
echo [1/2] Stage 1 done.
echo.

echo ============================================================
echo  Stage 1 complete. Starting Stage 2 model completion...
echo ============================================================
echo.

:: ----------------------------------------------------------
:: STAGE 2: Model completion on sign-flip CIFAR-100 checkpoint
::
:: Compare to:
::   Benign:           ~30.33%% (EXP-012, single seed — high variance)
::   No defense:        47.86%% (EXP-014)
::   Standard defense:  43.12%% (EXP-015)
::   Gradient noise n=2.0: 43.10%% (EXP-019 — best of Phase 9, still failed)
::   Sign-flip:           ??? (THIS RUN)
:: ----------------------------------------------------------
echo ===== [2/2] Stage 2 -- Model completion on sign-flip CIFAR-100 checkpoint =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% ^
  --dataset-path %DATAPATH% ^
  --n-labeled 400 ^
  --party-num 2 ^
  --half %HALF% ^
  --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-sf_half=16_ep150.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.
echo ============================================================
echo  Phase 13 COMPLETE -- Results in:
echo  %MODELSDIR%\
echo.
echo  model_completion_..._mal_asym_def-a=1.0-t=0.1-b=8-sf_half=16_ep150.pth_..._nlabeled=400.txt
echo.
echo  BASELINES:
echo    Benign:           ~30.33%% (EXP-012, seed-0)
echo    No defense:        47.86%% (EXP-014)
echo    Standard defense:  43.12%% (EXP-015)
echo    Gradient noise best (n=2.0): 43.10%% (EXP-019)
echo.
echo  TARGETS:
echo    Below 43.10%% = improvement over all prior CIFAR-100 defense attempts
echo    Below 30.33%% = below benign = CIFAR-100 story is complete
echo ============================================================
echo.
pause
