@echo off
:: ============================================================
:: Phase 12 — Sign-Flip Momentum Disruption, CIFAR-10, 100 epochs
::
:: WHY THIS BAT EXISTS:
::   All gradient-space noise variants (Phase 9, EXP-019) FAILED for
::   CIFAR-100 because noise injected into grad_output_A preserves
::   magnitude consistency that MaliciousSGD's ratio computation
::   exploits.  Two new mechanisms are now tested:
::
::   Sign-flip (this bat): alternate the sign of grad_output_A every
::   batch when the defense fires.  MaliciousSGD ratio =
::   clamp(1 + gamma*(−g/g), 1, 5) = 1.0 on every batch — amplification
::   is eliminated and weight updates average to zero over pairs of batches.
::
::   z_a corruption (Phase 14): corrupt the embedding z_a BEFORE the
::   top model sees it, forcing the bottom model to learn diffuse
::   (non-discriminative) representations.
::
:: WHY CIFAR-10 FIRST:
::   CIFAR-10 runs in ~3 hours vs ~8 hours for CIFAR-100.  The
::   benign (83.11%) and undefended-attack (94.95%) baselines are
::   already done (Phase 4, EXP-013/EXP-014).  A CIFAR-10 run here
::   verifies the sign-flip theory before committing 8 hours to CIFAR-100.
::
:: WHAT RUNS:
::   Stage 1: Attack + sign-flip, 100 epochs  (new condition only;
::            baselines already on disk from Phase 4)
::   Stage 2: Model completion on the sign-flip checkpoint
::
:: CHECKPOINT NAMING:
::   vfl_framework.py will save:
::     CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-sf_half=16.pth
::   Renamed immediately to _ep100.pth for consistency with Phase 4 naming.
::
:: SUCCESS CRITERIA:
::   Sign-flip MC < 83.11%%  → below benign baseline → sign-flip works on CIFAR-10
::   Sign-flip MC < 94.95%%  → at least partial disruption
::   Run Phase 13 (CIFAR-100, 150ep) ONLY if this result is promising.
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
echo  PHASE 12 -- CIFAR-10 Sign-Flip Defense, 100 Epochs
echo ============================================================
echo.

:: ----------------------------------------------------------
:: STAGE 1: Attack + Sign-Flip, 100 epochs
::
:: Benign (normal_half=16.pth) and undefended attack (mal_half=16.pth)
:: already exist from Phase 4.  Only the new sign-flip condition runs here.
::
:: The sign-flip mechanism: --asymmetric-sign-flip True alternates the sign
:: of grad_output_A every batch when Fisher divergence exceeds tau.
:: MaliciousSGD sees ratio = clamp(1 + gamma*(-g/g), 1, 5) = 1.0
:: on every batch — amplification is completely neutralised.
:: ----------------------------------------------------------
echo ===== [1/2] Stage 1 -- Attack + Sign-Flip, 100 epochs =====
echo Saves to: mal_asym_def-a=1.0-t=0.1-b=8-sf_half=16.pth
echo Renames to: mal_asym_def-a=1.0-t=0.1-b=8-sf_half=16_ep100.pth
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
echo Renaming sign-flip checkpoint to _ep100 variant...
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-sf_half=16.pth" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-sf_half=16_ep100.pth"
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-sf_half=16.txt" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-sf_half=16_ep100.txt"
echo [1/2] Stage 1 done.
echo.

echo ============================================================
echo  Stage 1 complete. Starting Stage 2 model completion...
echo ============================================================
echo.

:: ----------------------------------------------------------
:: STAGE 2: Model completion on sign-flip checkpoint
::
:: Loads the sign-flip checkpoint (attack disrupted by alternating signs).
:: Compare result to:
::   Benign:    83.11% (EXP-013 mean, 4 seeds)
::   No defense: 94.95% (EXP-014, Phase 4)
::   Sign-flip:   ??? (THIS RUN)
::
:: If sign-flip MC < 83.11% -> below benign -> defense works -> run Phase 13
:: If sign-flip MC >> 83.11% -> sign-flip not sufficient alone
:: ----------------------------------------------------------
echo ===== [2/2] Stage 2 -- Model completion on sign-flip checkpoint =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% ^
  --dataset-path %DATAPATH% ^
  --n-labeled 40 ^
  --party-num 2 ^
  --half %HALF% ^
  --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-sf_half=16_ep100.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.
echo ============================================================
echo  Phase 12 COMPLETE -- Results in:
echo  %MODELSDIR%\
echo.
echo  model_completion_..._mal_asym_def-a=1.0-t=0.1-b=8-sf_half=16_ep100.pth_..._nlabeled=40.txt
echo.
echo  BASELINES (from Phase 4):
echo    Benign:     83.11%% (EXP-013 mean)
echo    No defense: 94.95%% (EXP-014)
echo.
echo  SIGN-FLIP TARGET: below 83.11%% (benign) to confirm theory
echo  If below benign: run Phase 13 (CIFAR-100 sign-flip, 150ep)
echo  If above benign: report partial disruption; focus on Phase 14 (z_a corruption)
echo ============================================================
echo.
pause
