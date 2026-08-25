@echo off
:: ============================================================
:: Phase 9 — CIFAR100 Option B: Gradient Noise Injection
::
:: WHY THIS BAT EXISTS:
::   Phase 6A showed that stronger suppression (alpha=2.0) is
::   counterproductive on CIFAR-100: MC goes UP to 48.31%
::   (ABOVE the undefended 47.86%). Root cause: when scale -> 0,
::   the task gradient to Party A becomes exactly zero. MaliciousSGD
::   amplifies zero = no update, so the model freezes on the
::   class-discriminative structure built during burn-in.
::
::   Option B fix: when suppressing, inject calibrated Gaussian
::   noise so that MaliciousSGD amplifies noise instead of a
::   coherent class signal. Formula (in AsymmetricAdaptivePerturbation):
::     grad_output_a = scale*grad + noise_std*(1-scale)*E[|grad|]*randn()
::   At scale=0: all-noise gradient -> MaliciousSGD amplifies random
::   directions -> no coherent cluster formation in checkpoint weights.
::
:: WHAT RUNS (Stage 1 + Stage 2 for each noise_std variant):
::   Baselines:   benign and undefended attack already exist on disk
::                (normal_half=16.pth, mal_half=16.pth from Phase 5).
::                NOT re-run here; Stage 2 baseline numbers from EXP-010/EXP-011.
::
::   Option B variants (all at a=1.0, t=0.10, b=8, 150 epochs):
::     [1] noise_std=0.5  -> ..._asym_def-a=1.0-t=0.1-b=8-n=0.5_half=16_ep150
::     [2] noise_std=1.0  -> ..._asym_def-a=1.0-t=0.1-b=8-n=1.0_half=16_ep150
::     [3] noise_std=2.0  -> ..._asym_def-a=1.0-t=0.1-b=8-n=2.0_half=16_ep150
::
::   If one of these works (MC < benign 30.33%), add a=2.0 with that
::   noise_std in Phase 9b to get stronger suppression.
::
:: REFERENCE RESULTS (from research_log.md):
::   Benign MC (EXP-010):    30.33%
::   Attack MC (EXP-010):    47.86%
::   Defended a=1.0 (EXP-011): 43.12%  <- baseline to beat (still above benign)
::   Defended a=2.0 (EXP-013): 48.31%  <- FAILED (above attack!)
::
:: SUCCESS CRITERION: MC < 30.33% (below benign) for at least one variant.
::   Acceptable: MC < 43.12% (below EXP-011 no-noise baseline).
::
:: KEY PARAMETERS:
::   EPOCHS   : 150
::   K        : 5
::   HALF     : 16
::   n_labeled: 400
::   TAU      : 0.10 (same as EXP-011 for direct comparability)
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
echo  PHASE 9 -- CIFAR100 Option B: Gradient Noise Injection
echo  alpha=%ALPHA%, tau=%TAU%, burn_in=%BURNIN%
echo  Testing noise_std: 0.5, 1.0, 2.0
echo ============================================================
echo.

:: ----------------------------------------------------------
:: VARIANT 1: noise_std = 0.5
:: ----------------------------------------------------------
echo ===== [1/6] Stage 1 -- noise_std=0.5 =====
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
  --asymmetric-noise-std 0.5 ^
  --if-cluster-outputsA True

echo.
echo Renaming n=0.5 checkpoint to _ep150 variant...
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-n=0.5_half=16.pth" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-n=0.5_half=16_ep150.pth"
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-n=0.5_half=16.txt" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-n=0.5_half=16_ep150.txt"
echo [1/6] done.
echo.

echo ===== [2/6] Stage 2 -- noise_std=0.5 model completion =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% ^
  --dataset-path %DATAPATH% ^
  --n-labeled 400 ^
  --party-num 2 ^
  --half %HALF% ^
  --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-n=0.5_half=16_ep150.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.
echo [2/6] done.
echo.

:: ----------------------------------------------------------
:: VARIANT 2: noise_std = 1.0
:: ----------------------------------------------------------
echo ===== [3/6] Stage 1 -- noise_std=1.0 =====
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
  --asymmetric-noise-std 1.0 ^
  --if-cluster-outputsA True

echo.
echo Renaming n=1.0 checkpoint to _ep150 variant...
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-n=1.0_half=16.pth" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-n=1.0_half=16_ep150.pth"
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-n=1.0_half=16.txt" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-n=1.0_half=16_ep150.txt"
echo [3/6] done.
echo.

echo ===== [4/6] Stage 2 -- noise_std=1.0 model completion =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% ^
  --dataset-path %DATAPATH% ^
  --n-labeled 400 ^
  --party-num 2 ^
  --half %HALF% ^
  --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-n=1.0_half=16_ep150.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.
echo [4/6] done.
echo.

:: ----------------------------------------------------------
:: VARIANT 3: noise_std = 2.0
:: ----------------------------------------------------------
echo ===== [5/6] Stage 1 -- noise_std=2.0 =====
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
  --asymmetric-noise-std 2.0 ^
  --if-cluster-outputsA True

echo.
echo Renaming n=2.0 checkpoint to _ep150 variant...
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-n=2.0_half=16.pth" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-n=2.0_half=16_ep150.pth"
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-n=2.0_half=16.txt" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-n=2.0_half=16_ep150.txt"
echo [5/6] done.
echo.

echo ===== [6/6] Stage 2 -- noise_std=2.0 model completion =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% ^
  --dataset-path %DATAPATH% ^
  --n-labeled 400 ^
  --party-num 2 ^
  --half %HALF% ^
  --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-n=2.0_half=16_ep150.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.
echo [6/6] done.
echo.

echo ============================================================
echo  Phase 9 COMPLETE -- Results in:
echo  %MODELSDIR%\
echo.
echo  Baseline references (already on disk from Phase 5):
echo    benign:  30.33%%  (EXP-010, normal_half=16.pth)
echo    attack:  47.86%%  (EXP-010, mal_half=16.pth)
echo    no-noise defended: 43.12%%  (EXP-011, a=1.0,t=0.10,b=8)
echo.
echo  Option B results (this run):
echo    n=0.5: model_completion_...n=0.5...nlabeled=400.txt
echo    n=1.0: model_completion_...n=1.0...nlabeled=400.txt
echo    n=2.0: model_completion_...n=2.0...nlabeled=400.txt
echo.
echo  SUCCESS if MC < 30.33%% (below benign) for any variant.
echo  ACCEPTABLE if MC < 43.12%% (below EXP-011 no-noise).
echo ============================================================
echo.
pause
