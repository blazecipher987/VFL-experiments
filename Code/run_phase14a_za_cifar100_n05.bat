@echo off
:: ============================================================
:: Phase 14a — z_a Embedding Corruption, CIFAR-100, noise_std=0.5, 150 epochs
::
:: WHY THIS BAT EXISTS:
::   Phase 9 (gradient-space noise) failed because noise injected into
::   grad_output_A preserves gradient magnitude consistency that MaliciousSGD
::   exploits.  This bat tests the CORRECT surface: the embedding z_a BEFORE
::   the top model forward pass.
::
::   Mechanism (new --asymmetric-za-noise-std flag):
::     After detection fires (Fisher divergence > tau):
::       noisy_z_a = z_a + noise_std*(1 - scale)*randn(z_a)
::       top model trains on noisy_z_a (top model sees inconsistent class structure)
::       grad_output_A is computed for noisy_z_a (weaker class signal)
::       bottom model still gets clean z_a for monitor + backward path
::     Result: MaliciousSGD receives a noisier, weaker task gradient ->
::     harder to build discriminative embeddings over 150 epochs
::
::   Key difference from Phase 9:
::     Phase 9: noise added to grad_output_A AFTER backward (gradient space)
::     Phase 14: noise added to z_a BEFORE forward (embedding space)
::     The embedding-space corruption corrupts the TOP MODEL's view of Party A's
::     contribution; the gradient-space approach only added post-hoc noise.
::
:: THIS FILE: noise_std = 0.5 (moderate; strong enough to disrupt class structure
::   without completely destroying the top model's ability to use Party B)
::
:: INSTANCE ASSIGNMENT:
::   Run this on Instance 2 in parallel with Phase 12/13 (Instance 1) and
::   Phase 14b noise_std=1.0 (Instance 3).
::
:: CHECKPOINT NAMING:
::   vfl_framework.py saves:
::     CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-za=0.5_half=16.pth
::   Renamed to _ep150.pth for consistency.
::
:: SUCCESS CRITERIA:
::   Below 43.10%% (best Phase 9 result) = improvement
::   Below 30.33%% (benign seed-0)       = CIFAR-100 story complete
:: ============================================================

set DATASET=CIFAR100
set DATAPATH=.\data\CIFAR100
set EPOCHS=150
set HALF=16
set K=5

set TAU=0.10
set ALPHA=1.0
set BURNIN=8
set ZA_STD=0.5

set MODELSDIR=.\saved_experiment_results\saved_models\CIFAR100_saved_models
set RESUMEDIR=.\saved_experiment_results\saved_models\

echo.
echo ============================================================
echo  PHASE 14a -- CIFAR-100 z_a Corruption (noise_std=0.5), 150 Epochs
echo ============================================================
echo.

:: ----------------------------------------------------------
:: STAGE 1: Attack + z_a corruption (noise_std=0.5), 150 epochs
::
:: --asymmetric-za-noise-std 0.5: when defense fires, noisy z_a fed to top model.
:: The monitor and bottom model backward path see CLEAN z_a.
:: Gradient suppression (--asymmetric-defense True) still runs simultaneously.
:: ----------------------------------------------------------
echo ===== [1/2] Stage 1 -- Attack + z_a corruption (std=0.5), 150 epochs =====
echo Saves to: mal_asym_def-a=1.0-t=0.1-b=8-za=0.5_half=16.pth
echo Renames to: mal_asym_def-a=1.0-t=0.1-b=8-za=0.5_half=16_ep150.pth
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
  --asymmetric-za-noise-std %ZA_STD% ^
  --if-cluster-outputsA True

echo.
echo Renaming z_a checkpoint to _ep150 variant...
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-za=0.5_half=16.pth" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-za=0.5_half=16_ep150.pth"
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-za=0.5_half=16.txt" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-za=0.5_half=16_ep150.txt"
echo [1/2] Stage 1 done.
echo.

echo ============================================================
echo  Stage 1 complete. Starting Stage 2 model completion...
echo ============================================================
echo.

:: ----------------------------------------------------------
:: STAGE 2: Model completion on z_a=0.5 CIFAR-100 checkpoint
:: ----------------------------------------------------------
echo ===== [2/2] Stage 2 -- Model completion on z_a=0.5 checkpoint =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% ^
  --dataset-path %DATAPATH% ^
  --n-labeled 400 ^
  --party-num 2 ^
  --half %HALF% ^
  --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8-za=0.5_half=16_ep150.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.
echo ============================================================
echo  Phase 14a COMPLETE -- Results in:
echo  %MODELSDIR%\
echo.
echo  model_completion_..._mal_asym_def-a=1.0-t=0.1-b=8-za=0.5_half=16_ep150.pth_..._nlabeled=400.txt
echo.
echo  BASELINES:
echo    Benign (seed-0):          30.33%% (EXP-012 -- high run-to-run variance)
echo    No defense:               47.86%% (EXP-014)
echo    Standard defense:         43.12%% (EXP-015)
echo    Gradient noise best n=2.0: 43.10%% (EXP-019)
echo.
echo  TARGETS:
echo    Below 43.10%% = improvement over all prior CIFAR-100 attempts
echo    Below 30.33%% = below benign = success
echo  Compare with Phase 14b (z_a std=1.0) to select best noise level.
echo ============================================================
echo.
pause
