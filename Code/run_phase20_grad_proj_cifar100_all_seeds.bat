@echo off
:: ============================================================
:: Phase 20 (MERGED) -- CIFAR-100 Gradient Projection Defense, Seeds 42 / 123 / 456
::
:: REPLACES: run_phase20a/20b/20c_grad_proj_cifar100_seed*.bat
::   Those scripts required separate machines due to checkpoint collision.
::   This merged version runs all three seeds SEQUENTIALLY on ONE machine.
::   Seed N is fully renamed before seed N+1 starts -- no collision risk.
::
:: WHY:
::   Phase 19 (seed-0) achieved 26.97% MC < 29.56% benign mean -- first
::   CIFAR-100 defense success. Mechanism: catastrophic single-activation
::   collapse at epoch 11 (intra_var_A: 0.16 -> 141,644).
::   These runs validate whether the collapse replicates across seeds.
::
::   Benign per-seed references (Phase 16):
::     Seed-0:   30.33%   Seed-42:  26.19%
::     Seed-123: 28.56%   Seed-456: 33.14%
::
:: SAFETY STEP 0:
::   Renames the existing seed-0 grad_proj .pth checkpoint (from Phase 19,
::   unsuffixed) to _seed0_ep150 so seed-42 Stage 1 does not overwrite it.
::   Uses 2>nul so it fails silently if already renamed or not present.
::
:: PAPER CRITERION per seed: defended MC < that seed's benign MC.
::   If all 3 seeds pass, report 4-seed mean +/- std (including seed-0: 26.97%).
::   If any seed fails, check CSV for collapse epoch (intra_var_A spike).
::
:: PARAMETERS: 150 epochs, half=16, k=5, tau=0.10, burn_in=8, proj_lr=1e-3.
::   Identical to Phase 19 (seed-0).
::
:: TOTAL RUNTIME: ~18h (3 x ~6h Stage 1 + 3 x ~20min Stage 2)
::
:: OUTPUTS (per seed, N = 42 / 123 / 456):
::   Stage 1:  CIFAR100_saved_framework_lr=0.1_mal_grad_proj_half=16_seed{N}_ep150.pth/.txt
::   Stage 2:  model_completion_..._mal_grad_proj_half=16_seed{N}_ep150.pth_..._nlabeled=400.txt
::   CSV:      separability_CIFAR100_lr=0.1_mal_grad_proj_half=16_seed{N}.csv
:: ============================================================

set DATASET=CIFAR100
set DATAPATH=.\data\CIFAR100
set EPOCHS=150
set HALF=16
set K=5

set TAU=0.10
set BURNIN=8
set PROJ_LR=1e-3
set EMBED_DIM=100
set NUM_CLASSES=100

set MODELSDIR=.\saved_experiment_results\saved_models\CIFAR100_saved_models
set CSVDIR=.\saved_experiment_results\csv_files\CIFAR100_csv_files
set RESUMEDIR=.\saved_experiment_results\saved_models\

echo.
echo ============================================================
echo  PHASE 20 (MERGED) -- CIFAR-100 Gradient Projection, Seeds 42/123/456
echo  Sequential to avoid checkpoint collision on shared machine.
echo  Seed-0 reference: 26.97%% (Phase 19). Benign mean: 29.56%%.
echo ============================================================
echo.

:: ----------------------------------------------------------
:: STEP 0 -- Protect seed-0 checkpoint (Phase 19 result)
:: ----------------------------------------------------------
echo ===== [STEP 0] Protect seed-0 grad_proj checkpoint =====
move "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_grad_proj_half=16.pth" ^
     "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_grad_proj_half=16_seed0_ep150.pth" 2>nul
echo  (seed-0 .pth renamed to _seed0_ep150 if it existed; silent otherwise)
echo.

:: ==============================================================
:: SEED 42
:: ==============================================================
set SEED=42

echo ===== [1/6] Stage 1 -- Attack + Gradient Projection, seed=%SEED% =====
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --gradient-projection-defense True ^
  --gradient-proj-embedding-dim %EMBED_DIM% ^
  --gradient-proj-num-classes %NUM_CLASSES% ^
  --gradient-proj-lr %PROJ_LR% ^
  --asymmetric-burn-in %BURNIN% ^
  --asymmetric-tau %TAU% ^
  --if-cluster-outputsA True ^
  --manual-seed %SEED%

move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_grad_proj_half=16.pth" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_grad_proj_half=16_seed%SEED%_ep150.pth"
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_grad_proj_half=16.txt" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_grad_proj_half=16_seed%SEED%_ep150.txt"
move /Y "%CSVDIR%\separability_CIFAR100_lr=0.1_mal_grad_proj_half=16.csv" ^
         "%CSVDIR%\separability_CIFAR100_lr=0.1_mal_grad_proj_half=16_seed%SEED%.csv"
echo [1/6] Stage 1 seed=%SEED% done.
echo.

echo ===== [2/6] Stage 2 -- Model Completion, seed=%SEED% =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 400 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_grad_proj_half=16_seed%SEED%_ep150.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed %SEED%
echo [2/6] MC seed=%SEED% done.
echo.
echo  Target: below 26.19%% (benign seed-42 reference from Phase 16)
echo.

:: ==============================================================
:: SEED 123
:: ==============================================================
set SEED=123

echo ===== [3/6] Stage 1 -- Attack + Gradient Projection, seed=%SEED% =====
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --gradient-projection-defense True ^
  --gradient-proj-embedding-dim %EMBED_DIM% ^
  --gradient-proj-num-classes %NUM_CLASSES% ^
  --gradient-proj-lr %PROJ_LR% ^
  --asymmetric-burn-in %BURNIN% ^
  --asymmetric-tau %TAU% ^
  --if-cluster-outputsA True ^
  --manual-seed %SEED%

move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_grad_proj_half=16.pth" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_grad_proj_half=16_seed%SEED%_ep150.pth"
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_grad_proj_half=16.txt" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_grad_proj_half=16_seed%SEED%_ep150.txt"
move /Y "%CSVDIR%\separability_CIFAR100_lr=0.1_mal_grad_proj_half=16.csv" ^
         "%CSVDIR%\separability_CIFAR100_lr=0.1_mal_grad_proj_half=16_seed%SEED%.csv"
echo [3/6] Stage 1 seed=%SEED% done.
echo.

echo ===== [4/6] Stage 2 -- Model Completion, seed=%SEED% =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 400 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_grad_proj_half=16_seed%SEED%_ep150.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed %SEED%
echo [4/6] MC seed=%SEED% done.
echo.
echo  Target: below 28.56%% (benign seed-123 reference from Phase 16)
echo.

:: ==============================================================
:: SEED 456
:: ==============================================================
set SEED=456

echo ===== [5/6] Stage 1 -- Attack + Gradient Projection, seed=%SEED% =====
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --gradient-projection-defense True ^
  --gradient-proj-embedding-dim %EMBED_DIM% ^
  --gradient-proj-num-classes %NUM_CLASSES% ^
  --gradient-proj-lr %PROJ_LR% ^
  --asymmetric-burn-in %BURNIN% ^
  --asymmetric-tau %TAU% ^
  --if-cluster-outputsA True ^
  --manual-seed %SEED%

move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_grad_proj_half=16.pth" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_grad_proj_half=16_seed%SEED%_ep150.pth"
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_grad_proj_half=16.txt" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_grad_proj_half=16_seed%SEED%_ep150.txt"
move /Y "%CSVDIR%\separability_CIFAR100_lr=0.1_mal_grad_proj_half=16.csv" ^
         "%CSVDIR%\separability_CIFAR100_lr=0.1_mal_grad_proj_half=16_seed%SEED%.csv"
echo [5/6] Stage 1 seed=%SEED% done.
echo.

echo ===== [6/6] Stage 2 -- Model Completion, seed=%SEED% =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 400 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_grad_proj_half=16_seed%SEED%_ep150.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed %SEED%
echo [6/6] MC seed=%SEED% done.
echo.
echo  Target: below 33.14%% (benign seed-456 reference from Phase 16)
echo.

echo ============================================================
echo  PHASE 20 (MERGED) COMPLETE
echo.
echo  Gradient Projection Defense (4-seed CIFAR-100):
echo    Seed-0  (Phase 19): MC = 26.97%%   Target was below 30.33%%  PASS
echo    Seed-42:   see model_completion_..._mal_grad_proj_half=16_seed42_ep150.pth_..._nlabeled=400.txt
echo    Seed-123:  see model_completion_..._mal_grad_proj_half=16_seed123_ep150.pth_..._nlabeled=400.txt
echo    Seed-456:  see model_completion_..._mal_grad_proj_half=16_seed456_ep150.pth_..._nlabeled=400.txt
echo.
echo  Benign references per seed (Phase 16):
echo    Seed-0=30.33%%  Seed-42=26.19%%  Seed-123=28.56%%  Seed-456=33.14%%
echo.
echo  PAPER CRITERION: defended MC less than benign MC at each seed.
echo  If all 4 seeds pass, report 4-seed mean +/- std.
echo  If any seed fails, check CSV for collapse epoch (look for intra_var_A spike).
echo.
echo  Combine with Phase 21 (attack) for full 3x4 comparison table.
echo ============================================================
echo.
pause
