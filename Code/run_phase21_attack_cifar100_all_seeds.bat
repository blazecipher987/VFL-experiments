@echo off
:: ============================================================
:: Phase 21 (MERGED) -- CIFAR-100 Attack Baseline, Seeds 42 / 123 / 456
::
:: REPLACES: run_phase21a/21b/21c_attack_cifar100_seed*.bat
::   Those scripts required separate machines due to checkpoint collision.
::   This merged version runs all three seeds SEQUENTIALLY on ONE machine,
::   eliminating checkpoint collision: seed N is fully renamed before seed N+1 starts.
::
:: WHY:
::   MaliciousSGD attack sweep for CIFAR-100. Completes the 4-seed attack
::   column of the comparison table.
::   Seed-0 already done: EXP-014, MC = 47.86%%.
::   Seeds 42/123/456 run here in order.
::
:: SAFETY STEP 0:
::   Renames the existing seed-0 attack .pth checkpoint (unsuffixed, from EXP-014)
::   to _seed0_ep150 so seed-42 Stage 1 does not overwrite it.
::   Uses 2>nul so it fails silently if already renamed or not present.
::
:: PARAMETERS: 150 epochs, half=16, k=5 (identical to EXP-014 / Phase 5).
::
:: TOTAL RUNTIME: ~12h (3 x ~4h Stage 1 + 3 x ~20min Stage 2)
::
:: OUTPUTS (per seed, N = 42 / 123 / 456):
::   Stage 1:  CIFAR100_saved_framework_lr=0.1_mal_half=16_seed{N}_ep150.pth/.txt
::   Stage 2:  model_completion_..._mal_half=16_seed{N}_ep150.pth_layer=1_func=ReLU_bn=True_nlabeled=400.txt
::   CSV:      separability_CIFAR100_lr=0.1_mal_half=16_seed{N}.csv
:: ============================================================

set DATASET=CIFAR100
set DATAPATH=.\data\CIFAR100
set EPOCHS=150
set HALF=16
set K=5

set MODELSDIR=.\saved_experiment_results\saved_models\CIFAR100_saved_models
set CSVDIR=.\saved_experiment_results\csv_files\CIFAR100_csv_files
set RESUMEDIR=.\saved_experiment_results\saved_models\

echo.
echo ============================================================
echo  PHASE 21 (MERGED) -- CIFAR-100 Attack Baseline, Seeds 42/123/456
echo  Sequential to avoid checkpoint collision on shared machine.
echo  Seed-0 reference: 47.86%% (EXP-014)
echo ============================================================
echo.

:: ----------------------------------------------------------
:: STEP 0 -- Protect seed-0 checkpoint
:: ----------------------------------------------------------
echo ===== [STEP 0] Protect seed-0 attack checkpoint =====
move "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_half=16.pth" ^
     "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_half=16_seed0_ep150.pth" 2>nul
echo  (seed-0 .pth renamed to _seed0_ep150 if it existed; silent otherwise)
echo.

:: ==============================================================
:: SEED 42
:: ==============================================================
set SEED=42

echo ===== [1/6] Stage 1 -- Attack VFL, seed=%SEED% =====
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --manual-seed %SEED%

move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_half=16.pth" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_half=16_seed%SEED%_ep150.pth"
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_half=16.txt" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_half=16_seed%SEED%_ep150.txt"
move /Y "%CSVDIR%\separability_CIFAR100_lr=0.1_mal_half=16.csv" ^
         "%CSVDIR%\separability_CIFAR100_lr=0.1_mal_half=16_seed%SEED%.csv"
echo [1/6] Stage 1 seed=%SEED% done.
echo.

echo ===== [2/6] Stage 2 -- Model Completion, seed=%SEED% =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 400 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_half=16_seed%SEED%_ep150.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed %SEED%
echo [2/6] MC seed=%SEED% done.
echo.

:: ==============================================================
:: SEED 123
:: ==============================================================
set SEED=123

echo ===== [3/6] Stage 1 -- Attack VFL, seed=%SEED% =====
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --manual-seed %SEED%

move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_half=16.pth" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_half=16_seed%SEED%_ep150.pth"
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_half=16.txt" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_half=16_seed%SEED%_ep150.txt"
move /Y "%CSVDIR%\separability_CIFAR100_lr=0.1_mal_half=16.csv" ^
         "%CSVDIR%\separability_CIFAR100_lr=0.1_mal_half=16_seed%SEED%.csv"
echo [3/6] Stage 1 seed=%SEED% done.
echo.

echo ===== [4/6] Stage 2 -- Model Completion, seed=%SEED% =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 400 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_half=16_seed%SEED%_ep150.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed %SEED%
echo [4/6] MC seed=%SEED% done.
echo.

:: ==============================================================
:: SEED 456
:: ==============================================================
set SEED=456

echo ===== [5/6] Stage 1 -- Attack VFL, seed=%SEED% =====
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --manual-seed %SEED%

move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_half=16.pth" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_half=16_seed%SEED%_ep150.pth"
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_half=16.txt" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_half=16_seed%SEED%_ep150.txt"
move /Y "%CSVDIR%\separability_CIFAR100_lr=0.1_mal_half=16.csv" ^
         "%CSVDIR%\separability_CIFAR100_lr=0.1_mal_half=16_seed%SEED%.csv"
echo [5/6] Stage 1 seed=%SEED% done.
echo.

echo ===== [6/6] Stage 2 -- Model Completion, seed=%SEED% =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 400 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_half=16_seed%SEED%_ep150.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed %SEED%
echo [6/6] MC seed=%SEED% done.
echo.

echo ============================================================
echo  PHASE 21 (MERGED) COMPLETE
echo.
echo  Attack baseline (4-seed CIFAR-100):
echo    Seed-0  (EXP-014): MC = 47.86%%
echo    Seed-42:   see model_completion_..._mal_half=16_seed42_ep150.pth_..._nlabeled=400.txt
echo    Seed-123:  see model_completion_..._mal_half=16_seed123_ep150.pth_..._nlabeled=400.txt
echo    Seed-456:  see model_completion_..._mal_half=16_seed456_ep150.pth_..._nlabeled=400.txt
echo.
echo  Benign references per seed (Phase 16):
echo    Seed-0=30.33%%  Seed-42=26.19%%  Seed-123=28.56%%  Seed-456=33.14%%
echo.
echo  Combine with Phase 20 (grad_proj defense) for per-seed privacy gap table.
echo ============================================================
echo.
pause
