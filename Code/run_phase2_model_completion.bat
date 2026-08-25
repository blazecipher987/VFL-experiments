@echo off
:: ============================================================
:: Phase 2 — Model Completion on Defended Checkpoints
::
:: Run this AFTER run_phase2_defense_cifar10.bat and
:: run_phase2_defense_cifar100.bat have finished and saved
:: their .pth checkpoints.
::
:: What this evaluates:
::   For each dataset, we run model_completion.py (MixMatch SSL)
::   on three checkpoints and compare:
::     1. Active (no defense)   — baseline attack accuracy  [already have from Phase 1]
::     2. Active + defense ON   — does the defense lower attack accuracy?  [NEW]
::     3. Benign + defense ON   — does the defense hurt honest utility?    [NEW]
::
:: Results are saved as .txt files alongside the .pth checkpoints in:
::   saved_experiment_results\saved_models\CIFAR10_saved_models\
::   saved_experiment_results\saved_models\CIFAR100_saved_models\
::
:: Naming convention for defended checkpoints (set by vfl_framework.py setting_str):
::   CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16.pth
::   CIFAR10_saved_framework_lr=0.1_normal_asym_def-a=1.0-t=0.1-b=8_half=16.pth
::   CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16.pth
::   CIFAR100_saved_framework_lr=0.1_normal_asym_def-a=1.0-t=0.1-b=8_half=16.pth
:: ============================================================

set RESUMEDIR=.\saved_experiment_results\saved_models\
set CIFAR10PATH=.\data\CIFAR10
set CIFAR100PATH=.\data\CIFAR100

echo.
echo ============================================================
echo  CIFAR10 Model Completion
echo ============================================================

echo.
echo ----- [CIFAR10] Active + Defense ON (key result) -----
echo Expected: attack accuracy LOWER than Phase 1 active baseline (94.99%% train / 84.61%% test)
echo.
python model_completion.py ^
  --dataset-name CIFAR10 ^
  --dataset-path %CIFAR10PATH% ^
  --n-labeled 40 ^
  --party-num 2 ^
  --half 16 ^
  --k 4 ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16.pth ^
  --num-layer 1 ^
  --activation_func_type ReLU ^
  --use-bn True ^
  --epochs 25 ^
  --print-to-txt 1

echo.
echo ----- [CIFAR10] Benign + Defense ON (false-positive utility check) -----
echo Expected: utility MATCHES Phase 1 benign baseline (defense should not have fired)
echo.
python model_completion.py ^
  --dataset-name CIFAR10 ^
  --dataset-path %CIFAR10PATH% ^
  --n-labeled 40 ^
  --party-num 2 ^
  --half 16 ^
  --k 4 ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_normal_asym_def-a=1.0-t=0.1-b=8_half=16.pth ^
  --num-layer 1 ^
  --activation_func_type ReLU ^
  --use-bn True ^
  --epochs 25 ^
  --print-to-txt 1

echo.
echo ============================================================
echo  CIFAR100 Model Completion
echo ============================================================

echo.
echo ----- [CIFAR100] Active + Defense ON (key result) -----
echo Expected: attack accuracy LOWER than Phase 1 active baseline (43.35%% train / 22.38%% test)
echo.
python model_completion.py ^
  --dataset-name CIFAR100 ^
  --dataset-path %CIFAR100PATH% ^
  --n-labeled 400 ^
  --party-num 2 ^
  --half 16 ^
  --k 5 ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.1-b=8_half=16.pth ^
  --num-layer 1 ^
  --activation_func_type ReLU ^
  --use-bn True ^
  --epochs 25 ^
  --print-to-txt 1

echo.
echo ----- [CIFAR100] Benign + Defense ON (false-positive utility check) -----
echo Expected: utility MATCHES Phase 1 benign baseline (defense should not have fired)
echo.
python model_completion.py ^
  --dataset-name CIFAR100 ^
  --dataset-path %CIFAR100PATH% ^
  --n-labeled 400 ^
  --party-num 2 ^
  --half 16 ^
  --k 5 ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_normal_asym_def-a=1.0-t=0.1-b=8_half=16.pth ^
  --num-layer 1 ^
  --activation_func_type ReLU ^
  --use-bn True ^
  --epochs 25 ^
  --print-to-txt 1

echo.
echo ============================================================
echo  All Phase 2 model completion runs done.
echo.
echo  Key comparisons to make (open the .txt files):
echo.
echo  CIFAR10:
echo    Undefended active  : 94.99%% train / 84.61%% test  (from Phase 1)
echo    Defended active    : look for mal_asym_def... txt
echo    Benign + defense   : look for normal_asym_def... txt (should match benign baseline)
echo.
echo  CIFAR100:
echo    Undefended active  : 43.35%% train / 22.38%% test  (from Phase 1)
echo    Defended active    : look for mal_asym_def... txt
echo    Benign + defense   : look for normal_asym_def... txt (should match benign baseline)
echo ============================================================
echo.
pause
