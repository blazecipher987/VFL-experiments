@echo off
:: ============================================================
:: Phase 3 — Model Completion for All Ablation Checkpoints
::
:: Run this AFTER both ablation Stage 1 bats complete:
::   run_phase3_ablation_cifar10.bat
::   run_phase3_ablation_cifar100.bat
::
:: Evaluates label inference accuracy (model_completion.py) for:
::
:: CIFAR10 — 4 active ablation conditions:
::   alpha=0.5/2.0 with tau=0.10 (alpha ablation)
::   alpha=1.0 with tau=0.05/0.15 (tau ablation)
::
:: CIFAR100 — 4 active + 4 benign ablation conditions:
::   Same alpha/tau grid as CIFAR10
::   Benign conditions give 4 new baseline samples to average
::   (resolves the single-run variance issue from Phase 2)
::
:: After running, build Table 3 (ablation) in the paper with:
::   - Attack accuracy vs alpha (tau fixed at 0.10)
::   - Attack accuracy vs tau (alpha fixed at 1.0)
::   - CIFAR100 benign baseline mean ± std from 5 samples
::     (11.27% from Phase 2 + 4 new runs from this bat)
:: ============================================================

set RESUMEDIR=.\saved_experiment_results\saved_models\
set CIFAR10PATH=.\data\CIFAR10
set CIFAR100PATH=.\data\CIFAR100

echo.
echo ========================================================
echo  CIFAR10 — Active Ablation Conditions
echo ========================================================

echo.
echo ----- [C10-1] Active, alpha=0.5, tau=0.10 -----
python model_completion.py ^
  --dataset-name CIFAR10 --dataset-path %CIFAR10PATH% ^
  --n-labeled 40 --party-num 2 --half 16 --k 4 ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=0.5-t=0.1-b=8_half=16.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True --epochs 25 --print-to-txt 1

echo.
echo ----- [C10-2] Active, alpha=2.0, tau=0.10 -----
python model_completion.py ^
  --dataset-name CIFAR10 --dataset-path %CIFAR10PATH% ^
  --n-labeled 40 --party-num 2 --half 16 --k 4 ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=2.0-t=0.1-b=8_half=16.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True --epochs 25 --print-to-txt 1

echo.
echo ----- [C10-3] Active, alpha=1.0, tau=0.05 -----
python model_completion.py ^
  --dataset-name CIFAR10 --dataset-path %CIFAR10PATH% ^
  --n-labeled 40 --party-num 2 --half 16 --k 4 ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.05-b=8_half=16.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True --epochs 25 --print-to-txt 1

echo.
echo ----- [C10-4] Active, alpha=1.0, tau=0.15 -----
python model_completion.py ^
  --dataset-name CIFAR10 --dataset-path %CIFAR10PATH% ^
  --n-labeled 40 --party-num 2 --half 16 --k 4 ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.15-b=8_half=16.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True --epochs 25 --print-to-txt 1

echo.
echo ========================================================
echo  CIFAR100 — Active Ablation Conditions
echo ========================================================

echo.
echo ----- [C100-1] Active, alpha=0.5, tau=0.10 -----
python model_completion.py ^
  --dataset-name CIFAR100 --dataset-path %CIFAR100PATH% ^
  --n-labeled 400 --party-num 2 --half 16 --k 5 ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=0.5-t=0.1-b=8_half=16.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True --epochs 25 --print-to-txt 1

echo.
echo ----- [C100-2] Active, alpha=2.0, tau=0.10 -----
python model_completion.py ^
  --dataset-name CIFAR100 --dataset-path %CIFAR100PATH% ^
  --n-labeled 400 --party-num 2 --half 16 --k 5 ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=2.0-t=0.1-b=8_half=16.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True --epochs 25 --print-to-txt 1

echo.
echo ----- [C100-3] Active, alpha=1.0, tau=0.05 -----
python model_completion.py ^
  --dataset-name CIFAR100 --dataset-path %CIFAR100PATH% ^
  --n-labeled 400 --party-num 2 --half 16 --k 5 ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.05-b=8_half=16.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True --epochs 25 --print-to-txt 1

echo.
echo ----- [C100-4] Active, alpha=1.0, tau=0.15 -----
python model_completion.py ^
  --dataset-name CIFAR100 --dataset-path %CIFAR100PATH% ^
  --n-labeled 400 --party-num 2 --half 16 --k 5 ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_asym_def-a=1.0-t=0.15-b=8_half=16.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True --epochs 25 --print-to-txt 1

echo.
echo ========================================================
echo  CIFAR100 — Benign Baseline Samples (for variance fix)
echo  NOTE: these 4 results + Phase 2 result (11.27%) give
echo  5 samples; average them for the stable benign baseline.
echo ========================================================

echo.
echo ----- [C100-B2] Benign, alpha=0.5, tau=0.10 — baseline sample 2 -----
python model_completion.py ^
  --dataset-name CIFAR100 --dataset-path %CIFAR100PATH% ^
  --n-labeled 400 --party-num 2 --half 16 --k 5 ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_normal_asym_def-a=0.5-t=0.1-b=8_half=16.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True --epochs 25 --print-to-txt 1

echo.
echo ----- [C100-B3] Benign, alpha=2.0, tau=0.10 — baseline sample 3 -----
python model_completion.py ^
  --dataset-name CIFAR100 --dataset-path %CIFAR100PATH% ^
  --n-labeled 400 --party-num 2 --half 16 --k 5 ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_normal_asym_def-a=2.0-t=0.1-b=8_half=16.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True --epochs 25 --print-to-txt 1

echo.
echo ----- [C100-B4] Benign, alpha=1.0, tau=0.05 — baseline sample 4 -----
python model_completion.py ^
  --dataset-name CIFAR100 --dataset-path %CIFAR100PATH% ^
  --n-labeled 400 --party-num 2 --half 16 --k 5 ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_normal_asym_def-a=1.0-t=0.05-b=8_half=16.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True --epochs 25 --print-to-txt 1

echo.
echo ----- [C100-B5] Benign, alpha=1.0, tau=0.15 — baseline sample 5 -----
python model_completion.py ^
  --dataset-name CIFAR100 --dataset-path %CIFAR100PATH% ^
  --n-labeled 400 --party-num 2 --half 16 --k 5 ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_normal_asym_def-a=1.0-t=0.15-b=8_half=16.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True --epochs 25 --print-to-txt 1

echo.
echo ========================================================
echo  Phase 3 Model Completion COMPLETE
echo.
echo  Results to collect from TXT files:
echo.
echo  ALPHA ABLATION TABLE (tau=0.10 fixed):
echo    alpha=0.5 : C10-1 and C100-1
echo    alpha=1.0 : Phase 2 (52.28%% C10, 21.35%% C100)   [already done]
echo    alpha=2.0 : C10-2 and C100-2
echo.
echo  TAU ABLATION TABLE (alpha=1.0 fixed):
echo    tau=0.05  : C10-3 and C100-3
echo    tau=0.10  : Phase 2 (52.28%% C10, 21.35%% C100)   [already done]
echo    tau=0.15  : C10-4 and C100-4
echo.
echo  CIFAR100 BENIGN BASELINE (average these 5 values):
echo    Sample 1  : 11.27%% (Phase 2, already recorded)
echo    Sample 2  : C100-B2 (look in normal_asym_def-a=0.5 txt)
echo    Sample 3  : C100-B3 (look in normal_asym_def-a=2.0 txt)
echo    Sample 4  : C100-B4 (look in normal_asym_def-a=1.0-t=0.05 txt)
echo    Sample 5  : C100-B5 (look in normal_asym_def-a=1.0-t=0.15 txt)
echo.
echo  TXT files are in:
echo    .\saved_experiment_results\saved_models\CIFAR10_saved_models\
echo    .\saved_experiment_results\saved_models\CIFAR100_saved_models\
echo ========================================================
echo.
pause
