@echo off
:: ============================================================
:: Phase 7 — Competitor Defense Model Completion
::
:: WHAT THIS DOES:
::   Runs Stage 2 (model_completion.py) on the competitor defense
::   checkpoints that were trained during Phase 2 but never had
::   label inference evaluated. Both GC (gradient compression at
::   75% sparsity) and Laplace DP noise are competitor baselines
::   from the original VFL literature.
::
:: CHECKPOINTS USED (all 30-epoch Stage 1 from Phase 2):
::   CIFAR-10:
::     mal_gc-preserved_percent=0.75_half=16.pth
::     mal_lap_noise-scale=0.001_half=16.pth
::     normal_lap_noise-scale=0.001_half=16.pth   (benign, for comparison)
::   CIFAR-100:
::     mal_gc-preserved_percent=0.75_half=16.pth
::     mal_lap_noise-scale=0.001_half=16.pth
::     normal_lap_noise-scale=0.001_half=16.pth
::
:: WHY NOW:
::   Phase 6A results will determine if our defense beats alpha=2.0
::   but the competitor comparison table needs GC and Laplace ASR
::   regardless. These are pure Stage 2 runs — no GPU-heavy Stage 1
::   needed. Total runtime ~2-3 hrs.
::
:: NOTE ON 30-EPOCH CHECKPOINTS:
::   These checkpoints are only 30-epoch Stage 1. For CIFAR-10, the
::   attack is only partially converged at 30 epochs (MaliciousSGD
::   needs ~100 epochs). This means GC and Laplace results will be
::   measured on a partially-converged attack — which *underestimates*
::   how hard those defenses need to work. Note this limitation in
::   the paper. For CIFAR-100 the attack converges earlier so these
::   results are more representative.
::
:: PAPER USE:
::   Table: Competitor defense comparison
::     Defense    | CIFAR-10 ASR | CIFAR-100 ASR
::     GC (75%)   | [C10-GC]     | [C100-GC]
::     Laplace DP | [C10-LAP]    | [C100-LAP]
::     Ours (AsymDef) | 84.27% | [Phase 6A result]
::   Ours should beat (i.e. lower ASR than) GC and Laplace.
:: ============================================================

set RESUMEDIR=.\saved_experiment_results\saved_models\
set CIFAR10PATH=.\data\CIFAR10
set CIFAR100PATH=.\data\CIFAR100

echo.
echo ============================================================
echo  PHASE 7 -- Competitor Defense Model Completion
echo ============================================================
echo.

:: ===========================================================
:: CIFAR-10 SECTION
:: ===========================================================

echo ========================================================
echo  CIFAR-10 -- Competitor Conditions (K=4, n_labeled=40)
echo ========================================================
echo.

echo ----- [C10-GC] Active + GC (75%%) -----
echo Checkpoint: mal_gc-preserved_percent=0.75_half=16.pth
echo.
python model_completion.py ^
  --dataset-name CIFAR10 ^
  --dataset-path %CIFAR10PATH% ^
  --n-labeled 40 ^
  --party-num 2 ^
  --half 16 ^
  --k 4 ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_mal_gc-preserved_percent=0.75_half=16.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.
echo [C10-GC] done.
echo.

echo ----- [C10-LAP] Active + Laplace DP (scale=0.001) -----
echo Checkpoint: mal_lap_noise-scale=0.001_half=16.pth
echo.
python model_completion.py ^
  --dataset-name CIFAR10 ^
  --dataset-path %CIFAR10PATH% ^
  --n-labeled 40 ^
  --party-num 2 ^
  --half 16 ^
  --k 4 ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_mal_lap_noise-scale=0.001_half=16.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.
echo [C10-LAP] done.
echo.

echo ----- [C10-LAP-N] Benign + Laplace DP (baseline for LAP condition) -----
echo Checkpoint: normal_lap_noise-scale=0.001_half=16.pth
echo.
python model_completion.py ^
  --dataset-name CIFAR10 ^
  --dataset-path %CIFAR10PATH% ^
  --n-labeled 40 ^
  --party-num 2 ^
  --half 16 ^
  --k 4 ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_normal_lap_noise-scale=0.001_half=16.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.
echo [C10-LAP-N] done.
echo.

:: ===========================================================
:: CIFAR-100 SECTION
:: ===========================================================

echo ========================================================
echo  CIFAR-100 -- Competitor Conditions (K=5, n_labeled=400)
echo ========================================================
echo.

echo ----- [C100-GC] Active + GC (75%%) -----
echo Checkpoint: CIFAR100 mal_gc-preserved_percent=0.75_half=16.pth
echo.
python model_completion.py ^
  --dataset-name CIFAR100 ^
  --dataset-path %CIFAR100PATH% ^
  --n-labeled 400 ^
  --party-num 2 ^
  --half 16 ^
  --k 5 ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_gc-preserved_percent=0.75_half=16.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.
echo [C100-GC] done.
echo.

echo ----- [C100-LAP] Active + Laplace DP (scale=0.001) -----
echo Checkpoint: CIFAR100 mal_lap_noise-scale=0.001_half=16.pth
echo.
python model_completion.py ^
  --dataset-name CIFAR100 ^
  --dataset-path %CIFAR100PATH% ^
  --n-labeled 400 ^
  --party-num 2 ^
  --half 16 ^
  --k 5 ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_lap_noise-scale=0.001_half=16.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.
echo [C100-LAP] done.
echo.

echo ----- [C100-LAP-N] Benign + Laplace DP (baseline for LAP condition) -----
echo Checkpoint: CIFAR100 normal_lap_noise-scale=0.001_half=16.pth
echo.
python model_completion.py ^
  --dataset-name CIFAR100 ^
  --dataset-path %CIFAR100PATH% ^
  --n-labeled 400 ^
  --party-num 2 ^
  --half 16 ^
  --k 5 ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_normal_lap_noise-scale=0.001_half=16.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.
echo [C100-LAP-N] done.
echo.

echo ============================================================
echo  Phase 7 COMPLETE
echo.
echo  KEY RESULTS (look for Best Train Top-1 in each TXT file):
echo.
echo  CIFAR-10 (saved in CIFAR10_saved_models/):
echo    [C10-GC]    model_completion_..._mal_gc-preserved_percent=0.75_half=16.pth_..._nlabeled=40.txt
echo    [C10-LAP]   model_completion_..._mal_lap_noise-scale=0.001_half=16.pth_..._nlabeled=40.txt
echo    [C10-LAP-N] model_completion_..._normal_lap_noise-scale=0.001_half=16.pth_..._nlabeled=40.txt
echo.
echo  CIFAR-100 (saved in CIFAR100_saved_models/):
echo    [C100-GC]    model_completion_..._mal_gc-preserved_percent=0.75_half=16.pth_..._nlabeled=400.txt
echo    [C100-LAP]   model_completion_..._mal_lap_noise-scale=0.001_half=16.pth_..._nlabeled=400.txt
echo    [C100-LAP-N] model_completion_..._normal_lap_noise-scale=0.001_half=16.pth_..._nlabeled=400.txt
echo.
echo  PAPER CLAIM: Our AsymDef ASR (84.27%% C10, Phase6A result C100)
echo  should be lower than both C10-GC and C10-LAP to show our defense
echo  is stronger than prior art while preserving VFL accuracy.
echo.
echo  NOTE: These are 30-epoch checkpoints. CIFAR-10 attack is only
echo  partially converged at 30 epochs -- results underestimate what
echo  GC/Laplace need to handle. Note this limitation in the paper.
echo ============================================================
echo.
pause
