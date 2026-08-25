@echo off
:: ============================================================
:: Criteo Full Experiment Suite
::
:: PREREQUISITE: Run run_criteo_preprocess.bat first.
::   Output must exist at: .\data\Criteo\criteo.csv
::
:: HYPERPARAMETERS (faithful to original run_training.bat):
::   --lr 5e-2   --momentum 0.5   --epochs 7   --k 2   --half 4096
::
:: BURN-IN NOTE:
::   Original experiments used 7 epochs total. The default --asymmetric-burn-in 8
::   would outlast the entire training run (defense never activates).
::   We use --asymmetric-burn-in 2 for Criteo: proportionally equivalent to
::   burn_in=8 for 30-epoch image runs (~28%% of total epochs in both cases).
::
:: EXPERIMENTS (8 stages):
::   [1/8] Stage 1 — Benign baseline
::   [2/8] Stage 2 — Model completion on benign
::   [3/8] Stage 1 — Attack: Party A uses MaliciousSGD only
::   [4/8] Stage 2 — Model completion on attack
::   [5/8] Stage 1 — Attack-all: Party A AND Party B use MaliciousSGD
::   [6/8] Stage 2 — Model completion on attack-all
::   [7/8] Stage 1 — Attack (Party A only) + AsymmetricAdaptivePerturbation defense
::   [8/8] Stage 2 — Model completion on defended checkpoint
::
:: CHECKPOINT NAMES:
::   Benign:     Criteo_saved_framework_lr=0.05_normal_half=4096.pth
::   Attack:     Criteo_saved_framework_lr=0.05_mal_half=4096.pth
::   Attack-all: Criteo_saved_framework_lr=0.05_mal-all_half=4096.pth
::   Defense:    Criteo_saved_framework_lr=0.05_mal_asym_def-a=1.0-t=0.1-b=2_half=4096.pth
:: ============================================================

set DATASET=Criteo
set DATAPATH=.\data\Criteo\criteo.csv
set EPOCHS=7
set HALF=4096
set K=2
set LR=5e-2
set MOM=0.5

set TAU=0.10
set ALPHA=1.0
set BURNIN=2
set AUX_EMB=4
set AUX_CLS=2

set MODELSDIR=.\saved_experiment_results\saved_models\Criteo_saved_models
set RESUMEDIR=.\saved_experiment_results\saved_models\

echo.
echo ============================================================
echo  CRITEO EXPERIMENT SUITE  (8 stages)
echo  Dataset : %DATAPATH%
echo  lr=%LR%  momentum=%MOM%  epochs=%EPOCHS%  half=%HALF%  k=%K%
echo ============================================================
echo.

:: ============================================================
:: [1/8] STAGE 1 — BENIGN BASELINE
:: Standard SGD for all parties. MC result = privacy floor.
:: ============================================================
echo ===== [1/8] Stage 1 -- Benign baseline =====
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --lr %LR% --momentum %MOM% ^
  --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability False
echo.
echo [1/8] Benign Stage 1 done.
echo.

:: ============================================================
:: [2/8] STAGE 2 — MODEL COMPLETION ON BENIGN
:: ============================================================
echo ===== [2/8] Stage 2 -- Model completion (benign) =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% ^
  --dataset-path %DATAPATH% ^
  --n-labeled 400 ^
  --party-num 2 ^
  --half %HALF% ^
  --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name Criteo_saved_framework_lr=0.05_normal_half=4096.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1
echo.
echo [2/8] Benign Stage 2 done.
echo.

echo ============================================================
echo  Benign done. Starting attack (Party A only)...
echo ============================================================
echo.

:: ============================================================
:: [3/8] STAGE 1 — ATTACK: Party A uses MaliciousSGD only
:: Party B and top model use standard SGD.
:: ============================================================
echo ===== [3/8] Stage 1 -- MaliciousSGD attack (Party A only) =====
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --lr %LR% --momentum %MOM% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability False
echo.
echo [3/8] Attack Stage 1 done.
echo.

:: ============================================================
:: [4/8] STAGE 2 — MODEL COMPLETION ON ATTACK
:: ============================================================
echo ===== [4/8] Stage 2 -- Model completion (attack) =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% ^
  --dataset-path %DATAPATH% ^
  --n-labeled 400 ^
  --party-num 2 ^
  --half %HALF% ^
  --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name Criteo_saved_framework_lr=0.05_mal_half=4096.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1
echo.
echo [4/8] Attack Stage 2 done.
echo.

echo ============================================================
echo  Attack done. Starting attack-all (Party A + Party B)...
echo ============================================================
echo.

:: ============================================================
:: [5/8] STAGE 1 — ATTACK-ALL: Both parties use MaliciousSGD
:: Stronger attack scenario from original run_training.bat.
:: Our defense only targets Party A — this is an ablation to
:: see whether symmetric MaliciousSGD creates stronger leakage.
:: ============================================================
echo ===== [5/8] Stage 1 -- MaliciousSGD attack (all parties) =====
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --lr %LR% --momentum %MOM% ^
  --use-mal-optim True --use-mal-optim-all True --use-mal-optim-top False ^
  --monitor-separability False
echo.
echo [5/8] Attack-all Stage 1 done.
echo.

:: ============================================================
:: [6/8] STAGE 2 — MODEL COMPLETION ON ATTACK-ALL
:: ============================================================
echo ===== [6/8] Stage 2 -- Model completion (attack-all) =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% ^
  --dataset-path %DATAPATH% ^
  --n-labeled 400 ^
  --party-num 2 ^
  --half %HALF% ^
  --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name Criteo_saved_framework_lr=0.05_mal-all_half=4096.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1
echo.
echo [6/8] Attack-all Stage 2 done.
echo.

echo ============================================================
echo  Attack-all done. Starting defense...
echo ============================================================
echo.

:: ============================================================
:: [7/8] STAGE 1 — ATTACK + ASYMMETRIC ADAPTIVE PERTURBATION
:: Party A uses MaliciousSGD. Server suppresses Party A's
:: gradient when Fisher divergence > tau=0.10.
::
:: --asymmetric-burn-in 2: defense activates from epoch 2 onward.
::   With 7 total epochs, burn_in=2 gives 5 active defense epochs
::   (~71%% of training). Proportionally equivalent to burn_in=8
::   on 30-epoch image runs (~73%% of training).
::
:: --monitor-separability True: required for Fisher detection.
::   Fisher divergence CSV will be saved for inspection.
::
:: IF DEFENSE NEVER FIRES (scale=1.0 every epoch):
::   Benign Fisher divergence for Criteo may be > 0.10.
::   Run benign with --monitor-separability True, check the CSV,
::   and re-run this stage with --asymmetric-tau 0.05.
:: ============================================================
echo ===== [7/8] Stage 1 -- Attack + Asymmetric Defense =====
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --lr %LR% --momentum %MOM% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --asymmetric-defense True ^
  --asymmetric-alpha %ALPHA% ^
  --asymmetric-tau %TAU% ^
  --asymmetric-burn-in %BURNIN%
echo.
echo [7/8] Defense Stage 1 done.
echo.

:: ============================================================
:: [8/8] STAGE 2 — MODEL COMPLETION ON DEFENDED CHECKPOINT
:: ============================================================
echo ===== [8/8] Stage 2 -- Model completion (defense) =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% ^
  --dataset-path %DATAPATH% ^
  --n-labeled 400 ^
  --party-num 2 ^
  --half %HALF% ^
  --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name Criteo_saved_framework_lr=0.05_mal_asym_def-a=1.0-t=0.1-b=2_half=4096.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1
echo.
echo [8/8] Defense Stage 2 done.
echo.

echo ============================================================
echo  ALL CRITEO EXPERIMENTS COMPLETE
echo.
echo  Results in: %MODELSDIR%\
echo.
echo  Stage 2 model completion results:
echo    model_completion_..._normal_half=4096.pth_..._nlabeled=400.txt     <- benign
echo    model_completion_..._mal_half=4096.pth_..._nlabeled=400.txt        <- attack
echo    model_completion_..._mal-all_half=4096.pth_..._nlabeled=400.txt    <- attack-all
echo    model_completion_..._asym_def-a=1.0-t=0.1-b=2_half=4096.pth_..._nlabeled=400.txt  <- defense
echo.
echo  INTERPRETATION:
echo    Benign MC  = privacy floor (no attack)
echo    Attack MC  ^> Benign MC     = MaliciousSGD is effective on Criteo
echo    Attack-all MC vs Attack MC = how much Party B amplifies leakage
echo    Defense MC ^< Attack MC     = defense reduces leakage
echo    Defense MC ^< Benign MC     = full privacy restoration
echo ============================================================
echo.
pause
