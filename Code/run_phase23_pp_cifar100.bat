@echo off
:: ============================================================
:: Phase 23 — Persistent Projection Defense, CIFAR-100, Seed-0
::
:: PREREQUISITE: Run Phase 22 (CIFAR-10) first and confirm at least
::   one alpha_ema value passes the gate test there. Use the best
::   alpha_ema from Phase 22 as the PRIMARY run here (alpha_ema=0.2
::   is the default; adjust SET EMA_PRIMARY below if Phase 22 shows
::   a different value is better).
::
:: PURPOSE: Test whether PersistentProjectionDefense generalizes
::   from 10-class (CIFAR-10) to 100-class (CIFAR-100) — this is
::   the key question for the unified defense story.
::
:: WHAT THIS TESTS:
::   PersistentProjectionDefense with three alpha_ema values:
::     alpha_ema=0.1, 0.2, 0.3 (same sweep as Phase 22)
::   All conditions: seed=0, 150 epochs, CIFAR-100, MaliciousSGD attack.
::
:: HOW TO READ THE RESULTS:
::   Open the Stage 2 .txt for each alpha_ema (3 files listed below).
::   Find the line: "mc_best_train_top1"
::
::   DEFENSE WORKS if: mc_best_train_top1 < 30.33%%
::     (seed-0 benign MC from Phase 16)
::
::   CRITICAL DIAGNOSTIC (check the CSV):
::     separability_CIFAR100_lr=0.1_mal_pers_proj-ema={N}_half=16_seed0.csv
::     Column fisher_divergence: must cross tau=0.10 MULTIPLE times.
::     Column intra_var_A: must NOT jump 6+ orders of magnitude in one epoch.
::     If ONLY one spike in intra_var_A: same one-shot collapse as Phase 19/20.
::     If gradual / multi-epoch activity: defense is working as intended.
::
::   PAPER CRITERION:
::     mc_best_train_top1 < 30.33%% (seed-0 benign) = PASS for this seed
::     mc_best_train_top1 < 29.56%% (4-seed benign mean) = strong result
::
:: REFERENCES:
::   Benign CIFAR-100 (Phase 16):
::     Seed-0: 30.33%%   Seed-42: 26.19%%  Seed-123: 28.56%%  Seed-456: 33.14%%
::     4-seed mean: 29.56%% +/- 2.81%%
::   Attack CIFAR-100 (Phase 21):
::     Seed-0: 47.86%%  4-seed mean: ~40-48%% (see Phase 21 results)
::   GradProj (one-shot collapse, Phase 19/20):
::     Seed-0: 26.97%%  — this is the previous best at CIFAR-100.
::     PP should match or exceed this WITHOUT catastrophic collapse.
::
:: OUTPUTS (per alpha_ema N = 0.1, 0.2, 0.3):
::   Stage 1 pth: CIFAR100_saved_framework_lr=0.1_mal_pers_proj-ema={N}_half=16_seed0.pth
::   Stage 1 txt: CIFAR100_saved_framework_lr=0.1_mal_pers_proj-ema={N}_half=16_seed0.txt
::   Stage 2 txt: model_completion_CIFAR100_..._pers_proj-ema={N}_half=16_seed0.pth_..._nlabeled=400.txt
::   CSV:         separability_CIFAR100_lr=0.1_mal_pers_proj-ema={N}_half=16_seed0.csv
::     (CSV is in: saved_experiment_results/csv_files/CIFAR100_csv_files/)
::
:: RUNTIME ESTIMATE: ~3 runs x ~6h = ~18h total (CIFAR-100 150ep is slow)
:: ============================================================

set DATASET=CIFAR100
set DATAPATH=.\data\CIFAR100
set EPOCHS=150
set HALF=16
set K=5
set SEED=0

set TAU=0.10
set BURNIN=4
set PROJ_LR=1e-3
set EMBED_DIM=100
set NUM_CLASSES=100

set MODELSDIR=.\saved_experiment_results\saved_models\CIFAR100_saved_models
set CSVDIR=.\saved_experiment_results\csv_files\CIFAR100_csv_files
set RESUMEDIR=.\saved_experiment_results\saved_models\

echo.
echo ============================================================
echo  PHASE 23 -- Persistent Projection, CIFAR-100, Seed-0
echo  Testing alpha_ema in {0.1, 0.2, 0.3}
echo  SUCCESS CRITERION: mc_best_train_top1 below 30.33%% (benign seed-0)
echo  AND fisher_divergence crosses tau MULTIPLE epochs in CSV
echo ============================================================
echo.

:: ==============================================================
:: RUN 1: alpha_ema = 0.1
:: ==============================================================
set EMA=0.1

echo ===== [1/6] Stage 1 -- Persistent Projection, alpha_ema=%EMA% =====
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --persistent-projection True ^
  --gradient-proj-embedding-dim %EMBED_DIM% ^
  --gradient-proj-num-classes %NUM_CLASSES% ^
  --gradient-proj-lr %PROJ_LR% ^
  --persistent-proj-alpha-ema %EMA% ^
  --persistent-proj-burn-in %BURNIN% ^
  --asymmetric-tau %TAU% ^
  --if-cluster-outputsA True ^
  --stone1 75 --stone2 120 ^
  --manual-seed %SEED%

move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%.pth" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_seed%SEED%.pth"
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%.txt" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_seed%SEED%.txt"
move /Y "%CSVDIR%\separability_CIFAR100_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%.csv" ^
         "%CSVDIR%\separability_CIFAR100_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_seed%SEED%.csv"
echo [1/6] Stage 1 alpha_ema=%EMA% done.
echo.

echo ===== [2/6] Stage 2 -- Model Completion, alpha_ema=%EMA% =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 400 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_seed%SEED%.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed %SEED%
echo [2/6] MC alpha_ema=%EMA% done.
echo  >> Target: mc_best_train_top1 below 30.33%% (benign seed-0 from Phase 16)
echo.

:: ==============================================================
:: RUN 2: alpha_ema = 0.2
:: ==============================================================
set EMA=0.2

echo ===== [3/6] Stage 1 -- Persistent Projection, alpha_ema=%EMA% =====
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --persistent-projection True ^
  --gradient-proj-embedding-dim %EMBED_DIM% ^
  --gradient-proj-num-classes %NUM_CLASSES% ^
  --gradient-proj-lr %PROJ_LR% ^
  --persistent-proj-alpha-ema %EMA% ^
  --persistent-proj-burn-in %BURNIN% ^
  --asymmetric-tau %TAU% ^
  --if-cluster-outputsA True ^
  --stone1 75 --stone2 120 ^
  --manual-seed %SEED%

move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%.pth" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_seed%SEED%.pth"
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%.txt" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_seed%SEED%.txt"
move /Y "%CSVDIR%\separability_CIFAR100_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%.csv" ^
         "%CSVDIR%\separability_CIFAR100_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_seed%SEED%.csv"
echo [3/6] Stage 1 alpha_ema=%EMA% done.
echo.

echo ===== [4/6] Stage 2 -- Model Completion, alpha_ema=%EMA% =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 400 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_seed%SEED%.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed %SEED%
echo [4/6] MC alpha_ema=%EMA% done.
echo  >> Target: mc_best_train_top1 below 30.33%%
echo.

:: ==============================================================
:: RUN 3: alpha_ema = 0.3
:: ==============================================================
set EMA=0.3

echo ===== [5/6] Stage 1 -- Persistent Projection, alpha_ema=%EMA% =====
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --persistent-projection True ^
  --gradient-proj-embedding-dim %EMBED_DIM% ^
  --gradient-proj-num-classes %NUM_CLASSES% ^
  --gradient-proj-lr %PROJ_LR% ^
  --persistent-proj-alpha-ema %EMA% ^
  --persistent-proj-burn-in %BURNIN% ^
  --asymmetric-tau %TAU% ^
  --if-cluster-outputsA True ^
  --stone1 75 --stone2 120 ^
  --manual-seed %SEED%

move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%.pth" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_seed%SEED%.pth"
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%.txt" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_seed%SEED%.txt"
move /Y "%CSVDIR%\separability_CIFAR100_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%.csv" ^
         "%CSVDIR%\separability_CIFAR100_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_seed%SEED%.csv"
echo [5/6] Stage 1 alpha_ema=%EMA% done.
echo.

echo ===== [6/6] Stage 2 -- Model Completion, alpha_ema=%EMA% =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 400 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_seed%SEED%.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed %SEED%
echo [6/6] MC alpha_ema=%EMA% done.
echo.

echo ============================================================
echo  PHASE 23 COMPLETE
echo.
echo  Compare mc_best_train_top1 across the three Stage 2 txt files:
echo    alpha_ema=0.1: model_completion_CIFAR100_..._pers_proj-ema=0.1_half=16_seed0.pth_..._nlabeled=400.txt
echo    alpha_ema=0.2: model_completion_CIFAR100_..._pers_proj-ema=0.2_half=16_seed0.pth_..._nlabeled=400.txt
echo    alpha_ema=0.3: model_completion_CIFAR100_..._pers_proj-ema=0.3_half=16_seed0.pth_..._nlabeled=400.txt
echo.
echo  BENIGN REFERENCES (Phase 16):
echo    Seed-0: 30.33%%   4-seed mean: 29.56%%
echo  ATTACK BASELINE:
echo    Seed-0: 47.86%% (Phase 21)
echo  PREVIOUS BEST DEFENSE (GradProj one-shot collapse, Phase 19):
echo    Seed-0: 26.97%% -- PP should match or exceed this.
echo.
echo  DECISION TREE:
echo    Case A (PP beats GradProj, multi-epoch CSV activity):
echo      → UNIFIED DEFENSE story confirmed. Run seed sweep (Phase 23B).
echo      → This is the top-tier paper result: one mechanism, two datasets.
echo    Case B (PP beats benign but not GradProj, multi-epoch CSV):
echo      → Still a PASS. The multi-epoch behavior IS the theoretical improvement.
echo      → Run seed sweep. Report both GradProj (lower MC) and PP (stable design).
echo    Case C (PP passes but CSV shows one-shot collapse again):
echo      → Same failure mode. Check burn_in and alpha_ema. GradProj collapse
echo      →   seems unavoidable — may need a fundamentally different mechanism.
echo    Case D (PP fails — MC above benign):
echo      → Fragmented story. CIFAR-10 uses AAP, CIFAR-100 uses GradProj.
echo      → This weakens the paper claim significantly; investigate alternatives.
echo.
echo  CSV DIAGNOSTICS (in CIFAR100_csv_files/):
echo    fisher_divergence: number of epochs above tau=0.10 (want: MANY, not just 1)
echo    intra_var_A: should stay below ~1000 (one-shot collapse = jump to 100k+)
echo    inter_var_A: should decrease under effective defense
echo ============================================================
echo.
pause
