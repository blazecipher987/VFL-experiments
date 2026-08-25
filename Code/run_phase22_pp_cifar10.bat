@echo off
:: ============================================================
:: Phase 22 — Persistent Projection Defense, CIFAR-10, Seed-0
::
:: PURPOSE: "Does it work?" gate test. Run ONLY IF this passes
::   proceed to Phase 22B (seed sweep) and then Phase 23.
::   Do NOT run seed sweep or CIFAR-100 before seeing these results.
::
:: WHAT THIS TESTS:
::   PersistentProjectionDefense with three alpha_ema values:
::     alpha_ema=0.1  (slow EMA — direction changes slowly)
::     alpha_ema=0.2  (default — balanced adaptation)
::     alpha_ema=0.3  (fast EMA — direction tracks current batch closely)
::   All conditions: seed=0, 100 epochs, CIFAR-10, MaliciousSGD attack.
::
:: HOW TO READ THE RESULTS:
::   Open the Stage 2 .txt for each alpha_ema (3 files listed below).
::   Find the line: "mc_best_train_top1"
::
::   DEFENSE WORKS if: mc_best_train_top1 < benign MC reference
::     Benign MC reference (seed-0, 100ep):
::       model_completion_CIFAR10_..._normal_half=16.pth_..._nlabeled=40.txt
::       (from Phase 4 — run it again if the file is missing)
::     Attack baseline (seed-0, 100ep): ~81-85%% (Phase 4)
::
::   DEFENSE IS ONE-SHOT (BAD) if:
::     Check the CSV file: separability_CIFAR10_lr=0.1_mal_pers_proj-ema={N}_half=16.csv
::     Column fisher_divergence: if it crosses tau=0.10 ONLY ONCE (and then
::     goes negative forever), that is the same collapse as GradientProjectionDefense.
::     A working defense shows fisher_divergence crossing tau MULTIPLE times, OR
::     remaining below tau after the defense successfully suppresses the attack.
::
::   BEST alpha_ema: lowest mc_best_train_top1 that is ALSO below the benign reference.
::   Take that alpha_ema to Phase 23 (CIFAR-100).
::
:: OUTPUTS (per alpha_ema N = 0.1, 0.2, 0.3):
::   Stage 1 pth: CIFAR10_saved_framework_lr=0.1_mal_pers_proj-ema={N}_half=16_seed0.pth
::   Stage 1 txt: CIFAR10_saved_framework_lr=0.1_mal_pers_proj-ema={N}_half=16_seed0.txt
::   Stage 2 txt: model_completion_CIFAR10_..._pers_proj-ema={N}_half=16_seed0.pth_..._nlabeled=40.txt
::   CSV:         separability_CIFAR10_lr=0.1_mal_pers_proj-ema={N}_half=16_seed0.csv
::     (CSV is in: saved_experiment_results/csv_files/CIFAR10_csv_files/)
::
:: RUNTIME ESTIMATE: ~3 runs x ~2.5h = ~7.5h total
::
:: BENIGN REFERENCE (re-run if missing):
::   python vfl_framework.py --dataset CIFAR10 --path-dataset .\data\CIFAR10
::     --epochs 100 --half 16 --k 4
::     --use-mal-optim False --use-mal-optim-all False
:: ============================================================

set DATASET=CIFAR10
set DATAPATH=.\data\CIFAR10
set EPOCHS=100
set HALF=16
set K=4
set SEED=0

set TAU=0.10
set BURNIN=4
set PROJ_LR=1e-3
set EMBED_DIM=10
set NUM_CLASSES=10

set MODELSDIR=.\saved_experiment_results\saved_models\CIFAR10_saved_models
set CSVDIR=.\saved_experiment_results\csv_files\CIFAR10_csv_files
set RESUMEDIR=.\saved_experiment_results\saved_models\

echo.
echo ============================================================
echo  PHASE 22 -- Persistent Projection, CIFAR-10, Seed-0
echo  Testing alpha_ema in {0.1, 0.2, 0.3}
echo  SUCCESS CRITERION: mc_best_train_top1 below benign reference
echo  AND fisher_divergence crosses tau multiple epochs (no one-shot collapse)
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
  --manual-seed %SEED%

move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%.pth" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_seed%SEED%.pth"
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%.txt" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_seed%SEED%.txt"
move /Y "%CSVDIR%\separability_CIFAR10_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%.csv" ^
         "%CSVDIR%\separability_CIFAR10_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_seed%SEED%.csv"
echo [1/6] Stage 1 alpha_ema=%EMA% done.
echo.

echo ===== [2/6] Stage 2 -- Model Completion, alpha_ema=%EMA% =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 40 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_seed%SEED%.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed %SEED%
echo [2/6] MC alpha_ema=%EMA% done.
echo  >> Look for mc_best_train_top1 in the Stage 2 txt above.
echo  >> Defense PASSES if that value is BELOW the benign reference.
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
  --manual-seed %SEED%

move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%.pth" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_seed%SEED%.pth"
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%.txt" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_seed%SEED%.txt"
move /Y "%CSVDIR%\separability_CIFAR10_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%.csv" ^
         "%CSVDIR%\separability_CIFAR10_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_seed%SEED%.csv"
echo [3/6] Stage 1 alpha_ema=%EMA% done.
echo.

echo ===== [4/6] Stage 2 -- Model Completion, alpha_ema=%EMA% =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 40 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_seed%SEED%.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed %SEED%
echo [4/6] MC alpha_ema=%EMA% done.
echo  >> Look for mc_best_train_top1 in the Stage 2 txt above.
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
  --manual-seed %SEED%

move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%.pth" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_seed%SEED%.pth"
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%.txt" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_seed%SEED%.txt"
move /Y "%CSVDIR%\separability_CIFAR10_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%.csv" ^
         "%CSVDIR%\separability_CIFAR10_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_seed%SEED%.csv"
echo [5/6] Stage 1 alpha_ema=%EMA% done.
echo.

echo ===== [6/6] Stage 2 -- Model Completion, alpha_ema=%EMA% =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 40 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_seed%SEED%.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed %SEED%
echo [6/6] MC alpha_ema=%EMA% done.
echo.

echo ============================================================
echo  PHASE 22 COMPLETE
echo.
echo  Compare mc_best_train_top1 across the three Stage 2 txt files:
echo    alpha_ema=0.1: model_completion_CIFAR10_..._pers_proj-ema=0.1_half=16_seed0.pth_..._nlabeled=40.txt
echo    alpha_ema=0.2: model_completion_CIFAR10_..._pers_proj-ema=0.2_half=16_seed0.pth_..._nlabeled=40.txt
echo    alpha_ema=0.3: model_completion_CIFAR10_..._pers_proj-ema=0.3_half=16_seed0.pth_..._nlabeled=40.txt
echo.
echo  Reference baselines:
echo    Benign (seed-0):  model_completion_CIFAR10_..._normal_half=16.pth_..._nlabeled=40.txt
echo    Attack (seed-0):  model_completion_CIFAR10_..._mal_half=16.pth_..._nlabeled=40.txt (should be ~81-85%%)
echo    AAP defense mean: 30-35%% (Phase 4 / Phase 6B results)
echo.
echo  DECISION TREE:
echo    If ANY alpha_ema gives MC below benign reference:
echo      → PASS. Choose best alpha_ema. Run Phase 23 (CIFAR-100).
echo    If MC above benign reference BUT CSV shows multi-epoch projection firing:
echo      → Partial success. Tune burn_in or tau before Phase 23.
echo    If MC above benign reference AND CSV shows one-shot collapse:
echo      → Same problem as GradProj. Investigate EMA warmup or reduce alpha_ema.
echo.
echo  CSV diagnostics (in CIFAR10_csv_files/):
echo    Check column: fisher_divergence (should cross tau=0.10 MULTIPLE times)
echo    Check column: intra_var_A (should NOT spike 6+ orders of magnitude)
echo ============================================================
echo.
pause
