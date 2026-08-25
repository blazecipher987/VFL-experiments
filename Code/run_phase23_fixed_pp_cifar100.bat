@echo off
:: ============================================================
:: Phase 23 (FIXED) — Persistent Projection Defense, CIFAR-100, Seed-0
::
:: PREREQUISITE: Run Phase 22-fixed (CIFAR-10) first.
::   DO NOT run this file until at least one alpha_ema from Phase 22-fixed
::   gives mc_best_train_top1 < 87.23%% on CIFAR-10.
::   Use the best alpha_ema from Phase 22-fixed as the primary run.
::   Adjust EMA_BEST below if Phase 22-fixed shows a value other than 0.2.
::
:: WHY THIS EXISTS:
::   Phase 23 (EXP-043/044, 2026-07-13) produced mc_best_train_top1 of
::   49.42%% and 49.44%% (both ABOVE the attack baseline of 47.86%%) due to
::   the same d_ema direction estimation bug as Phase 22. Results are
::   invalidated. This re-run uses the fixed possible_defenses.py.
::
:: EXPECTED IMPROVEMENT vs BUGGY PHASE 23:
::   Buggy: 49.42 - 49.44%%  (above attack level 47.86%%)
::   Target: mc_best_train_top1 < 30.33%%  (seed-0 benign reference)
::   Stretch: mc_best_train_top1 < 26.97%%  (best GradProj result, EXP-032)
::
:: PAPER CRITERION (for publication):
::   mc_best_train_top1 < 30.33%%  = PASS (seed-0)
::   mc_best_train_top1 < 29.56%%  = STRONG (below 4-seed benign mean)
::   mc_best_train_top1 < 26.97%%  = BEST (beats GradProj one-shot collapse)
::   The last would confirm PP as a strictly better mechanism than GradProj.
::
:: KEY DIAGNOSTIC (more important here than on CIFAR-10):
::   Check separability CSV after each run:
::     intra_var_A: if it spikes 6+ orders of magnitude in ONE epoch,
::     that is GradProj's one-shot collapse returning — not the desired
::     gradual multi-epoch suppression. A spike is not necessarily bad
::     (it does lower MC) but it is NOT the PP mechanism working correctly.
::   The ideal CSV shows:
::     - fisher_divergence crossing tau MANY epochs (not just once)
::     - intra_var_A decreasing GRADUALLY, not spiking
::     - silhouette_a decreasing or staying low over 150 epochs
::
:: OUTPUT NAMING:
::   All outputs have '_fixed' inserted before '_seed0'.
::   e.g.: CIFAR100_saved_framework_lr=0.1_mal_pers_proj-ema=0.1_half=16_fixed_seed0.pth
::
:: RUNTIME ESTIMATE: ~3 runs x ~6h = ~18h total
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
echo  PHASE 23 (FIXED) -- Persistent Projection, CIFAR-100, Seed-0
echo  Bug fix: per-sample gradient normalization before EMA update
echo  Testing alpha_ema in {0.1, 0.2, 0.3}
echo  SUCCESS CRITERION: mc_best_train_top1 below 30.33%% (benign seed-0)
echo  STRETCH: below 26.97%% (GradProj one-shot result, EXP-032)
echo ============================================================
echo.

:: ==============================================================
:: RUN 1: alpha_ema = 0.1
:: ==============================================================
set EMA=0.1

echo ===== [1/6] Stage 1 -- PP (FIXED), alpha_ema=%EMA%, CIFAR-100 =====
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
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_fixed_seed%SEED%.pth"
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%.txt" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_fixed_seed%SEED%.txt"
move /Y "%CSVDIR%\separability_CIFAR100_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%.csv" ^
         "%CSVDIR%\separability_CIFAR100_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_fixed_seed%SEED%.csv"
echo [1/6] Stage 1 alpha_ema=%EMA% done.
echo.

echo ===== [2/6] Stage 2 -- Model Completion (FIXED), alpha_ema=%EMA% =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 400 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_fixed_seed%SEED%.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed %SEED%
echo [2/6] MC alpha_ema=%EMA% done.
echo  >> PASS if mc_best_train_top1 < 30.33%% (benign seed-0)
echo  >> Buggy Phase 23 got 49.42%% here (above attack 47.86%%)
echo.

:: ==============================================================
:: RUN 2: alpha_ema = 0.2
:: ==============================================================
set EMA=0.2

echo ===== [3/6] Stage 1 -- PP (FIXED), alpha_ema=%EMA%, CIFAR-100 =====
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
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_fixed_seed%SEED%.pth"
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%.txt" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_fixed_seed%SEED%.txt"
move /Y "%CSVDIR%\separability_CIFAR100_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%.csv" ^
         "%CSVDIR%\separability_CIFAR100_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_fixed_seed%SEED%.csv"
echo [3/6] Stage 1 alpha_ema=%EMA% done.
echo.

echo ===== [4/6] Stage 2 -- Model Completion (FIXED), alpha_ema=%EMA% =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 400 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_fixed_seed%SEED%.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed %SEED%
echo [4/6] MC alpha_ema=%EMA% done.
echo  >> Buggy Phase 23 got 49.44%% here
echo.

:: ==============================================================
:: RUN 3: alpha_ema = 0.3
:: ==============================================================
set EMA=0.3

echo ===== [5/6] Stage 1 -- PP (FIXED), alpha_ema=%EMA%, CIFAR-100 =====
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
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_fixed_seed%SEED%.pth"
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%.txt" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_fixed_seed%SEED%.txt"
move /Y "%CSVDIR%\separability_CIFAR100_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%.csv" ^
         "%CSVDIR%\separability_CIFAR100_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_fixed_seed%SEED%.csv"
echo [5/6] Stage 1 alpha_ema=%EMA% done.
echo.

echo ===== [6/6] Stage 2 -- Model Completion (FIXED), alpha_ema=%EMA% =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 400 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_fixed_seed%SEED%.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed %SEED%
echo [6/6] MC alpha_ema=%EMA% done.
echo  >> alpha_ema=0.3 was NOT run in buggy Phase 23; no comparison baseline here
echo.

echo ============================================================
echo  PHASE 23 (FIXED) COMPLETE
echo.
echo  Stage 2 results (look for mc_best_train_top1):
echo    alpha_ema=0.1: ..._pers_proj-ema=0.1_half=16_fixed_seed0.pth_..._nlabeled=400.txt
echo    alpha_ema=0.2: ..._pers_proj-ema=0.2_half=16_fixed_seed0.pth_..._nlabeled=400.txt
echo    alpha_ema=0.3: ..._pers_proj-ema=0.3_half=16_fixed_seed0.pth_..._nlabeled=400.txt
echo.
echo  BASELINES (CIFAR-100):
echo    Benign seed-0:      30.33%%  <- must be below this to PASS
echo    Benign 4-seed mean: 29.56%%  <- strong result if below this
echo    Attack seed-0:      47.86%%
echo    GradProj (EXP-032): 26.97%%  <- best prior result; target to beat
echo    Buggy PP:           49.42%% / 49.44%% (above attack; Phase 23 EXP-043/044)
echo.
echo  DECISION TREE:
echo    Case A -- mc < 26.97%% AND CSV shows NO one-shot intra_var_A spike:
echo      -> UNIFIED DEFENSE CONFIRMED. PP beats GradProj AND works differently.
echo      -> Run Phase 23B (seed sweep) immediately. This is the paper result.
echo.
echo    Case B -- mc < 30.33%% AND CSV shows gradual multi-epoch suppression:
echo      -> PASS (unified story holds). Not as strong as GradProj on raw MC,
echo      -> but the mechanism is correct. Run seed sweep.
echo.
echo    Case C -- mc < 30.33%% BUT CSV shows single large intra_var_A spike:
echo      -> One-shot collapse returned even with the fix. PP is collapsing
echo      -> rather than projecting persistently. This is a degenerate case.
echo      -> Still passes the gate test (MC is below benign), but the mechanism
echo      -> claim is weakened. Document and run seed sweep regardless.
echo.
echo    Case D -- mc >= 30.33%% (FAIL):
echo      -> Fixed PP fails on CIFAR-100. Options:
echo      -> (1) MDPP (K=10 directions, see Code/next_direction.md) -- next step
echo      -> (2) Fall back to GradProj for CIFAR-100 + AAP for CIFAR-10 (two-defense story)
echo.
echo  CSV DIAGNOSTICS (CIFAR100_csv_files/):
echo    intra_var_A: check if epoch-11-12 spike returns or if suppression is gradual
echo    fisher_divergence: count epochs where it exceeds tau=0.10 (want: many, not 1)
echo    inter_dist_A: should decrease under effective projection defense
echo ============================================================
echo.
pause
