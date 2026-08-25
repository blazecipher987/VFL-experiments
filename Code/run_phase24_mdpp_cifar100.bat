@echo off
:: ============================================================
:: Phase 24 — Multi-Direction Persistent Projection (MDPP), CIFAR-100, Seed-0
::
:: MOTIVATION
:: ----------
:: PersistentProjection (Phase 22/23) removes ONE direction per epoch.
:: For CIFAR-100 (C=100 classes, embed_dim=100), the discriminative subspace
:: has dimension 99. DCR(K=1, C=100) = 1/99 ≈ 1%% — virtually nothing.
:: MaliciousSGD can re-route attack signal through the remaining 98 directions.
::
:: This phase tests K=5, 10, 20 SVD-derived directions projected simultaneously:
::   DCR(5,100)  = 5/99  ≈  5%%
::   DCR(10,100) = 10/99 ≈ 10%%
::   DCR(20,100) = 20/99 ≈ 20%%
::
:: MECHANISM (MultiDirectionPersistentProjection in possible_defenses.py)
:: -----------------------------------------------------------------------
::   1. Train W_aux [100, 100] on current z_a every batch (same as GradProj).
::   2. SVD of W_aux → top-K right singular vectors D_inst [K, 100].
::   3. EMA + QR re-orthogonalise: D_ema stays orthonormal [K, 100].
::   4. When detected: grad = grad - (grad @ D_ema.T) @ D_ema
::      → removes ALL K components simultaneously in one matrix multiply.
::
:: SUCCESS CRITERION
:: -----------------
::   mc_best_test_top1 < 29.56%% (4-seed benign mean from Phase 16)
::   AND mc_best_test_top1 < 28.32%% (PersProjDef Phase 23 best, ema=0.2)
::   → i.e., MDPP should OUT-PERFORM Phase 23 PersistentProjection.
::
::   Benign CIFAR-100 references (Phase 16):
::     Seed-0: 30.33%%   Seed-42: 26.19%%  Seed-123: 28.56%%  Seed-456: 33.14%%
::     4-seed mean: 29.56%% ± 2.81%%
::   Attack CIFAR-100 (Phase 21, seed-0): 47.86%%
::   PersistentProj best (Phase 23, ema=0.2, seed-0): 28.32%% test MC
::
:: HOW TO READ RESULTS
:: -------------------
::   For each K, open the Stage 2 .txt file and find:
::     "top 1 accuracy" on the TESTING dataset — scan across 25 epochs,
::     take the peak test accuracy (the defense is working if this < benign).
::   Also check the CSV:
::     fisher_divergence: should cross tau=0.10 in MULTIPLE epochs (not just 1).
::     intra_var_A: must NOT spike 6+ orders of magnitude (no one-shot collapse).
::
:: DECISION TREE
:: -------------
::   Case A (K=10 or K=20 MC < Phase 23 MC=28.32%%):
::     → MDPP is strictly better than PersistentProj. Use K=10 as the
::       canonical CIFAR-100 defense. Run Phase 24B (seed sweep).
::   Case B (similar MC across K values, all < benign):
::     → DCR coverage not the bottleneck; something else limits defense.
::       Investigate if D_ema converges to same directions as K=1 case.
::   Case C (MC above benign reference):
::     → Defense weaker than Phase 23 PP. Possible: SVD directions from W_aux
::       mismatch the actual gradient directions that MaliciousSGD exploits.
::       Consider projecting BOTH W_aux directions AND d_aux gradient directions.
::
:: OUTPUTS (per K in {5, 10, 20}):
::   Stage 1 pth: CIFAR100_saved_framework_lr=0.1_mal_mdpp-k={K}-ema=0.2_half=16_seed0.pth
::   Stage 1 txt: CIFAR100_saved_framework_lr=0.1_mal_mdpp-k={K}-ema=0.2_half=16_seed0.txt
::   Stage 2 txt: model_completion_CIFAR100_..._mdpp-k={K}-ema=0.2_half=16_seed0.pth_..._nlabeled=400.txt
::   CSV:         separability_CIFAR100_lr=0.1_mal_mdpp-k={K}-ema=0.2_half=16_seed0.csv
::
:: RUNTIME ESTIMATE: 3 runs x ~6h = ~18h (same as Phase 23)
:: ============================================================

set DATASET=CIFAR100
set DATAPATH=.\data\CIFAR100
set EPOCHS=150
set HALF=16
set K_DATA=5
set SEED=0

set EMA=0.2
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
echo  PHASE 24 -- Multi-Direction Persistent Projection, CIFAR-100, Seed-0
echo  Testing K-directions in {5, 10, 20}, alpha_ema=0.2
echo  SUCCESS: mc_best_test_top1 below 28.32%% (Phase 23 PersistentProj best)
echo  AND below 29.56%% (4-seed benign mean, Phase 16)
echo ============================================================
echo.

:: ==============================================================
:: RUN 1: K = 5  (DCR ≈ 5%%)
:: ==============================================================
set KDIR=5

echo ===== [1/6] Stage 1 -- MDPP K=%KDIR% =====
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K_DATA% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --mdpp True ^
  --mdpp-k-directions %KDIR% ^
  --mdpp-alpha-ema %EMA% ^
  --mdpp-burn-in %BURNIN% ^
  --gradient-proj-embedding-dim %EMBED_DIM% ^
  --gradient-proj-num-classes %NUM_CLASSES% ^
  --gradient-proj-lr %PROJ_LR% ^
  --asymmetric-tau %TAU% ^
  --if-cluster-outputsA True ^
  --stone1 75 --stone2 120 ^
  --manual-seed %SEED%

move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_mdpp-k=%KDIR%-ema=%EMA%_half=%HALF%.pth" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_mdpp-k=%KDIR%-ema=%EMA%_half=%HALF%_seed%SEED%.pth"
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_mdpp-k=%KDIR%-ema=%EMA%_half=%HALF%.txt" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_mdpp-k=%KDIR%-ema=%EMA%_half=%HALF%_seed%SEED%.txt"
move /Y "%CSVDIR%\separability_CIFAR100_lr=0.1_mal_mdpp-k=%KDIR%-ema=%EMA%_half=%HALF%.csv" ^
         "%CSVDIR%\separability_CIFAR100_lr=0.1_mal_mdpp-k=%KDIR%-ema=%EMA%_half=%HALF%_seed%SEED%.csv"
echo [1/6] Stage 1 K=%KDIR% done.
echo.

echo ===== [2/6] Stage 2 -- Model Completion, K=%KDIR% =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 400 --party-num 2 --half %HALF% --k %K_DATA% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_mdpp-k=%KDIR%-ema=%EMA%_half=%HALF%_seed%SEED%.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed %SEED%
echo [2/6] MC K=%KDIR% done.
echo  >> Scan testing top-1 across 25 epochs; take peak. Target: below 28.32%%
echo.

:: ==============================================================
:: RUN 2: K = 10  (DCR ≈ 10%%)
:: ==============================================================
set KDIR=10

echo ===== [3/6] Stage 1 -- MDPP K=%KDIR% =====
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K_DATA% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --mdpp True ^
  --mdpp-k-directions %KDIR% ^
  --mdpp-alpha-ema %EMA% ^
  --mdpp-burn-in %BURNIN% ^
  --gradient-proj-embedding-dim %EMBED_DIM% ^
  --gradient-proj-num-classes %NUM_CLASSES% ^
  --gradient-proj-lr %PROJ_LR% ^
  --asymmetric-tau %TAU% ^
  --if-cluster-outputsA True ^
  --stone1 75 --stone2 120 ^
  --manual-seed %SEED%

move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_mdpp-k=%KDIR%-ema=%EMA%_half=%HALF%.pth" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_mdpp-k=%KDIR%-ema=%EMA%_half=%HALF%_seed%SEED%.pth"
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_mdpp-k=%KDIR%-ema=%EMA%_half=%HALF%.txt" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_mdpp-k=%KDIR%-ema=%EMA%_half=%HALF%_seed%SEED%.txt"
move /Y "%CSVDIR%\separability_CIFAR100_lr=0.1_mal_mdpp-k=%KDIR%-ema=%EMA%_half=%HALF%.csv" ^
         "%CSVDIR%\separability_CIFAR100_lr=0.1_mal_mdpp-k=%KDIR%-ema=%EMA%_half=%HALF%_seed%SEED%.csv"
echo [3/6] Stage 1 K=%KDIR% done.
echo.

echo ===== [4/6] Stage 2 -- Model Completion, K=%KDIR% =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 400 --party-num 2 --half %HALF% --k %K_DATA% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_mdpp-k=%KDIR%-ema=%EMA%_half=%HALF%_seed%SEED%.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed %SEED%
echo [4/6] MC K=%KDIR% done.
echo  >> Scan testing top-1 across 25 epochs; take peak. Target: below 28.32%%
echo.

:: ==============================================================
:: RUN 3: K = 20  (DCR ≈ 20%%)
:: ==============================================================
set KDIR=20

echo ===== [5/6] Stage 1 -- MDPP K=%KDIR% =====
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K_DATA% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --mdpp True ^
  --mdpp-k-directions %KDIR% ^
  --mdpp-alpha-ema %EMA% ^
  --mdpp-burn-in %BURNIN% ^
  --gradient-proj-embedding-dim %EMBED_DIM% ^
  --gradient-proj-num-classes %NUM_CLASSES% ^
  --gradient-proj-lr %PROJ_LR% ^
  --asymmetric-tau %TAU% ^
  --if-cluster-outputsA True ^
  --stone1 75 --stone2 120 ^
  --manual-seed %SEED%

move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_mdpp-k=%KDIR%-ema=%EMA%_half=%HALF%.pth" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_mdpp-k=%KDIR%-ema=%EMA%_half=%HALF%_seed%SEED%.pth"
move /Y "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_mdpp-k=%KDIR%-ema=%EMA%_half=%HALF%.txt" ^
         "%MODELSDIR%\CIFAR100_saved_framework_lr=0.1_mal_mdpp-k=%KDIR%-ema=%EMA%_half=%HALF%_seed%SEED%.txt"
move /Y "%CSVDIR%\separability_CIFAR100_lr=0.1_mal_mdpp-k=%KDIR%-ema=%EMA%_half=%HALF%.csv" ^
         "%CSVDIR%\separability_CIFAR100_lr=0.1_mal_mdpp-k=%KDIR%-ema=%EMA%_half=%HALF%_seed%SEED%.csv"
echo [5/6] Stage 1 K=%KDIR% done.
echo.

echo ===== [6/6] Stage 2 -- Model Completion, K=%KDIR% =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 400 --party-num 2 --half %HALF% --k %K_DATA% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR100_saved_framework_lr=0.1_mal_mdpp-k=%KDIR%-ema=%EMA%_half=%HALF%_seed%SEED%.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed %SEED%
echo [6/6] MC K=%KDIR% done.
echo.

echo ============================================================
echo  PHASE 24 COMPLETE
echo.
echo  Compare peak testing top-1 across the three Stage 2 files:
echo    K=5:  model_completion_CIFAR100_..._mdpp-k=5-ema=0.2_half=16_seed0.pth_..._nlabeled=400.txt
echo    K=10: model_completion_CIFAR100_..._mdpp-k=10-ema=0.2_half=16_seed0.pth_..._nlabeled=400.txt
echo    K=20: model_completion_CIFAR100_..._mdpp-k=20-ema=0.2_half=16_seed0.pth_..._nlabeled=400.txt
echo.
echo  REFERENCE BASELINES:
echo    Benign seed-0 (Phase 16):     30.33%%
echo    4-seed benign mean (Phase 16): 29.56%% ± 2.81%%
echo    Attack seed-0 (Phase 21):     47.86%%
echo    PersistentProj ema=0.2 (Phase 23, seed-0): 28.32%%   <-- beat this
echo.
echo  DECISION TREE:
echo    Case A (any K gives MC < 28.32%% AND < benign):
echo      → MDPP beats PersistentProj. Take best K for Phase 24B seed sweep.
echo      → Report DCR(K,C) as the key theoretical insight.
echo    Case B (MC plateau across K, all < benign):
echo      → K coverage not the bottleneck. Investigate direction alignment.
echo    Case C (MC above benign for all K):
echo      → SVD directions miss the attack subspace. Try projecting d_aux
echo        gradient directions alongside W_aux SVD directions.
echo.
echo  CSV DIAGNOSTICS (in CIFAR100_csv_files/):
echo    fisher_divergence: must cross tau=0.10 multiple epochs (not one-shot).
echo    intra_var_A: must NOT jump 6+ orders of magnitude.
echo    inter_dist_A: should decrease as K increases if MDPP is working.
echo ============================================================
echo.
pause
