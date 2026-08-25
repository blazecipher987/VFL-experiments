@echo off
:: ============================================================
:: Phase 22 (FIXED) — Persistent Projection Defense, CIFAR-10, Seed-0
::
:: WHY THIS EXISTS:
::   Phase 22 (EXP-040/041/042, 2026-07-13) failed because
::   PersistentProjectionDefense computed the EMA direction as:
::
::     d_mean = d_inst.mean(dim=0)        <- near-zero for balanced batches
::     d_mean_norm = d_mean / ||d_mean||  <- normalizing noise → random direction
::
::   For cross-entropy loss on a balanced batch, per-sample gradients
::   partially cancel when averaged across classes, producing a near-zero
::   vector. Normalizing it gives a random unit vector, so d_ema tracked
::   noise instead of the discriminative direction. The projection removed
::   random noise rather than the label-information-bearing component.
::
::   FIX (applied to possible_defenses.py, 2026-07-13):
::     d_inst_norm = d_inst / ||d_inst||  (per-sample, dim=-1)
::     d_mean      = d_inst_norm.mean(dim=0)   <- mean of unit vectors, no cancellation
::     d_mean_norm = d_mean / ||d_mean||
::
:: WHAT TO EXPECT (vs Phase 22 buggy results):
::   Buggy results:  mc_best_train_top1 = 92.75 - 94.50%  (FAIL; benign = 87.23%)
::   Target:         mc_best_train_top1 < 87.23%  (PASS)
::   Stretch target: mc_best_train_top1 < 83.11%  (4-seed benign mean; strong)
::
::   The epoch-50 discriminability jump seen in the buggy CSVs may or may not
::   persist after the fix. If it persists AND mc still fails, that jump is a
::   genuine signal to investigate (not just a bug artifact).
::
:: OUTPUT NAMING:
::   All outputs have '_fixed' inserted before '_seed0' to avoid overwriting
::   Phase 22 buggy results (kept for comparison).
::   e.g.: CIFAR10_saved_framework_lr=0.1_mal_pers_proj-ema=0.1_half=16_fixed_seed0.pth
::
:: DO NOT run Phase 23-fixed until at least one alpha_ema here PASSES the gate.
::
:: RUNTIME ESTIMATE: ~3 runs x ~2.5h = ~7.5h total
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
echo  PHASE 22 (FIXED) -- Persistent Projection, CIFAR-10, Seed-0
echo  Bug fix: per-sample gradient normalization before EMA update
echo  Testing alpha_ema in {0.1, 0.2, 0.3}
echo  SUCCESS CRITERION: mc_best_train_top1 below 87.23%% (benign seed-0)
echo ============================================================
echo.

:: ==============================================================
:: RUN 1: alpha_ema = 0.1
:: ==============================================================
set EMA=0.1

echo ===== [1/6] Stage 1 -- PP (FIXED), alpha_ema=%EMA% =====
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
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_fixed_seed%SEED%.pth"
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%.txt" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_fixed_seed%SEED%.txt"
move /Y "%CSVDIR%\separability_CIFAR10_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%.csv" ^
         "%CSVDIR%\separability_CIFAR10_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_fixed_seed%SEED%.csv"
echo [1/6] Stage 1 alpha_ema=%EMA% done.
echo.

echo ===== [2/6] Stage 2 -- Model Completion (FIXED), alpha_ema=%EMA% =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 40 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_fixed_seed%SEED%.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed %SEED%
echo [2/6] MC alpha_ema=%EMA% done.
echo  >> PASS if mc_best_train_top1 < 87.23%% (benign seed-0)
echo  >> Buggy Phase 22 got 93.73%% -- anything lower is improvement
echo.

:: ==============================================================
:: RUN 2: alpha_ema = 0.2
:: ==============================================================
set EMA=0.2

echo ===== [3/6] Stage 1 -- PP (FIXED), alpha_ema=%EMA% =====
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
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_fixed_seed%SEED%.pth"
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%.txt" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_fixed_seed%SEED%.txt"
move /Y "%CSVDIR%\separability_CIFAR10_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%.csv" ^
         "%CSVDIR%\separability_CIFAR10_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_fixed_seed%SEED%.csv"
echo [3/6] Stage 1 alpha_ema=%EMA% done.
echo.

echo ===== [4/6] Stage 2 -- Model Completion (FIXED), alpha_ema=%EMA% =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 40 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_fixed_seed%SEED%.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed %SEED%
echo [4/6] MC alpha_ema=%EMA% done.
echo  >> PASS if mc_best_train_top1 < 87.23%% (benign seed-0)
echo  >> Buggy Phase 22 got 92.75%% here
echo.

:: ==============================================================
:: RUN 3: alpha_ema = 0.3
:: ==============================================================
set EMA=0.3

echo ===== [5/6] Stage 1 -- PP (FIXED), alpha_ema=%EMA% =====
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
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_fixed_seed%SEED%.pth"
move /Y "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%.txt" ^
         "%MODELSDIR%\CIFAR10_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_fixed_seed%SEED%.txt"
move /Y "%CSVDIR%\separability_CIFAR10_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%.csv" ^
         "%CSVDIR%\separability_CIFAR10_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_fixed_seed%SEED%.csv"
echo [5/6] Stage 1 alpha_ema=%EMA% done.
echo.

echo ===== [6/6] Stage 2 -- Model Completion (FIXED), alpha_ema=%EMA% =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% --dataset-path %DATAPATH% ^
  --n-labeled 40 --party-num 2 --half %HALF% --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name CIFAR10_saved_framework_lr=0.1_mal_pers_proj-ema=%EMA%_half=%HALF%_fixed_seed%SEED%.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1 --manualSeed %SEED%
echo [6/6] MC alpha_ema=%EMA% done.
echo  >> Buggy Phase 22 got 94.50%% here
echo.

echo ============================================================
echo  PHASE 22 (FIXED) COMPLETE
echo.
echo  Stage 2 results to check (look for mc_best_train_top1):
echo    alpha_ema=0.1: ..._pers_proj-ema=0.1_half=16_fixed_seed0.pth_..._nlabeled=40.txt
echo    alpha_ema=0.2: ..._pers_proj-ema=0.2_half=16_fixed_seed0.pth_..._nlabeled=40.txt
echo    alpha_ema=0.3: ..._pers_proj-ema=0.3_half=16_fixed_seed0.pth_..._nlabeled=40.txt
echo.
echo  BASELINES:
echo    Benign (seed-0):    87.23%%  <- must be below this to PASS
echo    Attack (seed-0):    95.42%%
echo    AAP defense (mean): 81.80%%  <- this is what we want to match or beat
echo    Buggy PP results:   92.75 - 94.50%%  (Phase 22, all failed)
echo.
echo  DECISION TREE:
echo    Any alpha_ema gives MC < 87.23%%:
echo      -> PASS gate test. Run Phase 23-fixed (CIFAR-100).
echo      -> If MC < 81.80%%: PP beats AAP; PP becomes the sole defense mechanism.
echo      -> If MC 81.80 - 87.23%%: PP passes but is weaker than AAP; run seed sweep.
echo    All alpha_ema give MC >= 87.23%%:
echo      -> FAIL. Check CSVs.
echo      -> If intra_var_A is still NOT collapsing and MC is still near attack:
echo           The per-sample normalization alone is insufficient.
echo           Pivot to MDPP (K=10 principal directions) per Code/next_direction.md.
echo      -> If intra_var_A IS collapsing (spike) but MC still fails:
echo           One-shot collapse returned; the EMA didn't stabilize direction fast enough.
echo           Try burn_in=8 and alpha_ema=0.05 (slower EMA) in a Phase 22B-fixed run.
echo.
echo  CSV DIAGNOSTICS (compare to buggy Phase 22 CSVs):
echo    intra_var_A: should no longer reach 0.008-0.017 by epoch 99 (was very tight
echo      under buggy PP; correct PP should keep clusters more diffuse)
echo    silhouette_a: should stay low / negative (not rise to +0.30 like buggy runs)
echo    fisher_divergence: watch whether the epoch-50 jump persists or disappears;
echo      if it disappears, the jump was a bug artifact; if it persists, it is real
echo      and needs investigation as a separate mechanism
echo    grad_norm_ratio: should remain clearly below 1.0 after burn_in if projection
echo      is removing a meaningful component (was only slightly below 1.0 in buggy runs)
echo ============================================================
echo.
pause
