@echo off
:: ============================================================
:: Phase 17 -- Yahoo Answers: Modality Generalization Test
::
:: WHY THIS EXISTS:
::   The defense has been validated on two IMAGE datasets (CIFAR-10
::   4-seed, CINIC10L 1-seed). If it also works on a TEXT dataset
::   (Yahoo Answers, 10-class), the paper can claim modality
::   independence -- the Fisher divergence detection is not an
::   image-specific artifact.
::
::   Yahoo Answers (10 classes: society, science, health, education,
::   computers, sports, business, entertainment, relationship, politics)
::   is a text classification dataset. Each sample is split between
::   Party A (first text segment) and Party B (second segment) by the
::   read_data_text module -- NOT by the --half pixel-slice mechanism.
::   The --half=16 arg is passed but IGNORED by the Yahoo model
::   architecture; it only appears in the checkpoint filename.
::
:: !! INFRASTRUCTURE WARNING !!
::   Yahoo has never been run in this codebase before (no saved
::   models exist). This bat file is the first test. Potential failure
::   modes:
::     (A) MixText/BERT model load fails -- check transformers version
::     (B) read_data_text data loading is slow (~minutes for first batch)
::     (C) GPU OOM if batch size too large for BERT -- reduce --batch-size
::     (D) Checkpoint naming mismatch -- lr=0.001 NOT lr=0.1 (Yahoo uses
::         lr=1e-3, so filename has _lr=0.001_ not _lr=0.1_)
::
::   If Stage 1 benign fails, check the error and debug before continuing.
::   If Stage 1 benign+attack both succeed and show an attack advantage
::   in their VFL accuracy txt files, then Stage 1 defense + all Stage 2
::   runs are safe to continue.
::
:: CRITICAL: Yahoo LR = 1e-3, NOT 1e-1
::   Image datasets use lr=0.1. Yahoo uses BERT-based MixText, which
::   needs lr=0.001. Using 0.1 will cause divergence. This is set below.
::
:: PARAMETERS:
::   EPOCHS    : 100 (same as CIFAR-10 -- adjust if convergence is slow)
::   HALF      : 16  (ignored by Yahoo model; used only in filename)
::   K         : 4   (top-4 for 10-class, consistent with CIFAR-10)
::   LR        : 1e-3  !! DIFFERENT from image datasets !!
::   n_labeled : 40  (4 per class x 10 classes, same as CIFAR-10)
::   STONE1    : 50  (LR decay milestone, same as default)
::   STONE2    : 85  (LR decay milestone, same as default)
::
:: CHECKPOINT NAMES (note lr=0.001 in filename):
::   Benign : Yahoo_saved_framework_lr=0.001_normal_half=16.pth
::   Attack : Yahoo_saved_framework_lr=0.001_mal_half=16.pth
::   Defense: Yahoo_saved_framework_lr=0.001_mal_asym_def-a=1.0-t=0.1-b=8_half=16.pth
::
:: DATA PATH:
::   .\datasets\yahoo_answers_csv\  (train.csv + test.csv + classes.txt)
::   NOTE: trailing backslash is required -- read_data_text.py concatenates
::   the path and filename with no separator: data_path+'train.csv'.
::   .\datasets\ not .\data\ -- Yahoo is in the datasets source folder.
::
:: SUCCESS CRITERION (same as all other datasets):
::   Defended MC < Benign MC
:: ============================================================

set DATASET=Yahoo
set DATAPATH=.\datasets\yahoo_answers_csv\
set EPOCHS=100
set HALF=16
set K=4
set LR=0.001
set STONE1=50
set STONE2=85

set TAU=0.10
set ALPHA=1.0
set BURNIN=8

set MODELSDIR=.\saved_experiment_results\saved_models\Yahoo_saved_models
set RESUMEDIR=.\saved_experiment_results\saved_models\

echo.
echo ============================================================
echo  PHASE 17 -- Yahoo Answers Modality Generalization Test
echo  LR=0.001 (BERT-based model -- NOT the usual 0.1!)
echo  Data: .\datasets\yahoo_answers_csv\
echo ============================================================
echo.
echo  WARNING: This is the FIRST run of Yahoo in this codebase.
echo  If Stage 1 benign errors out, stop and debug before continuing.
echo  Common issues: transformers version, OOM, data path.
echo.

:: ----------------------------------------------------------
:: STAGE 1, CONDITION 1: Benign, 100 epochs
::
:: Run this first as an infrastructure smoke test. If it
:: completes and produces a .pth and .txt file, proceed.
:: ----------------------------------------------------------
echo ===== [1/3] Stage 1 -- Benign 100ep =====
echo Saves to: Yahoo_saved_framework_lr=0.001_normal_half=16.pth
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --lr %LR% --stone1 %STONE1% --stone2 %STONE2% ^
  --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False

echo.
echo [1/3] done -- check for Yahoo_saved_framework_lr=0.001_normal_half=16.pth
echo.

:: ----------------------------------------------------------
:: STAGE 1, CONDITION 2: Active attack, no defense, 100 epochs
::
:: Expected: VFL accuracy similar to benign (text not pixel-split,
:: so MaliciousSGD operates on the MixText embedding gradients).
:: Attack advantage will be visible in Stage 2 model completion.
:: ----------------------------------------------------------
echo ===== [2/3] Stage 1 -- Active no defense 100ep =====
echo Saves to: Yahoo_saved_framework_lr=0.001_mal_half=16.pth
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --lr %LR% --stone1 %STONE1% --stone2 %STONE2% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True

echo.
echo [2/3] done.
echo.

:: ----------------------------------------------------------
:: STAGE 1, CONDITION 3: Active attack + defense, 100 epochs
::
:: Same defense params as CIFAR-10 (alpha=1.0, tau=0.10, burn_in=8).
:: Fisher divergence threshold tau=0.10 is designed to be above the
:: benign Fisher ceiling. For Yahoo, the benign Fisher ceiling is
:: unknown -- if divergence is much larger or smaller, tau may need
:: tuning. But start with the same params for direct comparability.
:: ----------------------------------------------------------
echo ===== [3/3] Stage 1 -- Active + Defense 100ep =====
echo Saves to: Yahoo_saved_framework_lr=0.001_mal_asym_def-a=1.0-t=0.1-b=8_half=16.pth
echo.
python vfl_framework.py ^
  --dataset %DATASET% --path-dataset %DATAPATH% ^
  --epochs %EPOCHS% --half %HALF% --k %K% ^
  --lr %LR% --stone1 %STONE1% --stone2 %STONE2% ^
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False ^
  --monitor-separability True ^
  --asymmetric-defense True ^
  --asymmetric-alpha %ALPHA% ^
  --asymmetric-tau %TAU% ^
  --asymmetric-burn-in %BURNIN% ^
  --if-cluster-outputsA True

echo.
echo [3/3] done.
echo.

echo ============================================================
echo  Stage 1 complete. Starting Stage 2 model completion...
echo ============================================================
echo.

:: ----------------------------------------------------------
:: STAGE 2, CONDITION 4: Benign -- model completion
:: ----------------------------------------------------------
echo ===== [4/6] Stage 2 -- Benign MC =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% ^
  --dataset-path %DATAPATH% ^
  --n-labeled 40 ^
  --party-num 2 ^
  --half %HALF% ^
  --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name Yahoo_saved_framework_lr=0.001_normal_half=16.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.
echo [4/6] done.
echo.

:: ----------------------------------------------------------
:: STAGE 2, CONDITION 5: Active no defense -- model completion
:: ----------------------------------------------------------
echo ===== [5/6] Stage 2 -- Active no defense MC (ATTACK BASELINE) =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% ^
  --dataset-path %DATAPATH% ^
  --n-labeled 40 ^
  --party-num 2 ^
  --half %HALF% ^
  --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name Yahoo_saved_framework_lr=0.001_mal_half=16.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.
echo [5/6] done.
echo.

:: ----------------------------------------------------------
:: STAGE 2, CONDITION 6: Active + defense -- model completion
:: ----------------------------------------------------------
echo ===== [6/6] Stage 2 -- Active + Defense MC (KEY RESULT) =====
echo.
python model_completion.py ^
  --dataset-name %DATASET% ^
  --dataset-path %DATAPATH% ^
  --n-labeled 40 ^
  --party-num 2 ^
  --half %HALF% ^
  --k %K% ^
  --resume-dir %RESUMEDIR% ^
  --resume-name Yahoo_saved_framework_lr=0.001_mal_asym_def-a=1.0-t=0.1-b=8_half=16.pth ^
  --num-layer 1 --activation_func_type ReLU --use-bn True ^
  --epochs 25 --print-to-txt 1

echo.
echo [6/6] done.
echo.

echo ============================================================
echo  Phase 17 COMPLETE -- Results in:
echo  %MODELSDIR%\
echo.
echo  [4] model_completion_..._normal_half=16.pth..._nlabeled=40.txt
echo      (benign baseline -- what passive party achieves)
echo.
echo  [5] model_completion_..._mal_half=16.pth..._nlabeled=40.txt
echo      (attack -- should be well above [4] if MaliciousSGD works on text)
echo.
echo  [6] model_completion_..._mal_asym_def-a=1.0-t=0.1-b=8_half=16.pth..._nlabeled=40.txt
echo      (defended -- KEY RESULT: should be BELOW [4] for defense to work)
echo.
echo  PAPER CLAIM holds if [5] >> [4] and [6] < [4].
echo  If [5] ~= [4] (no attack advantage on text), Yahoo is uninformative.
echo  If [6] > [4] (defense fails), need tau/alpha tuning for Yahoo.
echo.
echo  NOTE: Check the VFL accuracy .txt files in %MODELSDIR%\ first.
echo  The benign/attack training accuracy gap shows if MaliciousSGD
echo  is creating discriminative text embeddings as expected.
echo ============================================================
echo.
pause
