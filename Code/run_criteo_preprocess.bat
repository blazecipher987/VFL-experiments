@echo off
:: ============================================================
:: Criteo Preprocessing — train.txt → criteo.csv
::
:: Input:
::   .\data\kaggle-display-advertising-challenge-dataset\train.txt
::   (Kaggle Display Advertising Challenge, tab-separated, ~45M rows)
::
:: Output:
::   .\data\kaggle-display-advertising-challenge-dataset\criteo.csv
::   100,000 rows × 8193 columns (8192 hashed features + label)
::   ~3 GB on disk
::
:: Runtime: ~15-30 minutes (CPU-bound, pure Python row-by-row hashing)
::
:: NOTE: Run this ONCE before any Criteo VFL experiments.
::       The criteo.csv output is the --path-dataset argument for all
::       vfl_framework.py and model_completion.py Criteo runs.
::
:: NOTE: test.txt from the Kaggle dataset has NO LABELS (competition blind set).
::       The code only uses train.txt and does its own 80/20 split internally.
:: ============================================================

echo.
echo ============================================================
echo  Criteo Preprocessing: train.txt ^-^> criteo.csv
echo  Input : .\data\kaggle-display-advertising-challenge-dataset\train.txt
echo  Output: .\data\Criteo\criteo.csv
echo  Size  : ~3 GB   Runtime: ~15-30 min
echo ============================================================
echo.

python datasets_preprocess\criteo_preprocess.py

echo.
echo ============================================================
echo  Preprocessing complete.
echo  criteo.csv is ready at:
echo  .\data\Criteo\criteo.csv
echo.
echo  Next step: run run_phase_criteo_full.bat
echo ============================================================
echo.
pause
