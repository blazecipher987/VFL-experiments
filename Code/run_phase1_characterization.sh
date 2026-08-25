#!/usr/bin/env bash
# Phase 1 Characterization Experiments
#
# Run these from the Code/ directory.
# Each run saves a separability_*.csv in:
#   ./saved_experiment_results/csv_files/CIFAR10_csv_files/
#
# Total wall-clock time estimate (CIFAR10, 30 epochs, GPU):
#   ~15-25 min per run × 6 runs = ~2-3 hours total
#   Tip: run conditions 1-3 first so you can check the signal early.
#
# After all runs complete, generate figures with:
#   python plot_characterization.py --dataset CIFAR10

DATASET=CIFAR10
DATAPATH=./data/CIFAR10
EPOCHS=30
HALF=16
K=4

echo "===== CONDITION 1: Benign baseline (no attack) ====="
python vfl_framework.py \
  --dataset $DATASET --path-dataset $DATAPATH \
  --epochs $EPOCHS --half $HALF --k $K \
  --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False \
  --monitor-separability True \
  --if-cluster-outputsA True

echo ""
echo "===== CONDITION 2: Active attack — Party A only (default gamma=1.0) ====="
python vfl_framework.py \
  --dataset $DATASET --path-dataset $DATAPATH \
  --epochs $EPOCHS --half $HALF --k $K \
  --use-mal-optim True --use-mal-optim-all False --use-mal-optim-top False \
  --monitor-separability True \
  --if-cluster-outputsA True

echo ""
echo "===== CONDITION 3: Active attack — all parties (gamma=1.0) ====="
python vfl_framework.py \
  --dataset $DATASET --path-dataset $DATAPATH \
  --epochs $EPOCHS --half $HALF --k $K \
  --use-mal-optim True --use-mal-optim-all True --use-mal-optim-top False \
  --monitor-separability True \
  --if-cluster-outputsA True

echo ""
echo "===== CONDITION 4: Benign baseline + DP Laplace noise defense ====="
python vfl_framework.py \
  --dataset $DATASET --path-dataset $DATAPATH \
  --epochs $EPOCHS --half $HALF --k $K \
  --use-mal-optim False --use-mal-optim-all False \
  --lap-noise True --noise-scale 1e-3 \
  --monitor-separability True \
  --if-cluster-outputsA True

echo ""
echo "===== CONDITION 5: Active attack (Party A) + DP Laplace noise defense ====="
python vfl_framework.py \
  --dataset $DATASET --path-dataset $DATAPATH \
  --epochs $EPOCHS --half $HALF --k $K \
  --use-mal-optim True --use-mal-optim-all False \
  --lap-noise True --noise-scale 1e-3 \
  --monitor-separability True \
  --if-cluster-outputsA True

echo ""
echo "===== CONDITION 6: Active attack + Gradient Compression defense ====="
python vfl_framework.py \
  --dataset $DATASET --path-dataset $DATAPATH \
  --epochs $EPOCHS --half $HALF --k $K \
  --use-mal-optim True --use-mal-optim-all False \
  --gc True --gc-preserved-percent 0.75 \
  --monitor-separability True \
  --if-cluster-outputsA True

echo ""
echo "===== All runs done. Generating plots... ====="
python plot_characterization.py --dataset $DATASET --save-dir ./saved_experiment_results
