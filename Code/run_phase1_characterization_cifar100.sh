#!/usr/bin/env bash
# Phase 1 Characterization Experiments — CIFAR100
#
# Run these from the Code/ directory:
#   bash run_phase1_characterization_cifar100.sh
#
# Each run saves a separability_*.csv in:
#   ./saved_experiment_results/csv_files/CIFAR100_csv_files/
#
# Wall-clock estimate (CIFAR100, 30 epochs, GPU):
#   ~25-40 min per run × 6 runs = ~3-4 hours total
#   Run conditions 1-3 first to verify the core signal before running 4-6.
#
# NOTE: Conditions 4 and 5 (Laplace DP) showed training instability on CIFAR10
#   (gradient collapse in one party).  Run them to characterize the behaviour,
#   but treat those CSVs with caution — the Fisher metrics may be artifacts.

DATASET=CIFAR100
DATAPATH=./data/CIFAR100
EPOCHS=30
HALF=16
K=5

echo "===== CONDITION 1: Benign baseline (no attack) ====="
python vfl_framework.py \
  --dataset $DATASET --path-dataset $DATAPATH \
  --epochs $EPOCHS --half $HALF --k $K \
  --use-mal-optim False --use-mal-optim-all False --use-mal-optim-top False \
  --monitor-separability True \
  --if-cluster-outputsA True

echo ""
echo "===== CONDITION 2: Active attack — Party A only (gamma=1.0) ====="
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
