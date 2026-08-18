#!/usr/bin/env bash
# Automated overnight batch runner for 4 physics ablation experiments
set -e

# Navigate to pump2d_smart directory
cd "$(dirname "$0")"

echo "=========================================================================="
echo "STARTING OVERNIGHT BATCH RUN OF 4 ABLATION EXPERIMENTS"
echo "Directory: $(pwd)"
echo "Python environment: uv run"
echo "=========================================================================="

export MLFLOW_ALLOW_FILE_STORE=true
PYTHONPATH=./src:../smart/smart uv run python run_experiments.py

echo "=========================================================================="
echo "ALL EXPERIMENTS SUCCESSFULLY COMPLETED!"
echo "=========================================================================="
