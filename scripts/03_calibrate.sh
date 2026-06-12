#!/usr/bin/env bash
# scripts/03_calibrate.sh
# ─────────────────────────────────────────────────────────────────────────────
# Stage 8 — Metric Calibration Study (HHEM vs DeBERTa-NLI correlation)
# ─────────────────────────────────────────────────────────────────────────────
# Parameters (number of calibration examples, condition, scorer models) are
# read from the calibration: block in configs/experiment.yaml.
# The script auto-selects the most recent results_*.jsonl in results/.
#
# Usage
# ─────
#   bash scripts/03_calibrate.sh
#
# Optional CLI overrides:
#   bash scripts/03_calibrate.sh --eval_gpu 2
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── System paths ───────────────────────────────────────────────────────────────
# Colab: USER is unset and /scratch is not writable; default to /content.
export SCRATCH_DIR="${SCRATCH_DIR:-/content/turborag_quant}"
export HF_HOME="${HF_HOME:-/content/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

# ── Virtual environment ────────────────────────────────────────────────────────
if [[ -f "/home/${USER}/venvs/crisp/bin/activate" ]]; then
    source "/home/${USER}/venvs/crisp/bin/activate"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "[03] Using active venv: $VIRTUAL_ENV"
fi

echo "[03_calibrate] Reading all settings from configs/experiment.yaml"
python src/run_experiment.py --stages calib "$@"
