#!/usr/bin/env bash
# scripts/02_evaluate.sh
# ─────────────────────────────────────────────────────────────────────────────
# Stages 3–7 — Evaluate all conditions × K values × datasets
# ─────────────────────────────────────────────────────────────────────────────
# All parameters (model, GPU, datasets, K values, conditions, HF dataset
# names, faithfulness scorers …) are read from configs/experiment.yaml.
#
# To change anything — number of examples, which conditions to run, which
# datasets, the evaluation GPU — edit configs/experiment.yaml.
#
# Usage
# ─────
#   bash scripts/02_evaluate.sh
#
# Optional CLI overrides:
#   bash scripts/02_evaluate.sh --model_name /new/path --eval_gpu 2
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── System paths ───────────────────────────────────────────────────────────────
export SCRATCH_DIR="${SCRATCH_DIR:-/scratch/${USER}/turborag_quant}"
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

# ── Virtual environment ────────────────────────────────────────────────────────
if [[ -f "/home/${USER}/venvs/crisp/bin/activate" ]]; then
    source "/home/${USER}/venvs/crisp/bin/activate"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "[02] Using active venv: $VIRTUAL_ENV"
fi

echo "[02_evaluate] Reading all settings from configs/experiment.yaml"
python src/run_experiment.py --stages eval "$@"
