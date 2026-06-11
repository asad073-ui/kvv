#!/usr/bin/env bash
# scripts/04_analyze.sh
# ─────────────────────────────────────────────────────────────────────────────
# Stages 9–11 — Hypothesis Analysis + Figure CSVs (CPU-only)
# ─────────────────────────────────────────────────────────────────────────────
# Output directory is read from paths.analysis_dir in configs/experiment.yaml.
# The script auto-selects the most recent summary_*.json in results/.
#
# Usage
# ─────
#   bash scripts/04_analyze.sh
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
    echo "[04] Using active venv: $VIRTUAL_ENV"
fi

echo "[04_analyze] Reading all settings from configs/experiment.yaml"
python src/run_experiment.py --stages analyze "$@"
