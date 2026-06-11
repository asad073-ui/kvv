#!/usr/bin/env bash
# scripts/run_full_pipeline.sh
# ─────────────────────────────────────────────────────────────────────────────
# Full end-to-end pipeline — all four stages in order
# ─────────────────────────────────────────────────────────────────────────────
# Stage 01 (build)   – Download wiki passages + build FP16/INT8/INT4 KV caches
# Stage 02 (eval)    – Evaluate C0–C3 × K × datasets (questions from HF Hub)
# Stage 03 (calib)   – HHEM vs DeBERTa-NLI correlation on 50 examples
# Stage 04 (analyze) – H1/H2/H3 hypothesis tests + figure CSVs
#
# All parameters come from configs/experiment.yaml — edit that file only.
# GPU assignments: gpu.chunk_cache_gpu (stage 01) and gpu.evaluate_gpu (02+03).
#
# Usage
# ─────
#   bash scripts/run_full_pipeline.sh
#
# Optional CLI overrides (passed through to run_experiment.py):
#   bash scripts/run_full_pipeline.sh --model_name /new/path --cache_gpu 0 --eval_gpu 1
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
    echo "[pipeline] Using active venv: $VIRTUAL_ENV"
else
    echo "[pipeline] WARNING: no virtual environment detected; using system Python"
fi

echo "[run_full_pipeline] Reading all settings from configs/experiment.yaml"
python src/run_experiment.py --stages all "$@"
