#!/usr/bin/env bash
# scripts/01_build_chunk_cache.sh
# ─────────────────────────────────────────────────────────────────────────────
# Stage 1+2 — Build offline KV caches from Wikipedia passages
# ─────────────────────────────────────────────────────────────────────────────
# All parameters (model path, GPU, chunk size, number of wiki passages, etc.)
# are read directly from configs/experiment.yaml — edit that file only.
#
# The only values set here are the two system-level paths that must exist as
# real environment variables so that config.py can expand ${SCRATCH_DIR} and
# ${HF_HOME} when it loads the YAML.
#
# Usage
# ─────
#   bash scripts/01_build_chunk_cache.sh
#
# Optional CLI overrides (passed through to run_experiment.py):
#   bash scripts/01_build_chunk_cache.sh --model_name /new/path --cache_gpu 2
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── System paths ───────────────────────────────────────────────────────────────
# Required for ${SCRATCH_DIR} / ${HF_HOME} expansion inside experiment.yaml.
# Everything else (model, GPU, chunk size, wiki_dpr config …) lives in the YAML.
export SCRATCH_DIR="${SCRATCH_DIR:-/scratch/${USER}/turborag_quant}"
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

# ── Virtual environment ────────────────────────────────────────────────────────
if [[ -f "/home/${USER}/venvs/crisp/bin/activate" ]]; then
    source "/home/${USER}/venvs/crisp/bin/activate"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "[01] Using active venv: $VIRTUAL_ENV"
else
    echo "[01] WARNING: no virtual environment detected; using system Python"
fi

echo "[01_build_chunk_cache] Reading all settings from configs/experiment.yaml"
python src/run_experiment.py --stages build "$@"
