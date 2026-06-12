#!/usr/bin/env bash
# scripts/mve.sh
# ─────────────────────────────────────────────────────────────────────────────
# Minimum Viable Experiment — build + evaluate + analyze in one shot
# ─────────────────────────────────────────────────────────────────────────────
# Activates MVE mode from the mve: block in configs/experiment.yaml:
#   mve.datasets     ["hotpotqa"]   (HotpotQA from HF; corpus = its own paragraphs)
#   mve.num_examples 100             (examples per dataset)
#   mve.k_values     [1, 3, 5]
#
# Conditions evaluated: all four from the YAML conditions: list — C0 (Gold Oracle),
# C1 (FP16), C2 (INT8), C3 (INT4). Edit conditions: in the YAML to change this.
#
# Sufficient to test H1 (asymmetric degradation) and H2 (chunk amplification).
#
# Usage
# ─────
#   bash scripts/mve.sh
#
# Optional CLI overrides:
#   bash scripts/mve.sh --model_name /new/path --cache_gpu 0 --eval_gpu 1
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
    echo "[MVE] Using active venv: $VIRTUAL_ENV"
else
    echo "[MVE] WARNING: no virtual environment detected; using system Python"
fi

echo "[mve] Reading all settings from configs/experiment.yaml (MVE mode)"
python src/run_experiment.py --stages build eval analyze --mve "$@"
