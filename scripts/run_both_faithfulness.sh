#!/usr/bin/env bash
# A1 acceptance: run the paper evaluation in BOTH faithfulness modes and report
# both. The generations are identical across modes (deterministic greedy decode);
# only the HHEM/NLI faithfulness scoring differs. We keep each pass in its OWN
# SCRATCH dir so the mode-dependent HHEM flags never cross-contaminate.
#
#   Pass 1  full_context   – paper-faithful (H2 multi-chunk claim). FULL pipeline,
#                            including the LLM-judge third anchor.
#   Pass 2  per_chunk_max   – lenient robustness check. SAME pipeline WITHOUT the
#                            judge stage, so the Anthropic API is billed only once.
#
# If the INT4 effect holds in BOTH passes it is robust to the truncation question.
#
# Requirements for the judge (pass 1 only): pip install anthropic, and
#   export ANTHROPIC_API_KEY=...   (see RUN notes / docs). Without the key the
# judge stage skips gracefully and you still get HHEM + NLI.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

#  System paths (BASE is shared; each pass gets its own scratch subtree)
export BASE_SCRATCH_DIR="${SCRATCH_DIR:-/scratch/${USER}/turborag_quant}"
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

#  Virtual environment
if [[ -f "/home/${USER}/venvs/crisp/bin/activate" ]]; then
    source "/home/${USER}/venvs/crisp/bin/activate"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "[both] Using active venv: $VIRTUAL_ENV"
fi

echo "==============================================================="
echo "[both] PASS 1/2  faithfulness_mode=full_context  (with LLM judge)"
echo "==============================================================="
SCRATCH_DIR="${BASE_SCRATCH_DIR}/full_context" \
    python src/run_experiment.py --stages all \
        --faithfulness_mode full_context "$@"

echo "==============================================================="
echo "[both] PASS 2/2  faithfulness_mode=per_chunk_max  (no judge — saves API budget)"
echo "==============================================================="
SCRATCH_DIR="${BASE_SCRATCH_DIR}/per_chunk_max" \
    python src/run_experiment.py \
        --stages build stage0 eval calib analyze tables \
        --faithfulness_mode per_chunk_max "$@"

echo "==============================================================="
echo "[both] Done. Compare the two reports:"
echo "  full_context : ${BASE_SCRATCH_DIR}/full_context/analysis/<model>/report.txt"
echo "  per_chunk_max: ${BASE_SCRATCH_DIR}/per_chunk_max/analysis/<model>/report.txt"
echo "[both] If the INT4 hallucination effect holds in BOTH, it is robust to the"
echo "[both] 512-token truncation confound."
echo "==============================================================="
