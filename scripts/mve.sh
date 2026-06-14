
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

#  System paths 
export SCRATCH_DIR="${SCRATCH_DIR:-/content/turborag_quant}"
export HF_HOME="${HF_HOME:-/content/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

#  Virtual environment 
if [[ -f "/home/${USER}/venvs/crisp/bin/activate" ]]; then
    source "/home/${USER}/venvs/crisp/bin/activate"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "[MVE] Using active venv: $VIRTUAL_ENV"
else
    echo "[MVE] WARNING: no virtual environment detected; using system Python"
fi

echo "[mve] Reading all settings from configs/experiment.yaml (MVE mode)"
python src/run_experiment.py --stages build eval analyze --mve "$@"
