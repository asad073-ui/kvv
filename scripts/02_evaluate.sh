
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
    echo "[02] Using active venv: $VIRTUAL_ENV"
fi

echo "[02_evaluate] Reading all settings from configs/experiment.yaml"
python src/run_experiment.py --stages eval "$@"
