"""
run_experiment.py  –  Config-driven experiment runner.

This is the single Python entry point that reads configs/experiment.yaml
and runs whichever stages you specify.  It calls chunk_cache.py and
evaluate.py as sub-processes so each stage gets a clean GPU context.

Stages
──────
  build   – Stage 1+2: build offline chunk KV caches (FP16/INT8/INT4)
  eval    – Stage 3–7: evaluate all conditions × K × datasets
  calib   – Stage 8:   metric calibration (HHEM vs DeBERTa-NLI)
  analyze – Stages 9–11: hypothesis analysis + figure CSVs
  all     – run all four stages in order

Usage
─────
    # Print the resolved config (no stages run)
    python src/run_experiment.py --config configs/experiment.yaml --dry_run

    # Run everything end-to-end
    python src/run_experiment.py --stages all

    # Run only the evaluation stage (caches already built)
    python src/run_experiment.py --stages eval

    # MVE: override num_examples and datasets
    python src/run_experiment.py --stages all --mve
"""

from __future__ import annotations
import argparse
import glob
import os
import subprocess
import sys

# ── locate project root ────────────────────────────────────────────────────────
HERE    = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from config import load_config, config_to_chunk_cache_args, config_to_evaluate_args


def _run(cmd: list[str], env: dict | None = None):
    """Run a command, inheriting the current environment, with optional overrides."""
    merged = {**os.environ, **(env or {})}
    print(f"\n[run_experiment] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=merged, cwd=PROJECT)
    if result.returncode != 0:
        sys.exit(result.returncode)


def _python(script_rel: str, extra_args: list[str], gpu_id: str, env: dict | None = None):
    """Run a src/ Python script on a specific GPU."""
    script = os.path.join(PROJECT, script_rel)
    gpu_env = {"CUDA_VISIBLE_DEVICES": gpu_id}
    if env:
        gpu_env.update(env)
    _run([sys.executable, script] + extra_args, env=gpu_env)


# ──────────────────────────────────────────────────────────────────────────────
# Stage runners
# ──────────────────────────────────────────────────────────────────────────────

def stage_build(cfg, args):
    print("\n━━━ Stage: Build Chunk KV Caches ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    os.makedirs(cfg.paths.kvcache_dir, exist_ok=True)
    os.makedirs(cfg.paths.storage_dir,  exist_ok=True)
    wiki_docs = getattr(cfg, "wiki_docs", None)
    if wiki_docs and getattr(wiki_docs, "save_dir", None):
        os.makedirs(wiki_docs.save_dir, exist_ok=True)
    cli_args = config_to_chunk_cache_args(cfg)
    _python("src/chunk_cache.py", cli_args, str(cfg.gpu.chunk_cache_gpu))


def stage_eval(cfg, args):
    print("\n━━━ Stage: Evaluate All Conditions ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    cli_args = config_to_evaluate_args(cfg)
    _python("src/evaluate.py", cli_args, str(cfg.gpu.evaluate_gpu))


def stage_calib(cfg, args):
    print("\n━━━ Stage: Metric Calibration ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    results_files = sorted(glob.glob(os.path.join(PROJECT, cfg.paths.output_dir, "results_*.jsonl")))
    if not results_files:
        print("[calib] No results JSONL found – skipping calibration.")
        return
    latest = results_files[-1]
    calib_out = os.path.join(PROJECT, cfg.paths.output_dir, "calibration")
    os.makedirs(calib_out, exist_ok=True)
    cli_args = [
        "--results_jsonl", latest,
        "--condition",     cfg.calibration.condition,
        "--n_calibration", str(cfg.calibration.n_examples),
        "--output_dir",    calib_out,
    ]
    _python("src/calibrate_metrics.py", cli_args, str(cfg.gpu.evaluate_gpu))


def stage_analyze(cfg, args):
    print("\n━━━ Stage: Hypothesis Analysis ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    summary_files = sorted(glob.glob(os.path.join(PROJECT, cfg.paths.output_dir, "summary_*.json")))
    if not summary_files:
        print("[analyze] No summary JSON found – skipping analysis.")
        return
    latest = summary_files[-1]
    os.makedirs(os.path.join(PROJECT, cfg.paths.analysis_dir), exist_ok=True)
    cli_args = [
        "--summary_json", latest,
        "--output_dir",   os.path.join(PROJECT, cfg.paths.analysis_dir),
    ]
    _python("src/analyze_results.py", cli_args, gpu_id="0")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

STAGE_FNS = {
    "build":   stage_build,
    "eval":    stage_eval,
    "calib":   stage_calib,
    "analyze": stage_analyze,
}

def main():
    parser = argparse.ArgumentParser(description="Config-driven TurboRAG experiment runner")
    parser.add_argument("--config", type=str,
                        default=os.path.join(PROJECT, "configs", "experiment.yaml"),
                        help="Path to experiment YAML config")
    parser.add_argument("--stages", nargs="+",
                        choices=["build", "eval", "calib", "analyze", "all"],
                        default=["all"],
                        help="Which stages to run")
    parser.add_argument("--mve", action="store_true",
                        help="Force MVE mode (overrides config mve.enabled)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Load and print config without running anything")
    # Optional single-value overrides (take precedence over the YAML)
    parser.add_argument("--model_name", type=str, default=None,
                        help="Override model.name from YAML")
    parser.add_argument("--cache_gpu",  type=int, default=None,
                        help="Override gpu.chunk_cache_gpu from YAML")
    parser.add_argument("--eval_gpu",   type=int, default=None,
                        help="Override gpu.evaluate_gpu from YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Apply any CLI overrides on top of the YAML values
    if args.model_name:
        cfg.model.name = args.model_name
    if args.cache_gpu is not None:
        cfg.gpu.chunk_cache_gpu = args.cache_gpu
    if args.eval_gpu is not None:
        cfg.gpu.evaluate_gpu = args.eval_gpu

    # Override MVE from CLI flag
    if args.mve:
        cfg.mve.enabled = True
        # Re-apply MVE overrides
        mve = cfg.mve
        cfg.datasets_list = mve.datasets
        cfg.k_values      = mve.k_values
        for ds_name in mve.datasets:
            ds_entry = cfg.datasets.__dict__.get(ds_name)
            if ds_entry is not None:
                ds_entry.num_examples = mve.num_examples

    if args.dry_run:
        import pprint
        print("── Resolved config ──────────────────────────────────")
        pprint.pprint(vars(cfg), width=100)
        print("\n── chunk_cache.py args ──────────────────────────────")
        print(" ".join(config_to_chunk_cache_args(cfg)))
        print("\n── evaluate.py args ─────────────────────────────────")
        print(" ".join(config_to_evaluate_args(cfg)))
        return

    stages = args.stages
    if "all" in stages:
        stages = ["build", "eval", "calib", "analyze"]

    print(f"[run_experiment] Stages to run: {stages}")
    print(f"[run_experiment] Config       : {args.config}")
    print(f"[run_experiment] MVE mode     : {getattr(cfg.mve, 'enabled', False)}")

    for stage in stages:
        STAGE_FNS[stage](cfg, args)

    print("\n[run_experiment] All stages complete.")


if __name__ == "__main__":
    main()
