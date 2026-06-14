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
import json
import os
import subprocess
import sys
from datetime import datetime

# ── locate project root ────────────────────────────────────────────────────────
HERE    = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from config import (
    load_config, config_to_chunk_cache_args, config_to_evaluate_args,
    precisions_for_conditions,
)

# Set by main() once the output dir is known, so every sub-stage streams into one
# logs.txt while still printing live to the console.
_LOG_PATH: str | None = None


def _run(cmd: list[str], env: dict | None = None):
    """Run a command live, mirroring combined stdout/stderr into _LOG_PATH."""
    merged = {**os.environ, **(env or {})}
    header = f"\n[run_experiment] Running: {' '.join(cmd)}"
    print(header)
    logf = open(_LOG_PATH, "a", encoding="utf-8") if _LOG_PATH else None
    if logf:
        logf.write(header + "\n")
        logf.flush()
    proc = subprocess.Popen(
        cmd, env=merged, cwd=PROJECT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, universal_newlines=True, encoding="utf-8", errors="replace",
    )
    for line in proc.stdout:
        sys.stdout.write(line)
        if logf:
            logf.write(line)
    proc.wait()
    if logf:
        logf.flush()
        logf.close()
    if proc.returncode != 0:
        sys.exit(proc.returncode)


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
    # Seed the orchestration process for reproducibility; evaluate.py seeds itself.
    import random as _random
    import numpy as _np
    import torch as _torch
    GLOBAL_SEED = 42
    _random.seed(GLOBAL_SEED)
    _np.random.seed(GLOBAL_SEED)
    _torch.manual_seed(GLOBAL_SEED)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(GLOBAL_SEED)
    _torch.backends.cudnn.deterministic = True
    _torch.backends.cudnn.benchmark = False

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


def stage_tables(cfg, args):
    print("\n━━━ Stage: Publication Tables ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    out_dir = os.path.join(PROJECT, cfg.paths.output_dir)
    summary_files = sorted(glob.glob(os.path.join(out_dir, "summary_*.json")))
    if not summary_files:
        print("[tables] No summary JSON found – skipping table generation.")
        return
    latest = summary_files[-1]
    analysis_dir = os.path.join(PROJECT, cfg.paths.analysis_dir)
    os.makedirs(analysis_dir, exist_ok=True)
    cli_args = [
        "--summary_json", latest,
        "--output_dir",   analysis_dir,
    ]
    cfg_json = os.path.join(out_dir, "config.json")
    if os.path.exists(cfg_json):
        cli_args += ["--config_json", cfg_json]
    _python("src/make_paper_tables.py", cli_args, gpu_id="0")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

STAGE_FNS = {
    "build":   stage_build,
    "eval":    stage_eval,
    "calib":   stage_calib,
    "analyze": stage_analyze,
    "tables":  stage_tables,
}

def main():
    parser = argparse.ArgumentParser(description="Config-driven TurboRAG experiment runner")
    parser.add_argument("--config", type=str,
                        default=os.path.join(PROJECT, "configs", "experiment.yaml"),
                        help="Path to experiment YAML config")
    parser.add_argument("--stages", nargs="+",
                        choices=["build", "eval", "calib", "analyze", "tables", "all"],
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

    # ── Full-experiment CLI surface (no source edits required) ────────────────
    parser.add_argument("--wiki_pages", type=int, default=None,
                        help="DPR Wikipedia passages to ingest (drives NQ-Open coverage). "
                             "0 disables the wiki source.")
    parser.add_argument("--num_nq_examples",     type=int, default=None,
                        help="NQ-Open eval examples (0 to skip the dataset)")
    parser.add_argument("--num_hotpot_examples", type=int, default=None,
                        help="HotpotQA eval examples (0 to skip the dataset)")
    parser.add_argument("--num_rgb_examples",    type=int, default=None,
                        help="RGB eval examples (0 to skip the dataset)")
    parser.add_argument("--k_values", type=int, nargs="+", default=None,
                        help="Number(s) of retrieved chunks, e.g. --k_values 1 3 5")
    parser.add_argument("--conditions", type=str, nargs="+", default=None,
                        choices=["C0", "C1", "C2", "C3"],
                        help="Conditions to run, e.g. --conditions C0 C1 C2 C3")
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

    # ── Full-experiment overrides: any explicit count/condition/K/wiki flag
    #    switches OFF MVE mode and rebuilds the active dataset list from the
    #    per-dataset example counts (a count of 0 drops that dataset). ─────────
    full_flags = [args.wiki_pages, args.num_nq_examples, args.num_hotpot_examples,
                  args.num_rgb_examples, args.k_values, args.conditions]
    if any(f is not None for f in full_flags):
        # Reload WITHOUT MVE flattening so leftover MVE counts never leak in.
        cfg = load_config(args.config, apply_mve=False)
        if args.model_name:
            cfg.model.name = args.model_name
        if args.cache_gpu is not None:
            cfg.gpu.chunk_cache_gpu = args.cache_gpu
        if args.eval_gpu is not None:
            cfg.gpu.evaluate_gpu = args.eval_gpu
        cfg.mve.enabled = False
        ds_count_overrides = {
            "nq_open":  args.num_nq_examples,
            "hotpotqa": args.num_hotpot_examples,
            "rgb":      args.num_rgb_examples,
        }
        for ds_name, n in ds_count_overrides.items():
            if n is not None:
                entry = getattr(cfg.datasets, ds_name, None)
                if entry is not None:
                    entry.num_examples = n
        # Active datasets = those (in canonical order) with num_examples > 0.
        all_ds = list(cfg.datasets.__dict__.keys())
        cfg.datasets_list = [
            ds for ds in all_ds
            if int(getattr(getattr(cfg.datasets, ds), "num_examples", 0) or 0) > 0
        ]
        if args.k_values is not None:
            cfg.k_values = args.k_values
        if args.conditions is not None:
            cfg.conditions = args.conditions
        if args.wiki_pages is not None:
            cfg.wiki_docs.num_docs = args.wiki_pages

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
        stages = ["build", "eval", "calib", "analyze", "tables"]

    # ── Persist the fully-resolved run config + open the shared log ───────────
    global _LOG_PATH
    out_dir = os.path.join(PROJECT, cfg.paths.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _LOG_PATH = os.path.join(out_dir, f"logs_{timestamp}.txt")

    run_config = {
        "timestamp":     timestamp,
        "config_file":   args.config,
        "model":         cfg.model.name,
        "dtype":         getattr(cfg.model, "dtype", "float16"),
        "mve_enabled":   bool(getattr(cfg.mve, "enabled", False)),
        "datasets":      list(cfg.datasets_list),
        "num_examples":  {ds: getattr(getattr(cfg.datasets, ds), "num_examples", None)
                          for ds in cfg.datasets_list},
        "k_values":      list(cfg.k_values),
        "conditions":    list(cfg.conditions),
        "precisions":    precisions_for_conditions(list(cfg.conditions)),
        "wiki_pages":    int(getattr(cfg.wiki_docs, "num_docs", 0) or 0),
        "chunk_size":    cfg.chunking.chunk_size,
        "chunk_overlap": cfg.chunking.chunk_overlap,
        "similarity_top_k": cfg.retrieval.similarity_top_k,
        "stages":        stages,
    }
    cfg_path = os.path.join(out_dir, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print(f"[run_experiment] Stages to run: {stages}")
    print(f"[run_experiment] Config       : {args.config}")
    print(f"[run_experiment] MVE mode     : {getattr(cfg.mve, 'enabled', False)}")
    print(f"[run_experiment] Datasets     : {cfg.datasets_list}")
    print(f"[run_experiment] K values     : {cfg.k_values}")
    print(f"[run_experiment] Conditions   : {cfg.conditions}")
    print(f"[run_experiment] Wiki pages   : {run_config['wiki_pages']}")
    print(f"[run_experiment] Run config   → {cfg_path}")
    print(f"[run_experiment] Log file     → {_LOG_PATH}")

    for stage in stages:
        STAGE_FNS[stage](cfg, args)

    print("\n[run_experiment] All stages complete.")


if __name__ == "__main__":
    main()
