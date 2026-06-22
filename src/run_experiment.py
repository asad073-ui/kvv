from __future__ import annotations
import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime

#  locate project root 
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



# Stage runners


def stage_build(cfg, args):
    print("\n Stage: Build Chunk KV Caches ")
    os.makedirs(cfg.paths.kvcache_dir, exist_ok=True)
    os.makedirs(cfg.paths.storage_dir,  exist_ok=True)
    wiki_docs = getattr(cfg, "wiki_docs", None)
    if wiki_docs and getattr(wiki_docs, "save_dir", None):
        os.makedirs(wiki_docs.save_dir, exist_ok=True)
    cli_args = config_to_chunk_cache_args(cfg)
    _python("src/chunk_cache.py", cli_args, str(cfg.gpu.chunk_cache_gpu))


def stage_eval(cfg, args):
    print("\n Stage: Evaluate All Conditions ")
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


def _read_latest_summary(out_dir_rel: str):
    """Load the most recent summary_*.json written under out_dir_rel (rel to PROJECT)."""
    pattern = os.path.join(PROJECT, out_dir_rel, "summary_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    with open(files[-1], encoding="utf-8") as f:
        return json.load(f)


def _to_float(v):
    try:
        if v is None or (isinstance(v, str) and v.strip().upper() == "N/A"):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def stage_stage0(cfg, args):
    """Stage-0 baseline-fidelity gate (ROOT CAUSE 1 + 2).

    Runs ONLY C0 (no cache) and C1 (lossless bf16 cache) on a small slice, then
    refuses to let the pipeline collect C2/C3 numbers unless the baseline is
    trustworthy:
      (i)   zero non-finite-logit failures,
      (ii)  degeneracy rate < 2% per condition,
      (iii) C1 EM/F1 within ~10% of C0,
      (iv)  C1 hallucination rate within 10pp of C0.
    """
    import copy
    print("\n Stage 0: Baseline Fidelity Gate (C0 vs C1) ")

    n_gate = int(getattr(getattr(cfg, "stage0", None), "n_examples", 50) or 50)
    tol_acc = float(getattr(getattr(cfg, "stage0", None), "acc_rel_tol", 0.10) or 0.10)
    tol_faith = float(getattr(getattr(cfg, "stage0", None), "faith_abs_tol", 0.10) or 0.10)
    max_degen = float(getattr(getattr(cfg, "stage0", None), "max_degenerate_rate", 0.02) or 0.02)

    stage0_rel = os.path.join(cfg.paths.output_dir, "stage0")
    os.makedirs(os.path.join(PROJECT, stage0_rel), exist_ok=True)

    gate_cfg = copy.deepcopy(cfg)
    gate_cfg.conditions = ["C0", "C1"]
    gate_cfg.paths.output_dir = stage0_rel
    for ds in gate_cfg.datasets_list:
        entry = getattr(gate_cfg.datasets, ds)
        cur = int(getattr(entry, "num_examples", 0) or 0)
        entry.num_examples = min(cur, n_gate) if cur > 0 else n_gate

    cli_args = config_to_evaluate_args(gate_cfg)
    _python("src/evaluate.py", cli_args, str(cfg.gpu.evaluate_gpu))

    summary = _read_latest_summary(stage0_rel)
    if not summary:
        sys.exit("[stage0] No summary written — evaluate.py produced no rows. Gate FAILED.")

    failures = []
    # group rows by (dataset, k)
    groups = {}
    for r in summary:
        groups.setdefault((r["dataset"], r["k"]), {})[r["condition"]] = r

    for r in summary:
        nf = int(r.get("n_nonfinite", 0) or 0)
        if nf > 0:
            failures.append(f"{r['dataset']} K={r['k']} {r['condition']}: {nf} non-finite-logit failures")
        dr = _to_float(r.get("degenerate_rate"))
        if dr is not None and dr > max_degen:
            failures.append(f"{r['dataset']} K={r['k']} {r['condition']}: degeneracy {dr:.3f} > {max_degen:.3f}")

    for (ds, k), conds in groups.items():
        c0, c1 = conds.get("C0"), conds.get("C1")
        if not c0 or not c1:
            failures.append(f"{ds} K={k}: missing C0 or C1 row")
            continue
        for metric in ("EM", "F1"):
            v0, v1 = _to_float(c0.get(metric)), _to_float(c1.get(metric))
            if v0 is None or v1 is None:
                continue
            allowed = max(tol_acc * v0, 0.05)  # relative 10% with a small absolute floor
            if v1 < v0 - allowed:
                failures.append(
                    f"{ds} K={k}: C1 {metric}={v1:.3f} far below C0 {metric}={v0:.3f} "
                    f"(allowed drop {allowed:.3f})"
                )
        h0, h1 = _to_float(c0.get("hallucination_rate")), _to_float(c1.get("hallucination_rate"))
        if h0 is not None and h1 is not None and abs(h1 - h0) > tol_faith:
            failures.append(
                f"{ds} K={k}: C1 hallucination {h1:.3f} not within {tol_faith:.2f} of C0 {h0:.3f}"
            )

    if failures:
        print("\n[stage0] GATE FAILED — baseline is not trustworthy. "
              "Do NOT collect C2/C3 numbers until these are resolved:")
        for f in failures:
            print(f"  ✗ {f}")
        sys.exit(
            "[stage0] Halting before quantized conditions. "
            "Check dtype=bfloat16 and precompute_mode=standard_causal."
        )
    print("[stage0] GATE PASSED — C1 ≈ C0, zero non-finite logits, degeneracy under threshold. "
          "Proceeding to full evaluation.")


def stage_judge(cfg, args):
    """A3: optional LLM-as-judge third anchor over the latest results JSONL.

    Runs OUTSIDE the GPU eval loop. Skips gracefully (no hard failure) when the
    judge is disabled in config or ANTHROPIC_API_KEY is not set, so it never
    breaks the pipeline on the GPU instance.
    """
    print("\n Stage: LLM-as-Judge (3rd faithfulness/accuracy anchor) ")
    jcfg = getattr(cfg, "llm_judge", None)
    if jcfg is not None and not bool(getattr(jcfg, "enabled", True)):
        print("[judge] llm_judge.enabled=false — skipping.")
        return
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[judge] ANTHROPIC_API_KEY not set — skipping LLM judge stage.")
        return

    results_files = sorted(
        f for f in glob.glob(os.path.join(PROJECT, cfg.paths.output_dir, "results_*.jsonl"))
        if "judged" not in os.path.basename(f)
    )
    if not results_files:
        print("[judge] No results JSONL found — skipping.")
        return
    latest = results_files[-1]

    judge_out = os.path.join(PROJECT, cfg.paths.output_dir, "llm_judge")
    os.makedirs(judge_out, exist_ok=True)
    cli_args = ["--results_jsonl", latest, "--output_dir", judge_out]
    if jcfg is not None:
        if getattr(jcfg, "model", None):
            cli_args += ["--model", str(jcfg.model)]
        if getattr(jcfg, "max_records", None) is not None:
            cli_args += ["--max_records", str(jcfg.max_records)]
        if getattr(jcfg, "validate_n", None) is not None:
            cli_args += ["--validate_n", str(jcfg.validate_n)]
        # Budget controls: restrict to the H1 pair and/or cap per group.
        conds = getattr(jcfg, "conditions", None)
        if conds:
            cli_args += ["--conditions"] + [str(c) for c in conds]
        mpg = getattr(jcfg, "max_per_group", 0)
        if mpg:
            cli_args += ["--max_per_group", str(mpg)]
    _python("src/llm_judge.py", cli_args, str(cfg.gpu.evaluate_gpu))


def stage_calib(cfg, args):
    print("\n Stage: Metric Calibration ")
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
    print("\n Stage: Hypothesis Analysis ")
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
    # Pair the analysis with the matching raw JSONL so H1 (McNemar) + H3
    # (noise slope) + paired-delta CIs can run on per-example data. Prefer the
    # LLM-judge-augmented JSONL when present (superset of the raw records plus
    # llm_correct/llm_faithful) so report.txt can show HHEM/NLI/LLM side by side.
    judged = os.path.join(PROJECT, cfg.paths.output_dir, "llm_judge", "results_judged.jsonl")
    results_files = sorted(
        f for f in glob.glob(os.path.join(PROJECT, cfg.paths.output_dir, "results_*.jsonl"))
        if "judged" not in os.path.basename(f)
    )
    chosen = judged if os.path.exists(judged) else (results_files[-1] if results_files else None)
    if chosen:
        cli_args += ["--results_jsonl", chosen]
    _python("src/analyze_results.py", cli_args, gpu_id="0")


def stage_tables(cfg, args):
    print("\n Stage: Publication Tables ")
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



# Main


STAGE_FNS = {
    "build":   stage_build,
    "stage0":  stage_stage0,
    "eval":    stage_eval,
    "judge":   stage_judge,
    "calib":   stage_calib,
    "analyze": stage_analyze,
    "tables":  stage_tables,
}

def main():
    parser = argparse.ArgumentParser(description="Config-driven TurboRAG experiment runner")
    parser.add_argument("--config", type=str,
                        default=os.path.join(PROJECT, "configs", "full_experiment.yaml"),
                        help="Path to experiment YAML config")
    parser.add_argument("--stages", nargs="+",
                        choices=["build", "stage0", "eval", "judge", "calib",
                                 "analyze", "tables", "all"],
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

    #  Full-experiment CLI surface (no source edits required) 
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
    parser.add_argument("--faithfulness_mode", type=str, default=None,
                        choices=["per_chunk_max", "full_context"],
                        help="Override evaluation.faithfulness_mode from YAML. Run "
                             "the eval twice (once per mode) to report both (A1).")
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

    # A1: faithfulness-mode override (applies after any reload above). Lets the
    # caller run the eval twice — full_context and per_chunk_max — and report both.
    if args.faithfulness_mode is not None:
        ev = getattr(cfg, "evaluation", None)
        if ev is not None:
            ev.faithfulness_mode = args.faithfulness_mode
        print(f"[run_experiment] faithfulness_mode override -> {args.faithfulness_mode}")

    if args.dry_run:
        import pprint
        print(" Resolved config ")
        pprint.pprint(vars(cfg), width=100)
        print("\n chunk_cache.py args ")
        print(" ".join(config_to_chunk_cache_args(cfg)))
        print("\n evaluate.py args ")
        print(" ".join(config_to_evaluate_args(cfg)))
        return

    stages = args.stages
    if "all" in stages:
        # stage0 runs between build and the full eval: it halts the pipeline if
        # the C0/C1 baseline is not trustworthy, so no C2/C3 numbers are produced
        # against a failing gate.
        stages = ["build", "stage0", "eval", "judge", "calib", "analyze", "tables"]

    #  Persist the fully-resolved run config + open the shared log 
    global _LOG_PATH
    out_dir = os.path.join(PROJECT, cfg.paths.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _LOG_PATH = os.path.join(out_dir, f"logs_{timestamp}.txt")

    run_config = {
        "timestamp":     timestamp,
        "config_file":   args.config,
        "model":         cfg.model.name,
        "dtype":         getattr(cfg.model, "dtype", "bfloat16"),
        "precompute_mode": getattr(cfg, "precompute_mode", "standard_causal"),
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
