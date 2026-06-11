"""
config.py  –  YAML configuration loader for the TurboRAG quantization study.

Reads configs/experiment.yaml (or a user-specified path), expands
${USER} / ${SCRATCH_DIR} / ${HF_HOME} shell-style variables,
and returns a clean namespace object that every script can consume.

Usage inside Python
───────────────────
    from config import load_config
    cfg = load_config()               # reads configs/experiment.yaml by default
    cfg = load_config("configs/experiment.yaml")

    # Access fields
    print(cfg.model.name)
    print(cfg.paths.kvcache_dir)
    print(cfg.k_values)              # [1, 3, 5]
    print(cfg.datasets["hotpotqa"]["query_file"])

Usage from CLI (print resolved config)
───────────────────────────────────────
    python src/config.py
    python src/config.py configs/experiment.yaml
"""

from __future__ import annotations
import os
import re
import sys
import yaml
from types import SimpleNamespace
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# Env-var expansion
# ──────────────────────────────────────────────────────────────────────────────

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _expand(value: Any) -> Any:
    """Recursively expand ${VAR} in string values."""
    if isinstance(value, str):
        def _replace(m):
            var = m.group(1)
            return os.environ.get(var, m.group(0))   # leave unexpanded if not set
        return _ENV_PATTERN.sub(_replace, value)
    elif isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand(v) for v in value]
    return value


# ──────────────────────────────────────────────────────────────────────────────
# Dict → SimpleNamespace (recursive)
# ──────────────────────────────────────────────────────────────────────────────

def _to_ns(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str | None = None) -> SimpleNamespace:
    """
    Load and return the experiment config as a SimpleNamespace tree.

    Parameters
    ----------
    path : str, optional
        Path to the YAML file.  Defaults to configs/experiment.yaml
        relative to the project root (two levels up from this file).
    """
    if path is None:
        here    = os.path.dirname(os.path.abspath(__file__))
        project = os.path.dirname(here)          # src/ → project root
        path    = os.path.join(project, "configs", "experiment.yaml")

    with open(path) as f:
        raw = yaml.safe_load(f)

    expanded = _expand(raw)
    cfg      = _to_ns(expanded)

    # ── Convenience: flatten MVE overrides if mve.enabled ────────────────────
    if getattr(getattr(cfg, "mve", None), "enabled", False):
        mve = cfg.mve
        cfg.datasets_list    = mve.datasets
        cfg.k_values         = mve.k_values
        # Override num_examples on each dataset entry
        for ds_name in mve.datasets:
            ds_entry = cfg.datasets.__dict__.get(ds_name)
            if ds_entry is not None:
                ds_entry.num_examples = mve.num_examples
    else:
        cfg.datasets_list = list(cfg.datasets.__dict__.keys())

    return cfg


def config_to_evaluate_args(cfg: SimpleNamespace) -> list[str]:
    """
    Convert the loaded config into a flat list of CLI arguments
    suitable for passing to evaluate.py via subprocess or argparse.

    Datasets are loaded from HuggingFace when hf_name is set; otherwise the
    local query_file path is used (e.g. for the rgb dataset).
    """
    args = []
    args += ["--model_name",           cfg.model.name]
    args += ["--embedding_model_name", cfg.embedding.name]
    args += ["--storage_dir",          cfg.paths.storage_dir]
    args += ["--output_dir",           cfg.paths.output_dir]
    args += ["--similarity_top_k",     str(cfg.retrieval.similarity_top_k)]

    ds_names        = cfg.datasets_list
    hf_names        = []
    hf_configs      = []
    hf_splits       = []
    question_fields = []
    answer_fields   = []
    query_files     = []
    n_ex_list       = []

    for ds in ds_names:
        entry = cfg.datasets.__dict__[ds]
        hf_names.append(getattr(entry, "hf_name",        "") or "")
        hf_configs.append(getattr(entry, "hf_config",    "") or "")
        hf_splits.append(getattr(entry, "hf_split",      "validation") or "validation")
        question_fields.append(getattr(entry, "question_field", "question"))
        answer_fields.append(getattr(entry, "answer_field",     "answer"))
        query_files.append(getattr(entry, "query_file",  "") or "")
        n_ex_list.append(str(entry.num_examples))

    args += ["--datasets"]        + ds_names
    args += ["--hf_names"]        + hf_names
    args += ["--hf_configs"]      + hf_configs
    args += ["--hf_splits"]       + hf_splits
    args += ["--question_fields"] + question_fields
    args += ["--answer_fields"]   + answer_fields
    args += ["--query_files"]     + query_files
    # Use the first dataset's num_examples (they're equal in the MVE case)
    args += ["--num_examples", n_ex_list[0]]

    args += ["--k_values"] + [str(k) for k in cfg.k_values]
    args += ["--conditions"] + cfg.conditions

    if cfg.evaluation.eval_hhem:
        args.append("--eval_hhem")
    if cfg.evaluation.eval_nli:
        args.append("--eval_nli")
    if cfg.model.use_flash_attn:
        args.append("--use_flash_attn")

    return args


def config_to_chunk_cache_args(cfg: SimpleNamespace) -> list[str]:
    """
    Convert the loaded config into CLI arguments for chunk_cache.py.
    Uses the DPR TSV direct download via cfg.wiki_docs when available.
    """
    wd = getattr(cfg, "wiki_docs", None)
    args = [
        "--model_name",           cfg.model.name,
        "--embedding_model_name", cfg.embedding.name,
        "--output_path",          cfg.paths.kvcache_dir,
        "--storage_dir",          cfg.paths.storage_dir,
        "--chunk_size",           str(cfg.chunking.chunk_size),
        "--chunk_overlap",        str(cfg.chunking.chunk_overlap),
    ]
    if wd and getattr(wd, "download_url", None):
        args += [
            "--wiki_docs_url",      wd.download_url,
            "--wiki_docs_num",      str(getattr(wd, "num_docs", 10000)),
            "--wiki_docs_save_dir", wd.save_dir,
        ]
    else:
        # Fall back to local documents_dir if wiki_docs is not configured
        docs_dir = getattr(getattr(cfg, "paths", None), "documents_dir", "documents")
        args += ["--documents_dir", docs_dir]
    return args


# ──────────────────────────────────────────────────────────────────────────────
# CLI: print resolved config
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pprint
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else None
    cfg      = load_config(cfg_path)
    pprint.pprint(vars(cfg), width=100)
