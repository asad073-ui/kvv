from __future__ import annotations
import os
import re
import sys
import yaml
from types import SimpleNamespace
from typing import Any



# Env-var expansion


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



# Dict → SimpleNamespace (recursive)


def _to_ns(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj



# Public API


def load_config(path: str | None = None, apply_mve: bool = True) -> SimpleNamespace:
    """
    Load and return the experiment config as a SimpleNamespace tree.

    Parameters
    ----------
    path : str, optional
        Path to the YAML file.  Defaults to configs/experiment.yaml
        relative to the project root (two levels up from this file).
    apply_mve : bool
        When True (default) and mve.enabled is set, the MVE block is flattened
        onto the active dataset list / k_values / per-dataset num_examples.
        Pass False to obtain the raw YAML values untouched (used by the full-
        experiment CLI path so leftover MVE counts never leak into a full run).
    """
    if path is None:
        here    = os.path.dirname(os.path.abspath(__file__))
        project = os.path.dirname(here)          # src/ → project root
        path    = os.path.join(project, "configs", "experiment.yaml")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    expanded = _expand(raw)
    cfg      = _to_ns(expanded)

    #  Convenience: flatten MVE overrides if mve.enabled 
    if apply_mve and getattr(getattr(cfg, "mve", None), "enabled", False):
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
    # Pass per-dataset num_examples (evaluate.py pads if fewer values than datasets)
    args += ["--num_examples"] + n_ex_list

    args += ["--k_values"] + [str(k) for k in cfg.k_values]
    args += ["--conditions"] + cfg.conditions

    if cfg.evaluation.eval_hhem:
        args.append("--eval_hhem")
    if cfg.evaluation.eval_nli:
        args.append("--eval_nli")
    if cfg.model.use_flash_attn:
        args.append("--use_flash_attn")

    # Record the wiki corpus size in every results/summary row for provenance.
    wd = getattr(cfg, "wiki_docs", None)
    n_wiki = int(getattr(wd, "num_docs", 0) or 0) if wd is not None else 0
    args += ["--wiki_pages", str(n_wiki)]

    # Faithfulness scoring mode + scorer batch sizes (full-run throughput knobs).
    ev = getattr(cfg, "evaluation", None)
    args += ["--faithfulness_mode", getattr(ev, "faithfulness_mode", "per_chunk_max")]
    args += ["--hhem_batch_size",   str(getattr(ev, "hhem_batch_size", 16))]
    args += ["--nli_batch_size",    str(getattr(ev, "nli_batch_size", 16))]

    # Generation length (paper uses a fixed short budget across all conditions).
    gen = getattr(cfg, "generation", None)
    args += ["--max_new_tokens", str(getattr(gen, "max_new_tokens", 64))]

    return args


# Default DPR Wikipedia passages URL (Facebook CDN).  Used when wiki_docs.num_docs
# > 0 but no explicit download_url is set in the YAML.
DEFAULT_DPR_TSV_URL = "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"


def precisions_for_conditions(conditions: list[str]) -> list[str]:
    """Map experimental conditions to the KV-cache precisions that must be built.

    C0 (Gold Oracle) needs no precomputed cache; C1→fp16, C2→int8, C3→int4.
    Building only the required precisions saves large amounts of disk.
    If only C0 is requested we still emit fp16 (argparse requires >=1 precision
    and it keeps the index usable if a quant condition is added later).
    """
    cond_to_prec = {"C1": "fp16", "C2": "int8", "C3": "int4"}
    precs = [cond_to_prec[c] for c in conditions if c in cond_to_prec]
    # Preserve canonical order fp16 < int8 < int4, dedup.
    ordered = [p for p in ("fp16", "int8", "int4") if p in precs]
    return ordered or ["fp16"]


def config_to_chunk_cache_args(cfg: SimpleNamespace) -> list[str]:
    """
    Convert the loaded config into CLI arguments for chunk_cache.py.

    The corpus is now built from the UNION of every source that the active
    datasets require:
      * HotpotQA paragraphs   – when 'hotpotqa' is an active dataset
      * RGB positives/negatives – when 'rgb' is an active dataset (local JSONL)
      * DPR Wikipedia passages – when wiki_docs.num_docs > 0 (NQ-Open coverage)
    """
    active = set(getattr(cfg, "datasets_list", list(cfg.datasets.__dict__.keys())))

    args = [
        "--model_name",           cfg.model.name,
        "--embedding_model_name", cfg.embedding.name,
        "--output_path",          cfg.paths.kvcache_dir,
        "--storage_dir",          cfg.paths.storage_dir,
        "--chunk_size",           str(cfg.chunking.chunk_size),
        "--chunk_overlap",        str(cfg.chunking.chunk_overlap),
    ]

    # Only build the precisions the active conditions actually need.
    args += ["--precisions"] + precisions_for_conditions(list(cfg.conditions))

    mve = getattr(cfg, "mve", None)
    mve_on = bool(getattr(mve, "enabled", False))

    def _ds_num(name, fallback=200):
        entry = getattr(cfg.datasets, name, None)
        if entry is None:
            return fallback
        if mve_on:
            return getattr(mve, "num_examples", fallback)
        return getattr(entry, "num_examples", fallback)

    #  HotpotQA paragraphs 
    if "hotpotqa" in active:
        args += ["--hotpotqa_corpus",
                 "--hotpotqa_num_examples", str(_ds_num("hotpotqa"))]

    # RGB positives + negatives (local JSONL) 
    if "rgb" in active:
        rgb_entry = getattr(cfg.datasets, "rgb", None)
        rgb_file  = getattr(rgb_entry, "query_file", "") if rgb_entry else ""
        if rgb_file:
            args += ["--rgb_corpus", rgb_file,
                     "--rgb_num_examples", str(_ds_num("rgb"))]

    #  DPR Wikipedia passages (drives NQ-Open coverage) 
    wd = getattr(cfg, "wiki_docs", None)
    n_wiki = int(getattr(wd, "num_docs", 0) or 0) if wd is not None else 0
    if n_wiki > 0:
        url = getattr(wd, "download_url", "") or DEFAULT_DPR_TSV_URL
        save_dir = getattr(wd, "save_dir", "") or os.path.join(
            os.path.dirname(cfg.paths.kvcache_dir), "wiki_dpr_docs")
        args += [
            "--wiki_docs_url",      url,
            "--wiki_docs_num",      str(n_wiki),
            "--wiki_docs_save_dir", save_dir,
        ]

    return args



# CLI: print resolved config


if __name__ == "__main__":
    import pprint
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else None
    cfg      = load_config(cfg_path)
    pprint.pprint(vars(cfg), width=100)
