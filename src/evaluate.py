"""
evaluate.py  –  Main evaluation script for the TurboRAG KV-cache quantization study.

Implements all experimental conditions:
  C0  Gold Oracle RAG   – full raw-text context, no precomputed cache
  C1  FP16 TurboRAG    – precomputed FP16 chunk caches stitched at query time
  C2  INT8 TurboRAG    – offline INT8 quantized chunk caches
  C3  INT4 TurboRAG    – offline INT4 quantized chunk caches

Design (transformers==4.51.3)
─────────────────────────────────────────────────────────────────────
  * Chunk + prefix KV caches store RAW (un-rotated) keys.  At decode time the
    modified attention in qwen2.py re-applies RoPE to the full stitched key
    sequence with global positions, so independently-built caches compose.
  * Cache handling uses ONLY the stable DynamicCache.to_legacy_cache() /
    from_legacy_cache() round-trip — never key_cache / value_cache / _seen_tokens.
  * The cached path (C1-C3) generates with a manual greedy-decode loop, because
    4.51.3's generate() recomputes cache_position as arange(input_len)[past_len:],
    which is empty when the pre-filled cache is longer than the suffix.
  * Everything runs in float16 (T4 / sm_75 has no native bfloat16).
  * Refusals ("I can not answer …") are tracked separately and PAIRED across
    conditions before faithfulness scoring, so hallucination_rate denominators
    are consistent and H1/H2 comparisons are valid.
  * NQ-Open multi-answer: gold answers are kept as a list; EM/F1/contain_EM use
    max-over-golds scoring.
  * Faithfulness is scored per-chunk (max across chunks) to avoid the K-dependent
    truncation artifact that would otherwise inflate hallucination at high K.
"""


import os
import sys
import json
import time
import logging
import argparse
import csv
import math
import re
import random
import platform
import subprocess
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple


import torch
import numpy as np
from tqdm import tqdm
from tabulate import tabulate


import transformers
from transformers import AutoTokenizer, DynamicCache


sys.path.insert(0, os.path.dirname(__file__))
from qwen2 import Qwen2ModifiedForCausalLM
from kv_quantization import decompress_kvcache, cache_size_bytes
from metrics import (
    batch_em, batch_f1,
    HHEMScorer, DeBERTaNLIScorer,
    hallucination_rate, mean_entailment,
    hallucination_rate_per_chunk, entailment_per_chunk,
    _normalize as _normalize_answer,
)


from llama_index.core import Settings, load_index_from_storage, StorageContext, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


GLOBAL_SEED = 42


# ──────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ──────────────────────────────────────────────────────────────────────────────


SYSTEM_PROMPT = (
    "<|im_start|>system\n"
    "You are an accurate and reliable AI assistant that can answer questions with the "
    "help of external documents. Please note that external documents may contain noisy "
    "information. If the information in the document contains the correct answer, you will "
    "give an accurate answer. If the information in the document does not contain the "
    "answer, you will generate 'I can not answer the question because of the insufficient "
    "information in documents.'.<|im_end|><|im_start|>user\nDocs:"
)


QUERY_SUFFIX_TEMPLATE = "\n\nQuestion: {query}\nAnswer in one concise phrase or sentence.<|im_end|><|im_start|>assistant\n"



def build_full_prompt(chunks: List[str], query: str) -> str:
    return SYSTEM_PROMPT + "".join(chunks) + QUERY_SUFFIX_TEMPLATE.format(query=query)



def build_query_suffix(query: str) -> str:
    return QUERY_SUFFIX_TEMPLATE.format(query=query)



# ──────────────────────────────────────────────────────────────────────────────
# KV cache stitching
# ──────────────────────────────────────────────────────────────────────────────
# transformers==4.51.3: use the stable to_legacy_cache() / from_legacy_cache()
# round-trip.  We never touch cache.key_cache / cache.value_cache / _seen_tokens
# (deprecated / removed in later releases).


def legacy_to_dynamic(legacy: tuple) -> DynamicCache:
    """Convert a disk-loaded legacy cache tuple (per-layer (k, v)) to a DynamicCache."""
    return DynamicCache.from_legacy_cache(legacy)



def stack_past_key_values(caches: List[DynamicCache]) -> DynamicCache:
    """
    Concatenate DynamicCache objects along the sequence dimension.

    Each cache contributes its per-layer (key, value) tensors; we cat along the
    sequence axis (dim=2) and rebuild a DynamicCache via from_legacy_cache().
    All input caches must hold tensors on the same device and dtype.
    """
    legacy_list = [c.to_legacy_cache() for c in caches]
    num_layers  = len(legacy_list[0])
    if num_layers == 0:
        log.error("stack_past_key_values: first cache has 0 layers.")
        return DynamicCache()
    stacked = tuple(
        (
            torch.cat([lc[layer][0] for lc in legacy_list], dim=2),
            torch.cat([lc[layer][1] for lc in legacy_list], dim=2),
        )
        for layer in range(num_layers)
    )
    return DynamicCache.from_legacy_cache(stacked)



# ──────────────────────────────────────────────────────────────────────────────
# Context helpers
# ──────────────────────────────────────────────────────────────────────────────


def trim_context_for_hhem(context: str, max_tokens: int = 400) -> str:
    """Truncate context to fit inside the scorer's token window (~4 chars/token).

    HHEM (flan-t5-base, 512-token limit) handles its own truncation internally,
    so this is a safety net.  For DeBERTa-NLI (512-token hard limit) the caller
    passes max_tokens=200 to leave room for the answer segment.
    """
    return context[: max_tokens * 4]



# ──────────────────────────────────────────────────────────────────────────────
# Accuracy helpers
# ──────────────────────────────────────────────────────────────────────────────


# FIX-NORMDUP: _normalize_answer is now imported from metrics.py as
# _normalize_answer (aliased from _normalize) to avoid duplicate
# implementations that could silently diverge.



def _extract_short_answer(text: str) -> str:
    """
    FIX-EXTRACT: Pull a short answer span from a free-form generation.

    Used ONLY for EM computation.  F1, HHEM, and NLI intentionally keep raw
    preds because they reward partial matches / faithfulness of the full response.

    Strategy (multi-tier, from most to least specific):
      1. Explicit answer markers: "the answer is", "answer is", "answer:"
      2. First sentence: most QA models front-load the answer.
      3. First N words: last resort for single-phrase outputs.
    """
    text = text.strip()
    if not text:
        return ""

    lower = text.lower()

    # ── Tier 1: explicit answer-introducing phrases (high confidence) ──
    explicit_markers = [
        "the answer is ",
        "answer is ",
        "answer: ",
        "the answer to the question is ",
        "the answer to this question is ",
    ]
    for marker in explicit_markers:
        idx = lower.find(marker)
        if idx != -1:
            span = text[idx + len(marker):]
            for sep in ('.', ',', ';', '\n'):
                span = span.split(sep)[0]
            span = span.strip().rstrip('.')
            if 1 <= len(span.split()) <= 12:
                return span

    # ── Tier 2: first sentence ──
    sentences = re.split(r'(?<=\w)\.\s', text, maxsplit=1)
    first_sent = sentences[0].strip().rstrip('.')
    if 1 <= len(first_sent.split()) <= 15:
        return first_sent

    # ── Tier 3: first N words ──
    words = text.split()
    return " ".join(words[:10]).rstrip(".,;:!?")



def max_over_golds_em(predictions: List[str], gold_lists: List[List[str]]) -> float:
    """EM: fraction of predictions that exactly match ANY gold alias."""
    if not predictions:
        return 0.0
    from metrics import exact_match
    return sum(
        max((exact_match(_extract_short_answer(p), g) for g in golds), default=0)
        for p, golds in zip(predictions, gold_lists)
    ) / len(predictions)



def max_over_golds_f1(predictions: List[str], gold_lists: List[List[str]]) -> float:
    """F1: average over examples of max token-F1 across gold aliases.

    Uses extracted short-answer spans (same as EM) so token overlap is measured
    against the answer phrase, not a full verbose generation.  This matches standard
    SQuAD-style evaluation methodology.
    """
    if not predictions:
        return 0.0
    from metrics import token_f1
    return sum(
        max((token_f1(_extract_short_answer(p), g) for g in golds), default=0.0)
        for p, golds in zip(predictions, gold_lists)
    ) / len(predictions)



def batch_contain_em(predictions: List[str], gold_lists: List[List[str]]) -> float:
    """Contain-EM: fraction where normalised gold is a substring of normalised prediction.

    Checks if ANY gold alias is contained (max-over-golds).
    """
    if not predictions:
        return 0.0
    return sum(
        1 for p, golds in zip(predictions, gold_lists)
        if any(_normalize_answer(g) in _normalize_answer(p) for g in golds)
    ) / len(predictions)



# ──────────────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────────────


EOS_TOKEN_IDS  = [151645, 151643]
MAX_NEW_TOKENS = 64

# The system prompt instructs the model to emit this exact phrase when the
# retrieved documents do not contain the answer.  A refusal is a RETRIEVAL
# failure, not a faithfulness failure, so it is tracked separately and excluded
# from HHEM/NLI scoring (otherwise it inflates every hallucination_rate).
REFUSAL_MARKER = "I can not answer the question"



def _get_cache_seq_len(cache: DynamicCache) -> int:
    """Number of tokens already stored in the cache (stable cross-version API)."""
    return cache.get_seq_length()



def generate_with_cache(
    model, tokenizer, device,
    past_kvcache: Optional[DynamicCache],
    query: str,
    chunks: List[str],
) -> str:
    """Generate an answer given either a stitched KV cache (C1-C3) or raw chunks (C0).

    The cached path uses a manual greedy-decode loop instead of model.generate().
    transformers 4.51.3's generate() recomputes cache_position as
    arange(input_len)[past_len:] (see GenerationMixin._get_initial_cache_position),
    which is EMPTY when the pre-filled cache is longer than the new suffix — our
    exact situation.  Driving position_ids / cache_position / attention_mask by
    hand sidesteps that entirely.
    """
    with torch.no_grad():
        if past_kvcache is not None:
            cached_len     = _get_cache_seq_len(past_kvcache)
            n_model_layers = model.config.num_hidden_layers
            n_cache_layers = len(past_kvcache.to_legacy_cache()) if cached_len > 0 else 0

            # Fall back to C0-style generation if the cache is empty or malformed
            # (a stitched cache should have exactly num_hidden_layers layers).
            if cached_len == 0 or n_cache_layers != n_model_layers:
                log.error(
                    "generate_with_cache: unusable cache (cached_len=%d, "
                    "cache_layers=%d, model_layers=%d). Falling back to C0-style "
                    "full-prompt generation.",
                    cached_len, n_cache_layers, n_model_layers,
                )
                return generate_with_cache(model, tokenizer, device, None, query, chunks)

            suffix     = build_query_suffix(query)
            suffix_ids = tokenizer.encode(suffix, return_tensors="pt").to(device)

            past      = past_kvcache
            cur_ids   = suffix_ids
            total_len = cached_len            # tokens already in the cache
            generated: List[int] = []

            for _ in range(MAX_NEW_TOKENS):
                q_len          = cur_ids.shape[1]
                position_ids   = torch.arange(
                    total_len, total_len + q_len, device=device
                ).unsqueeze(0)
                cache_position = torch.arange(
                    total_len, total_len + q_len, device=device
                )
                attention_mask = torch.ones(
                    1, total_len + q_len, device=device, dtype=torch.long
                )
                out = model(
                    input_ids=cur_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past,
                    cache_position=cache_position,
                    use_cache=True,
                )
                next_id = int(out.logits[0, -1].argmax())
                past      = out.past_key_values
                total_len += q_len
                if next_id in EOS_TOKEN_IDS:
                    break
                generated.append(next_id)
                cur_ids = torch.tensor([[next_id]], device=device, dtype=suffix_ids.dtype)

            return tokenizer.decode(generated, skip_special_tokens=True).strip()

        else:
            prompt    = build_full_prompt(chunks, query)
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            outputs   = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=1.0,                    # FIX-TEMP
                eos_token_id=EOS_TOKEN_IDS,
            )
            new_tokens = outputs[0][input_ids.shape[1]:]
            return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()



# ──────────────────────────────────────────────────────────────────────────────
# Per-query inference for each condition
# ──────────────────────────────────────────────────────────────────────────────


PRECISION_MAP = {"C0": None, "C1": "fp16", "C2": "int8", "C3": "int4"}
CONDITION_LABELS = {
    "C0": "Gold Oracle RAG",
    "C1": "FP16 TurboRAG",
    "C2": "INT8 TurboRAG",
    "C3": "INT4 TurboRAG",
}



def _move_compressed_to_device(compressed: list, device: torch.device) -> list:
    """Move compressed KV cache tensors to `device` before dequantization.

    For FP16: moves the raw float16 tensors.
    For INT8: moves quantized (uint8), scale (fp32), zero_point (int32).
    For INT4: moves packed (uint8) and scale (fp32).
    Operates on the list-of-dicts structure from compress_kvcache().
    Running dequantization on GPU (bit-unpacking, scale multiply) is faster and
    avoids a large transient on-GPU memory spike from loading already-inflated tensors.
    """
    moved = []
    for layer_data in compressed:
        new_layer = {}
        for kv_key in ("k", "v"):
            d = layer_data[kv_key]
            if isinstance(d, torch.Tensor):
                new_layer[kv_key] = d.to(device)
            elif isinstance(d, dict):
                new_layer[kv_key] = {
                    dk: dv.to(device) if isinstance(dv, torch.Tensor) else dv
                    for dk, dv in d.items()
                }
            else:
                new_layer[kv_key] = d
        moved.append(new_layer)
    return moved



def run_query(
    query: str,
    retrieved_nodes,
    k: int,
    condition: str,
    model,
    tokenizer,
    device,
    prefix_kvcache: DynamicCache,
) -> Dict[str, Any]:
    """
    Run a single query under a given condition.

    Returns io_seconds (disk load + dequant) and ttft_seconds (first model forward
    to first token) separately so callers can attribute latency correctly.

    prefix_kvcache is shared (read-only) across all queries.  stack_past_key_values
    materialises brand-new tensors via torch.cat, so the prefix is never mutated.
    """
    nodes_k       = retrieved_nodes[:k]
    precision     = PRECISION_MAP[condition]
    chunk_texts   = []
    kv_size_bytes = 0

    if condition == "C0":
        # Task 5: wrap raw text with the same delimiters used for C1-C3 chunk caches
        # so C0 and C1 differ only by retrieval condition (not by delimiter tokens).
        for nws in nodes_k:
            raw = nws.node.metadata.get("raw_text", nws.node.text)
            chunk_texts.append(f"<|doc_start|>{raw}<|doc_end|>")

        io_time = 0.0
        ttft_start = time.perf_counter()
        answer = generate_with_cache(model, tokenizer, device, None, query, chunk_texts)
        ttft = time.perf_counter() - ttft_start

    else:
        kvcache_list = [prefix_kvcache]

        # Task 4: measure disk I/O + dequant separately from model forward
        io_start = time.perf_counter()
        for nws in nodes_k:
            node      = nws.node
            cache_key = f"kvcache_{precision}"

            if cache_key not in node.metadata:
                raise KeyError(
                    f"Node '{node.id_}' missing '{cache_key}'. "
                    f"Did you run --stages build first? "
                    f"Available keys: {list(node.metadata.keys())}"
                )
            fpath = node.metadata[cache_key]
            if not os.path.exists(fpath):
                raise FileNotFoundError(
                    f"Cache file not found: '{fpath}'. Re-run --stages build."
                )

            # Task 3: Load to CPU first (avoids GPU OOM spike on large caches),
            # then move the compressed dict's tensors to device BEFORE dequantization
            # so that all INT4/INT8 arithmetic (bit-unpacking, scale multiply) runs on GPU.
            compressed_cpu = torch.load(fpath, map_location="cpu", weights_only=False)
            compressed = _move_compressed_to_device(compressed_cpu, device)
            legacy = decompress_kvcache(compressed, precision)
            # Safety cast: ensure tensors are on device with float16 dtype.
            legacy = tuple((kk.to(device), v.to(device)) for kk, v in legacy)
            kvcache_list.append(legacy_to_dynamic(legacy))
            chunk_texts.append(node.metadata.get("raw_text", node.text))
            kv_size_bytes += cache_size_bytes(compressed, precision)

        stitched = stack_past_key_values(kvcache_list)
        io_time = time.perf_counter() - io_start

        ttft_start = time.perf_counter()
        answer = generate_with_cache(model, tokenizer, device, stitched, query, [])
        ttft = time.perf_counter() - ttft_start

    return {
        "answer":        answer,
        "context":       "\n\n".join(chunk_texts),
        "chunk_texts":   chunk_texts,           # per-chunk list for per-chunk faithfulness scoring
        "ttft_seconds":  ttft,
        "io_seconds":    io_time,
        "kv_size_bytes": kv_size_bytes,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loaders
# ──────────────────────────────────────────────────────────────────────────────


def _load_from_hf(
    hf_name, hf_config, hf_split, num_examples,
    question_field, answer_field, hf_cache_dir,
) -> List[Dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install with: pip install datasets")

    kwargs = {}
    if hf_config:
        kwargs["name"] = hf_config
    if hf_cache_dir:
        kwargs["cache_dir"] = hf_cache_dir

    log.info(f"  Loading HF dataset: {hf_name} / {hf_config or 'default'} split={hf_split}")
    ds = load_dataset(hf_name, split=hf_split, **kwargs)

    examples = []
    for row in ds:
        if num_examples != -1 and len(examples) >= num_examples:
            break
        q = row.get(question_field) or row.get("question", "")
        # Task 6: keep full answer list for multi-answer datasets (e.g. NQ-Open).
        a_raw = row.get(answer_field) or row.get("answer", "")
        if isinstance(a_raw, list) and all(isinstance(x, str) for x in a_raw):
            # Multi-answer: keep the full list for max-over-golds scoring.
            a_normalized = a_raw
        elif isinstance(a_raw, list):
            a_raw = a_raw[0] if a_raw else ""
            while isinstance(a_raw, list):
                a_raw = a_raw[0] if a_raw else ""
            a_normalized = [str(a_raw)]
        elif isinstance(a_raw, dict):
            a_raw = a_raw.get("text", "") or a_raw.get("answer", "")
            while isinstance(a_raw, list):
                a_raw = a_raw[0] if a_raw else ""
            a_normalized = [str(a_raw)]
        else:
            a_normalized = [str(a_raw)]
        examples.append({"query": q, "answer": a_normalized})
    return examples



def _load_from_jsonl(query_file: str, num_examples: int) -> List[Dict]:
    examples = []
    with open(query_file) as f:
        for line in f:
            data   = json.loads(line.strip())
            query  = data.get("query") or data.get("question", "")
            a_raw  = data.get("answer") or data.get("answers", "")
            # Task 6: same multi-answer normalization as _load_from_hf.
            if isinstance(a_raw, list) and all(isinstance(x, str) for x in a_raw):
                a_normalized = a_raw
            elif isinstance(a_raw, list):
                a_raw = a_raw[0] if a_raw else ""
                while isinstance(a_raw, list):
                    a_raw = a_raw[0] if a_raw else ""
                a_normalized = [str(a_raw)]
            elif isinstance(a_raw, dict):
                a_raw = a_raw.get("text", "") or a_raw.get("answer", "")
                while isinstance(a_raw, list):
                    a_raw = a_raw[0] if a_raw else ""
                a_normalized = [str(a_raw)]
            else:
                a_normalized = [str(a_raw)]
            examples.append({"query": query, "answer": a_normalized})
            if num_examples != -1 and len(examples) >= num_examples:
                break
    return examples



def load_dataset_examples(
    dataset_name, query_file, num_examples,
    hf_name=None, hf_config=None, hf_split="validation",
    question_field="question", answer_field="answer", hf_cache_dir=None,
) -> List[Dict]:
    if hf_name:
        return _load_from_hf(hf_name, hf_config or None, hf_split,
                              num_examples, question_field, answer_field, hf_cache_dir)
    if query_file:
        return _load_from_jsonl(query_file, num_examples)
    raise ValueError(f"Dataset '{dataset_name}': provide either --hf_names or --query_files")


# ──────────────────────────────────────────────────────────────────────────────
# Metadata capture helpers (Task 12)
# ──────────────────────────────────────────────────────────────────────────────


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _get_pip_freeze() -> List[str]:
    try:
        return subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            stderr=subprocess.DEVNULL
        ).decode().strip().split("\n")
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="TurboRAG KV-cache quantization evaluation")
    parser.add_argument("--model_name",            type=str, required=True)
    parser.add_argument("--embedding_model_name",  type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--storage_dir",           type=str, default="doc_emb")
    parser.add_argument("--datasets",              type=str, nargs="+",
                        default=["nq_open", "hotpotqa", "rgb"])
    parser.add_argument("--hf_names",        type=str, nargs="+", default=[])
    parser.add_argument("--hf_configs",      type=str, nargs="+", default=[])
    parser.add_argument("--hf_splits",       type=str, nargs="+", default=[])
    parser.add_argument("--question_fields", type=str, nargs="+", default=[])
    parser.add_argument("--answer_fields",   type=str, nargs="+", default=[])
    parser.add_argument("--hf_cache_dir",    type=str, default=None)
    parser.add_argument("--query_files",     type=str, nargs="+", default=[])
    parser.add_argument("--num_examples",    type=int, nargs="+", default=[200])
    parser.add_argument("--k_values",        type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--conditions",      type=str, nargs="+",
                        default=["C0", "C1", "C2", "C3"],
                        choices=["C0", "C1", "C2", "C3"])
    parser.add_argument("--use_flash_attn",   action="store_true")
    parser.add_argument("--output_dir",       type=str, default="results")
    parser.add_argument("--eval_hhem",        action="store_true")
    parser.add_argument("--eval_nli",         action="store_true")
    parser.add_argument("--similarity_top_k", type=int, default=10)
    args = parser.parse_args()

    # Task 14: guard against the dead use_flash_attn lever
    if getattr(args, "use_flash_attn", False):
        raise ValueError(
            "use_flash_attn=True is not supported: Qwen2ModifiedAttention always uses "
            "eager attention to maintain raw-key storage semantics. "
            "Set use_flash_attn: false in experiment.yaml."
        )

    assert args.similarity_top_k >= max(args.k_values), \
        "--similarity_top_k must be >= max(k_values)"

    # Task 12: global seeding for reproducibility
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GLOBAL_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log.info(f"transformers version: {transformers.__version__}  "
             f"(DynamicCache via to_legacy_cache/from_legacy_cache)")

    n_ds = len(args.datasets)
    def _pad(lst, default):
        return list(lst) + [default] * (n_ds - len(lst))

    hf_names          = _pad(args.hf_names,       "")
    hf_configs        = _pad(args.hf_configs,      "")
    hf_splits         = _pad(args.hf_splits,       "validation")
    question_fields   = _pad(args.question_fields, "question")
    answer_fields     = _pad(args.answer_fields,   "answer")
    query_files       = _pad(args.query_files,     "")
    num_examples_list = _pad(args.num_examples,    200)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path  = os.path.join(args.output_dir, f"results_{timestamp}.jsonl")
    csv_path  = os.path.join(args.output_dir, f"summary_{timestamp}.csv")
    json_path = os.path.join(args.output_dir, f"summary_{timestamp}.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    attn_impl = "eager"  # use_flash_attn is guarded above; always eager here
    log.info(f"Loading model: {args.model_name} (attn={attn_impl})")

    # T4 (sm_75): float16 throughout, never bfloat16.
    model = Qwen2ModifiedForCausalLM.from_pretrained(
        args.model_name,
        attn_implementation=attn_impl,
        torch_dtype=torch.float16,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    log.info("Pre-computing system prefix KV cache ...")
    prefix_inputs = tokenizer([SYSTEM_PROMPT], return_tensors="pt")
    with torch.no_grad():
        prefix_out = model(
            prefix_inputs["input_ids"].to(device),
            attention_mask=prefix_inputs["attention_mask"].to(device),
            use_cache=True,
        )
    prefix_kvcache = prefix_out.past_key_values

    # Convert to DynamicCache if the model returned a legacy tuple (defensive).
    if not isinstance(prefix_kvcache, DynamicCache):
        prefix_kvcache = DynamicCache.from_legacy_cache(prefix_kvcache)

    log.info(
        "Prefix KV cache ready: %d layers, seq_len=%d",
        len(prefix_kvcache.to_legacy_cache()),
        prefix_kvcache.get_seq_length(),
    )

    log.info("Loading retrieval index ...")
    # Task 13: pin embedding model to GPU to avoid multi-hour CPU bottleneck
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=args.embedding_model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    storage_ctx = StorageContext.from_defaults(persist_dir=args.storage_dir)
    index       = load_index_from_storage(storage_ctx)
    retriever   = index.as_retriever(similarity_top_k=args.similarity_top_k)

    hhem_scorer = HHEMScorer(device=device)       if args.eval_hhem else None
    nli_scorer  = DeBERTaNLIScorer(device=device) if args.eval_nli  else None

    all_records  = []
    summary_rows = []

    # ── Outer loop: datasets ──────────────────────────────────────────────────
    for ds_name, qfile, hf_name, hf_cfg, hf_split, q_field, a_field, n_examples in zip(
        args.datasets, query_files, hf_names, hf_configs,
        hf_splits, question_fields, answer_fields, num_examples_list,
    ):
        log.info(f"\n{'='*60}\nDataset: {ds_name}\n{'='*60}")
        examples = load_dataset_examples(
            dataset_name=ds_name, query_file=qfile, num_examples=n_examples,
            hf_name=hf_name or None, hf_config=hf_cfg or None, hf_split=hf_split,
            question_field=q_field, answer_field=a_field, hf_cache_dir=args.hf_cache_dir,
        )
        log.info(f"  {len(examples)} examples loaded")
        n_ex = len(examples)

        for k in args.k_values:
            # ── Pass 1: collect per-condition per-example results ──────────────
            # all_cond_results[condition][i] is either a dict of results or None
            # (if the example raised an exception).  Keeping the per-example
            # structure lets us compute the refusal intersection mask in Pass 2.
            all_cond_results: Dict[str, List[Optional[Dict]]] = {}
            cond_ex_to_rec:   Dict[str, Dict[int, int]]       = {}  # condition->ex_idx->rec_idx

            for condition in args.conditions:
                per_ex: List[Optional[Dict]] = [None] * n_ex
                ex_to_rec: Dict[int, int]    = {}

                for i, ex in enumerate(tqdm(examples, desc=f"{ds_name} K={k} {condition}")):
                    query  = ex["query"]
                    answer = ex["answer"]   # List[str] after Task 6 loaders
                    try:
                        qb    = QueryBundle(query_str=query)
                        nodes = retriever.retrieve(qb)
                        result = run_query(
                            query=query, retrieved_nodes=nodes, k=k,
                            condition=condition, model=model, tokenizer=tokenizer,
                            device=device, prefix_kvcache=prefix_kvcache,
                        )
                        is_refusal = REFUSAL_MARKER in result["answer"]
                        per_ex[i] = {
                            "answer":      result["answer"],
                            "context":     result["context"],
                            "chunk_texts": result["chunk_texts"],
                            "ttft":        result["ttft_seconds"],
                            "io_seconds":  result["io_seconds"],
                            "kv_bytes":    result["kv_size_bytes"],
                            "is_refusal":  is_refusal,
                        }
                        rec_idx = len(all_records)
                        all_records.append({
                            "dataset":    ds_name,
                            "k":          k,
                            "condition":  condition,
                            "query":      query,
                            "gold":       answer,   # serializes fine as JSON list
                            "prediction": result["answer"],
                            "context":    result["context"],
                            "ttft":       result["ttft_seconds"],
                            "io_seconds": result["io_seconds"],
                            "kv_bytes":   result["kv_size_bytes"],
                            "is_refusal": is_refusal,
                        })
                        ex_to_rec[i] = rec_idx
                    except Exception as e:
                        log.error(
                            f"Skipped query [{ds_name} K={k} {condition} example={i}]: {e}",
                            exc_info=True,
                        )

                all_cond_results[condition] = per_ex
                cond_ex_to_rec[condition]   = ex_to_rec

            # ── Task 7: paired non-refusal mask (intersection across C1/C2/C3 only) ──
            # C0 is the oracle upper-bound — its refusal pattern is driven by raw-text
            # retrieval, not by quantization noise, so including it in the pairing
            # requirement would incorrectly shrink the faithfulness denominator for H1/H2.
            # The paired mask covers only the quantization conditions (C1/C2/C3).
            # C0 is scored separately on its own non-refusal set in Pass 2.
            quant_conditions = [c for c in args.conditions if c != "C0"]
            paired_mask = [
                all(
                    all_cond_results[c][i] is not None and not all_cond_results[c][i]["is_refusal"]
                    for c in quant_conditions
                ) if quant_conditions else False
                for i in range(n_ex)
            ]
            paired_n = sum(paired_mask)
            nr_paired_idx = [i for i in range(n_ex) if paired_mask[i]]

            # ── Pass 2: compute metrics per condition ──────────────────────────
            for condition in args.conditions:
                per_ex    = all_cond_results[condition]
                ex_to_rec = cond_ex_to_rec[condition]

                # Valid examples (no exception) — used for EM / F1 / timing
                valid_idx = [i for i, r in enumerate(per_ex) if r is not None]
                preds = [per_ex[i]["answer"] for i in valid_idx]
                refs  = [examples[i]["answer"] for i in valid_idx]  # List[List[str]]

                # EM/F1 run over ALL valid examples (refusals count as wrong answers)
                em_score   = max_over_golds_em(preds, refs)
                f1_score   = max_over_golds_f1(preds, refs)
                contain_em = batch_contain_em(preds, refs)

                if preds and log.isEnabledFor(logging.DEBUG):
                    for ii in range(min(3, len(preds))):
                        log.debug(
                            "  [%d] raw=%r  extracted=%r  gold=%r",
                            ii, preds[ii][:80],
                            _extract_short_answer(preds[ii]),
                            refs[ii],
                        )

                ttft_list = [per_ex[i]["ttft"]      for i in valid_idx]
                io_list   = [per_ex[i]["io_seconds"] for i in valid_idx]
                kv_list   = [per_ex[i]["kv_bytes"]   for i in valid_idx]

                avg_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else float("nan")
                avg_io   = sum(io_list)   / len(io_list)   if io_list   else float("nan")
                avg_kv   = sum(kv_list)   / len(kv_list)   if kv_list   else 0

                is_refusals  = [per_ex[i]["is_refusal"] for i in valid_idx]
                refusal_rate = sum(is_refusals) / max(len(is_refusals), 1)

                # Faithfulness set selection:
                # C0 uses its own non-refusal examples (oracle has independent refusals).
                # C1/C2/C3 use the paired set (intersection of non-refusals across all
                # quant conditions) so hallucination_rate denominators are consistent
                # and H1/H2 δ comparisons are valid.
                if condition == "C0":
                    nr_idx = [i for i in valid_idx if not per_ex[i]["is_refusal"]]
                else:
                    nr_idx = [i for i in nr_paired_idx if per_ex[i] is not None]

                hall_rate   = float("nan")
                ent_score   = float("nan")
                n_hall      = 0
                n_total     = len(nr_idx)
                HHEM_THRESHOLD = 0.5

                # Task 8: per-chunk scoring — avoids K-dependent truncation artifact
                if hhem_scorer and nr_idx:
                    chunk_texts_list = [per_ex[i]["chunk_texts"] for i in nr_idx]
                    nr_preds         = [per_ex[i]["answer"]      for i in nr_idx]
                    faith_scores, hallucinated_flags = hallucination_rate_per_chunk(
                        hhem_scorer, chunk_texts_list, nr_preds,
                        threshold=HHEM_THRESHOLD,
                    )
                    hall_rate = sum(hallucinated_flags) / len(hallucinated_flags)
                    n_total   = len(hallucinated_flags)
                    n_hall    = sum(hallucinated_flags)
                    for j, i in enumerate(nr_idx):
                        rec_idx = ex_to_rec.get(i)
                        if rec_idx is not None:
                            all_records[rec_idx]["hhem_faithfulness"] = faith_scores[j]
                            all_records[rec_idx]["hhem_hallucinated"] = hallucinated_flags[j]

                if nli_scorer and nr_idx:
                    chunk_texts_list = [per_ex[i]["chunk_texts"] for i in nr_idx]
                    nr_preds         = [per_ex[i]["answer"]      for i in nr_idx]
                    nli_scores = entailment_per_chunk(nli_scorer, chunk_texts_list, nr_preds)
                    ent_score  = sum(e for e, _, _ in nli_scores) / len(nli_scores) \
                                 if nli_scores else float("nan")
                    for j, i in enumerate(nr_idx):
                        rec_idx = ex_to_rec.get(i)
                        if rec_idx is not None:
                            all_records[rec_idx]["nli_entailment"]    = nli_scores[j][0]
                            all_records[rec_idx]["nli_neutral"]       = nli_scores[j][1]
                            all_records[rec_idx]["nli_contradiction"] = nli_scores[j][2]

                row = {
                    "dataset":            ds_name,
                    "k":                  k,
                    "condition":          condition,
                    "condition_label":    CONDITION_LABELS[condition],
                    "n_examples":         len(preds),
                    "n_hall":             n_hall,
                    "n_total":            n_total,
                    "paired_n":           paired_n,
                    "refusal_rate":       round(refusal_rate, 4),
                    "EM":                 round(em_score,   4),
                    "contain_EM":         round(contain_em, 4),
                    "F1":                 round(f1_score,   4),
                    "hallucination_rate": round(hall_rate, 4) if not math.isnan(hall_rate) else "N/A",
                    "entailment_score":   round(ent_score,  4) if not math.isnan(ent_score)  else "N/A",
                    "avg_ttft_s":         round(avg_ttft, 4) if not math.isnan(avg_ttft) else "N/A",
                    "avg_io_s":           round(avg_io,   4) if not math.isnan(avg_io)   else "N/A",
                    "avg_kv_bytes":       int(avg_kv),
                }
                summary_rows.append(row)
                log.info(
                    f"    n={len(preds)}  EM={em_score:.3f}  ContEM={contain_em:.3f}  "
                    f"F1={f1_score:.3f}  Refuse={refusal_rate:.3f}  "
                    f"Paired={paired_n}  "
                    f"Hall={'N/A' if math.isnan(hall_rate) else f'{hall_rate:.3f}'}  "
                    f"Ent={'N/A' if math.isnan(ent_score) else f'{ent_score:.3f}'}  "
                    f"TTFT={'N/A' if math.isnan(avg_ttft) else f'{avg_ttft:.3f}s'}  "
                    f"IO={'N/A' if math.isnan(avg_io) else f'{avg_io:.3f}s'}  "
                    f"KV={avg_kv/1e6:.2f}MB"
                )

    # ── Write raw JSONL ──────────────────────────────────────────────────────
    with open(raw_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")
    log.info(f"Raw records → {raw_path}")

    # ── Write summary CSV ────────────────────────────────────────────────────
    fieldnames = [
        "dataset", "k", "condition", "condition_label", "n_examples",
        "n_hall", "n_total", "paired_n", "refusal_rate",
        "EM", "contain_EM", "F1",
        "hallucination_rate", "entailment_score",
        "avg_ttft_s", "avg_io_s", "avg_kv_bytes",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames,
            quoting=csv.QUOTE_ALL,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    log.info(f"Summary CSV → {csv_path}")

    # ── Write summary JSON ───────────────────────────────────────────────────
    with open(json_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    log.info(f"Summary JSON → {json_path}")

    # ── Task 12: Write experiment metadata ───────────────────────────────────
    meta = {
        "git_commit":             _get_git_commit(),
        "python_version":         sys.version,
        "platform":               platform.platform(),
        "torch_version":          torch.__version__,
        "transformers_version":   transformers.__version__,
        "cuda_available":         torch.cuda.is_available(),
        "cuda_device":            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "global_seed":            GLOBAL_SEED,
        "pip_freeze":             _get_pip_freeze(),
        "resolved_args":          vars(args),
        "timestamp":              timestamp,
    }
    meta_path = os.path.join(args.output_dir, f"meta_{timestamp}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Experiment metadata → {meta_path}")

    # ── Print table ──────────────────────────────────────────────────────────
    headers = ["Dataset", "K", "Cond", "n", "Paired", "Refuse",
               "EM", "ContEM", "F1", "Hall↓", "Ent↑", "TTFT(s)", "IO(s)", "KV(MB)"]
    table = [
        [r["dataset"], r["k"], r["condition"], r["n_examples"], r["paired_n"],
         r["refusal_rate"], r["EM"], r["contain_EM"], r["F1"],
         r["hallucination_rate"], r["entailment_score"],
         r["avg_ttft_s"], r["avg_io_s"], round(r["avg_kv_bytes"] / 1e6, 2)]
        for r in summary_rows
    ]
    print("\n" + tabulate(table, headers=headers, tablefmt="grid"))



if __name__ == "__main__":
    main()
