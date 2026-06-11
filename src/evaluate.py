"""
evaluate.py  –  Main evaluation script for the TurboRAG KV-cache quantization study.

Implements all experimental conditions:
  C0  Gold Oracle RAG   – full raw-text context, no precomputed cache
  C1  FP16 TurboRAG    – precomputed FP16 chunk caches stitched at query time
  C2  INT8 TurboRAG    – offline INT8 quantized chunk caches
  C3  INT4 TurboRAG    – offline INT4 quantized chunk caches

FIX LOG
────────────────────────────────────────────────────────────────────
  FIX-2LAYER    _normalise_legacy misdetected 2-layer old-format cache as new format.
                Resolved via from_dynamic_cache flag — caller always knows the source.

  FIX-VERCHECK  transformers 4.47+ reverted from_legacy_cache to old per-layer format.
                _legacy_fmt_is_new() detects version at runtime and passes the correct
                format so the code works on 4.44, 4.45, 4.46, and 4.47+.

  FIX-TEMP      do_sample=False with model config temperature!=1.0 raised UserWarning.
                Fixed by passing temperature=1.0 explicitly in all generate() calls.

  FIX-EM        batch_em now receives _extract_short_answer(pred) instead of raw pred.
                Raw multi-sentence generations never matched gold spans → EM≈0.
                F1 and HHEM continue to use raw preds (intentional).

  FIX-ROOT      from_legacy_cache on transformers>=4.45 requires new format.
  FIX-SEEN      _seen_tokens not updated by from_legacy_cache (4.45-4.47). Patched.
  FIX-MASK      cached_len read from key tensor shape, not get_seq_length().
  FIX-CSV       DictWriter quoting=QUOTE_ALL prevents pandas ParserError.
  FIX-NAN       avg_ttft NaN guard prevents tabulate crash on empty preds.
  FIX-1         torch_dtype= (not dtype=) in from_pretrained.
  FIX-2         context joined with "\\n\\n".
  FIX-3         trim_context_for_hhem uses full joined context up to 512 tokens.
  FIX-4         F1 computed on raw preds; EM on extracted short answer.
  FIX-5         batch_contain_em added.
  FIX-6         Scorers receive raw preds.
  FIX-7         n_hall and n_total written to summary row.
  FIX-8         _load_from_hf uses while loop for nested list unwrapping.
  FIX-DEEPCOPY  prefix_kvcache deep-copied per query to prevent mutation.
  FIX-ERRLOG    log.error with exc_info=True for full tracebacks.
"""

import os
import sys
import json
import time
import copy
import logging
import argparse
import csv
import math
import re
from datetime import datetime
from typing import List, Optional, Dict, Any

import torch
from tqdm import tqdm
from tabulate import tabulate

# Change 3: import top-level transformers for version check
import transformers
from transformers import AutoTokenizer, DynamicCache

sys.path.insert(0, os.path.dirname(__file__))
from qwen2 import Qwen2ModifiedForCausalLM
from kv_quantization import decompress_kvcache, cache_size_bytes
from metrics import (
    batch_em, batch_f1,
    HHEMScorer, DeBERTaNLIScorer,
    hallucination_rate, mean_entailment,
)

from llama_index.core import Settings, load_index_from_storage, StorageContext, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


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

QUERY_SUFFIX_TEMPLATE = "\n\nQuestion: {query}<|im_end|><|im_start|>assistant\n"


def build_full_prompt(chunks: List[str], query: str) -> str:
    return SYSTEM_PROMPT + "".join(chunks) + QUERY_SUFFIX_TEMPLATE.format(query=query)


def build_query_suffix(query: str) -> str:
    return QUERY_SUFFIX_TEMPLATE.format(query=query)


# ──────────────────────────────────────────────────────────────────────────────
# KV cache stitching  (FIX-ROOT + FIX-SEEN + FIX-2LAYER + FIX-VERCHECK)
# ──────────────────────────────────────────────────────────────────────────────

def _normalise_legacy(legacy_cache, from_dynamic_cache: bool = False):
    """
    Normalise any legacy cache to a list of per-layer (k, v) pairs.

    Args:
        legacy_cache: from DynamicCache.to_legacy_cache() or torch.load from disk.
        from_dynamic_cache: True when the source is DynamicCache.to_legacy_cache().
            This flag is the ONLY reliable disambiguator for 2-layer models.

    Why the flag is necessary (FIX-2LAYER):
        transformers>=4.45 new format: ((k0,k1,...,kL-1), (v0,v1,...,vL-1))  len==2
        Old per-layer format:          ((k0,v0),(k1,v1),...,(kL-1,vL-1))     len==L
        For L==2 both have len==2 with tensors at [0][0], so shape probing fails.
        The caller always knows the source, so we require it to pass that explicitly.

    Returns:
        list of (k_tensor, v_tensor) tuples, one per layer.
    """
    if from_dynamic_cache:
        # DynamicCache.to_legacy_cache() on 4.45/4.46 returns new format.
        # On 4.47+ it reverts to old per-layer format — but we call
        # _normalise_legacy here only in stack_past_key_values which calls
        # to_legacy_cache(). We detect both cases correctly:
        if (
            len(legacy_cache) == 2
            and hasattr(legacy_cache[0], "__len__")
            and len(legacy_cache[0]) > 0
            and isinstance(legacy_cache[0][0], torch.Tensor)
            # Distinguish: in new format legacy_cache[0] is the all-keys tuple (len==L);
            # in old format legacy_cache[0] is (k0, v0) — a pair.
            # We check that legacy_cache[0][0] and legacy_cache[1][0] are both tensors
            # AND the inner length is > 1 element per slot (new format has L keys/values).
            # For old format with L==2: legacy_cache[0] == (k0,v0), len==2, inner[0] is tensor.
            # For new format with L==2: legacy_cache[0] == (k0,k1), len==2, inner[0] is tensor.
            # These are truly ambiguous from shape alone — hence the flag.
        ):
            # If from_dynamic_cache=True, trust the caller: it IS new format.
            # (4.47+ to_legacy_cache reverts to old format — see _legacy_fmt_is_new below)
            if _legacy_fmt_is_new():
                all_keys, all_values = legacy_cache
                return list(zip(all_keys, all_values))
            else:
                # 4.47+ to_legacy_cache returns old per-layer format again.
                return list(legacy_cache)
        log.warning("from_dynamic_cache=True but unexpected structure; falling back.")
        return list(legacy_cache)

    # ── Disk-loaded cache: always old per-layer format ──────────────────────
    if len(legacy_cache) != 2:
        # L != 2: len==L unambiguously means old per-layer format.
        return list(legacy_cache)

    # len==2 on disk: could be new format saved from 4.45/4.46, or old 2-layer format.
    inner0 = legacy_cache[0]
    if (
        hasattr(inner0, "__len__")
        and len(inner0) > 2  # all-keys tuple has L>2 elements
        and isinstance(inner0[0], torch.Tensor)
    ):
        # New format saved to disk (uncommon but valid).
        all_keys, all_values = legacy_cache
        return list(zip(all_keys, all_values))

    # 2-layer old per-layer format: treat as [(k0,v0),(k1,v1)].
    return list(legacy_cache)


def _to_new_legacy_format(per_layer_pairs):
    """
    Convert per-layer (k, v) pairs → new ((k0,k1,...), (v0,v1,...)) format.
    Required by DynamicCache.from_legacy_cache on transformers 4.45–4.46.
    """
    all_keys   = tuple(kv[0] for kv in per_layer_pairs)
    all_values = tuple(kv[1] for kv in per_layer_pairs)
    return (all_keys, all_values)


def _legacy_fmt_is_new() -> bool:
    """
    FIX-VERCHECK: detect which format from_legacy_cache expects at runtime.

    transformers 4.45–4.46 → expects ((k0,...,kL-1), (v0,...,vL-1))  [new format]
    transformers 4.44 and 4.47+ → expects ((k0,v0),(k1,v1),...)       [old format]

    Parsing the version string defensively to handle pre-release suffixes
    like "4.47.0.dev0" or "4.46.1".
    """
    try:
        parts = transformers.__version__.split(".")
        major, minor = int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        # Unparseable version — assume old format (safe default).
        log.warning(f"Cannot parse transformers version '{transformers.__version__}'; "
                    "assuming old-format from_legacy_cache.")
        return False
    return (major == 4) and (minor in (45, 46))


def _set_seen_tokens(cache: DynamicCache) -> DynamicCache:
    """
    FIX-SEEN: transformers 4.45–4.47 does not update _seen_tokens in
    from_legacy_cache. Patch it from the key tensor shape so
    get_seq_length() returns the correct value.
    """
    if hasattr(cache, "key_cache") and len(cache.key_cache) > 0:
        cache._seen_tokens = cache.key_cache[0].shape[2]  # (batch, heads, seq, head_dim)
    return cache


def legacy_to_dynamic(legacy, from_dynamic_cache: bool = False) -> DynamicCache:
    """
    Convert a legacy cache (disk or live DynamicCache) to a DynamicCache.

    FIX-ROOT + FIX-2LAYER + FIX-VERCHECK: normalise to per-layer pairs first,
    then build the format that the installed transformers version expects.
    FIX-SEEN: patch _seen_tokens immediately after construction.
    """
    per_layer = _normalise_legacy(legacy, from_dynamic_cache=from_dynamic_cache)
    # Pass the correct format based on installed transformers version.
    fmt = _to_new_legacy_format(per_layer) if _legacy_fmt_is_new() else tuple(per_layer)
    cache = DynamicCache.from_legacy_cache(fmt)
    return _set_seen_tokens(cache)


def stack_past_key_values(past_key_values_list: List[DynamicCache]) -> DynamicCache:
    """
    Concatenate a list of DynamicCache objects along the sequence dimension.

    FIX-VERCHECK: builds the correct format for the installed transformers version.
    FIX-2LAYER: uses from_dynamic_cache=True when source is a live DynamicCache.
    """
    legacy_list = []
    for c in past_key_values_list:
        if hasattr(c, "to_legacy_cache"):
            raw = c.to_legacy_cache()
            legacy_list.append(_normalise_legacy(raw, from_dynamic_cache=True))
        else:
            legacy_list.append(_normalise_legacy(c, from_dynamic_cache=False))

    num_layers = len(legacy_list[0])
    stacked_pairs = [
        (
            torch.cat([c[layer][0] for c in legacy_list], dim=2),
            torch.cat([c[layer][1] for c in legacy_list], dim=2),
        )
        for layer in range(num_layers)
    ]
    fmt = _to_new_legacy_format(stacked_pairs) if _legacy_fmt_is_new() else tuple(stacked_pairs)
    cache = DynamicCache.from_legacy_cache(fmt)
    return _set_seen_tokens(cache)


# ──────────────────────────────────────────────────────────────────────────────
# Context helpers
# ──────────────────────────────────────────────────────────────────────────────

def trim_context_for_hhem(context: str, max_tokens: int = 512) -> str:
    """Truncate context to fit inside the scorer's token window. (FIX-3)"""
    return context[: max_tokens * 4]


# ──────────────────────────────────────────────────────────────────────────────
# Accuracy helpers
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_answer(text: str) -> str:
    """Lower-case, strip punctuation and articles, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _extract_short_answer(text: str) -> str:
    """
    FIX-EM / Change 2: Pull a short answer span from a free-form generation.

    The model outputs multi-sentence explanations; EM against a gold span like
    "Paris" will always be 0 on raw text. This heuristic extracts the likely
    answer span so EM is meaningful.

    Used ONLY for EM computation. F1, HHEM, and NLI intentionally keep raw preds
    because they reward partial matches / faithfulness of the full response.

    Strategy:
      1. Look for explicit answer markers ("answer is", "is", "was", etc.)
         and take the first short span after the marker.
      2. If no marker found, return the first sentence (up to first '.').
    """
    text = text.strip()
    # Common answer-introducing phrases, ordered by specificity.
    markers = [
        "answer is ",
        "answer: ",
        " is ",
        " was ",
        " are ",
        " were ",
    ]
    for marker in markers:
        idx = text.lower().find(marker)
        if idx != -1:
            span = text[idx + len(marker):]
            # Trim at first sentence boundary or comma.
            span = span.split(".")[0].split(",")[0].strip()
            # Only trust short spans — long ones are likely still a sentence.
            if 1 <= len(span.split()) <= 10:
                return span
    # Fallback: first sentence.
    return text.split(".")[0].strip()


def batch_contain_em(predictions: List[str], ground_truths: List[str]) -> float:
    """
    FIX-5: Contain-EM — fraction where normalised gold is a substring of
    normalised prediction.
    """
    if not predictions:
        return 0.0
    return sum(
        1 for p, g in zip(predictions, ground_truths)
        if _normalize_answer(g) in _normalize_answer(p)
    ) / len(predictions)


# ──────────────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────────────

EOS_TOKEN_IDS  = [151645, 151643]
MAX_NEW_TOKENS = 64


def _get_cache_seq_len(cache: DynamicCache) -> int:
    """FIX-MASK: read from key tensor shape, not _seen_tokens."""
    if hasattr(cache, "key_cache") and len(cache.key_cache) > 0:
        return cache.key_cache[0].shape[2]
    return cache.get_seq_length()


def generate_with_cache(
    model, tokenizer, device,
    past_kvcache: Optional[DynamicCache],
    query: str,
    chunks: List[str],
) -> str:
    """Generate an answer given either a stitched KV cache (C1-C3) or raw chunks (C0)."""
    with torch.no_grad():
        if past_kvcache is not None:
            suffix         = build_query_suffix(query)
            new_input_ids  = tokenizer.encode(suffix, return_tensors="pt").to(device)
            cached_len     = _get_cache_seq_len(past_kvcache)
            attention_mask = torch.ones(
                1, cached_len + new_input_ids.shape[1],
                device=device, dtype=torch.long,
            )
            outputs = model.generate(
                new_input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                past_key_values=past_kvcache,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=1.0,       # FIX-TEMP: suppress UserWarning
                eos_token_id=EOS_TOKEN_IDS,
            )
            new_tokens = outputs[0][new_input_ids.shape[1]:]
            return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        else:
            prompt    = build_full_prompt(chunks, query)
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            outputs   = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=1.0,       # FIX-TEMP: suppress UserWarning
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

    FIX-DEEPCOPY: prefix_kvcache is shared across all queries. Deep-copy it
    at the start of every C1/C2/C3 call so stitching never mutates the original.
    """
    nodes_k       = retrieved_nodes[:k]
    precision     = PRECISION_MAP[condition]
    chunk_texts   = []
    kv_size_bytes = 0

    t0 = time.perf_counter()

    if condition == "C0":
        for nws in nodes_k:
            chunk_texts.append(nws.node.metadata.get("raw_text", nws.node.text))
        answer = generate_with_cache(model, tokenizer, device, None, query, chunk_texts)

    else:
        # FIX-DEEPCOPY
        prefix_copy  = copy.deepcopy(prefix_kvcache)
        kvcache_list = [prefix_copy]

        for nws in nodes_k:
            node       = nws.node
            cache_key  = f"kvcache_{precision}"
            fpath      = node.metadata[cache_key]
            compressed = torch.load(fpath, weights_only=True)
            legacy     = decompress_kvcache(compressed, precision)
            # Disk-loaded → from_dynamic_cache=False (always old per-layer format)
            kvcache_list.append(legacy_to_dynamic(legacy, from_dynamic_cache=False))
            chunk_texts.append(node.metadata.get("raw_text", node.text))
            kv_size_bytes += cache_size_bytes(compressed, precision)

        stitched = stack_past_key_values(kvcache_list)
        answer   = generate_with_cache(model, tokenizer, device, stitched, query, [])

    ttft = time.perf_counter() - t0

    return {
        "answer":        answer,
        "context":       "\n\n".join(chunk_texts),   # FIX-2
        "ttft_seconds":  ttft,
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
        a = row.get(answer_field)   or row.get("answer", "")
        # FIX-8: peel nested list/dict until plain string
        while isinstance(a, list):
            a = a[0] if a else ""
        if isinstance(a, dict):
            a = a.get("text", "") or a.get("answer", "")
            while isinstance(a, list):
                a = a[0] if a else ""
        examples.append({"query": q, "answer": str(a)})
    return examples


def _load_from_jsonl(query_file: str, num_examples: int) -> List[Dict]:
    examples = []
    with open(query_file) as f:
        for line in f:
            data   = json.loads(line.strip())
            query  = data.get("query") or data.get("question", "")
            answer = data.get("answer") or data.get("answers", "")
            while isinstance(answer, list):
                answer = answer[0] if answer else ""
            examples.append({"query": query, "answer": answer})
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
    parser.add_argument("--similarity_top_k", type=int, default=5)
    args = parser.parse_args()

    assert args.similarity_top_k >= max(args.k_values), \
        "--similarity_top_k must be >= max(k_values)"

    log.info(f"transformers version: {transformers.__version__}  "
             f"(from_legacy_cache expects {'NEW' if _legacy_fmt_is_new() else 'OLD'} format)")

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

    attn_impl = "flash_attention_2" if args.use_flash_attn else "eager"
    log.info(f"Loading model: {args.model_name} (attn={attn_impl})")
    model = Qwen2ModifiedForCausalLM.from_pretrained(
        args.model_name,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,   # FIX-1
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    log.info("Pre-computing system prefix KV cache ...")
    prefix_inputs = tokenizer([SYSTEM_PROMPT], return_tensors="pt", padding=True)
    with torch.no_grad():
        prefix_out = model(
            prefix_inputs["input_ids"].to(device),
            attention_mask=prefix_inputs["attention_mask"].to(device),
            use_cache=True,
        )
    prefix_kvcache = prefix_out.past_key_values
    # Defensive: ensure _seen_tokens is correct even for the prefix cache.
    _set_seen_tokens(prefix_kvcache)

    log.info("Loading retrieval index ...")
    Settings.embed_model = HuggingFaceEmbedding(model_name=args.embedding_model_name)
    storage_ctx = StorageContext.from_defaults(persist_dir=args.storage_dir)
    index       = load_index_from_storage(storage_ctx)
    retriever   = index.as_retriever(similarity_top_k=args.similarity_top_k)

    hhem_scorer = HHEMScorer(device=device)       if args.eval_hhem else None
    nli_scorer  = DeBERTaNLIScorer(device=device) if args.eval_nli  else None

    all_records  = []
    summary_rows = []

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

        for k in args.k_values:
            for condition in args.conditions:
                log.info(f"  Condition={condition}  K={k}")
                preds, refs, contexts = [], [], []
                ttft_list, kv_size_list = [], []

                for ex in tqdm(examples, desc=f"{ds_name} K={k} {condition}"):
                    query  = ex["query"]
                    answer = ex["answer"]
                    try:
                        qb    = QueryBundle(query_str=query)
                        nodes = retriever.retrieve(qb)
                        result = run_query(
                            query=query, retrieved_nodes=nodes, k=k,
                            condition=condition, model=model, tokenizer=tokenizer,
                            device=device, prefix_kvcache=prefix_kvcache,
                        )
                        preds.append(result["answer"])
                        refs.append(answer)
                        contexts.append(result["context"])
                        ttft_list.append(result["ttft_seconds"])
                        kv_size_list.append(result["kv_size_bytes"])
                        all_records.append({
                            "dataset":    ds_name,
                            "k":          k,
                            "condition":  condition,
                            "query":      query,
                            "gold":       answer,
                            "prediction": result["answer"],
                            "context":    result["context"],
                            "ttft":       result["ttft_seconds"],
                            "kv_bytes":   result["kv_size_bytes"],
                        })
                    except Exception as e:
                        # FIX-ERRLOG: full traceback, never silent
                        log.error(
                            f"Skipped query [{ds_name} K={k} {condition}]: {e}",
                            exc_info=True,
                        )

                # FIX-EM (Change 2): EM uses extracted short answer spans.
                # F1 keeps raw preds intentionally.
                em_preds   = [_extract_short_answer(p) for p in preds]
                f1_score   = batch_f1(preds, refs)          # raw
                em_score   = batch_em(em_preds, refs)       # extracted
                contain_em = batch_contain_em(preds, refs)  # FIX-5, raw

                # FIX-NAN: guard against empty preds
                avg_ttft = (sum(ttft_list) / len(ttft_list)) if ttft_list else float("nan")
                avg_kv   = (sum(kv_size_list) / len(kv_size_list)) if kv_size_list else 0

                hall_rate = float("nan")
                ent_score = float("nan")
                n_hall    = 0
                n_total   = len(preds)
                HHEM_THRESHOLD = 0.5

                if hhem_scorer and contexts and preds:
                    hhem_contexts = [trim_context_for_hhem(c) for c in contexts]
                    faith_scores  = hhem_scorer.batch_score(hhem_contexts, preds)  # FIX-6
                    hall_rate     = hallucination_rate(faith_scores)
                    n_total       = len(faith_scores)
                    n_hall        = sum(1 for s in faith_scores if s < HHEM_THRESHOLD)
                    matching = [
                        r for r in all_records
                        if r["dataset"] == ds_name and r["k"] == k and r["condition"] == condition
                    ]
                    for rec, fs in zip(matching, faith_scores):
                        rec["hhem_faithfulness"] = fs
                        rec["hhem_hallucinated"] = fs < HHEM_THRESHOLD

                if nli_scorer and contexts and preds:
                    nli_contexts = [trim_context_for_hhem(c, max_tokens=512) for c in contexts]
                    nli_scores = nli_scorer.batch_score(nli_contexts, preds)  # FIX-6
                    ent_score  = mean_entailment(nli_scores)
                    matching = [
                        r for r in all_records
                        if r["dataset"] == ds_name and r["k"] == k and r["condition"] == condition
                    ]
                    for rec, ns in zip(matching, nli_scores):
                        rec["nli_entailment"]    = ns[0]
                        rec["nli_neutral"]       = ns[1]
                        rec["nli_contradiction"] = ns[2]

                row = {
                    "dataset":            ds_name,
                    "k":                  k,
                    "condition":          condition,
                    "condition_label":    CONDITION_LABELS[condition],
                    "n_examples":         len(preds),
                    "n_hall":             n_hall,    # FIX-7
                    "n_total":            n_total,   # FIX-7
                    "EM":                 round(em_score,   4),
                    "contain_EM":         round(contain_em, 4),
                    "F1":                 round(f1_score,   4),
                    "hallucination_rate": round(hall_rate, 4) if not math.isnan(hall_rate) else "N/A",
                    "entailment_score":   round(ent_score,  4) if not math.isnan(ent_score)  else "N/A",
                    "avg_ttft_s":         round(avg_ttft, 4) if not math.isnan(avg_ttft) else "N/A",
                    "avg_kv_bytes":       int(avg_kv),
                }
                summary_rows.append(row)
                log.info(
                    f"    n={len(preds)}  EM={em_score:.3f}  ContEM={contain_em:.3f}  "
                    f"F1={f1_score:.3f}  "
                    f"Hall={'N/A' if math.isnan(hall_rate) else f'{hall_rate:.3f}'}  "
                    f"Ent={'N/A' if math.isnan(ent_score) else f'{ent_score:.3f}'}  "
                    f"TTFT={'N/A' if math.isnan(avg_ttft) else f'{avg_ttft:.3f}s'}  "
                    f"KV={avg_kv/1e6:.2f}MB"
                )

    # ── Write raw JSONL ──────────────────────────────────────────────────────
    with open(raw_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")
    log.info(f"Raw records → {raw_path}")

    # ── Write summary CSV (FIX-CSV) ──────────────────────────────────────────
    fieldnames = [
        "dataset", "k", "condition", "condition_label", "n_examples",
        "n_hall", "n_total", "EM", "contain_EM", "F1",
        "hallucination_rate", "entailment_score", "avg_ttft_s", "avg_kv_bytes",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames,
            quoting=csv.QUOTE_ALL,   # FIX-CSV
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    log.info(f"Summary CSV → {csv_path}")

    # ── Write summary JSON ───────────────────────────────────────────────────
    with open(json_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    log.info(f"Summary JSON → {json_path}")

    # ── Print table ──────────────────────────────────────────────────────────
    headers = ["Dataset", "K", "Cond", "n", "EM", "ContEM", "F1",
               "Hall↓", "Ent↑", "TTFT(s)", "KV(MB)"]
    table = [
        [r["dataset"], r["k"], r["condition"], r["n_examples"],
         r["EM"], r["contain_EM"], r["F1"],
         r["hallucination_rate"], r["entailment_score"],
         r["avg_ttft_s"], round(r["avg_kv_bytes"] / 1e6, 2)]
        for r in summary_rows
    ]
    print("\n" + tabulate(table, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()