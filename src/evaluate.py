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

  FIX-EXTRACT   _extract_short_answer rewritten: removed greedy " is "/" was " markers
                that matched mid-sentence and produced wrong spans → EM≈0 on all conds.
  FIX-SAFECACHE _safe_from_legacy_cache: auto-detect format with fallback if
                from_legacy_cache produces an empty DynamicCache.
  FIX-F1DIV     Guard against ZeroDivisionError when prediction normalises to empty.
  FIX-HHEM-IDX  HHEM/NLI record lookup uses index slicing instead of O(n²) filter.
  FIX-TLOAD     torch.load uses weights_only=False for dicts with mixed Python types.
  FIX-NORMDUP   Removed duplicate _normalize_answer; uses _normalize from metrics.py.
  FIX-GUARD     generate_with_cache: if stacked cache is empty (cached_len==0), fall
                back to C0-style full-prompt generation instead of silently generating
                a context-free answer that inflates C1/C2/C3 scores.

  FIX-CACHEPOS  generate_with_cache: pass cache_position explicitly to model.generate()
                when using a pre-filled DynamicCache. Without it, transformers tries to
                auto-compute cache_position from scratch, produces an empty tensor of
                size 0, and crashes with:
                  IndexError: index -1 is out of bounds for dimension 0 with size 0
                in prepare_inputs_for_generation (utils.py line 388).
                Fix: cache_position=torch.arange(cached_len, cached_len + new_len).

  FIX-BUGA      prefix_kvcache was a raw tuple when model() returns legacy format
                (transformers<4.38 or direct model() call without DynamicCache).
                hasattr guard checked but never converted → tuple flowed to
                stack_past_key_values() → crash at first.key_cache.
                Fix: pass empty DynamicCache() to forward; defensive isinstance check.

  FIX-BUGB      Stitched cache keys are un-rotated (TurboRAG design). During generate(),
                incremental decode steps were re-applying RoPE to ALL cached keys on
                every step → double/triple/N-tuple RoPE on previously decoded tokens.
                Fix: _rope_applied_to_prefix flag on stitched cache; qwen2.py checks it.

  FIX-BUGD      generate_with_cache did not validate that the stitched cache layer count
                matches model.config.num_hidden_layers. Mismatch causes silent wrong
                answers. Fix: validate and fall back to C0 if mismatched.

  FIX-BUGE      prefix_kvcache was computed with past_key_values=DynamicCache() which
                triggered Regime 2 → keys stored WITH RoPE.  Chunk caches use Regime 3
                → keys stored WITHOUT RoPE.  Stitching them and re-applying RoPE in
                Regime 1 caused double-RoPE on prefix keys → garbage attention.
                Fix: pass past_key_values=None for prefix computation (Regime 3).

  FIX-BUGF      Regime 1 in qwen2.py applied RoPE to the full key sequence but only
                updated the local variable — the cache still held un-rotated keys.
                Subsequent Regime-2 decode steps read wrong keys → garbage after token 1.
                Fix: write rotated keys back with past_key_values.key_cache[layer] = key_states.

  FIX-BUGG      chunk_cache.py compute_and_save_chunk did not explicitly pass
                past_key_values=None, making it fragile if transformers auto-creates a
                cache.  Fix: add explicit past_key_values=None.
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


import transformers
from transformers import AutoTokenizer, DynamicCache


sys.path.insert(0, os.path.dirname(__file__))
from qwen2 import Qwen2ModifiedForCausalLM
from kv_quantization import decompress_kvcache, cache_size_bytes
from metrics import (
    batch_em, batch_f1,
    HHEMScorer, DeBERTaNLIScorer,
    hallucination_rate, mean_entailment,
    _normalize as _normalize_answer,
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


def _normalise_legacy(legacy_cache):
    """
    Normalise any legacy cache (from disk) to a list of per-layer (k, v) pairs.

    Disk caches from decompress_kvcache are always old per-layer format:
        ((k0,v0), (k1,v1), ..., (kL-1,vL-1))   len == L

    But caches saved on transformers 4.45/4.46 could be new format:
        ((k0,k1,...,kL-1), (v0,v1,...,vL-1))     len == 2

    We detect both and always return: list of (k_tensor, v_tensor) tuples.
    """
    if len(legacy_cache) == 2:
        inner0 = legacy_cache[0]
        if (
            hasattr(inner0, "__len__")
            and len(inner0) > 2
            and isinstance(inner0[0], torch.Tensor)
        ):
            # New format: ((all_keys), (all_values))
            all_keys, all_values = legacy_cache
            return list(zip(all_keys, all_values))
        # Old format with exactly 2 layers: [(k0,v0), (k1,v1)]
        return list(legacy_cache)
    # Old format with L != 2 layers
    return list(legacy_cache)



def _build_dynamic_cache(per_layer_pairs) -> DynamicCache:
    """
    Build a DynamicCache by directly populating key_cache/value_cache lists.

    This completely bypasses DynamicCache.from_legacy_cache() which has
    format differences across transformers versions (4.44 old, 4.45-4.46 new,
    4.47+ old again) and causes cache_position crashes in generate().

    DynamicCache.key_cache and .value_cache are simple Python lists of tensors
    on ALL transformers versions — this is the only stable API.
    """
    cache = DynamicCache()
    for k_tensor, v_tensor in per_layer_pairs:
        cache.key_cache.append(k_tensor)
        cache.value_cache.append(v_tensor)
    # Patch _seen_tokens from actual tensor shape (FIX-SEEN)
    if cache.key_cache:
        cache._seen_tokens = cache.key_cache[0].shape[2]
    return cache



def legacy_to_dynamic(legacy) -> DynamicCache:
    """
    Convert a disk-loaded legacy cache to a DynamicCache.

    Normalises the legacy format to per-layer (k, v) pairs, then directly
    populates the DynamicCache — no from_legacy_cache() call needed.
    """
    per_layer = _normalise_legacy(legacy)
    return _build_dynamic_cache(per_layer)



def stack_past_key_values(past_key_values_list: List[DynamicCache]) -> DynamicCache:
    """
    Concatenate a list of DynamicCache objects along the sequence dimension.

    Directly reads from key_cache/value_cache lists and builds a new
    DynamicCache.  No to_legacy_cache()/from_legacy_cache() round-trip.
    """
    first = past_key_values_list[0]
    num_layers = len(first.key_cache)

    if num_layers == 0:
        log.error(
            "stack_past_key_values: first cache has 0 layers. "
            "key_cache lengths: %s",
            [len(c.key_cache) for c in past_key_values_list],
        )
        return DynamicCache()

    stacked_pairs = []
    for layer in range(num_layers):
        stacked_k = torch.cat(
            [c.key_cache[layer] for c in past_key_values_list], dim=2
        )
        stacked_v = torch.cat(
            [c.value_cache[layer] for c in past_key_values_list], dim=2
        )
        stacked_pairs.append((stacked_k, stacked_v))

    return _build_dynamic_cache(stacked_pairs)



# ──────────────────────────────────────────────────────────────────────────────
# Context helpers
# ──────────────────────────────────────────────────────────────────────────────


def trim_context_for_hhem(context: str, max_tokens: int = 512) -> str:
    """Truncate context to fit inside the scorer's token window. (FIX-3)"""
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
            # FIX-BUGD: validate that cache layers match model layers
            n_model_layers = model.config.num_hidden_layers
            n_cache_layers = len(past_kvcache.key_cache)
            if n_cache_layers != n_model_layers:
                log.error(
                    "Cache layer mismatch: cache has %d layers, model has %d. "
                    "Falling back to C0-style generation.",
                    n_cache_layers, n_model_layers,
                )
                return generate_with_cache(
                    model, tokenizer, device, None, query, chunks
                )


            suffix        = build_query_suffix(query)
            new_input_ids = tokenizer.encode(suffix, return_tensors="pt").to(device)
            cached_len    = _get_cache_seq_len(past_kvcache)

            # FIX-GUARD: if the stacked cache is empty fall back to C0-style.
            if cached_len == 0:
                log.error(
                    "generate_with_cache: stacked cache is empty (cached_len=0). "
                    "Falling back to C0-style full-prompt generation."
                )
                prompt    = build_full_prompt(chunks, query)
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                outputs   = model.generate(
                    input_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    temperature=1.0,
                    eos_token_id=EOS_TOKEN_IDS,
                )
                new_tokens = outputs[0][input_ids.shape[1]:]
                return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            new_len = new_input_ids.shape[1]

            attention_mask = torch.ones(
                1, cached_len + new_len,
                device=device, dtype=torch.long,
            )

            # ── FIX-CACHEPOS ────────────────────────────────────────────────
            # transformers.generate() with a pre-filled DynamicCache REQUIRES
            # an explicit cache_position tensor telling it where in the overall
            # sequence the new input tokens sit.
            #
            # Without it, generate() tries to infer cache_position internally.
            # The inference path calls get_seq_length() on the DynamicCache,
            # which on some transformers versions returns 0 (or builds an empty
            # tensor), leading to:
            #
            #   IndexError: index -1 is out of bounds for dimension 0 with size 0
            #   in prepare_inputs_for_generation (utils.py line 388)
            #
            # Fix: pass cache_position=torch.arange(cached_len, cached_len + new_len)
            # so generate() knows exactly where the suffix tokens start.
            # ────────────────────────────────────────────────────────────────
            cache_position = torch.arange(
                cached_len, cached_len + new_len,
                device=device, dtype=torch.long,
            )

            outputs = model.generate(
                new_input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                past_key_values=past_kvcache,
                attention_mask=attention_mask,
                cache_position=cache_position,      # FIX-CACHEPOS
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=1.0,                    # FIX-TEMP
                eos_token_id=EOS_TOKEN_IDS,
            )
            new_tokens = outputs[0][new_len:]
            return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

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

            # FIX-TLOAD: compressed caches are dicts with mixed Python types
            try:
                compressed = torch.load(fpath, weights_only=True)
            except Exception:
                compressed = torch.load(fpath, weights_only=False)
            legacy     = decompress_kvcache(compressed, precision)
            kvcache_list.append(legacy_to_dynamic(legacy))
            chunk_texts.append(node.metadata.get("raw_text", node.text))
            kv_size_bytes += cache_size_bytes(compressed, precision)

        stitched = stack_past_key_values(kvcache_list)
        # FIX-BUGB: signal to qwen2.py that prefix keys are UN-rotated and
        # need RoPE applied on the first decode step.
        stitched._rope_applied_to_prefix = False
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
             f"(using direct DynamicCache population — no from_legacy_cache)")

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

    # FIX-1: torch_dtype= (not dtype=)
    model = Qwen2ModifiedForCausalLM.from_pretrained(
        args.model_name,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    log.info("Pre-computing system prefix KV cache ...")
    prefix_inputs = tokenizer([SYSTEM_PROMPT], return_tensors="pt", padding=True)
    with torch.no_grad():
        # FIX-BUGE: pass past_key_values=None so Qwen2ModifiedAttention enters
        # Regime 3 (keys stored WITHOUT RoPE).  This matches the TurboRAG chunk
        # cache design — all cached keys are un-rotated; RoPE is applied once at
        # decode time via Regime 1.
        #
        # Previously FIX-BUGA passed DynamicCache() which triggered Regime 2
        # (RoPE applied to keys before storing).  Stitching those rotated prefix
        # keys with un-rotated chunk keys, then re-applying RoPE in Regime 1,
        # caused double-RoPE on the prefix → garbage attention.
        prefix_out = model(
            prefix_inputs["input_ids"].to(device),
            attention_mask=prefix_inputs["attention_mask"].to(device),
            past_key_values=None,     # Regime 3: un-rotated keys
            use_cache=True,
        )
    prefix_kvcache = prefix_out.past_key_values

    # Convert to DynamicCache if model returned a tuple (FIX-BUGA defensive)
    if not isinstance(prefix_kvcache, DynamicCache):
        log.warning(
            "prefix_kvcache is %s, not DynamicCache — converting. "
            "Check transformers version.",
            type(prefix_kvcache).__name__,
        )
        per_layer = _normalise_legacy(prefix_kvcache)
        prefix_kvcache = _build_dynamic_cache(per_layer)

    # Always patch _seen_tokens from actual tensor shape (FIX-SEEN)
    if len(prefix_kvcache.key_cache) > 0:
        prefix_kvcache._seen_tokens = prefix_kvcache.key_cache[0].shape[2]
    log.info(
        "Prefix KV cache ready: %d layers, seq_len=%d",
        len(prefix_kvcache.key_cache),
        prefix_kvcache._seen_tokens,
    )

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
                record_start_idx = len(all_records)  # FIX-HHEM-IDX

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
                        # FIX-ERRLOG
                        log.error(
                            f"Skipped query [{ds_name} K={k} {condition}]: {e}",
                            exc_info=True,
                        )

                # FIX-EM: EM uses extracted short answer spans; F1 uses raw preds.
                em_preds   = [_extract_short_answer(p) for p in preds]
                f1_score   = batch_f1(preds, refs)
                em_score   = batch_em(em_preds, refs)
                contain_em = batch_contain_em(preds, refs)

                if preds and log.isEnabledFor(logging.DEBUG):
                    for i in range(min(3, len(preds))):
                        log.debug(
                            "  [%d] raw=%r  extracted=%r  gold=%r  em=%d",
                            i, preds[i][:80], em_preds[i], refs[i],
                            1 if _normalize_answer(em_preds[i]) == _normalize_answer(refs[i]) else 0,
                        )

                # FIX-NAN
                avg_ttft = (sum(ttft_list) / len(ttft_list)) if ttft_list else float("nan")
                avg_kv   = (sum(kv_size_list) / len(kv_size_list)) if kv_size_list else 0

                hall_rate = float("nan")
                ent_score = float("nan")
                n_hall    = 0
                n_total   = len(preds)
                HHEM_THRESHOLD = 0.5

                if hhem_scorer and contexts and preds:
                    hhem_contexts = [trim_context_for_hhem(c) for c in contexts]
                    faith_scores  = hhem_scorer.batch_score(hhem_contexts, preds)
                    hall_rate     = hallucination_rate(faith_scores)
                    n_total       = len(faith_scores)
                    n_hall        = sum(1 for s in faith_scores if s < HHEM_THRESHOLD)
                    matching = all_records[record_start_idx:]
                    for rec, fs in zip(matching, faith_scores):
                        rec["hhem_faithfulness"] = fs
                        rec["hhem_hallucinated"] = fs < HHEM_THRESHOLD

                if nli_scorer and contexts and preds:
                    nli_contexts = [trim_context_for_hhem(c, max_tokens=512) for c in contexts]
                    nli_scores = nli_scorer.batch_score(nli_contexts, preds)
                    ent_score  = mean_entailment(nli_scores)
                    matching = all_records[record_start_idx:]
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
                    "n_hall":             n_hall,
                    "n_total":            n_total,
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