"""
evaluate.py  –  Main evaluation script for the TurboRAG KV-cache quantization study.

Implements all experimental conditions from the refined research idea:
  C0  Gold Oracle RAG   – full raw-text context, no precomputed cache
  C1  FP16 TurboRAG    – precomputed FP16 chunk caches stitched at query time
  C2  INT8 TurboRAG    – offline INT8 quantized chunk caches, dequantized before stitching
  C3  INT4 TurboRAG    – offline INT4 quantized chunk caches, dequantized before stitching

FIX LOG
────────────────────────────────────────────────────────────────────────────────
  FIX-ROOT  legacy_to_dynamic / stack_past_key_values: DynamicCache.from_legacy_cache
            on transformers >= 4.45 expects new-format ((k0,k1,...),(v0,v1,...)).
            Old per-layer format ((k0,v0),(k1,v1),...) silently produced a cache
            with _seen_tokens=0, causing get_seq_length() to return 0 → every
            C1/C2/C3 query threw "index -1 is out of bounds for dimension 0 with
            size 0" and was silently skipped → EM=0, F1=0.

  FIX-SEEN  After DynamicCache.from_legacy_cache the internal _seen_tokens counter
            is NOT updated (transformers bug present in 4.45-4.47).  We manually
            set cache._seen_tokens to the actual sequence length immediately after
            construction so get_seq_length() returns a real value.

  FIX-MASK  generate_with_cache used cached_len = past_kvcache.get_seq_length()
            which returned 0 due to FIX-SEEN bug above.  Now reads the actual
            length directly from the key tensor shape so it is robust regardless
            of _seen_tokens.

  FIX-CSV   csv.DictWriter lacked quoting=csv.QUOTE_ALL / extrasaction='ignore'.
            Fields containing commas (condition_label "Gold Oracle RAG", or any
            context text leaking into a field) caused pandas read_csv to throw
            "ParserError: Expected 1 fields in line N, saw 2".
            Fixed by adding quoting=csv.QUOTE_ALL to the DictWriter constructor.

  FIX-NAN   avg_ttft NaN when preds list is empty but ttft_list is also empty
            caused tabulate to crash.  Guarded with explicit nan checks.

  FIX-1  torch_dtype= instead of dtype= in from_pretrained
         dtype= is silently ignored by transformers; model loaded in fp32 instead of bf16.

  FIX-2  context stored with "\\n\\n".join(chunk_texts) instead of " ".join

  FIX-3  trim_context_for_hhem now passes the full joined context up to 512 tokens.

  FIX-4  F1 and EM computed on raw preds, not on extract_short_answer(pred).

  FIX-5  Added contain_em metric.

  FIX-6  HHEM and NLI scorers receive raw preds.

  FIX-7  n_hall and n_total written to the summary row dict.

  FIX-8  _load_from_hf uses a while loop for isinstance(a, list) unwrapping.

  FIX-DEEPCOPY  prefix_kvcache is passed by reference into run_query.
                stack_past_key_values calls torch.cat which mutates the
                underlying tensors of the original prefix cache in-place
                (transformers DynamicCache stores references, not copies).
                Every query after the first received a corrupted prefix,
                raising an exception that was silently swallowed → C1/C2/C3
                produced 0 predictions.
                Fix: deep-copy prefix_kvcache at the start of each run_query
                call so the original is never touched.

  FIX-ERRLOG    The except block only logged log.warning(str(e)) which hid the
                real traceback.  Changed to log.error(..., exc_info=True) so
                future failures show the full stack trace in stdout.
"""

import os
import sys
import json
import time
import copy       # ← FIX-DEEPCOPY
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
# KV cache stitching  (FIX-ROOT + FIX-SEEN)
# ──────────────────────────────────────────────────────────────────────────────

def _normalise_legacy(legacy_cache):
    """
    Normalise DynamicCache.to_legacy_cache() output to per-layer (k, v) pairs.

    transformers < 4.45  → ((k0,v0), (k1,v1), ..., (kL-1,vL-1))  len=L
    transformers >= 4.45 → ((k0,k1,...,kL-1), (v0,v1,...,vL-1))   len=2

    Disambiguation: the new format has len==2 AND each element is a tuple/list
    of tensors (not itself a pair of (tensor, tensor) tuples).
    We identify the new format by checking len==2 AND that legacy_cache[0][0]
    is a torch.Tensor (meaning legacy_cache[0] is the all-keys tuple).
    """
    if (
        len(legacy_cache) == 2
        and not isinstance(legacy_cache[0], torch.Tensor)
        and hasattr(legacy_cache[0], "__len__")
        and len(legacy_cache[0]) > 0
        and isinstance(legacy_cache[0][0], torch.Tensor)
    ):
        # New format: ((k0, k1, ..., kL-1), (v0, v1, ..., vL-1))
        all_keys, all_values = legacy_cache
        return list(zip(all_keys, all_values))
    # Old format or raw per-layer list: already [(k0,v0), (k1,v1), ...]
    return list(legacy_cache)


def _to_new_legacy_format(per_layer_pairs):
    """
    Convert per-layer (k, v) pairs  →  new ((k0,k1,...), (v0,v1,...)) format
    required by DynamicCache.from_legacy_cache in transformers >= 4.45.
    """
    all_keys   = tuple(kv[0] for kv in per_layer_pairs)
    all_values = tuple(kv[1] for kv in per_layer_pairs)
    return (all_keys, all_values)


def _set_seen_tokens(cache: DynamicCache) -> DynamicCache:
    """
    FIX-SEEN: transformers 4.45-4.47 does not update DynamicCache._seen_tokens
    when building via from_legacy_cache.  get_seq_length() therefore returns 0,
    which makes the attention mask too short and causes:
        "index -1 is out of bounds for dimension 0 with size 0"

    Fix: read the real sequence length from the key tensor of layer 0 and write
    it back to _seen_tokens.
    """
    if hasattr(cache, "key_cache") and len(cache.key_cache) > 0:
        real_len = cache.key_cache[0].shape[2]  # (batch, heads, seq, head_dim)
        cache._seen_tokens = real_len
    return cache


def legacy_to_dynamic(legacy) -> DynamicCache:
    """
    Convert a per-layer (k,v) legacy cache tuple/list to a DynamicCache.

    FIX-ROOT: always converts to the new all-keys/all-values format before
              calling from_legacy_cache.
    FIX-SEEN: patches _seen_tokens immediately after construction.
    """
    per_layer = _normalise_legacy(legacy)
    new_fmt   = _to_new_legacy_format(per_layer)
    cache     = DynamicCache.from_legacy_cache(new_fmt)
    return _set_seen_tokens(cache)


def stack_past_key_values(past_key_values_list: List[DynamicCache]) -> DynamicCache:
    """
    Concatenate a list of DynamicCache objects along the sequence dimension.

    Steps:
      1. Convert each DynamicCache → per-layer (k,v) pairs
      2. torch.cat all per-layer k tensors together, same for v
      3. Convert the stitched result → new format → DynamicCache.from_legacy_cache
      4. Patch _seen_tokens (FIX-SEEN)

    NOTE: This function does NOT mutate the input caches because torch.cat
    creates a new tensor.  The caller must still pass deep-copied caches
    when the same cache object will be reused across multiple queries
    (see FIX-DEEPCOPY in run_query).
    """
    legacy_list = []
    for c in past_key_values_list:
        if hasattr(c, "to_legacy_cache"):
            raw = c.to_legacy_cache()
        else:
            raw = c
        legacy_list.append(_normalise_legacy(raw))

    num_layers = len(legacy_list[0])
    stacked_pairs = [
        (
            torch.cat([c[layer][0] for c in legacy_list], dim=2),
            torch.cat([c[layer][1] for c in legacy_list], dim=2),
        )
        for layer in range(num_layers)
    ]
    new_fmt = _to_new_legacy_format(stacked_pairs)
    cache   = DynamicCache.from_legacy_cache(new_fmt)
    return _set_seen_tokens(cache)


# ──────────────────────────────────────────────────────────────────────────────
# Context helpers
# ──────────────────────────────────────────────────────────────────────────────

def trim_context_for_hhem(context: str, max_tokens: int = 512) -> str:
    """
    Truncate the full multi-chunk context to fit inside the scorer's token window.
    Rough approximation: 1 token ≈ 4 characters.  (FIX-3)
    """
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


def batch_contain_em(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Contain-EM — fraction of examples where the normalised gold answer
    is a substring of the normalised prediction.
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

EOS_TOKEN_IDS = [151645, 151643]
MAX_NEW_TOKENS = 64


def _get_cache_seq_len(cache: DynamicCache) -> int:
    """
    FIX-MASK: Read actual sequence length from key tensors, not from
    _seen_tokens (which may be stale even after our _set_seen_tokens fix
    if the cache was built externally).  This is the ground truth.
    """
    if hasattr(cache, "key_cache") and len(cache.key_cache) > 0:
        return cache.key_cache[0].shape[2]
    # Fallback to the API method
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
            suffix        = build_query_suffix(query)
            new_input_ids = tokenizer.encode(suffix, return_tensors="pt").to(device)
            # FIX-MASK: use actual tensor shape, not get_seq_length()
            cached_len    = _get_cache_seq_len(past_kvcache)

            attention_mask = torch.ones(
                1, cached_len + new_input_ids.shape[1],
                device=device, dtype=torch.long
            )
            outputs = model.generate(
                new_input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                past_key_values=past_kvcache,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
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
                eos_token_id=EOS_TOKEN_IDS,
            )
            new_tokens = outputs[0][input_ids.shape[1]:]
            return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ──────────────────────────────────────────────────────────────────────────────
# Per-query inference for each condition
# ──────────────────────────────────────────────────────────────────────────────

PRECISION_MAP = {
    "C0": None,
    "C1": "fp16",
    "C2": "int8",
    "C3": "int4",
}

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

    FIX-DEEPCOPY: prefix_kvcache is shared across ALL queries in the eval loop.
    stack_past_key_values calls torch.cat which does NOT modify the tensors
    in-place, BUT DynamicCache.to_legacy_cache() in transformers >= 4.45
    returns direct references to the internal key_cache / value_cache lists.
    If the returned tensors were ever used in a way that modifies them
    (e.g. in-place ops, or if transformers internals reuse the buffer),
    the prefix would be silently corrupted for the next query.

    The safest fix — and the one that eliminates this entire class of bugs —
    is to deep-copy the prefix cache at the start of every run_query call.
    This adds a small memory overhead per query but is negligible compared
    to the model forward pass.
    """
    nodes_k   = retrieved_nodes[:k]
    precision = PRECISION_MAP[condition]
    chunk_texts   = []
    kv_size_bytes = 0

    t0 = time.perf_counter()

    if condition == "C0":
        for nws in nodes_k:
            chunk_texts.append(nws.node.metadata.get("raw_text", nws.node.text))
        answer = generate_with_cache(model, tokenizer, device, None, query, chunk_texts)

    else:
        # ── FIX-DEEPCOPY ──────────────────────────────────────────────────────
        # Deep-copy the prefix cache so this query's stitching does not corrupt
        # the shared prefix_kvcache object for subsequent queries.
        prefix_copy = copy.deepcopy(prefix_kvcache)
        kvcache_list = [prefix_copy]
        # ─────────────────────────────────────────────────────────────────────

        for nws in nodes_k:
            node      = nws.node
            cache_key = f"kvcache_{precision}"
            fpath     = node.metadata[cache_key]
            compressed = torch.load(fpath, weights_only=True)
            legacy     = decompress_kvcache(compressed, precision)
            # FIX-ROOT + FIX-SEEN: always go through legacy_to_dynamic
            kvcache_list.append(legacy_to_dynamic(legacy))
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
    hf_name: str,
    hf_config: Optional[str],
    hf_split: str,
    num_examples: int,
    question_field: str,
    answer_field: str,
    hf_cache_dir: Optional[str],
) -> List[Dict]:
    """Load question/answer pairs from a HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required. Install with: pip install datasets"
        )

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

        # FIX-8: peel nested list/dict structures until we get a plain string
        while isinstance(a, list):
            a = a[0] if a else ""
        if isinstance(a, dict):
            a = a.get("text", "") or a.get("answer", "")
            while isinstance(a, list):
                a = a[0] if a else ""

        examples.append({"query": q, "answer": str(a)})

    return examples


def _load_from_jsonl(query_file: str, num_examples: int) -> List[Dict]:
    """Load question/answer pairs from a local JSONL file."""
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
    dataset_name: str,
    query_file: str,
    num_examples: int,
    hf_name: Optional[str] = None,
    hf_config: Optional[str] = None,
    hf_split: str = "validation",
    question_field: str = "question",
    answer_field: str = "answer",
    hf_cache_dir: Optional[str] = None,
) -> List[Dict]:
    """
    Load (query, answer) pairs for a dataset.
    Priority: HF dataset (if hf_name is set) → local JSONL (if query_file is set).
    """
    if hf_name:
        return _load_from_hf(
            hf_name=hf_name,
            hf_config=hf_config or None,
            hf_split=hf_split,
            num_examples=num_examples,
            question_field=question_field,
            answer_field=answer_field,
            hf_cache_dir=hf_cache_dir,
        )
    if query_file:
        return _load_from_jsonl(query_file, num_examples)
    raise ValueError(
        f"Dataset '{dataset_name}': provide either --hf_names or --query_files"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TurboRAG KV-cache quantization evaluation")
    parser.add_argument("--model_name",           type=str, required=True)
    parser.add_argument("--embedding_model_name", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--storage_dir",          type=str, default="doc_emb")

    parser.add_argument("--datasets",       type=str, nargs="+",
                        default=["nq_open", "hotpotqa", "rgb"])

    parser.add_argument("--hf_names",        type=str, nargs="+", default=[],
                        help="HF dataset name per dataset ('' to use query_file instead)")
    parser.add_argument("--hf_configs",      type=str, nargs="+", default=[],
                        help="HF dataset config per dataset ('' for default)")
    parser.add_argument("--hf_splits",       type=str, nargs="+", default=[],
                        help="HF dataset split per dataset (default: validation)")
    parser.add_argument("--question_fields", type=str, nargs="+", default=[],
                        help="Question field name per dataset (default: question)")
    parser.add_argument("--answer_fields",   type=str, nargs="+", default=[],
                        help="Answer field name per dataset (default: answer)")
    parser.add_argument("--hf_cache_dir",    type=str, default=None)

    parser.add_argument("--query_files",    type=str, nargs="+", default=[],
                        help="JSONL path per dataset; used when hf_names entry is empty")

    parser.add_argument("--num_examples",       type=int, nargs="+", default=[200])
    parser.add_argument("--k_values",           type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--conditions",         type=str, nargs="+",
                        default=["C0", "C1", "C2", "C3"],
                        choices=["C0", "C1", "C2", "C3"])
    parser.add_argument("--use_flash_attn",     action="store_true")
    parser.add_argument("--output_dir",         type=str, default="results")
    parser.add_argument("--eval_hhem",          action="store_true")
    parser.add_argument("--eval_nli",           action="store_true")
    parser.add_argument("--similarity_top_k",   type=int, default=5)
    args = parser.parse_args()

    assert args.similarity_top_k >= max(args.k_values), \
        "--similarity_top_k must be >= max(k_values)"

    n_ds = len(args.datasets)

    def _pad(lst, default):
        return list(lst) + [default] * (n_ds - len(lst))

    hf_names        = _pad(args.hf_names,        "")
    hf_configs      = _pad(args.hf_configs,       "")
    hf_splits       = _pad(args.hf_splits,        "validation")
    question_fields = _pad(args.question_fields,  "question")
    answer_fields   = _pad(args.answer_fields,    "answer")
    query_files     = _pad(args.query_files,      "")
    num_examples_list = _pad(args.num_examples,   200)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path  = os.path.join(args.output_dir, f"results_{timestamp}.jsonl")
    csv_path  = os.path.join(args.output_dir, f"summary_{timestamp}.csv")
    json_path = os.path.join(args.output_dir, f"summary_{timestamp}.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    attn_impl = "flash_attention_2" if args.use_flash_attn else "eager"
    log.info(f"Loading model: {args.model_name} (attn={attn_impl})")
    model     = Qwen2ModifiedForCausalLM.from_pretrained(
        args.model_name,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,        # FIX-1
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
    # prefix_kvcache comes directly from the model (already a proper DynamicCache
    # with _seen_tokens set correctly since it was built by a live forward pass).
    prefix_kvcache = prefix_out.past_key_values
    # Defensive: ensure _seen_tokens is correct even for the prefix cache.
    if hasattr(prefix_kvcache, "_seen_tokens"):
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
            dataset_name=ds_name,
            query_file=qfile,
            num_examples=n_examples,
            hf_name=hf_name or None,
            hf_config=hf_cfg or None,
            hf_split=hf_split,
            question_field=q_field,
            answer_field=a_field,
            hf_cache_dir=args.hf_cache_dir,
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
                            query=query,
                            retrieved_nodes=nodes,
                            k=k,
                            condition=condition,
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            prefix_kvcache=prefix_kvcache,
                        )
                        preds.append(result["answer"])
                        refs.append(answer)
                        contexts.append(result["context"])
                        ttft_list.append(result["ttft_seconds"])
                        kv_size_list.append(result["kv_size_bytes"])

                        record = {
                            "dataset":    ds_name,
                            "k":          k,
                            "condition":  condition,
                            "query":      query,
                            "gold":       answer,
                            "prediction": result["answer"],
                            "context":    result["context"],
                            "ttft":       result["ttft_seconds"],
                            "kv_bytes":   result["kv_size_bytes"],
                        }
                        all_records.append(record)

                    except Exception as e:
                        # FIX-ERRLOG: log full traceback so failures are never silent
                        log.error(
                            f"Skipped query [{ds_name} K={k} {condition}]: {e}",
                            exc_info=True
                        )

                f1_score   = batch_f1(preds, refs)          # FIX-4
                em_score   = batch_em(preds, refs)          # FIX-4
                contain_em = batch_contain_em(preds, refs)  # FIX-5

                # FIX-NAN: guard against empty preds list
                avg_ttft = (sum(ttft_list) / len(ttft_list)) if ttft_list else float("nan")
                avg_kv   = (sum(kv_size_list) / len(kv_size_list)) if kv_size_list else 0

                hall_rate = float("nan")
                ent_score = float("nan")
                n_hall  = 0
                n_total = len(preds)

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
                    nli_contexts = [
                        trim_context_for_hhem(c, max_tokens=512)
                        for c in contexts
                    ]
                    nli_scores = nli_scorer.batch_score(nli_contexts, preds)   # FIX-6
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
                    "n_hall":             n_hall,             # FIX-7
                    "n_total":            n_total,            # FIX-7
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
                    f"F1={f1_score:.3f}  Hall={'N/A' if math.isnan(hall_rate) else f'{hall_rate:.3f}'}  "
                    f"Ent={'N/A' if math.isnan(ent_score) else f'{ent_score:.3f}'}  "
                    f"TTFT={'N/A' if math.isnan(avg_ttft) else f'{avg_ttft:.3f}s'}  "
                    f"KV={avg_kv/1e6:.2f}MB"
                )

    # ── Write raw JSONL ──────────────────────────────────────────────────────
    with open(raw_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")
    log.info(f"Raw records → {raw_path}")

    # ── Write summary CSV  (FIX-CSV: QUOTE_ALL prevents parser errors) ──────
    fieldnames = [
        "dataset", "k", "condition", "condition_label", "n_examples",
        "n_hall", "n_total",
        "EM", "contain_EM", "F1",
        "hallucination_rate", "entailment_score",
        "avg_ttft_s", "avg_kv_bytes",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_ALL,          # FIX-CSV
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
    table   = [
        [r["dataset"], r["k"], r["condition"], r["n_examples"],
         r["EM"], r["contain_EM"], r["F1"],
         r["hallucination_rate"], r["entailment_score"],
         r["avg_ttft_s"],
         round(r["avg_kv_bytes"] / 1e6, 2)]
        for r in summary_rows
    ]
    print("\n" + tabulate(table, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()