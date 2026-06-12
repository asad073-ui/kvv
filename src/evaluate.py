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
  * Refusals ("I can not answer …") are tracked separately and excluded from the
    HHEM / NLI faithfulness scoring (they are retrieval failures, not
    hallucinations); EM / F1 still include them.
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

    prefix_kvcache is shared (read-only) across all queries.  stack_past_key_values
    materialises brand-new tensors via torch.cat, so the prefix is never mutated —
    no deep-copy needed.  Chunk caches are loaded onto CPU then moved to `device`
    so they cat cleanly with the on-device prefix (same device AND same float16
    dtype, avoiding the historical bf16/fp16 + cuda/cpu mismatch crashes).
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
        kvcache_list = [prefix_kvcache]

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

            # Compressed caches are dicts with mixed Python types → weights_only=False.
            # map_location="cpu" avoids a GPU OOM spike while loading.
            compressed = torch.load(fpath, map_location="cpu", weights_only=False)
            legacy     = decompress_kvcache(compressed, precision)
            # Move chunk tensors onto the model device so they cat with the prefix.
            legacy     = tuple((k.to(device), v.to(device)) for k, v in legacy)
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

    attn_impl = "flash_attention_2" if args.use_flash_attn else "eager"
    log.info(f"Loading model: {args.model_name} (attn={attn_impl})")

    # T4 (sm_75): float16 throughout, never bfloat16.  torch_dtype= is the correct
    # kwarg on transformers 4.51.3 (dtype= is the later alias).
    model = Qwen2ModifiedForCausalLM.from_pretrained(
        args.model_name,
        attn_implementation=attn_impl,
        torch_dtype=torch.float16,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    log.info("Pre-computing system prefix KV cache ...")
    prefix_inputs = tokenizer([SYSTEM_PROMPT], return_tensors="pt")
    with torch.no_grad():
        # use_cache=True → Qwen2Model auto-creates a DynamicCache; the modified
        # attention stores RAW (un-rotated) keys in it, matching the chunk caches.
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
                            "is_refusal": REFUSAL_MARKER in result["answer"],
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

                # ── Refusals: tracked separately, excluded from faithfulness ──
                # A refusal means retrieval gave irrelevant context; counting it as
                # a hallucination would bias every hall_rate upward and make H1/H2
                # uninterpretable.  EM/F1 still include refusals (they are wrong
                # answers); only HHEM/NLI exclude them.
                is_refusals  = [REFUSAL_MARKER in p for p in preds]
                refusal_rate = sum(is_refusals) / max(len(preds), 1)
                nr_idx       = [i for i, r in enumerate(is_refusals) if not r]

                hall_rate = float("nan")
                ent_score = float("nan")
                n_hall    = 0
                n_total   = len(nr_idx)
                HHEM_THRESHOLD = 0.5
                matching = all_records[record_start_idx:]

                if hhem_scorer and nr_idx:
                    nr_contexts  = [trim_context_for_hhem(contexts[i]) for i in nr_idx]
                    nr_preds     = [preds[i] for i in nr_idx]
                    faith_scores = hhem_scorer.batch_score(nr_contexts, nr_preds)
                    hall_rate    = hallucination_rate(faith_scores)
                    n_total      = len(faith_scores)
                    n_hall       = sum(1 for s in faith_scores if s < HHEM_THRESHOLD)
                    for j, i in enumerate(nr_idx):
                        matching[i]["hhem_faithfulness"] = faith_scores[j]
                        matching[i]["hhem_hallucinated"] = faith_scores[j] < HHEM_THRESHOLD

                if nli_scorer and nr_idx:
                    nr_contexts = [trim_context_for_hhem(contexts[i], max_tokens=200) for i in nr_idx]
                    nr_preds    = [preds[i] for i in nr_idx]
                    nli_scores  = nli_scorer.batch_score(nr_contexts, nr_preds)
                    ent_score   = mean_entailment(nli_scores)
                    for j, i in enumerate(nr_idx):
                        matching[i]["nli_entailment"]    = nli_scores[j][0]
                        matching[i]["nli_neutral"]       = nli_scores[j][1]
                        matching[i]["nli_contradiction"] = nli_scores[j][2]

                row = {
                    "dataset":            ds_name,
                    "k":                  k,
                    "condition":          condition,
                    "condition_label":    CONDITION_LABELS[condition],
                    "n_examples":         len(preds),
                    "n_hall":             n_hall,
                    "n_total":            n_total,
                    "refusal_rate":       round(refusal_rate, 4),
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
                    f"F1={f1_score:.3f}  Refuse={refusal_rate:.3f}  "
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
        "n_hall", "n_total", "refusal_rate", "EM", "contain_EM", "F1",
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
    headers = ["Dataset", "K", "Cond", "n", "Refuse", "EM", "ContEM", "F1",
               "Hall↓", "Ent↑", "TTFT(s)", "KV(MB)"]
    table = [
        [r["dataset"], r["k"], r["condition"], r["n_examples"], r["refusal_rate"],
         r["EM"], r["contain_EM"], r["F1"],
         r["hallucination_rate"], r["entailment_score"],
         r["avg_ttft_s"], round(r["avg_kv_bytes"] / 1e6, 2)]
        for r in summary_rows
    ]
    print("\n" + tabulate(table, headers=headers, tablefmt="grid"))



if __name__ == "__main__":
    main()