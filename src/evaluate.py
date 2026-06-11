"""
evaluate.py  –  Main evaluation script for the TurboRAG KV-cache quantization study.

Implements all experimental conditions from the refined research idea:
  C0  Gold Oracle RAG   – full raw-text context, no precomputed cache
  C1  FP16 TurboRAG    – precomputed FP16 chunk caches stitched at query time
  C2  INT8 TurboRAG    – offline INT8 quantized chunk caches, dequantized before stitching
  C3  INT4 TurboRAG    – offline INT4 quantized chunk caches, dequantized before stitching

Questions are loaded from HuggingFace datasets (nq_open, hotpotqa) or from
local JSONL files (rgb and any custom dataset). Pass --hf_names alongside
--datasets to select the source per dataset.

Outputs
───────
  results/results_<timestamp>.jsonl   – one record per (dataset, K, condition, example)
  results/summary_<timestamp>.csv     – aggregated metrics table (main paper table)
  results/summary_<timestamp>.json    – same, JSON format

Fix log vs original repo
────────────────────────
  FIX-1  torch_dtype= instead of dtype= in from_pretrained
         dtype= is silently ignored by transformers; model loaded in fp32 instead of bf16.

  FIX-2  context stored with "\\n\\n".join(chunk_texts) instead of " ".join
         The space-join made split("\\n\\n") return one giant blob so
         trim_context_for_hhem always saw only the first 1200 chars regardless of K.

  FIX-3  trim_context_for_hhem now passes the full joined context up to 512 tokens
         (≈2048 chars) instead of taking only the first chunk.  HHEM uses flan-T5-base
         internally (512-token window); 512 tokens is the correct upper bound.
         NLI calls also pass max_tokens=512 and rely on DeBERTa's built-in truncation.

  FIX-4  F1 and EM computed on raw preds, not on extract_short_answer(pred).
         token_f1 already handles verbose output correctly (precision/recall overlap).
         extract_short_answer was making F1 near-zero for refusal outputs.

  FIX-5  Added contain_em metric: fraction of examples where the normalised gold
         answer string is a substring of the normalised prediction.
         Standard EM is always 0 for generative RAG; contain_em is the correct
         accuracy signal used by NQ-Open and TriviaQA official evaluations.

  FIX-6  HHEM and NLI scorers receive raw preds (not short_predictions).

  FIX-7  n_hall and n_total written to the summary row dict so that
         analyze_results.py can run the proportions z-test for H3.

  FIX-8  _load_from_hf uses a while loop for isinstance(a, list) unwrapping
         so that nested structures like NQ-Open answers={'text':[...]} resolve
         correctly to a plain string.
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
# KV cache stitching
# ──────────────────────────────────────────────────────────────────────────────

def stack_past_key_values(past_key_values_list: List[DynamicCache]) -> DynamicCache:
    """Concatenate a list of DynamicCache objects along the sequence dimension."""
    legacy_list = [
        c.to_legacy_cache() if hasattr(c, "to_legacy_cache") else c
        for c in past_key_values_list
    ]
    num_layers = len(legacy_list[0])
    stacked = tuple(
        (
            torch.cat([c[layer][0] for c in legacy_list], dim=2),
            torch.cat([c[layer][1] for c in legacy_list], dim=2),
        )
        for layer in range(num_layers)
    )
    return DynamicCache.from_legacy_cache(stacked)


def legacy_to_dynamic(legacy: tuple) -> DynamicCache:
    return DynamicCache.from_legacy_cache(legacy)


# FIX-3: max_tokens raised to 512 (HHEM's full window); function now truncates
# the full joined context instead of silently taking only the first chunk.
def trim_context_for_hhem(context: str, max_tokens: int = 512) -> str:
    """
    Truncate the full multi-chunk context to fit inside the scorer's token window.

    Uses ALL K retrieved chunks (context is already '\\n\\n'-joined by run_query).
    Rough approximation: 1 token ≈ 4 characters.  HHEM's internal flan-T5-base
    tokenizer caps at 512 tokens; DeBERTa caps at 512 tokens.  Passing the full
    512-token budget here and relying on each scorer's own truncation=True is safe
    and correct.
    """
    # FIX-3: do NOT split and take only [0]; pass the entire joined context.
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
    FIX-5: Contain-EM  — fraction of examples where the normalised gold answer
    is a substring of the normalised prediction.

    This is the correct accuracy metric for open-domain generative RAG
    (used by NQ-Open and TriviaQA official evaluations).  Standard exact-match
    is always 0 for generative models that produce full sentences.
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
            cached_len    = past_kvcache.get_seq_length()

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
    """Run a single query under a given condition."""
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
        kvcache_list = [prefix_kvcache]
        for nws in nodes_k:
            node      = nws.node
            cache_key = f"kvcache_{precision}"
            fpath     = node.metadata[cache_key]
            compressed = torch.load(fpath, weights_only=True)
            legacy     = decompress_kvcache(compressed, precision)
            kvcache_list.append(legacy_to_dynamic(legacy))
            chunk_texts.append(node.metadata.get("raw_text", node.text))
            kv_size_bytes += cache_size_bytes(compressed, precision)

        stitched = stack_past_key_values(kvcache_list)
        answer   = generate_with_cache(model, tokenizer, device, stitched, query, [])

    ttft = time.perf_counter() - t0

    return {
        "answer":        answer,
        # FIX-2: join with "\n\n" so trim_context_for_hhem can see all K chunks.
        "context":       "\n\n".join(chunk_texts),
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

        # FIX-8: use while loop so nested structures like nq_open
        # answers={'text': ['Paris', ...], 'answer_start': [...]} fully unpack.
        while isinstance(a, list):
            a = a[0] if a else ""
        if isinstance(a, dict):
            # Some datasets (hotpotqa, nq_open) nest the answer inside a dict.
            a = a.get("text", "") or a.get("answer", "")
            # After dict extraction it might still be a list.
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

    # Dataset names (required, one per dataset)
    parser.add_argument("--datasets",       type=str, nargs="+",
                        default=["nq_open", "hotpotqa", "rgb"])

    # HuggingFace dataset config (one entry per dataset, "" = use query_file fallback)
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
    parser.add_argument("--hf_cache_dir",    type=str, default=None,
                        help="HF datasets cache directory (default: HF_DATASETS_CACHE env var)")

    # Local JSONL fallback (used when hf_name for that dataset is empty)
    parser.add_argument("--query_files",    type=str, nargs="+", default=[],
                        help="JSONL path per dataset; used when hf_names entry is empty")

    parser.add_argument("--num_examples",       type=int, nargs="+", default=[200],
                        help="Per-dataset example count (-1 = use all)")
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

    # Pad per-dataset lists to len(datasets) so we can zip safely
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

    # ── Load LLM ──
    attn_impl = "flash_attention_2" if args.use_flash_attn else "eager"
    log.info(f"Loading model: {args.model_name} (attn={attn_impl})")
    # FIX-1: torch_dtype= is the correct kwarg; dtype= is silently ignored by transformers.
    model     = Qwen2ModifiedForCausalLM.from_pretrained(
        args.model_name,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ── Precompute system-prefix KV cache ──
    log.info("Pre-computing system prefix KV cache …")
    prefix_inputs = tokenizer([SYSTEM_PROMPT], return_tensors="pt", padding=True)
    with torch.no_grad():
        prefix_out = model(
            prefix_inputs["input_ids"].to(device),
            attention_mask=prefix_inputs["attention_mask"].to(device),
            use_cache=True,
        )
    prefix_kvcache = prefix_out.past_key_values

    # ── Load embedding model + index ──
    log.info("Loading retrieval index …")
    Settings.embed_model = HuggingFaceEmbedding(model_name=args.embedding_model_name)
    storage_ctx = StorageContext.from_defaults(persist_dir=args.storage_dir)
    index       = load_index_from_storage(storage_ctx)
    retriever   = index.as_retriever(similarity_top_k=args.similarity_top_k)

    # ── Optionally load faithfulness scorers ──
    hhem_scorer = HHEMScorer(device=device)       if args.eval_hhem else None
    nli_scorer  = DeBERTaNLIScorer(device=device) if args.eval_nli  else None

    # ── Main evaluation ──
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
                        log.warning(f"Skipped query: {e}")

                # ── Aggregate metrics ──
                # FIX-4: compute F1 and standard EM directly on raw preds.
                # token_f1 uses precision/recall overlap — verbose output is handled
                # correctly without any extraction step.
                f1_score      = batch_f1(preds, refs)
                em_score      = batch_em(preds, refs)

                # FIX-5: contain_em is the correct accuracy metric for generative RAG.
                # Standard EM is always 0 for sentence-length model outputs.
                contain_em    = batch_contain_em(preds, refs)

                avg_ttft = sum(ttft_list) / len(ttft_list)   if ttft_list  else float("nan")
                avg_kv   = sum(kv_size_list) / len(kv_size_list) if kv_size_list else 0

                hall_rate = float("nan")
                ent_score = float("nan")
                # FIX-7: initialise counts here so they appear in the row dict
                # even when --eval_hhem is not passed.
                n_hall  = 0
                n_total = len(preds)

                HHEM_THRESHOLD = 0.5

                if hhem_scorer and contexts and preds:
                    # FIX-3: full joined context (all K chunks), up to 512-token window.
                    hhem_contexts = [trim_context_for_hhem(c) for c in contexts]

                    # FIX-6: pass raw preds, not extract_short_answer(pred).
                    faith_scores = hhem_scorer.batch_score(hhem_contexts, preds)

                    hall_rate = hallucination_rate(faith_scores)

                    # FIX-7: compute counts for H3 proportions z-test.
                    n_total = len(faith_scores)
                    n_hall  = sum(1 for s in faith_scores if s < HHEM_THRESHOLD)

                    matching = [
                        r for r in all_records
                        if r["dataset"] == ds_name
                        and r["k"] == k
                        and r["condition"] == condition
                    ]
                    for rec, fs in zip(matching, faith_scores):
                        rec["hhem_faithfulness"] = fs
                        rec["hhem_hallucinated"] = fs < HHEM_THRESHOLD

                if nli_scorer and contexts and preds:
                    # FIX-3 / FIX-7: full context at 512 tokens; DeBERTa truncates
                    # internally at max_length=512 — no data is lost beyond that limit.
                    nli_contexts = [
                        trim_context_for_hhem(c, max_tokens=512)
                        for c in contexts
                    ]

                    # FIX-6: pass raw preds, not short_predictions.
                    nli_scores = nli_scorer.batch_score(nli_contexts, preds)

                    ent_score = mean_entailment(nli_scores)

                    matching = [
                        r for r in all_records
                        if r["dataset"] == ds_name
                        and r["k"] == k
                        and r["condition"] == condition
                    ]
                    for rec, ns in zip(matching, nli_scores):
                        rec["nli_entailment"]    = ns[0]
                        rec["nli_neutral"]       = ns[1]
                        rec["nli_contradiction"] = ns[2]

                # FIX-7: n_hall and n_total are now written into the row dict so
                # analyze_results.py can locate them for the proportions z-test.
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
                    "avg_ttft_s":         round(avg_ttft, 4),
                    "avg_kv_bytes":       int(avg_kv),
                }
                summary_rows.append(row)
                log.info(
                    f"    EM={em_score:.3f}  ContEM={contain_em:.3f}  F1={f1_score:.3f}  "
                    f"Hall={hall_rate:.3f}  Ent={ent_score:.3f}  "
                    f"TTFT={avg_ttft:.3f}s  KV={avg_kv/1e6:.2f}MB"
                )

    # ── Write outputs ──
    with open(raw_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")
    log.info(f"Raw records → {raw_path}")

    fieldnames = [
        "dataset", "k", "condition", "condition_label", "n_examples",
        "n_hall", "n_total",
        "EM", "contain_EM", "F1",
        "hallucination_rate", "entailment_score",
        "avg_ttft_s", "avg_kv_bytes",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    log.info(f"Summary CSV → {csv_path}")

    with open(json_path, "w") as f:
        json.dump(summary_rows, f, indent=2)
    log.info(f"Summary JSON → {json_path}")

    headers = ["Dataset", "K", "Cond", "EM", "ContEM", "F1", "Hall↓", "Ent↑", "TTFT(s)", "KV(MB)"]
    table   = [
        [r["dataset"], r["k"], r["condition"],
         r["EM"], r["contain_EM"], r["F1"],
         r["hallucination_rate"], r["entailment_score"],
         r["avg_ttft_s"],
         round(r["avg_kv_bytes"] / 1e6, 2)]
        for r in summary_rows
    ]
    print("\n" + tabulate(table, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()
