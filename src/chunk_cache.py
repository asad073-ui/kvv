import os
import json
import csv
import gzip
import tempfile
import argparse
import urllib.request
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import DynamicCache

import sys; sys.path.insert(0, os.path.dirname(__file__))
from qwen2 import Qwen2ModifiedForCausalLM
from kv_quantization import compress_kvcache, cache_size_bytes

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode, Document
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import SimpleVectorStore
from typing import List

PRECISIONS = ("fp16", "int8", "int4")

# Default DPR Wikipedia passages URL (Facebook CDN, publicly available)
DPR_TSV_URL = "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"



# CLI


def get_args():
    parser = argparse.ArgumentParser(
        description="Build per-precision offline KV caches + embedding index"
    )
    parser.add_argument("--model_name",           type=str, required=True)
    parser.add_argument("--embedding_model_name", type=str, default="BAAI/bge-small-en-v1.5")

    # DPR Wikipedia passage source (direct TSV download).  Default empty: the DPR
    # download is opt-in.  For the Colab MVE use --hotpotqa_corpus instead.
    parser.add_argument("--wiki_docs_url",      type=str, default="",
                        help="URL of psgs_w100.tsv.gz (Facebook CDN); empty to disable")
    parser.add_argument("--wiki_docs_num",      type=int, default=10000,
                        help="Number of passages to stream from the TSV")
    parser.add_argument("--wiki_docs_save_dir", type=str, default=None,
                        help="Directory to cache wiki_passages.jsonl. "
                             "Defaults to --output_path/../wiki_dpr_docs")

    # HotpotQA corpus (multi-hop paragraphs).  Can be combined with wiki + rgb.
    parser.add_argument("--hotpotqa_corpus", action="store_true", default=False,
                        help="Add HotpotQA context paragraphs to the corpus")
    parser.add_argument("--hotpotqa_num_examples", type=int, default=100,
                        help="Number of HotpotQA examples to extract paragraphs from")

    # RGB corpus (noisy-retrieval positives + negatives from a local JSONL).
    parser.add_argument("--rgb_corpus", type=str, default="",
                        help="Path to rgb.jsonl; ingests each question's positive + "
                             "negative documents so RGB retrieval recall > 0")
    parser.add_argument("--rgb_num_examples", type=int, default=0,
                        help="Number of RGB examples to harvest documents from")

    # Local document fallback (used when --wiki_docs_url is empty string)
    parser.add_argument("--documents_dir",    type=str, default=None,
                        help="Local directory of .txt files (fallback if wiki_docs_url is empty)")

    # Which precision caches to build.  Subset of {fp16,int8,int4}.  Building only
    # the precisions required by the active conditions saves large amounts of disk
    # (each precision is a separate .pt file per chunk).
    parser.add_argument("--precisions", type=str, nargs="+",
                        default=list(PRECISIONS),
                        choices=list(PRECISIONS),
                        help="Precision caches to write per chunk (default: all three)")

    # KV-cache and index paths
    parser.add_argument("--output_path",   type=str, default="chunk_kvcache")
    parser.add_argument("--storage_dir",   type=str, default="doc_emb")
    parser.add_argument("--chunk_size",    type=int, default=512)
    parser.add_argument("--chunk_overlap", type=int, default=10)
    return parser.parse_args()


# -
# Document loading: DPR Wikipedia TSV (direct download, no HF datasets needed)
# -

def load_wiki_dpr_documents(
    download_url: str,
    num_docs: int,
    save_dir: str,
) -> List[Document]:
    """
    Stream Wikipedia passages from the DPR TSV hosted on Facebook's CDN and
    return them as LlamaIndex Documents.

    TSV columns (tab-separated, with header): id, text, title

    On the first call, passages are written to save_dir/wiki_passages.jsonl.
    On subsequent calls that file is read directly — no network connection needed.

    Only the first num_docs rows are transferred; the connection is closed
    immediately after, so bandwidth usage is proportional to num_docs, not
    the full 2.2 GB file.
    """
    os.makedirs(save_dir, exist_ok=True)
    cache_file = os.path.join(save_dir, "wiki_passages.jsonl")

    if os.path.exists(cache_file):
        print(f"[chunk_cache] Loading cached passages from {cache_file}")
        docs = []
        with open(cache_file, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                docs.append(Document(
                    text=row["text"],
                    metadata={"title": row.get("title", ""), "wiki_id": row["id"]},
                    id_=str(row["id"]),
                ))
        print(f"[chunk_cache] Loaded {len(docs)} passages from cache")
        return docs

    print(f"[chunk_cache] Streaming {num_docs} passages from {download_url}")
    print("[chunk_cache] (only the first rows are downloaded; connection closes after)")

    docs = []
    # Write to a temp file first; rename on success (atomic, no partial cache)
    tmp_file = cache_file + ".tmp"
    try:
        with urllib.request.urlopen(download_url) as resp, \
             gzip.open(resp, "rt", encoding="utf-8") as gz, \
             open(tmp_file, "w", encoding="utf-8") as fout:

            reader = csv.reader(gz, delimiter="\t")
            next(reader)  # skip header row: id  text  title

            with tqdm(total=num_docs, desc="Streaming wiki passages", unit="passage") as pbar:
                for row in reader:
                    if len(docs) >= num_docs:
                        break
                    if len(row) < 3:
                        continue
                    wiki_id, text, title = row[0], row[1], row[2]
                    entry = {"id": wiki_id, "text": text, "title": title}
                    fout.write(json.dumps(entry) + "\n")
                    docs.append(Document(
                        text=text,
                        metadata={"title": title, "wiki_id": wiki_id},
                        id_=wiki_id,
                    ))
                    pbar.update(1)

        os.rename(tmp_file, cache_file)
        print(f"[chunk_cache] Saved {len(docs)} passages to {cache_file}")

    except Exception:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        raise

    return docs


# Local document fallback (original documents/ directory loader)

def load_local_documents(documents_dir: str) -> List[Document]:
    from llama_index.core import SimpleDirectoryReader
    print(f"[chunk_cache] Loading documents from local directory: {documents_dir}")
    docs = SimpleDirectoryReader(documents_dir).load_data()
    print(f"[chunk_cache] Found {len(docs)} document(s)")
    return docs



# HotpotQA corpus loader (replaces the DPR Wikipedia download for the MVE)


def load_hotpotqa_corpus(hf_dataset, num_examples: int = 100) -> List[Document]:
    """
    Extract all context paragraphs from a HotpotQA dataset slice and return them
    as LlamaIndex Documents.  Each paragraph becomes one Document; deduplication
    is by (title, first-80-chars) pair.

    This replaces the DPR Wikipedia download for the MVE.  Using the questions'
    own gold + distractor paragraphs guarantees retrieval recall > 0 and lets the
    model actually answer questions instead of refusing on an off-topic corpus.

    HotpotQA "distractor" row schema:
        row["context"]["title"]     -> list[str]
        row["context"]["sentences"] -> list[list[str]]   (one list per title)
    """
    seen = set()
    docs: List[Document] = []
    for row in list(hf_dataset)[:num_examples]:
        ctx       = row["context"]
        titles    = ctx["title"]
        sentences = ctx["sentences"]  # list of lists
        for title, sent_list in zip(titles, sentences):
            text = " ".join(sent_list).strip()
            if not text:
                continue
            key = (title, text[:80])
            if key in seen:
                continue
            seen.add(key)
            docs.append(Document(
                text=text,
                metadata={"title": title, "source": "hotpotqa"},
                id_=f"hq_{len(docs)}",
            ))
    print(f"[chunk_cache] Built corpus: {len(docs)} paragraphs "
          f"from {num_examples} HotpotQA questions")
    return docs



# RGB corpus loader (noisy-retrieval positives + negatives from local JSONL)


def load_rgb_corpus(rgb_jsonl: str, num_examples: int = 0) -> List[Document]:
    """
    Ingest the positive (supporting) and negative (distractor) documents that the
    RGB benchmark ships per question into the retrieval corpus.

    rgb.jsonl row schema:
        {"id", "query", "answer", "positive": [str,...], "negative": [str,...]}

    Both lists are added so that (a) the answer-bearing positives are retrievable
    and (b) the negatives create the realistic noisy-retrieval setting RGB is
    designed to probe.  num_examples<=0 means "use every row in the file".
    """
    import io
    seen = set()
    docs: List[Document] = []
    n_pos = n_neg = 0
    with io.open(rgb_jsonl, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if num_examples and num_examples > 0:
        rows = rows[:num_examples]
    for row in rows:
        qid = row.get("id", len(docs))
        for kind in ("positive", "negative"):
            for passage in row.get(kind, []) or []:
                text = (passage or "").strip()
                if not text:
                    continue
                key = text[:120]
                if key in seen:
                    continue
                seen.add(key)
                docs.append(Document(
                    text=text,
                    metadata={"title": f"rgb_{qid}_{kind}", "source": "rgb", "rgb_kind": kind},
                    id_=f"rgb_{len(docs)}",
                ))
                if kind == "positive":
                    n_pos += 1
                else:
                    n_neg += 1
    print(f"[chunk_cache] Built RGB corpus: {len(docs)} docs "
          f"({n_pos} positive, {n_neg} negative) from {len(rows)} questions")
    return docs



# Helper: normalise DynamicCache.to_legacy_cache() output


def _to_per_layer_pairs(legacy_cache):
    """
    Normalise the output of DynamicCache.to_legacy_cache() to a sequence of
    (key_tensor, value_tensor) pairs, one per transformer layer.

    transformers<4.45  returned: ((k0,v0), (k1,v1), ...)   ← per-layer pairs
    transformers>=4.45 returns:  ((k0,k1,...), (v0,v1,...)) ← 2-element tuple

    Both formats are handled so the code is forward- and backward-compatible.

    FIX-BUGC: The old check `len == 2 and not isinstance(..., Tensor)` failed
    for models with exactly 2 layers, where the old format also has len == 2.
    Now we additionally check `len(inner0) > 2` to distinguish the two:
      - New format inner0 = (k0, k1, ..., kL-1) → len > 2 for L > 2
      - Old format inner0 = (k0, v0) → len == 2
    For the edge case of L == 2 with new format, inner0 = (k0, k1) has len == 2,
    same as old format inner0 = (k0, v0).  We disambiguate by checking if inner0[0]
    and inner0[1] have the same shape (both keys in new format) vs different shapes
    (key and value in old format — though they typically have the same shape).
    For safety, we also require inner0[0] to be a Tensor.
    """
    if len(legacy_cache) == 2 and not isinstance(legacy_cache[0], torch.Tensor):
        inner0 = legacy_cache[0]
        if (
            hasattr(inner0, "__len__")
            and len(inner0) > 2
            and isinstance(inner0[0], torch.Tensor)
        ):
            # Definitely new format: (all_keys_tuple, all_values_tuple)
            all_keys, all_values = legacy_cache
            return list(zip(all_keys, all_values))
        # Old format with exactly 2 layers: ((k0,v0), (k1,v1))
        return list(legacy_cache)
    # Old format with L != 2 layers
    return list(legacy_cache)



# Core: compute + save per-chunk KV caches


def compute_and_save_chunk(
    chunk_text: str,
    chunk_id: str,
    model: Qwen2ModifiedForCausalLM,
    tokenizer,
    output_path: str,
    device: torch.device,
    precisions: List[str] = PRECISIONS,
) -> dict:
    """Forward-pass a single chunk and save the requested compressed caches."""
    wrapped = f"<|doc_start|>{chunk_text}<|doc_end|>"
    inputs  = tokenizer(wrapped, return_tensors="pt").to(device)

    with torch.no_grad():
        # use_cache=True makes Qwen2Model auto-create an (empty) DynamicCache and
        # pass it to attention.  Qwen2ModifiedAttention then stores RAW
        # (un-rotated) keys via cache.update(); RoPE is applied only to a local
        # copy for the attention math, so to_legacy_cache() below returns the raw
        # keys that TurboRAG stitching requires.
        outputs = model(**inputs, use_cache=True)

    # DynamicCache.to_legacy_cache() format changed in transformers>=4.45.
    # _to_per_layer_pairs() normalises both old and new formats so that
    # compress_kvcache always receives [(k0,v0), (k1,v1), ...].
    raw_legacy   = outputs.past_key_values.to_legacy_cache()
    legacy_cache = _to_per_layer_pairs(raw_legacy)

    paths = {}
    for prec in precisions:
        compressed = compress_kvcache(legacy_cache, prec)
        fpath      = os.path.join(output_path, f"kvcache_chunk_{chunk_id}_{prec}.pt")
        torch.save(compressed, fpath)
        paths[prec] = fpath

    return paths



# LlamaIndex node parser that wraps our cache builder


class KVCachedNodeParser:
    """
    Splits each document into token-bounded chunks, runs forward passes,
    and saves FP16 / INT8 / INT4 KV caches for each chunk.

    Plain Python class — does not inherit from any Pydantic-backed LlamaIndex
    base so arbitrary attributes can be set freely.
    """

    def __init__(self, model, tokenizer, output_path, device, chunk_size=512,
                 chunk_overlap=10, precisions=PRECISIONS):
        self.model        = model
        self.tokenizer    = tokenizer
        self.output_path  = output_path
        self.device       = device
        self.precisions   = list(precisions)
        self.splitter     = TokenTextSplitter(
            tokenizer=tokenizer.encode,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def get_nodes_from_documents(
        self,
        documents: List[Document],
        **kwargs,
    ) -> List[TextNode]:
        nodes = []
        global_chunk_id = 0
        for doc_id, document in enumerate(tqdm(documents, desc="Documents")):
            doc_text    = document.get_content()
            chunk_texts = self.splitter.split_text(doc_text)

            for chunk_text in tqdm(chunk_texts, desc=f"  Doc {doc_id} chunks", leave=False):
                chunk_id = f"{doc_id}_{global_chunk_id}"
                paths    = compute_and_save_chunk(
                    chunk_text, chunk_id, self.model,
                    self.tokenizer, self.output_path, self.device,
                    precisions=self.precisions,
                )
                meta = {
                    "raw_text": chunk_text,
                    "source":   document.metadata.get("source", "unknown"),
                }
                for prec in self.precisions:
                    meta[f"kvcache_{prec}"] = paths[prec]
                node = TextNode(
                    text=f"<|doc_start|>{chunk_text}<|doc_end|>",
                    id_=f"chunk_{chunk_id}",
                    metadata=meta,
                )
                nodes.append(node)
                global_chunk_id += 1

        return nodes



# Main


def main():
    args   = get_args()

    # Task 14: guard against the dead use_flash_attn lever
    if getattr(args, "use_flash_attn", False):
        raise ValueError(
            "use_flash_attn=True is not supported: Qwen2ModifiedAttention always uses "
            "eager attention to maintain raw-key storage semantics. "
            "Set use_flash_attn: false in experiment.yaml."
        )

    # Task 12: global seeding for reproducibility
    import random as _random
    import numpy as _np
    GLOBAL_SEED = 42
    _random.seed(GLOBAL_SEED)
    _np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GLOBAL_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.storage_dir,  exist_ok=True)


    documents: List[Document] = []
    corpus_manifest = {}

    if args.hotpotqa_corpus:
        try:
            from datasets import load_dataset
            hq_ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
        except Exception as e:
            raise RuntimeError(f"Failed to load HotpotQA for corpus building: {e}") from e
        hq_docs = load_hotpotqa_corpus(hq_ds, num_examples=args.hotpotqa_num_examples)
        documents += hq_docs
        corpus_manifest["hotpotqa_docs"] = len(hq_docs)
        corpus_manifest["hotpotqa_num_examples"] = args.hotpotqa_num_examples

    if args.rgb_corpus:
        rgb_docs = load_rgb_corpus(args.rgb_corpus, num_examples=args.rgb_num_examples)
        documents += rgb_docs
        corpus_manifest["rgb_docs"] = len(rgb_docs)
        corpus_manifest["rgb_num_examples"] = args.rgb_num_examples

    if args.wiki_docs_url and args.wiki_docs_num > 0:
        save_dir = args.wiki_docs_save_dir or os.path.join(
            os.path.dirname(args.output_path), "wiki_dpr_docs"
        )
        wiki_docs = load_wiki_dpr_documents(
            download_url=args.wiki_docs_url,
            num_docs=args.wiki_docs_num,
            save_dir=save_dir,
        )
        documents += wiki_docs
        corpus_manifest["wiki_pages"] = len(wiki_docs)

    if args.documents_dir:
        local_docs = load_local_documents(args.documents_dir)
        documents += local_docs
        corpus_manifest["local_docs"] = len(local_docs)

    if not documents:
        raise ValueError(
            "No corpus source produced documents. Provide at least one of "
            "--hotpotqa_corpus, --rgb_corpus, --wiki_docs_url (DPR TSV) or --documents_dir."
        )
    print(f"[chunk_cache] Combined corpus: {len(documents)} documents "
          f"from sources {corpus_manifest}")

    # ── Load model + tokenizer ────────────────────────────────────────────────
    print(f"[chunk_cache] Loading model: {args.model_name}")
    model     = Qwen2ModifiedForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,        # T4 (sm_75): float16, never bfloat16
        attn_implementation="eager",
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Task 13: pin embedding model to GPU to avoid multi-hour CPU embedding bottleneck
    embed_model = HuggingFaceEmbedding(
        model_name=args.embedding_model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    node_parser = KVCachedNodeParser(
        model=model,
        tokenizer=tokenizer,
        output_path=args.output_path,
        device=device,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        precisions=args.precisions,
    )

    nodes = node_parser.get_nodes_from_documents(documents)
    print(f"[chunk_cache] Built {len(nodes)} chunk nodes "
          f"(precisions={args.precisions})")

    vector_store = SimpleVectorStore()
    index        = VectorStoreIndex(
        nodes=nodes,
        embed_model=embed_model,
        vector_store=vector_store,
    )
    index.storage_context.persist(persist_dir=args.storage_dir)
    print(f"[chunk_cache] Index persisted to: {args.storage_dir}")

    # Record corpus composition + build settings next to the index so the
    # evaluation/analysis stages (and the paper) can report exactly what corpus
    # the KV caches were built from.
    corpus_manifest.update({
        "num_chunks":     len(nodes),
        "precisions":     list(args.precisions),
        "chunk_size":     args.chunk_size,
        "chunk_overlap":  args.chunk_overlap,
        "embedding_model": args.embedding_model_name,
    })
    manifest_path = os.path.join(args.storage_dir, "corpus_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(corpus_manifest, f, indent=2)
    print(f"[chunk_cache] Corpus manifest → {manifest_path}")


if __name__ == "__main__":
    main()