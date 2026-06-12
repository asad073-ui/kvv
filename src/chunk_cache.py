"""
chunk_cache.py  –  Offline KV-cache builder for TurboRAG quantization study.

For every document chunk it:
  1. Tokenises the chunk with <|doc_start|> … <|doc_end|> delimiters.
  2. Runs a forward pass through Qwen2ModifiedForCausalLM to obtain the raw KV cache.
  3. Compresses the KV cache to FP16, INT8, and INT4 and saves each version.
  4. Builds a LlamaIndex VectorStore index over all nodes (one per chunk), where
     each node carries the paths to its three cached files.

Documents are sourced from the DPR Wikipedia passages TSV hosted by Facebook:
  https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz

On the first run the requested passages are streamed and saved to
--wiki_docs_save_dir/wiki_passages.jsonl.  Subsequent runs reload from that
file so no re-download is needed.  Only ~1 MB is transferred for 10k passages
(the first rows of the 2.2 GB gzip, then the connection is closed).

Usage
─────
python src/chunk_cache.py \
    --model_name       /path/to/Qwen2.5-3B-Instruct \
    --wiki_docs_url    https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz \
    --wiki_docs_num    10000 \
    --wiki_docs_save_dir /scratch/$USER/turborag_quant/wiki_dpr_docs \
    --output_path        /scratch/$USER/turborag_quant/chunk_kvcache \
    --storage_dir        /scratch/$USER/turborag_quant/doc_emb
"""

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


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(
        description="Build per-precision offline KV caches + embedding index"
    )
    parser.add_argument("--model_name",           type=str, required=True)
    parser.add_argument("--embedding_model_name", type=str, default="BAAI/bge-small-en-v1.5")

    # DPR Wikipedia passage source (direct TSV download)
    parser.add_argument("--wiki_docs_url",      type=str, default=DPR_TSV_URL,
                        help="URL of psgs_w100.tsv.gz (Facebook CDN)")
    parser.add_argument("--wiki_docs_num",      type=int, default=10000,
                        help="Number of passages to stream from the TSV")
    parser.add_argument("--wiki_docs_save_dir", type=str, default=None,
                        help="Directory to cache wiki_passages.jsonl. "
                             "Defaults to --output_path/../wiki_dpr_docs")

    # Local document fallback (used when --wiki_docs_url is empty string)
    parser.add_argument("--documents_dir",    type=str, default=None,
                        help="Local directory of .txt files (fallback if wiki_docs_url is empty)")

    # KV-cache and index paths
    parser.add_argument("--output_path",   type=str, default="chunk_kvcache")
    parser.add_argument("--storage_dir",   type=str, default="doc_emb")
    parser.add_argument("--chunk_size",    type=int, default=512)
    parser.add_argument("--chunk_overlap", type=int, default=10)
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Document loading: DPR Wikipedia TSV (direct download, no HF datasets needed)
# ──────────────────────────────────────────────────────────────────────────────

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
        with open(cache_file) as f:
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
             open(tmp_file, "w") as fout:

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


# ──────────────────────────────────────────────────────────────────────────────
# Local document fallback (original documents/ directory loader)
# ──────────────────────────────────────────────────────────────────────────────

def load_local_documents(documents_dir: str) -> List[Document]:
    from llama_index.core import SimpleDirectoryReader
    print(f"[chunk_cache] Loading documents from local directory: {documents_dir}")
    docs = SimpleDirectoryReader(documents_dir).load_data()
    print(f"[chunk_cache] Found {len(docs)} document(s)")
    return docs


# ──────────────────────────────────────────────────────────────────────────────
# Helper: normalise DynamicCache.to_legacy_cache() output
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Core: compute + save per-chunk KV caches
# ──────────────────────────────────────────────────────────────────────────────

def compute_and_save_chunk(
    chunk_text: str,
    chunk_id: str,
    model: Qwen2ModifiedForCausalLM,
    tokenizer,
    output_path: str,
    device: torch.device,
) -> dict:
    """Forward-pass a single chunk and save FP16/INT8/INT4 compressed caches."""
    wrapped = f"<|doc_start|>{chunk_text}<|doc_end|>"
    inputs  = tokenizer(wrapped, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    # DynamicCache.to_legacy_cache() format changed in transformers>=4.45.
    # _to_per_layer_pairs() normalises both old and new formats so that
    # compress_kvcache always receives [(k0,v0), (k1,v1), ...].
    raw_legacy   = outputs.past_key_values.to_legacy_cache()
    legacy_cache = _to_per_layer_pairs(raw_legacy)

    paths = {}
    for prec in PRECISIONS:
        compressed = compress_kvcache(legacy_cache, prec)
        fpath      = os.path.join(output_path, f"kvcache_chunk_{chunk_id}_{prec}.pt")
        torch.save(compressed, fpath)
        paths[prec] = fpath

    return paths


# ──────────────────────────────────────────────────────────────────────────────
# LlamaIndex node parser that wraps our cache builder
# ──────────────────────────────────────────────────────────────────────────────

class KVCachedNodeParser:
    """
    Splits each document into token-bounded chunks, runs forward passes,
    and saves FP16 / INT8 / INT4 KV caches for each chunk.

    Plain Python class — does not inherit from any Pydantic-backed LlamaIndex
    base so arbitrary attributes can be set freely.
    """

    def __init__(self, model, tokenizer, output_path, device, chunk_size=512, chunk_overlap=10):
        self.model        = model
        self.tokenizer    = tokenizer
        self.output_path  = output_path
        self.device       = device
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
                    self.tokenizer, self.output_path, self.device
                )
                node = TextNode(
                    text=f"<|doc_start|>{chunk_text}<|doc_end|>",
                    id_=f"chunk_{chunk_id}",
                    metadata={
                        "kvcache_fp16": paths["fp16"],
                        "kvcache_int8": paths["int8"],
                        "kvcache_int4": paths["int4"],
                        "raw_text":     chunk_text,
                    },
                )
                nodes.append(node)
                global_chunk_id += 1

        return nodes


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.storage_dir,  exist_ok=True)

    # ── Load documents ────────────────────────────────────────────────────────
    if args.wiki_docs_url:
        save_dir = args.wiki_docs_save_dir or os.path.join(
            os.path.dirname(args.output_path), "wiki_dpr_docs"
        )
        documents = load_wiki_dpr_documents(
            download_url=args.wiki_docs_url,
            num_docs=args.wiki_docs_num,
            save_dir=save_dir,
        )
    elif args.documents_dir:
        documents = load_local_documents(args.documents_dir)
    else:
        raise ValueError(
            "Provide either --wiki_docs_url (DPR TSV) or --documents_dir (local .txt files)"
        )

    # ── Load model + tokenizer ────────────────────────────────────────────────
    print(f"[chunk_cache] Loading model: {args.model_name}")
    model     = Qwen2ModifiedForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    embed_model = HuggingFaceEmbedding(model_name=args.embedding_model_name)

    node_parser = KVCachedNodeParser(
        model=model,
        tokenizer=tokenizer,
        output_path=args.output_path,
        device=device,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    nodes = node_parser.get_nodes_from_documents(documents)
    print(f"[chunk_cache] Built {len(nodes)} chunk nodes")

    vector_store = SimpleVectorStore()
    index        = VectorStoreIndex(
        nodes=nodes,
        embed_model=embed_model,
        vector_store=vector_store,
    )
    index.storage_context.persist(persist_dir=args.storage_dir)
    print(f"[chunk_cache] Index persisted to: {args.storage_dir}")


if __name__ == "__main__":
    main()