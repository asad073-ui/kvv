# TurboRAG KV-Cache Quantization Study

**Paper:** *Quantization-Induced Hallucination in TurboRAG: Does Offline KV Cache Compression Hurt Faithfulness More Than Factual Accuracy?*

---

## Research Question

In a TurboRAG-style precomputed RAG system, does offline chunk-level KV cache quantization increase hallucination **faster** than it decreases standard factual accuracy metrics (EM / F1)?

---

## ⚡ Run on Google Colab (T4) — fast sanity pass

The repo ships configured for a **Minimum Viable Experiment** that runs end-to-end
on a free Colab **T4** in roughly **10–15 minutes**. It builds the retrieval corpus
from HotpotQA's own paragraphs (no Wikipedia download), then evaluates all four
conditions (C0 Gold Oracle, C1 FP16, C2 INT8, C3 INT4) over K ∈ {1, 3, 5} with
HHEM + DeBERTa-NLI faithfulness scoring. Everything is pinned to
`transformers==4.51.3` and runs in **float16** (T4 has no native bfloat16).

> Goal of this pass: confirm the metrics come out sensible — C0/C1 should be the
> most faithful, C2 → C3 should degrade. Once that looks right, raise
> `mve.num_examples` in `configs/experiment.yaml` (e.g. 100–200) and run on a
> bigger GPU for paper-grade numbers.

**1. Runtime → Change runtime type → T4 GPU.** Upload the repo (or
`git clone`), then point Colab at its folder:

```python
%cd /content/hall_kvcache-master      # ← the uploaded/cloned repo folder
```

**2. Install dependencies** (this pins `transformers==4.51.3`; Colab keeps its
pre-installed CUDA `torch`):

```python
!pip install -q -r requirements.txt
# If Colab prints "RESTART RUNTIME", do Runtime → Restart session, then re-run %cd.
```

**3. Sanity-check the environment:**

```python
import torch, transformers
print("GPU:", torch.cuda.get_device_name(0))
print("transformers:", transformers.__version__)   # must be 4.51.3
```

**4. Run the full MVE** (build caches → evaluate → analyze). `SCRATCH_DIR` and
`HF_HOME` default to `/content/...`, which is writable on Colab:

```python
!bash scripts/mve.sh
```

**5. View the metrics table:**

```python
import glob, pandas as pd
csv = sorted(glob.glob('/content/turborag_quant/results/summary_*.csv'))[-1]
df = pd.read_csv(csv)
df[['dataset','k','condition','n_examples','refusal_rate','EM','F1',
    'hallucination_rate','entailment_score']]
```

```python
# Hypothesis report (H1 asymmetric degradation, H2 amplification, H3 complexity)
print(open(sorted(glob.glob('/content/turborag_quant/analysis/report.txt'))[-1]).read())
```

> With only 20 examples the hypothesis tests will not be statistically powered
> (that is expected for a smoke pass) — you are checking that the numbers are
> *coherent*, not that H1/H2/H3 are "SUPPORTED".

---

## Project Structure

```
turborag_quant/
├── configs/
│   └── experiment.yaml        # ← Single source of truth for ALL parameters
├── src/
│   ├── qwen2.py               # Modified Qwen2 with RoPE-free key caching
│   ├── kv_quantization.py     # FP16 / INT8 / INT4 offline KV cache compression
│   ├── chunk_cache.py         # Stage 1+2: stream wiki_dpr passages, build KV caches
│   ├── evaluate.py            # Stage 3–7: evaluation loop (conditions × K × datasets)
│   ├── metrics.py             # EM, F1, HHEM, DeBERTa-NLI scorers
│   ├── calibrate_metrics.py   # Stage 8: HHEM vs DeBERTa-NLI correlation
│   ├── analyze_results.py     # Stage 9–11: hypothesis testing + figure CSVs
│   ├── config.py              # YAML loader (expands ${SCRATCH_DIR}, ${HF_HOME})
│   └── run_experiment.py      # Master runner — reads YAML, orchestrates all stages
├── scripts/
│   ├── 01_build_chunk_cache.sh  # → python src/run_experiment.py --stages build
│   ├── 02_evaluate.sh           # → python src/run_experiment.py --stages eval
│   ├── 03_calibrate.sh          # → python src/run_experiment.py --stages calib
│   ├── 04_analyze.sh            # → python src/run_experiment.py --stages analyze
│   ├── mve.sh                   # → run_experiment.py --stages build eval analyze --mve
│   └── run_full_pipeline.sh     # → python src/run_experiment.py --stages all
├── questions/
│   └── rgb.jsonl              # Local JSONL for RGB dataset (nq_open + hotpotqa load from HF)
├── results/                   # Auto-created: JSONL + CSV outputs
├── analysis/                  # Auto-created: hypothesis CSVs + report
└── requirements.txt
```

> **Document corpus and question datasets are fetched automatically from HuggingFace Hub.**
> You do not need to populate a `documents/` folder or prepare JSONL files for NQ-Open or HotpotQA.

---

## Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Edit configs/experiment.yaml — set model.name to your checkpoint path
#    That is the only required edit before running.
```

---

## The Only File You Need to Edit: `configs/experiment.yaml`

All parameters — model path, GPU assignment, number of wiki passages, which datasets
to run, K values, chunk size, faithfulness scorers — live in one place.
The bash scripts and Python runners read them automatically.

### Key sections

```yaml
# ── The ONE field you must change ─────────────────────────────────────────────
model:
  name: "/scratch/${USER}/hf_cache/hub/models--Qwen--Qwen2.5-3B-Instruct/..."
  #       ↑ set this to your Qwen2 TurboRAG checkpoint

# ── Document corpus (DPR Wikipedia passages, streamed from Facebook CDN) ──────
wiki_docs:
  download_url: "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"
  num_docs:     10000     # ← increase for fuller coverage; 10k is good for MVE
  save_dir:     "${SCRATCH_DIR}/wiki_dpr_docs"   # saved once, reloaded on next run
  # Only ~1 MB is transferred for 10k passages (first rows of the 2.2 GB gzip)

# ── Question datasets (loaded from HuggingFace Hub automatically) ─────────────
datasets:
  nq_open:
    hf_name:  "nq_open"            # huggingface.co/datasets/nq_open
    hf_split: "validation"
    num_examples: 200
  hotpotqa:
    hf_name:   "hotpotqa"          # huggingface.co/datasets/hotpotqa
    hf_config: "distractor"
    hf_split:  "validation"
    num_examples: 200
  rgb:
    hf_name:    null               # RGB not on HF Hub → use local JSONL
    query_file: "questions/rgb.jsonl"
    num_examples: 200

# ── Where heavy files are stored (scratch filesystem) ─────────────────────────
paths:
  kvcache_dir: "${SCRATCH_DIR}/chunk_kvcache"   # per-chunk .pt files
  storage_dir: "${SCRATCH_DIR}/doc_emb"         # LlamaIndex embedding index
  output_dir:  "results"
  analysis_dir: "analysis"

# ── GPU assignment ─────────────────────────────────────────────────────────────
gpu:
  chunk_cache_gpu: 0    # Stage 01 (LLM forward passes over all chunks)
  evaluate_gpu:    1    # Stages 02+03 (generation + HHEM + DeBERTa-NLI)

# ── Experiment parameters ──────────────────────────────────────────────────────
k_values:   [1, 3, 5]
conditions: [C0, C1, C2, C3]
chunking:
  chunk_size:    512
  chunk_overlap: 10
retrieval:
  similarity_top_k: 5

# ── Minimum Viable Experiment ──────────────────────────────────────────────────
mve:
  enabled:      false     # set true here, or pass --mve flag to run_experiment.py
  num_examples: 100
  datasets:     ["nq_open", "hotpotqa"]
  k_values:     [1, 3, 5]

# ── Faithfulness evaluation ────────────────────────────────────────────────────
evaluation:
  eval_hhem: true
  eval_nli:  true
```

### Verify that the config resolves correctly

```bash
# Print all values with ${SCRATCH_DIR} / ${HF_HOME} fully expanded
python src/config.py

# Show what CLI arguments will be generated for each stage (no work done)
python src/run_experiment.py --dry_run
```

---

## Running Experiments

Every bash script reads `configs/experiment.yaml` automatically.
**Set `SCRATCH_DIR` and `HF_HOME` in your shell once** (or accept the defaults),
then just run the script:

```bash
# Optional: override defaults (already set inside each script)
export SCRATCH_DIR=/scratch/${USER}/turborag_quant
export HF_HOME=/scratch/${USER}/hf_cache
```

### Quickest path — Minimum Viable Experiment

```bash
bash scripts/mve.sh
```

Runs Stage 01 (build caches), Stage 02 (evaluate), Stage 04 (analyze) with the
`mve:` block in the YAML — by default `num_examples=20`, `datasets=[hotpotqa]`,
`k_values=[1,3,5]`, conditions `C0–C3`. The retrieval corpus is built from the
HotpotQA questions' own paragraphs, so no Wikipedia download is needed.

The **first** run downloads the Qwen2.5-3B model (~6 GB) and the HotpotQA dataset
(cached under `$HF_HOME`); subsequent stages in the same session reuse the cache.
Each `mve.sh` invocation rebuilds the chunk caches + index from scratch (cheap at
`num_examples=20`).

### Full experiment

```bash
bash scripts/run_full_pipeline.sh
```

Runs all four stages end-to-end using the GPU IDs from `gpu.chunk_cache_gpu`
and `gpu.evaluate_gpu` in the YAML.

### Individual stages

```bash
bash scripts/01_build_chunk_cache.sh   # Stage 1+2: stream wiki_dpr → build KV caches
bash scripts/02_evaluate.sh            # Stage 3–7: evaluate all conditions × K
bash scripts/03_calibrate.sh           # Stage 8:   HHEM vs NLI correlation
bash scripts/04_analyze.sh             # Stage 9–11: H1/H2/H3 tests + figure CSVs
```

### Using the Python runner directly (more control)

```bash
# Run all stages
python src/run_experiment.py --stages all

# MVE mode (overrides mve.enabled in YAML temporarily)
python src/run_experiment.py --stages all --mve

# Run only evaluation (caches already built)
python src/run_experiment.py --stages eval

# Override model or GPU without editing the YAML
python src/run_experiment.py --stages all --model_name /other/model --cache_gpu 2 --eval_gpu 3

# Use a non-default config file
python src/run_experiment.py --config configs/my_experiment.yaml --stages all
```

### Override a single value without editing the YAML

For quick one-off overrides, pass flags directly:

```bash
bash scripts/02_evaluate.sh --eval_gpu 0         # run eval on GPU 0 instead of 1
bash scripts/run_full_pipeline.sh --cache_gpu 2  # cache stage on GPU 2
```

These flags are forwarded to `run_experiment.py` via `"$@"` and take precedence over the YAML.

---

## What Gets Stored Where

| Location | Contents |
|---|---|
| `${SCRATCH_DIR}/wiki_dpr_docs/wiki_passages.jsonl` | Downloaded Wikipedia passages (built once, reloaded thereafter) |
| `${SCRATCH_DIR}/chunk_kvcache/` | Per-chunk `.pt` files at fp16, int8, int4 |
| `${SCRATCH_DIR}/doc_emb/` | LlamaIndex embedding index |
| `results/results_<ts>.jsonl` | Per-example: query, prediction, context, TTFT, KV bytes, HHEM, NLI |
| `results/summary_<ts>.csv` | Main paper table: EM, F1, hallucination rate, entailment, TTFT, KV size |
| `results/summary_<ts>.json` | Same in JSON |
| `results/calibration/` | HHEM vs NLI scatter + correlation summary |
| `analysis/` | H1/H2/H3 CSVs, figure data, report.txt |

---

## Experimental Conditions

| Condition | Label | Description |
|---|---|---|
| C0 | Gold Oracle RAG | Raw text context, standard generation. Upper-bound reference. |
| C1 | FP16 TurboRAG | Precomputed FP16 chunk caches stitched at query time. |
| C2 | INT8 TurboRAG | Offline INT8 asymmetric quantization, dequantized before stitching. |
| C3 | INT4 TurboRAG | Offline INT4 symmetric quantization (packed uint8), dequantized before stitching. |

---

## Hypotheses

| Hypothesis | Test | Supported when |
|---|---|---|
| H1 – Asymmetric Degradation | `hall_drop_pct > f1_drop_pct` | Hallucination worsens faster than F1 decreases |
| H2 – Multi-Chunk Amplification | `δ3 > 3·δ1` and `δ5 > 5·δ1` | INT4–FP16 hallucination gap grows super-linearly with K |
| H3 – Task-Complexity | NQ-Open gap < HotpotQA gap < RGB gap | Effect is strongest under noisy / multi-hop retrieval |

---

## RGB Dataset (local JSONL only)

NQ-Open and HotpotQA are loaded automatically from HuggingFace Hub.
RGB is not available on HF Hub; place it at `questions/rgb.jsonl`:

```json
{"question": "What year was the Eiffel Tower built?", "answer": "1889"}
{"question": "Were Scott Derrickson and Ed Wood both directors?", "answers": ["yes"]}
```

Accepted field names: `query` / `question`, `answer` / `answers` (list).
If you do not have an RGB file, remove `rgb` from `conditions` in the YAML.

---

## Key Design Notes

**RoPE-free key storage (`qwen2.py`):** Keys are stored raw (un-rotated) so that
RoPE can be reapplied at attention time with global reordered position IDs —
the core TurboRAG mechanism that makes independently-cached chunks composable.

**Quantization is offline (`kv_quantization.py`):** Compression happens during
`chunk_cache.py` (Stage 01), not at query time. This isolates the effect of
storage-level compression from inference-time compute.

**Three files per chunk:** Every chunk produces three `.pt` files on disk
(fp16, int8, int4). The retrieval index node stores all three paths.
`evaluate.py` loads the right file based on the condition being evaluated.

**Wiki passages downloaded once:** On the first run, `chunk_cache.py` streams
`num_docs` rows from `psgs_w100.tsv.gz` on Facebook's CDN (no HuggingFace
datasets library needed) and writes them to
`$SCRATCH_DIR/wiki_dpr_docs/wiki_passages.jsonl`. Only ~1 MB is transferred
for 10k passages; the connection closes immediately after. Every subsequent
run loads from the cached JSONL — no re-downloading needed.

---

## See also

- `LINEAGE.md` — function-level mapping from the original `turbo_rag.py` to the new codebase
- `src/run_experiment.py --dry_run` — inspect fully resolved config + generated CLI args
