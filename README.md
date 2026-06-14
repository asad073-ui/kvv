# TurboRAG KV-Cache Quantization Study

**Paper:** *Quantization-Induced Hallucination in TurboRAG: Does Offline KV Cache Compression Hurt Faithfulness More Than Factual Accuracy?*

---

## Research Question

In a TurboRAG-style precomputed RAG system, does offline chunk-level KV cache quantization increase hallucination **faster** than it decreases standard factual accuracy metrics (EM / F1)?

---

## 🖥️ Run on RTX 3090 (full paper experiment)

This repo is configured to run the **full 3-dataset paper experiment** on a single
**NVIDIA RTX 3090 (24 GB VRAM, Ampere sm_86)**: NQ-Open + HotpotQA + RGB, all four
conditions (C0 Gold Oracle, C1 FP16, C2 INT8, C3 INT4) over K ∈ {1, 3, 5}, with
HHEM-2.1-Open + DeBERTa-NLI faithfulness scoring. It runs in **float16** (the stored
KV caches are float16; the 3090 also supports bfloat16, but float16 keeps the model
numerically consistent with the caches).

> **Read [RUN_3090.md](RUN_3090.md) first** for VRAM / disk / runtime estimates.
> The binding constraint is **disk, not VRAM**: all three precisions cost ≈ 64.7 KB
> per token, so 10k wiki pages ≈ 92 GB of KV cache. Size your scratch disk before
> launching.

### 1. Install dependencies

```bash
pip install -r requirements.txt          # pins transformers==4.51.3
```

### 2. Point the env vars at fast LOCAL NVMe (the eval stage is I/O bound)

```bash
export SCRATCH_DIR=/mnt/nvme/turborag_quant
export HF_HOME=/mnt/nvme/hf_cache
```

### 3. Sanity-check the environment

```python
import torch, transformers
print("GPU:", torch.cuda.get_device_name(0))          # expect: NVIDIA GeForce RTX 3090
print("transformers:", transformers.__version__)      # must be 4.51.3
print("capability:", torch.cuda.get_device_capability(0))  # (8, 6) = Ampere
```

### 4. Small validation run first (~45 min, ~80 GB disk)

```bash
python src/run_experiment.py --config configs/full_experiment.yaml \
    --wiki_pages 1000 --num_nq_examples 50 --num_hotpot_examples 50 --num_rgb_examples 50
```

Confirm the numbers are coherent: `refusal_rate` is sane for **all three** datasets
(a high value for `nq_open` means the wiki corpus is too small to cover the
questions), `paired_n > 0`, and C0/C1 are the most faithful while C2 → C3 degrade.

### 5. Paper run (~6–9 h)

```bash
python src/run_experiment.py --config configs/full_experiment.yaml \
    --wiki_pages 10000 \
    --num_nq_examples 200 --num_hotpot_examples 200 --num_rgb_examples 200 \
    --k_values 1 3 5 --conditions C0 C1 C2 C3
```

### 6. View the results

```python
import glob, pandas as pd, os
SCRATCH = os.environ["SCRATCH_DIR"]
csv = sorted(glob.glob(f'{SCRATCH}/results/summary_*.csv'))[-1]
df = pd.read_csv(csv)
df[['dataset','k','condition','wiki_pages','n_examples','paired_n','refusal_rate',
    'EM','F1','hallucination_rate','entailment_score','avg_ttft_s','avg_latency_s']]
```

```python
# Publication master table + H1/H2/H3 hypothesis report
print(open(sorted(glob.glob(f'{SCRATCH}/analysis/paper_table.md'))[-1], encoding='utf-8').read())
print(open(sorted(glob.glob(f'{SCRATCH}/analysis/report.txt'))[-1], encoding='utf-8').read())
```

---

## ⚡ Quick local smoke test (MVE)

Before the long paper run you can run a HotpotQA-only **Minimum Viable Experiment**
(20 examples, ~10–15 min) to confirm the pipeline is wired correctly. It builds the
retrieval corpus from HotpotQA's own paragraphs (no Wikipedia download):

```bash
bash scripts/mve.sh         # uses the mve: block in configs/experiment.yaml
```

> With only 20 examples the H1/H2/H3 tests are not statistically powered (expected).
> You are checking that the numbers are *coherent*, not that the hypotheses are
> "SUPPORTED".

---

## New CLI surface (no source edits required)

Every experiment knob is a flag on `run_experiment.py` and overrides the YAML:

| Flag | Effect |
|---|---|
| `--wiki_pages N` | DPR Wikipedia passages to ingest (drives NQ-Open coverage). `0` disables the wiki source. |
| `--num_nq_examples N` | NQ-Open eval examples (`0` drops the dataset) |
| `--num_hotpot_examples N` | HotpotQA eval examples (`0` drops the dataset) |
| `--num_rgb_examples N` | RGB eval examples (`0` drops the dataset) |
| `--k_values 1 3 5` | Number(s) of retrieved chunks |
| `--conditions C0 C1 C2 C3` | Conditions to run (also selects which precisions get built) |
| `--model_name`, `--cache_gpu`, `--eval_gpu`, `--config`, `--stages`, `--mve` | as before |

Passing any of the first six flags switches OFF MVE mode and runs the full datasets.
The **resolved run config** (including `wiki_pages`) is saved to
`results/config.json`, echoed into every results/summary row, and the combined
stdout of all stages is teed to `results/logs_<ts>.txt`.

---

## Project Structure

```
turborag_quant/
├── configs/
│   ├── experiment.yaml         # MVE / smoke-test profile (HotpotQA only)
│   └── full_experiment.yaml    # ← FULL paper profile (NQ + HotpotQA + RGB)
├── src/
│   ├── qwen2.py                # Modified Qwen2 with RoPE-free (raw) key caching
│   ├── kv_quantization.py      # FP16 / INT8 / INT4 offline KV cache compression
│   ├── chunk_cache.py          # Stage 1+2: build COMBINED corpus + per-chunk KV caches
│   ├── evaluate.py             # Stage 3–7: evaluation loop (conditions × K × datasets)
│   ├── metrics.py              # EM, F1, HHEM, DeBERTa-NLI scorers
│   ├── calibrate_metrics.py    # Stage 8: HHEM vs DeBERTa-NLI correlation
│   ├── analyze_results.py      # Stage 9–11: hypothesis testing + figure CSVs
│   ├── make_paper_tables.py    # Publication master table (CSV / Markdown / LaTeX)
│   ├── config.py               # YAML loader (expands ${SCRATCH_DIR}, ${HF_HOME})
│   └── run_experiment.py       # Master runner — reads YAML + CLI, orchestrates all stages
├── scripts/                    # thin wrappers around run_experiment.py stages
├── questions/
│   └── rgb.jsonl               # RGB dataset: query, answer, positive[], negative[] docs
├── RUN_3090.md                 # VRAM / disk / runtime estimates + optimal settings
└── requirements.txt
```

> NQ-Open and HotpotQA questions are fetched automatically from HuggingFace Hub.
> RGB is local (`questions/rgb.jsonl`) and ships its own positive + negative documents.

---

## How the corpus is built (combined, single global index)

`chunk_cache.py` builds **one** retrieval index from the **union** of every source the
active datasets need — this is what makes the full 3-dataset run possible:

| Source | Enabled when | Gives recall for |
|---|---|---|
| HotpotQA gold + distractor paragraphs | `hotpotqa` is active | multi-hop QA (H2) |
| RGB positive + negative documents (`rgb.jsonl`) | `rgb` is active | noisy retrieval (H3) |
| DPR Wikipedia passages | `--wiki_pages > 0` | single-hop NQ-Open (H1) |

Each chunk produces a `.pt` file **per requested precision**. `--conditions`
determines which precisions get built (C1→fp16, C2→int8, C3→int4; C0 needs none),
so running fewer conditions saves disk. The corpus composition is recorded in
`${SCRATCH_DIR}/doc_emb/corpus_manifest.json`.

---

## The two config files

`configs/full_experiment.yaml` is the paper profile (all three datasets, wiki on,
single GPU). `configs/experiment.yaml` is the MVE smoke profile. Both are read
automatically; any value can be overridden from the CLI (above).

Key sections of `full_experiment.yaml`:

```yaml
model:
  name: "Qwen/Qwen2.5-3B-Instruct"   # ← set to your TurboRAG checkpoint if different
  dtype: "float16"                   # stored caches are float16; keep model fp16 to match

wiki_docs:
  download_url: "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"
  num_docs:     10000                # ← or override with --wiki_pages

datasets:
  nq_open:  { hf_name: "nq_open",           hf_split: "validation", num_examples: 200 }
  hotpotqa: { hf_name: "hotpotqa/hotpot_qa", hf_config: "distractor", num_examples: 200 }
  rgb:      { hf_name: null, query_file: "questions/rgb.jsonl",       num_examples: 200 }

k_values:   [1, 3, 5]
conditions: [C0, C1, C2, C3]
retrieval:  { similarity_top_k: 10 }          # must be >= max(k_values)

evaluation:
  eval_hhem: true
  eval_nli:  true
  faithfulness_mode: "per_chunk_max"          # see "Faithfulness modes" below
  hhem_batch_size: 16                          # raise to 32–64 on the 3090
  nli_batch_size:  16

gpu:
  chunk_cache_gpu: 0
  evaluate_gpu:    0                           # single RTX 3090
```

Verify the config resolves and inspect the generated stage args (no work done):

```bash
python src/config.py                                         # print fully-expanded config
python src/run_experiment.py --config configs/full_experiment.yaml --dry_run \
    --wiki_pages 10000 --num_nq_examples 200 --num_hotpot_examples 200 --num_rgb_examples 200
```

---

## Running individual stages

```bash
# Everything (build → eval → calib → analyze → tables)
python src/run_experiment.py --config configs/full_experiment.yaml --stages all \
    --wiki_pages 10000 --num_nq_examples 200 --num_hotpot_examples 200 --num_rgb_examples 200

# Just one stage (caches already built)
python src/run_experiment.py --config configs/full_experiment.yaml --stages eval
python src/run_experiment.py --config configs/full_experiment.yaml --stages analyze tables
```

Stages: `build` (KV caches + index) · `eval` (conditions × K × datasets) ·
`calib` (HHEM vs NLI) · `analyze` (H1/H2/H3 + figure CSVs) · `tables` (publication table).

---

## What Gets Stored Where

| Location | Contents |
|---|---|
| `${SCRATCH_DIR}/wiki_dpr_docs/wiki_passages.jsonl` | DPR Wikipedia passages (built once, reloaded thereafter) |
| `${SCRATCH_DIR}/chunk_kvcache/` | Per-chunk `.pt` files at the requested precisions — **delete after analysis** |
| `${SCRATCH_DIR}/doc_emb/` | LlamaIndex embedding index + `corpus_manifest.json` |
| `${SCRATCH_DIR}/results/config.json` | Fully-resolved run config (incl. `wiki_pages`) |
| `${SCRATCH_DIR}/results/logs_<ts>.txt` | Combined stdout/stderr of every stage |
| `${SCRATCH_DIR}/results/results_<ts>.jsonl` | Per-example: query, prediction, context, TTFT, latency, KV bytes, per-example HHEM/NLI |
| `${SCRATCH_DIR}/results/summary_<ts>.csv` / `.json` | Main metric table (EM, F1, hallucination, entailment, TTFT, latency, KV size, wiki_pages) |
| `${SCRATCH_DIR}/results/meta_<ts>.json` | git commit, pip freeze, torch/transformers versions, seed |
| `${SCRATCH_DIR}/analysis/paper_table.{csv,md,tex}` | Publication master table |
| `${SCRATCH_DIR}/analysis/figure{1,2,3}_data.csv` | H1/H2/H3 figure data |
| `${SCRATCH_DIR}/analysis/report.txt` | Human-readable H1/H2/H3 verdicts |

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
| H1 – Asymmetric Degradation | `hall_delta_pp > f1_delta_pp` | Hallucination worsens faster than F1 decreases |
| H2 – Multi-Chunk Amplification | `δ3 > 3·δ1` and `δ5 > 5·δ1` | INT4–FP16 hallucination gap grows super-linearly with K |
| H3 – Task-Complexity | NQ-Open gap < HotpotQA gap < RGB gap | Effect is strongest under noisy / multi-hop retrieval |

---

## Faithfulness modes (matters for H2)

`evaluation.faithfulness_mode` (or pass through `--faithfulness_mode`):

- **`per_chunk_max`** (default) — max faithfulness across retrieved chunks. Avoids the
  K-dependent truncation artifact, but is lenient and can **mask** the H2 multi-chunk
  amplification effect (more chunks → more chances some chunk supports the answer).
- **`full_context`** — score the answer against the concatenated context the model
  actually attended over (paper-faithful for H2; HHEM/NLI handle their own truncation).

**Recommendation: run both and report.** They probe different things.

---

## RGB Dataset (local JSONL)

NQ-Open and HotpotQA load from HuggingFace Hub. RGB is not on the Hub; it lives at
`questions/rgb.jsonl` and ships its own evidence documents:

```json
{"id": 0, "query": "...", "answer": [["1889", ...]], "positive": ["..."], "negative": ["...", "..."]}
```

Accepted field names: `query`/`question`, `answer`/`answers` (list). The `positive`
and `negative` documents are ingested into the retrieval corpus automatically when
`rgb` is an active dataset. To skip RGB, run with `--num_rgb_examples 0`.

---

## Key Design Notes

**RoPE-free key storage (`qwen2.py`):** Keys are stored raw (un-rotated) so RoPE can be
re-applied at attention time over the full stitched sequence with global reordered
position IDs — the core TurboRAG mechanism that makes independently-cached chunks
composable. Requires eager attention (FlashAttention is intentionally unsupported).

**Quantization is offline (`kv_quantization.py`):** Compression happens during the build
stage, not at query time, isolating storage-level compression from inference compute.

**Per-precision files per chunk:** Each chunk produces one `.pt` file per requested
precision. The retrieval node stores those paths; `evaluate.py` loads the right one per
condition. `--conditions` controls which precisions are built (storage saver).

**Reproducibility:** global seed 42 (Python/NumPy/torch), cuDNN deterministic, and a
`meta_<ts>.json` capturing git commit, `pip freeze`, and library versions per run.

---

## See also

- [RUN_3090.md](RUN_3090.md) — VRAM / disk / runtime estimates, bottlenecks, optimal settings
- `python src/run_experiment.py --dry_run` — inspect fully resolved config + generated CLI args
