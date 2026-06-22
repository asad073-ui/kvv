# Precomputed-KV-Cache RAG: Quantization-Induced Hallucination Study

**Paper:** *Quantization-Induced Hallucination in Precomputed-KV-Cache RAG: Does Offline KV Cache Compression Hurt Faithfulness More Than Factual Accuracy?*

---

## Research Question

In a precomputed-KV-cache RAG system, does offline chunk-level KV cache quantization increase hallucination **faster** than it decreases factual accuracy (containment-EM)?

> **Methodology note (v2).** The default precompute path is **`standard_causal`**:
> retrieved chunks are concatenated and prefilled with ONE stock causal pass, and
> *that* cache is what gets quantized. This is in-distribution, needs no fine-tuning,
> and isolates the quantization effect. The original TurboRAG **`reordered`**
> independent-attention scheme is OOD on an un-adapted checkpoint (the paper recovers
> accuracy only via SFT, which we do not have) and is kept as a labelled secondary
> baseline. We therefore title the work "precomputed-KV-cache RAG," not "TurboRAG."

---

## 🖥️ Run on RTX 3090

The repo is configured for the **headline RGB-only experiment** on a single
**NVIDIA RTX 3090 (24 GB VRAM, Ampere sm_86)**: RGB × C0/C1/C2/C3 × K ∈ {1, 3, 5},
with HHEM-2.1-Open + DeBERTa-NLI faithfulness scoring. It runs in **bfloat16**.

> **bfloat16 is required (ROOT CAUSE 1).** Qwen2.5 is bf16-native; loading it in
> float16 overflows its KV values to ±Inf → NaN logits → token 0 (`!`), which was
> the v1 corruption. A NaN/Inf logit guard now aborts and counts any such failure,
> and a **Stage-0 gate** halts the pipeline before C2/C3 unless C1 ≈ C0.

> **Read [RUN_3090.md](RUN_3090.md) first.** In `standard_causal` the binding
> constraint is neither disk nor VRAM — caches are prefilled at query time, so no
> per-chunk `.pt` files are written. (The disk table applies only to `reordered`.)

### 1. Install dependencies

```bash
pip install -r requirements.txt          # pins transformers==4.51.3
```

### 2. Set environment variables (override defaults as needed)

```bash
export SCRATCH_DIR="${SCRATCH_DIR:-/scratch/${USER}/turborag_quant}"
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
```

### 3. Sanity-check the environment

```python
import torch, transformers
print("GPU:", torch.cuda.get_device_name(0))          # expect: NVIDIA GeForce RTX 3090
print("transformers:", transformers.__version__)      # must be 4.51.3
print("capability:", torch.cuda.get_device_capability(0))  # (8, 6) = Ampere
```

### 4. Gate-check the baseline first

```bash
python src/run_experiment.py --config configs/full_experiment.yaml --stages build stage0
```

Stage-0 runs C0+C1 on 50 examples and **fails loudly** unless: zero non-finite
logits, degeneracy < 2% per condition, C1 EM/F1 within ~10% of C0, and C1
hallucination within 10pp of C0. If it fails, fix `dtype`/`precompute_mode` before
spending GPU hours — do **not** collect C2/C3 numbers against a failing gate.

### 5. Headline run (~2–3.5 h)

```bash
# RGB-only, bf16, standard_causal — all from the YAML defaults.
python src/run_experiment.py --config configs/full_experiment.yaml
```

Confirm coherence: `n_nonfinite == 0` everywhere, `degenerate_rate` low,
`paired_n > 0`, and C0/C1 most faithful while C2 → C3 degrade.

### 6. View the results

```python
import glob, pandas as pd, os
SCRATCH = os.environ["SCRATCH_DIR"]
MODEL_SLUG = os.environ["MODEL_SLUG"]
csv = sorted(glob.glob(f'{SCRATCH}/results/{MODEL_SLUG}/summary_*.csv'))[-1]
df = pd.read_csv(csv)
df[['dataset','k','condition','wiki_pages','n_examples','paired_n','refusal_rate',
    'EM','F1','hallucination_rate','entailment_score','avg_ttft_s','avg_latency_s']]
```

```python
# Publication master table + H1/H2/H3 hypothesis report
print(open(sorted(glob.glob(f'{SCRATCH}/analysis/{MODEL_SLUG}/paper_table.md'))[-1], encoding='utf-8').read())
print(open(sorted(glob.glob(f'{SCRATCH}/analysis/{MODEL_SLUG}/report.txt'))[-1], encoding='utf-8').read())
```

---

## ⚡ Quick local smoke test (MVE)

Before the long paper run you can run a HotpotQA-only **Minimum Viable Experiment**
(20 examples, ~10–15 min) to confirm the pipeline is wired correctly. It builds the
retrieval corpus from HotpotQA's own paragraphs (no Wikipedia download):

```bash
bash scripts/mve.sh         # uses the mve: block in configs/full_experiment.yaml
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
`results/${MODEL_SLUG}/config.json`, echoed into every results/summary row, and the combined
stdout of all stages is teed to `results/${MODEL_SLUG}/logs_<ts>.txt`.

---

## Project Structure

```
turborag_quant/
├── configs/
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
`${SCRATCH_DIR}/doc_emb/${MODEL_SLUG}/corpus_manifest.json`.

---

## The two config files

`configs/full_experiment.yaml` is the paper profile (all three datasets, wiki on,
single GPU). `configs/full_experiment.yaml` is the MVE smoke profile. Both are read
automatically; any value can be overridden from the CLI (above).

Key sections of `full_experiment.yaml`:

```yaml
model:
  name: "Qwen/Qwen2.5-3B-Instruct"   # debug on 3B; switch to 7B for the final run
  dtype: "bfloat16"                  # ROOT CAUSE 1: Qwen2.5 is bf16-native; fp16 → NaN→token-0

precompute_mode: "standard_causal"   # in-distribution (default). "reordered" = legacy TurboRAG (OOD, secondary)

wiki_docs:
  num_docs:     0                    # Fix 6: NQ-Open/DPR wiki DISABLED for the RGB-only headline run

datasets:                            # num_examples <= 0 disables a dataset entirely
  nq_open:  { hf_name: "nq_open",            num_examples: 0 }   # dropped (needs full DPR index)
  hotpotqa: { hf_name: "hotpotqa/hotpot_qa", hf_config: "distractor", num_examples: 0 }  # optional secondary
  rgb:      { hf_name: null, query_file: "questions/rgb.jsonl",  num_examples: 200 }     # headline dataset

k_values:   [1, 3, 5]
conditions: [C0, C1, C2, C3]
retrieval:  { similarity_top_k: 10 }          # must be >= max(k_values)

evaluation:
  eval_hhem: true
  eval_nli:  true
  faithfulness_mode: "full_context"           # run both modes and report (see below)
  hhem_batch_size: 16                          # raise to 32–64 on the 3090
  nli_batch_size:  16

stage0:           { n_examples: 50, acc_rel_tol: 0.10, faith_abs_tol: 0.10, max_degenerate_rate: 0.02 }
calibration:      { n_examples: 300, condition: "all" }   # variance-bearing set; gate on both p-values

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
| `${SCRATCH_DIR}/chunk_kvcache/${MODEL_SLUG}/` | Per-chunk `.pt` files at the requested precisions — **delete after analysis** |
| `${SCRATCH_DIR}/doc_emb/${MODEL_SLUG}/` | LlamaIndex embedding index + `corpus_manifest.json` |
| `${SCRATCH_DIR}/results/${MODEL_SLUG}/config.json` | Fully-resolved run config (incl. `wiki_pages`) |
| `${SCRATCH_DIR}/results/${MODEL_SLUG}/logs_<ts>.txt` | Combined stdout/stderr of every stage |
| `${SCRATCH_DIR}/results/${MODEL_SLUG}/results_<ts>.jsonl` | Per-example: query, prediction, context, TTFT, latency, KV bytes, per-example HHEM/NLI |
| `${SCRATCH_DIR}/results/${MODEL_SLUG}/summary_<ts>.csv` / `.json` | Main metric table (EM, F1, hallucination, entailment, TTFT, latency, KV size, wiki_pages) |
| `${SCRATCH_DIR}/results/${MODEL_SLUG}/meta_<ts>.json` | git commit, pip freeze, torch/transformers versions, seed |
| `${SCRATCH_DIR}/analysis/${MODEL_SLUG}/paper_table.{csv,md,tex}` | Publication master table |
| `${SCRATCH_DIR}/analysis/${MODEL_SLUG}/figure{1,2,3}_data.csv` | H1/H2/H3 figure data |
| `${SCRATCH_DIR}/analysis/${MODEL_SLUG}/report.txt` | Human-readable H1/H2/H3 verdicts |

---

## Experimental Conditions

In `standard_causal` (default) the "precomputed cache" is one stock causal prefill
over the concatenated retrieved chunks; the precision below is applied to *that*
cache. C1 is therefore a lossless **bf16** round-trip and should match C0 (the
Stage-0 gate enforces this).

| Condition | Label | Description |
|---|---|---|
| C0 | Gold Oracle RAG | Raw text context, standard generation. Upper-bound reference. |
| C1 | BF16 reference  | Prefilled cache round-tripped through the codec in bf16 (lossless baseline). |
| C2 | INT8 cache      | Offline INT8 asymmetric quantization of the prefilled cache, dequantized to bf16. |
| C3 | INT4 cache      | Offline INT4 symmetric quantization (packed uint8), dequantized to bf16. |

---

## Hypotheses

Per-example tests run on the raw `results_<ts>.jsonl` (pass `--results_jsonl` to
`analyze_results.py`; the orchestrator does this automatically).

| Hypothesis | Test | Supported when |
|---|---|---|
| H1 – Asymmetric Degradation | **McNemar** on C1→C3 faithfulness flips within the EM-preserved subset (b = faithful→hallucinated, c = reverse) | `b > c` and McNemar `p < 0.05` |
| H2 – Multi-Chunk Amplification | OLS slope of the paired INT4−FP16 hallucination gap on K, with bootstrap CIs (attention-JS mechanism test is a documented stretch) | slope > 0, `p < 0.05` |
| H3 – Noise Complexity (RGB only) | OLS regression of the paired INT4−FP16 gap on the retrieved-distractor (noise) ratio | slope > 0, `p < 0.05` |

---

## Faithfulness modes (matters for H2)

`evaluation.faithfulness_mode` (or pass through `--faithfulness_mode`):

- **`full_context`** (config default) — score the answer against the concatenated
  context the model actually attended over (paper-faithful for H2; HHEM/NLI handle
  their own truncation).
- **`per_chunk_max`** — max faithfulness across retrieved chunks. Avoids the
  K-dependent truncation artifact, but is lenient and can **mask** the H2 multi-chunk
  amplification effect (more chunks → more chances some chunk supports the answer).

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
