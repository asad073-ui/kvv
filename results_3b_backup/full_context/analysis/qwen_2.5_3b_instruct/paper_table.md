# Precomputed-KV-Cache RAG — Quantization Faithfulness — Main Results

_model_: `Qwen/Qwen2.5-3B-Instruct` · _wiki_pages_: 0 · _datasets_: rgb · _K_: [1, 3, 5] · _conditions_: ['C0', 'C1', 'C2', 'C3']

| Dataset | K | Condition | ContainEM (prim.) | EM | F1 (secondary) | Hallucination | Entailment | NonFinite | Degen. | KV Size (MB) | TTFT (s) | Latency (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rgb | 1 | Full-Context Oracle | 0.7700 | 0.0067 | 0.2593 | 0.0469 | 0.7729 | 0 | 0.0000 | 0.000 | 1.0378 | 1.0378 |
| rgb | 1 | FP16 Precomputed-KV | 0.7767 | 0.0067 | 0.2629 | 0.0469 | 0.7786 | 0 | 0.0000 | 10.202 | 0.0816 | 1.4529 |
| rgb | 1 | INT8 Precomputed-KV | 0.7767 | 0.0067 | 0.2638 | 0.0469 | 0.7782 | 0 | 0.0000 | 5.420 | 0.0752 | 1.3570 |
| rgb | 1 | INT4 Precomputed-KV | 0.7067 | 0.0433 | 0.2527 | 0.1211 | 0.6480 | 0 | 0.0300 | 2.710 | 0.0587 | 1.3238 |
| rgb | 3 | Full-Context Oracle | 0.9133 | 0.0133 | 0.3107 | 0.0275 | 0.6547 | 0 | 0.0000 | 0.000 | 1.1325 | 1.1325 |
| rgb | 3 | FP16 Precomputed-KV | 0.9100 | 0.0100 | 0.3082 | 0.0312 | 0.6605 | 0 | 0.0000 | 25.287 | 0.0561 | 1.1176 |
| rgb | 3 | INT8 Precomputed-KV | 0.9067 | 0.0100 | 0.3090 | 0.0278 | 0.6550 | 0 | 0.0000 | 13.434 | 0.0556 | 1.0656 |
| rgb | 3 | INT4 Precomputed-KV | 0.9067 | 0.0433 | 0.3123 | 0.0972 | 0.5989 | 0 | 0.0000 | 6.717 | 0.0590 | 1.3655 |
| rgb | 5 | Full-Context Oracle | 0.9600 | 0.0133 | 0.3197 | 0.0168 | 0.7257 | 0 | 0.0000 | 0.000 | 1.2559 | 1.2559 |
| rgb | 5 | FP16 Precomputed-KV | 0.9600 | 0.0100 | 0.3194 | 0.0168 | 0.7292 | 0 | 0.0000 | 40.063 | 0.0579 | 1.1908 |
| rgb | 5 | INT8 Precomputed-KV | 0.9600 | 0.0133 | 0.3217 | 0.0201 | 0.7213 | 0 | 0.0000 | 21.283 | 0.0602 | 1.1407 |
| rgb | 5 | INT4 Precomputed-KV | 0.9200 | 0.0433 | 0.3211 | 0.0403 | 0.6573 | 0 | 0.0100 | 10.642 | 0.0591 | 1.2437 |
