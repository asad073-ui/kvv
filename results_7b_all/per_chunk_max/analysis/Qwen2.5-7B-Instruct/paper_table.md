# Precomputed-KV-Cache RAG — Quantization Faithfulness — Main Results

_model_: `Qwen/Qwen2.5-7B-Instruct` · _wiki_pages_: 0 · _datasets_: rgb · _K_: [1, 3, 5] · _conditions_: ['C0', 'C1', 'C2', 'C3']

| Dataset | K | Condition | ContainEM (prim.) | EM | F1 (secondary) | Hallucination | Entailment | NonFinite | Degen. | KV Size (MB) | TTFT (s) | Latency (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rgb | 1 | Full-Context Oracle | 0.7433 | 0.0700 | 0.3209 | 0.0623 | 0.6782 | 0 | 0.0000 | 0.000 | 1.4363 | 1.4363 |
| rgb | 1 | FP16 Precomputed-KV | 0.7500 | 0.0733 | 0.3215 | 0.0602 | 0.6912 | 0 | 0.0000 | 15.870 | 0.0820 | 1.3371 |
| rgb | 1 | INT8 Precomputed-KV | 0.7267 | 0.0600 | 0.3092 | 0.1004 | 0.6877 | 0 | 0.0000 | 8.431 | 0.0834 | 1.3499 |
| rgb | 1 | INT4 Precomputed-KV | 0.3867 | 0.0567 | 0.2132 | 0.5060 | 0.3198 | 0 | 0.1000 | 4.216 | 0.0812 | 1.9138 |
| rgb | 3 | Full-Context Oracle | 0.8467 | 0.1333 | 0.3997 | 0.0481 | 0.7571 | 0 | 0.0000 | 0.000 | 1.4252 | 1.4252 |
| rgb | 3 | FP16 Precomputed-KV | 0.8500 | 0.1267 | 0.3931 | 0.0447 | 0.7543 | 0 | 0.0000 | 39.335 | 0.0830 | 1.4725 |
| rgb | 3 | INT8 Precomputed-KV | 0.8367 | 0.1200 | 0.3913 | 0.0962 | 0.7355 | 0 | 0.0000 | 20.897 | 0.0873 | 1.4320 |
| rgb | 3 | INT4 Precomputed-KV | 0.3233 | 0.0500 | 0.1780 | 0.6220 | 0.3612 | 0 | 0.2367 | 10.448 | 0.0930 | 3.3217 |
| rgb | 5 | Full-Context Oracle | 0.9100 | 0.1200 | 0.4152 | 0.0269 | 0.8329 | 0 | 0.0000 | 0.000 | 1.5943 | 1.5943 |
| rgb | 5 | FP16 Precomputed-KV | 0.9067 | 0.1333 | 0.4206 | 0.0272 | 0.8426 | 0 | 0.0000 | 62.320 | 0.0864 | 1.6073 |
| rgb | 5 | INT8 Precomputed-KV | 0.8700 | 0.1067 | 0.4008 | 0.0408 | 0.7885 | 0 | 0.0000 | 33.107 | 0.0936 | 1.5244 |
| rgb | 5 | INT4 Precomputed-KV | 0.2767 | 0.0267 | 0.1498 | 0.6905 | 0.4086 | 0 | 0.2033 | 16.554 | 0.0832 | 3.3248 |
