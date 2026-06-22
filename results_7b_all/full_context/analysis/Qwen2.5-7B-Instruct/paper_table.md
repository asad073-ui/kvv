# Precomputed-KV-Cache RAG — Quantization Faithfulness — Main Results

_model_: `Qwen/Qwen2.5-7B-Instruct` · _wiki_pages_: 0 · _datasets_: rgb · _K_: [1, 3, 5] · _conditions_: ['C0', 'C1', 'C2', 'C3']

| Dataset | K | Condition | ContainEM (prim.) | EM | F1 (secondary) | Hallucination | Entailment | NonFinite | Degen. | KV Size (MB) | TTFT (s) | Latency (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rgb | 1 | Full-Context Oracle | 0.7433 | 0.0700 | 0.3209 | 0.0623 | 0.7174 | 0 | 0.0000 | 0.000 | 1.4701 | 1.4701 |
| rgb | 1 | FP16 Precomputed-KV | 0.7500 | 0.0733 | 0.3215 | 0.0602 | 0.7307 | 0 | 0.0000 | 15.870 | 0.0867 | 1.4186 |
| rgb | 1 | INT8 Precomputed-KV | 0.7267 | 0.0600 | 0.3092 | 0.1004 | 0.7137 | 0 | 0.0000 | 8.431 | 0.0863 | 1.4091 |
| rgb | 1 | INT4 Precomputed-KV | 0.3867 | 0.0567 | 0.2132 | 0.5060 | 0.3239 | 0 | 0.1000 | 4.216 | 0.0882 | 2.0861 |
| rgb | 3 | Full-Context Oracle | 0.8467 | 0.1333 | 0.3997 | 0.0412 | 0.5613 | 0 | 0.0000 | 0.000 | 1.5447 | 1.5447 |
| rgb | 3 | FP16 Precomputed-KV | 0.8500 | 0.1267 | 0.3931 | 0.0344 | 0.5766 | 0 | 0.0000 | 39.335 | 0.0880 | 1.5543 |
| rgb | 3 | INT8 Precomputed-KV | 0.8367 | 0.1200 | 0.3913 | 0.0756 | 0.5666 | 0 | 0.0000 | 20.897 | 0.0890 | 1.4648 |
| rgb | 3 | INT4 Precomputed-KV | 0.3233 | 0.0500 | 0.1780 | 0.6151 | 0.2855 | 0 | 0.2367 | 10.448 | 0.0882 | 3.1740 |
| rgb | 5 | Full-Context Oracle | 0.9100 | 0.1200 | 0.4152 | 0.0202 | 0.6260 | 0 | 0.0000 | 0.000 | 1.8400 | 1.8400 |
| rgb | 5 | FP16 Precomputed-KV | 0.9067 | 0.1333 | 0.4206 | 0.0204 | 0.6241 | 0 | 0.0000 | 62.320 | 0.0877 | 1.6409 |
| rgb | 5 | INT8 Precomputed-KV | 0.8700 | 0.1067 | 0.4008 | 0.0306 | 0.6035 | 0 | 0.0000 | 33.107 | 0.0863 | 1.4047 |
| rgb | 5 | INT4 Precomputed-KV | 0.2767 | 0.0267 | 0.1498 | 0.7007 | 0.3407 | 0 | 0.2033 | 16.554 | 0.0866 | 3.4826 |
