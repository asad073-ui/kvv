# Precomputed-KV-Cache RAG — Quantization Faithfulness — Main Results

_model_: `Qwen/Qwen2.5-3B-Instruct` · _wiki_pages_: 0 · _datasets_: rgb · _K_: [1, 3, 5] · _conditions_: ['C0', 'C1', 'C2', 'C3']

| Dataset | K | Condition | ContainEM (prim.) | EM | F1 (secondary) | Hallucination | Entailment | NonFinite | Degen. | KV Size (MB) | TTFT (s) | Latency (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rgb | 1 | Full-Context Oracle | 0.7700 | 0.0067 | 0.2593 | 0.0469 | 0.7726 | 0 | 0.0000 | 0.000 | 1.0556 | 1.0556 |
| rgb | 1 | FP16 Precomputed-KV | 0.7767 | 0.0067 | 0.2629 | 0.0469 | 0.7722 | 0 | 0.0000 | 10.202 | 0.0581 | 1.0322 |
| rgb | 1 | INT8 Precomputed-KV | 0.7767 | 0.0067 | 0.2638 | 0.0469 | 0.7720 | 0 | 0.0000 | 5.420 | 0.0647 | 1.1618 |
| rgb | 1 | INT4 Precomputed-KV | 0.7067 | 0.0433 | 0.2527 | 0.1211 | 0.6303 | 0 | 0.0300 | 2.710 | 0.0689 | 1.5592 |
| rgb | 3 | Full-Context Oracle | 0.9133 | 0.0133 | 0.3107 | 0.0447 | 0.8678 | 0 | 0.0000 | 0.000 | 1.1836 | 1.1836 |
| rgb | 3 | FP16 Precomputed-KV | 0.9100 | 0.0100 | 0.3082 | 0.0451 | 0.8738 | 0 | 0.0000 | 25.287 | 0.0598 | 1.1837 |
| rgb | 3 | INT8 Precomputed-KV | 0.9067 | 0.0100 | 0.3090 | 0.0417 | 0.8666 | 0 | 0.0000 | 13.434 | 0.0597 | 1.1354 |
| rgb | 3 | INT4 Precomputed-KV | 0.9067 | 0.0433 | 0.3123 | 0.1250 | 0.8197 | 0 | 0.0000 | 6.717 | 0.0599 | 1.3831 |
| rgb | 5 | Full-Context Oracle | 0.9600 | 0.0133 | 0.3197 | 0.0235 | 0.9206 | 0 | 0.0000 | 0.000 | 1.2689 | 1.2689 |
| rgb | 5 | FP16 Precomputed-KV | 0.9600 | 0.0100 | 0.3194 | 0.0235 | 0.9222 | 0 | 0.0000 | 40.063 | 0.0609 | 1.2468 |
| rgb | 5 | INT8 Precomputed-KV | 0.9600 | 0.0133 | 0.3217 | 0.0235 | 0.9171 | 0 | 0.0000 | 21.283 | 0.0599 | 1.1303 |
| rgb | 5 | INT4 Precomputed-KV | 0.9200 | 0.0433 | 0.3211 | 0.0503 | 0.8847 | 0 | 0.0100 | 10.642 | 0.0593 | 1.2554 |
