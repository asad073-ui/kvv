"""
calibrate_metrics.py  –  Stage 8: Metric Calibration Study.

Runs HHEM-2.1-Open and DeBERTa-NLI on a calibration set of N examples
under the FP16 TurboRAG condition and reports:
  - Pearson / Spearman correlation between the two signals
  - Agreement rate (both flag same examples as hallucinated)
  - A scatter-plot CSV for plotting

Usage
─────
python src/calibrate_metrics.py \
    --results_jsonl results/results_<timestamp>.jsonl \
    --condition C1 \
    --n_calibration 50 \
    --output_dir results/calibration
"""

import os, sys, json, argparse
import torch
from scipy.stats import pearsonr, spearmanr
import csv

sys.path.insert(0, os.path.dirname(__file__))
from metrics import HHEMScorer, DeBERTaNLIScorer, hallucination_rate, mean_entailment
from evaluate import trim_context_for_hhem

def main():
    parser = argparse.ArgumentParser(description="Metric calibration: HHEM vs DeBERTa-NLI")
    parser.add_argument("--results_jsonl", type=str, required=True,
                        help="Raw results JSONL from evaluate.py")
    parser.add_argument("--condition",     type=str, default="C1")
    parser.add_argument("--dataset",       type=str, default=None,
                        help="Filter to a specific dataset (default: all)")
    parser.add_argument("--n_calibration", type=int, default=50)
    parser.add_argument("--output_dir",    type=str, default="results/calibration")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load records ──
    records = []
    with open(args.results_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            if rec["condition"] != args.condition:
                continue
            if args.dataset and rec["dataset"] != args.dataset:
                continue
            records.append(rec)
    records = records[:args.n_calibration]
    print(f"Calibration set: {len(records)} examples (condition={args.condition})")



    # ── Score ──
    print("Loading HHEM …")
    hhem   = HHEMScorer(device=device)
    
    contexts = [trim_context_for_hhem(r["context"]) for r in records]
    # FIX: use raw predictions (matches FIX-4/FIX-6 in evaluate.py)
    answers = [r["prediction"] for r in records]
    hhem_scores = hhem.batch_score(contexts, answers)

    print("Loading DeBERTa-NLI …")
    nli    = DeBERTaNLIScorer(device=device)
    
    # contexts are already trimmed to 512 tokens by trim_context_for_hhem above.
    nli_scores  = nli.batch_score(contexts, answers)

    ent_scores  = [e for e, _, _ in nli_scores]

    # ── Correlation ──
    pr, pp = pearsonr(hhem_scores, ent_scores)
    sr, sp = spearmanr(hhem_scores, ent_scores)
    hall_rate = hallucination_rate(hhem_scores)
    avg_ent   = mean_entailment(nli_scores)

    print(f"\nCalibration Results (N={len(records)}, condition={args.condition})")
    print(f"  HHEM avg faithfulness : {sum(hhem_scores)/len(hhem_scores):.4f}")
    print(f"  Hallucination rate    : {hall_rate:.4f}")
    print(f"  DeBERTa avg entailment: {avg_ent:.4f}")
    print(f"  Pearson r(HHEM, NLI)  : {pr:.4f}  (p={pp:.4e})")
    print(f"  Spearman r(HHEM, NLI) : {sr:.4f}  (p={sp:.4e})")

    # ── Save scatter CSV ──
    out_csv = os.path.join(args.output_dir, "calibration_scatter.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "hhem_faithfulness", "nli_entailment", "nli_neutral", "nli_contradiction"])
        for i, (hs, (e, n, c)) in enumerate(zip(hhem_scores, nli_scores)):
            w.writerow([i, hs, e, n, c])
    print(f"\nScatter data → {out_csv}")

    # ── Save summary ──
    summary = {
        "n":                len(records),
        "condition":        args.condition,
        "hhem_avg":         sum(hhem_scores)/len(hhem_scores),
        "hallucination_rate": hall_rate,
        "nli_avg_entailment": avg_ent,
        "pearson_r":        pr,
        "pearson_p":        pp,
        "spearman_r":       sr,
        "spearman_p":       sp,
    }
    with open(os.path.join(args.output_dir, "calibration_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nRecommendation:")
    if abs(pr) > 0.7 or abs(sr) > 0.7:
        print("  Metrics agree strongly. Use HHEM as primary, DeBERTa-NLI as secondary.")
    else:
        print("  Metrics disagree. Report both and discuss failure modes.")


if __name__ == "__main__":
    main()
