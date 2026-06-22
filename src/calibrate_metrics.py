import os, sys, json, argparse, random
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
    parser.add_argument("--condition",     type=str, default="all",
                        help="Condition to calibrate on, or 'all'/'any' to draw a "
                             "variance-bearing mix across C0-C3 (recommended).")
    parser.add_argument("--dataset",       type=str, default=None,
                        help="Filter to a specific dataset (default: all)")
    parser.add_argument("--n_calibration", type=int, default=300)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--output_dir",    type=str, default="results/calibration")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load records ──
    all_conditions = args.condition.strip().lower() in ("all", "any", "")
    records = []
    with open(args.results_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            if not all_conditions and rec["condition"] != args.condition:
                continue
            if args.dataset and rec["dataset"] != args.dataset:
                continue
            records.append(rec)
    # Fix 5: shuffle so the calibration sample is a representative mix across
    # conditions/datasets (not just the first cluster) before truncating to N.
    random.seed(args.seed)
    random.shuffle(records)
    records = records[:args.n_calibration]
    cond_label = "all" if all_conditions else args.condition
    n_by_cond = {}
    for r in records:
        n_by_cond[r["condition"]] = n_by_cond.get(r["condition"], 0) + 1
    print(f"Calibration set: {len(records)} examples (condition={cond_label}); "
          f"by condition: {n_by_cond}")

    if len(records) < 2:
        print(f"[calib] Only {len(records)} records found for condition={args.condition}. "
              "Need at least 2. Skipping calibration.")
        return

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

    # Fix 5: HHEM may only be declared primary when BOTH correlation coefficients
    # are meaningfully strong AND BOTH are statistically significant. A high r on a
    # near-zero-variance set with p > 0.05 is the exact spurious 'agree' verdict the
    # old single-condition calibration produced.
    both_strong      = (abs(pr) > 0.7) and (abs(sr) > 0.7)
    both_significant = (pp < 0.05) and (sp < 0.05)
    agree = both_strong and both_significant

    # ── Save summary ──
    summary = {
        "n":                len(records),
        "condition":        cond_label,
        "n_by_condition":   n_by_cond,
        "hhem_avg":         sum(hhem_scores)/len(hhem_scores),
        "hallucination_rate": hall_rate,
        "nli_avg_entailment": avg_ent,
        "pearson_r":        pr,
        "pearson_p":        pp,
        "spearman_r":       sr,
        "spearman_p":       sp,
        "both_strong":      bool(both_strong),
        "both_significant": bool(both_significant),
        "verdict_agree":    bool(agree),
    }
    with open(os.path.join(args.output_dir, "calibration_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item())

    print("\nRecommendation:")
    if agree:
        print("  Metrics agree strongly AND significantly (Pearson & Spearman, p<0.05). "
              "Use HHEM as primary, DeBERTa-NLI as secondary.")
    else:
        reason = []
        if not both_strong:
            reason.append("|r| not > 0.7 on both")
        if not both_significant:
            reason.append("not significant on both (p>=0.05)")
        print(f"  Metrics do NOT meet the primary-metric bar ({'; '.join(reason)}). "
              "Report both HHEM and DeBERTa-NLI and discuss failure modes.")


if __name__ == "__main__":
    main()
