"""
analyze_results.py  –  Hypothesis testing and figure generation.

Reads the summary JSON produced by evaluate.py and runs:
  H1 – Asymmetric Degradation: does hallucination drop faster than F1?
  H2 – Multi-Chunk Amplification: does the INT4–FP16 gap grow super-linearly with K?
  H3 – Task-Complexity: is the effect largest on RGB, then HotpotQA, then NQ-Open?
  Efficiency: tabulate KV storage size and TTFT by precision.

Outputs
───────
  analysis/h1_degradation.csv
  analysis/h2_amplification.csv
  analysis/h3_complexity.csv
  analysis/efficiency.csv
  analysis/figure1_data.csv     – relative F1 vs faithfulness drop by precision
  analysis/figure2_data.csv     – INT4−FP16 hallucination gap vs K
  analysis/figure3_data.csv     – storage size vs hallucination rate
  analysis/report.txt           – human-readable summary

Usage
─────
python src/analyze_results.py \
    --summary_json results/summary_<timestamp>.json \
    --output_dir analysis
"""

import os, sys, json, csv, argparse, math
try:
    from scipy.stats import proportions_ztest
except ImportError:
    proportions_ztest = None

from typing import List, Dict, Any


def get_count(row, possible_keys):
    """
    Try to read a count field from the summary row.
    Returns None if no matching key exists.
    """
    for key in possible_keys:
        if key in row:
            try:
                return int(row[key])
            except (TypeError, ValueError):
                return None
    return None


def get_hall_counts(row):
    """
    Try to recover hallucination count and total count from a summary row.

    This requires evaluate.py to store either:
      - n_hall / n_total
      - num_hallucinated / num_examples
      - hallucinated_count / total_count

    If counts are unavailable, return (None, None).
    """
    n_hall = get_count(row, [
        "n_hall",
        "num_hallucinated",
        "hallucinated_count",
        "hall_count",
    ])

    n_total = get_count(row, [
        "n_total",
        "num_examples",
        "total_count",
        "n",
    ])

    return n_hall, n_total


def load_summary(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def safe_float(v) -> float:
    try:
        f = float(v)
        return f if not math.isnan(f) else 0.0
    except (TypeError, ValueError):
        return 0.0


def get_row(data, dataset, k, condition):
    for r in data:
        if r["dataset"] == dataset and r["k"] == k and r["condition"] == condition:
            return r
    return None


def relative_drop(base_val: float, new_val: float) -> float:
    """Return percentage drop relative to base (positive = degradation)."""
    if base_val == 0:
        return 0.0
    return (base_val - new_val) / base_val * 100


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_json", type=str, required=True)
    parser.add_argument("--output_dir",   type=str, default="analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = load_summary(args.summary_json)

    datasets   = sorted({r["dataset"]   for r in data})
    conditions = sorted({r["condition"] for r in data})
    k_values   = sorted({r["k"]         for r in data})

    report_lines = ["=" * 70, "TurboRAG KV Quantization – Hypothesis Analysis", "=" * 70, ""]

    # ──────────────────────────────────────────────────────────────────────────
    # H1: Asymmetric Degradation (FP16 → INT8 → INT4)
    # ──────────────────────────────────────────────────────────────────────────
    h1_rows = []
    report_lines.append("H1 – Asymmetric Degradation (relative drop from FP16 baseline)")
    report_lines.append("-" * 70)
    for ds in datasets:
        for k in k_values:
            base = get_row(data, ds, k, "C1")
            if base is None:
                continue
            base_f1   = safe_float(base["F1"])
            base_hall = safe_float(base["hallucination_rate"])
            for cond in ["C2", "C3"]:
                row = get_row(data, ds, k, cond)
                if row is None:
                    continue
                row_f1   = safe_float(row["F1"])
                row_hall = safe_float(row["hallucination_rate"])
                f1_drop   = relative_drop(base_f1,   row_f1)
                faith_drop = relative_drop(1 - base_hall, 1 - row_hall)  # faithfulness drop
                asymmetric = faith_drop > f1_drop
                h1_rows.append({
                    "dataset":          ds,
                    "k":                k,
                    "condition":        cond,
                    "fp16_f1":          round(base_f1, 4),
                    "fp16_hall":        round(base_hall, 4),
                    "cond_f1":          round(row_f1, 4),
                    "cond_hall":        round(row_hall, 4),
                    "f1_drop_pct":      round(f1_drop, 2),
                    "faith_drop_pct":   round(faith_drop, 2),
                    "h1_supported":     asymmetric,
                })
                report_lines.append(
                    f"  {ds:12s} K={k} {cond}: F1 drop={f1_drop:.1f}%  "
                    f"Faith drop={faith_drop:.1f}%  "
                    f"H1={'SUPPORTED' if asymmetric else 'NOT supported'}"
                )

    write_csv(
        os.path.join(args.output_dir, "h1_degradation.csv"),
        ["dataset","k","condition","fp16_f1","fp16_hall","cond_f1","cond_hall",
         "f1_drop_pct","faith_drop_pct","h1_supported"],
        h1_rows
    )
    report_lines.append("")

    # Minimum gap threshold to avoid noise-level false positives (used by H2 and H3).
    MIN_GAP = 0.02

    # ──────────────────────────────────────────────────────────────────────────
    # H2: Multi-Chunk Amplification  δK = Hall(INT4,K) − Hall(FP16,K)
    # ──────────────────────────────────────────────────────────────────────────
    h2_rows = []
    report_lines.append("H2 – Multi-Chunk Amplification (INT4−FP16 hallucination gap vs K)")
    report_lines.append("-" * 70)
    for ds in datasets:
        delta_k = {}
        for k in k_values:
            r_fp16 = get_row(data, ds, k, "C1")
            r_int4 = get_row(data, ds, k, "C3")
            if r_fp16 is None or r_int4 is None:
                continue
            delta = safe_float(r_int4["hallucination_rate"]) - safe_float(r_fp16["hallucination_rate"])
            delta_k[k] = delta
            h2_rows.append({"dataset": ds, "k": k, "delta_hall_int4_fp16": round(delta, 4)})

        # Test super-linearity: δ3 > 3*δ1 and δ5 > 5*δ1
        # Test super-linearity with direction guard:
        # H2 should be supported only if hallucination actually increases under INT4.
        if 1 in delta_k and 3 in delta_k and 5 in delta_k:
            d1 = delta_k[1]
            d3 = delta_k[3]
            d5 = delta_k[5]

            # Direction guard + minimum gap guard:
            # Gaps must be positive and above noise level before claiming amplification.
            h2_3_supported = (d1 > 0) and (d3 > MIN_GAP) and (d3 > 3 * d1)
            h2_5_supported = (d1 > 0) and (d5 > MIN_GAP) and (d5 > 5 * d1)

            report_lines.append(
                f"  {ds:12s}  "
                f"δ1={d1:.4f}  δ3={d3:.4f}  δ5={d5:.4f}  "
                f"3δ1={3*d1:.4f}  5δ1={5*d1:.4f}  "
                f"H2(3)={'SUPPORTED' if h2_3_supported else 'NOT supported'}  "
                f"H2(5)={'SUPPORTED' if h2_5_supported else 'NOT supported'}"
            )

    write_csv(
        os.path.join(args.output_dir, "h2_amplification.csv"),
        ["dataset", "k", "delta_hall_int4_fp16"],
        h2_rows
    )
    report_lines.append("")

    # ──────────────────────────────────────────────────────────────────────────
    # H3: Task-Complexity  (NQ-Open < HotpotQA < RGB)
    # ──────────────────────────────────────────────────────────────────────────
    h3_rows = []



    report_lines.append("H3 – Task Complexity (expected: NQ-Open < HotpotQA < RGB)")
    report_lines.append("-" * 70)

    for k in k_values:
        ds_gaps = {}
        ds_rows = {}

        for ds in datasets:
            r_fp16 = get_row(data, ds, k, "C1")
            r_int4 = get_row(data, ds, k, "C3")

            if r_fp16 is None or r_int4 is None:
                continue

            fp16_hall = safe_float(r_fp16["hallucination_rate"])
            int4_hall = safe_float(r_int4["hallucination_rate"])

            gap = int4_hall - fp16_hall

            ds_gaps[ds] = gap
            ds_rows[ds] = {
                "fp16": r_fp16,
                "int4": r_int4,
                "fp16_hall": fp16_hall,
                "int4_hall": int4_hall,
                "gap": gap,
            }

            h3_rows.append({
                "k": k,
                "dataset": ds,
                "fp16_hall": round(fp16_hall, 4),
                "int4_hall": round(int4_hall, 4),
                "int4_fp16_hall_gap": round(gap, 4),
            })

        if not ds_gaps:
            continue

        sorted_ds = sorted(ds_gaps.items(), key=lambda x: x[1])

        report_lines.append(
            f"  K={k}  Ranking by INT4−FP16 gap ascending: "
            + " < ".join(f"{d}({g:.4f})" for d, g in sorted_ds)
        )

        # Dataset names may differ slightly in your summary JSON.
        # Adjust these strings if your dataset names are different.
        nq_name = None
        hotpot_name = None
        rgb_name = None

        for ds in ds_gaps:
            ds_lower = ds.lower()
            if "nq" in ds_lower:
                nq_name = ds
            elif "hotpot" in ds_lower:
                hotpot_name = ds
            elif "rgb" in ds_lower:
                rgb_name = ds

        if nq_name is None or hotpot_name is None or rgb_name is None:
            report_lines.append(
                f"  K={k}  H3=NOT evaluated "
                f"(missing one of NQ-Open, HotpotQA, RGB)"
            )
            continue

        gap_nq = ds_gaps[nq_name]
        gap_hotpot = ds_gaps[hotpot_name]
        gap_rgb = ds_gaps[rgb_name]

        nq_to_hotpot_diff = gap_hotpot - gap_nq
        hotpot_to_rgb_diff = gap_rgb - gap_hotpot

        # Direction + minimum-gap guard:
        # H3 requires RGB gap > HotpotQA gap > NQ gap,
        # and each adjacent difference must be larger than MIN_GAP.
        h3_direction_supported = (
            gap_nq < gap_hotpot < gap_rgb
            and nq_to_hotpot_diff > MIN_GAP
            and hotpot_to_rgb_diff > MIN_GAP
        )

        h3_significance_supported = True
        pval_nq_hotpot = None
        pval_hotpot_rgb = None

        # Optional statistical significance check.
        # This only works if your summary JSON contains hallucination counts.
        if proportions_ztest is not None:
            nq_hall, nq_total = get_hall_counts(ds_rows[nq_name]["int4"])
            hotpot_hall, hotpot_total = get_hall_counts(ds_rows[hotpot_name]["int4"])
            rgb_hall, rgb_total = get_hall_counts(ds_rows[rgb_name]["int4"])

            counts_available = all(v is not None for v in [
                nq_hall, nq_total,
                hotpot_hall, hotpot_total,
                rgb_hall, rgb_total,
            ])

            if counts_available:
                _, pval_nq_hotpot = proportions_ztest(
                    [nq_hall, hotpot_hall],
                    [nq_total, hotpot_total]
                )

                _, pval_hotpot_rgb = proportions_ztest(
                    [hotpot_hall, rgb_hall],
                    [hotpot_total, rgb_total]
                )

                h3_significance_supported = (
                    pval_nq_hotpot < 0.05
                    and pval_hotpot_rgb < 0.05
                )
            else:
                h3_significance_supported = False

        h3_supported = h3_direction_supported and h3_significance_supported

        report_lines.append(
            f"  K={k}  "
            f"gap_nq={gap_nq:.4f}  "
            f"gap_hotpot={gap_hotpot:.4f}  "
            f"gap_rgb={gap_rgb:.4f}  "
            f"ΔHotpot−NQ={nq_to_hotpot_diff:.4f}  "
            f"ΔRGB−Hotpot={hotpot_to_rgb_diff:.4f}  "
            f"H3={'SUPPORTED' if h3_supported else 'NOT supported'}"
        )

        if pval_nq_hotpot is not None and pval_hotpot_rgb is not None:
            report_lines.append(
                f"        p(NQ vs Hotpot)={pval_nq_hotpot:.4g}  "
                f"p(Hotpot vs RGB)={pval_hotpot_rgb:.4g}"
            )
        elif proportions_ztest is None:
            report_lines.append(
                "        Statistical test skipped because scipy is not installed."
            )
        else:
            report_lines.append(
                "        Statistical test skipped because hallucination counts are missing."
            )

    write_csv(
        os.path.join(args.output_dir, "h3_complexity.csv"),
        [
            "k",
            "dataset",
            "fp16_hall",
            "int4_hall",
            "int4_fp16_hall_gap",
        ],
        h3_rows
    )

    report_lines.append("")

    # ──────────────────────────────────────────────────────────────────────────
    # Efficiency (Stage 11)
    # ──────────────────────────────────────────────────────────────────────────
    eff_rows = []
    report_lines.append("Efficiency – KV storage size and TTFT by condition")
    report_lines.append("-" * 70)
    for ds in datasets:
        for k in k_values:
            for cond in conditions:
                row = get_row(data, ds, k, cond)
                if row is None:
                    continue
                kv_mb = row["avg_kv_bytes"] / 1e6
                eff_rows.append({
                    "dataset":       ds,
                    "k":             k,
                    "condition":     cond,
                    "avg_kv_mb":     round(kv_mb, 3),
                    "avg_ttft_s":    row["avg_ttft_s"],
                })
    write_csv(
        os.path.join(args.output_dir, "efficiency.csv"),
        ["dataset", "k", "condition", "avg_kv_mb", "avg_ttft_s"],
        eff_rows
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Figure CSVs
    # ──────────────────────────────────────────────────────────────────────────

    # Figure 1: F1 drop vs faithfulness drop by precision (averaged over all ds+k)
    fig1 = []
    for cond in ["C1", "C2", "C3"]:
        f1_drops, faith_drops = [], []
        for ds in datasets:
            for k in k_values:
                base = get_row(data, ds, k, "C1")
                row  = get_row(data, ds, k, cond)
                if base is None or row is None:
                    continue
                f1_drops.append(relative_drop(safe_float(base["F1"]), safe_float(row["F1"])))
                faith_drops.append(relative_drop(
                    1 - safe_float(base["hallucination_rate"]),
                    1 - safe_float(row["hallucination_rate"])
                ))
        if f1_drops:
            fig1.append({
                "condition":             cond,
                "avg_relative_f1_drop":  round(sum(f1_drops)  / len(f1_drops),  2),
                "avg_relative_faith_drop": round(sum(faith_drops)/ len(faith_drops), 2),
            })
    write_csv(
        os.path.join(args.output_dir, "figure1_data.csv"),
        ["condition", "avg_relative_f1_drop", "avg_relative_faith_drop"],
        fig1
    )

    # Figure 2: INT4–FP16 hallucination gap vs K (per dataset)
    fig2 = []
    for ds in datasets:
        for k in k_values:
            r_fp16 = get_row(data, ds, k, "C1")
            r_int4 = get_row(data, ds, k, "C3")
            if r_fp16 and r_int4:
                gap = safe_float(r_int4["hallucination_rate"]) - safe_float(r_fp16["hallucination_rate"])
                fig2.append({"dataset": ds, "k": k, "delta_hall": round(gap, 4)})
    write_csv(
        os.path.join(args.output_dir, "figure2_data.csv"),
        ["dataset", "k", "delta_hall"],
        fig2
    )

    # Figure 3: Storage size vs hallucination rate (storage–faithfulness tradeoff)
    fig3 = []
    for ds in datasets:
        for k in k_values:
            for cond in ["C1", "C2", "C3"]:
                row = get_row(data, ds, k, cond)
                if row:
                    fig3.append({
                        "dataset":   ds, "k": k, "condition": cond,
                        "kv_mb":     round(row["avg_kv_bytes"]/1e6, 3),
                        "hall_rate": safe_float(row["hallucination_rate"]),
                    })
    write_csv(
        os.path.join(args.output_dir, "figure3_data.csv"),
        ["dataset", "k", "condition", "kv_mb", "hall_rate"],
        fig3
    )

    # ── Write report ──
    report_path = os.path.join(args.output_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print("\n".join(report_lines))
    print(f"\nReport → {report_path}")
    print(f"Outputs → {args.output_dir}/")


if __name__ == "__main__":
    main()
