import os, sys, json, csv, argparse, math, random, re, string

# Fix 3: statsmodels is a hard dependency now (it is in requirements.txt). No more
# silent try/except skip — if it is missing the analysis is incomplete and we want
# a loud failure rather than a quietly degraded report.
from statsmodels.stats.contingency_tables import mcnemar
# scipy is also required (calibration already depends on it) for the noise-slope.
from scipy.stats import linregress

from typing import List, Dict, Any, Tuple, Optional


# ── basic IO / parsing ───────────────────────────────────────────────────────

def load_summary(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def load_records(path: str) -> List[Dict]:
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def safe_float(v) -> float:
    if v is None or (isinstance(v, str) and v.strip().upper() == "N/A"):
        return float("nan")
    try:
        f = float(v)
        return f if not math.isnan(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def get_count(row, possible_keys):
    for key in possible_keys:
        if key in row:
            try:
                return int(row[key])
            except (TypeError, ValueError):
                return None
    return None


def get_row(data, dataset, k, condition):
    for r in data:
        if r["dataset"] == dataset and r["k"] == k and r["condition"] == condition:
            return r
    return None


def relative_drop(base_val: float, new_val: float) -> float:
    if base_val == 0:
        return 0.0
    return (base_val - new_val) / base_val * 100


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ── answer-correctness (containment-EM, matches evaluate.batch_contain_em) ────

def _normalize(text: str) -> str:
    text = (text or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def em_correct(prediction: str, golds) -> bool:
    """Containment-EM: any normalized gold alias is a substring of the prediction."""
    if isinstance(golds, str):
        golds = [golds]
    np_ = _normalize(prediction)
    return any(_normalize(g) in np_ for g in (golds or []) if g)


# ── bootstrap (paired per-example delta CI) ──────────────────────────────────

def bootstrap_ci_delta(delta_values: List[float], n_boot: int = 2000,
                       ci: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    """Bootstrap CI for the mean of paired per-example deltas (Fix 3)."""
    random.seed(seed)
    n = len(delta_values)
    if n == 0:
        return (float("nan"), float("nan"))
    boot_means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += delta_values[random.randrange(n)]
        boot_means.append(s / n)
    boot_means.sort()
    lo = boot_means[int((1 - ci) / 2 * n_boot)]
    hi = boot_means[int((1 + ci) / 2 * n_boot)]
    return (lo, hi)


# ── per-example pairing index ────────────────────────────────────────────────

def index_records(records: List[Dict]) -> Dict[Tuple[str, int], Dict[str, Dict[str, Dict]]]:
    """{(dataset, k): {condition: {query: record}}}."""
    idx: Dict[Tuple[str, int], Dict[str, Dict[str, Dict]]] = {}
    for r in records:
        key = (r["dataset"], r["k"])
        idx.setdefault(key, {}).setdefault(r["condition"], {})[r["query"]] = r
    return idx


def _paired_hall(c_lo: Dict[str, Dict], c_hi: Dict[str, Dict],
                 require_em_preserved: bool = True):
    """Yield (query, hall_lo, hall_hi, noise_ratio) for paired examples.

    Only examples present in both conditions with HHEM flags available are used.
    When require_em_preserved, restrict to examples whose containment-EM status is
    unchanged between the two conditions (isolates faithfulness flips from
    accuracy changes).
    """
    for q in set(c_lo) & set(c_hi):
        r_lo, r_hi = c_lo[q], c_hi[q]
        h_lo = r_lo.get("hhem_hallucinated")
        h_hi = r_hi.get("hhem_hallucinated")
        if h_lo is None or h_hi is None:
            continue
        if require_em_preserved:
            em_lo = em_correct(r_lo.get("prediction", ""), r_lo.get("gold", []))
            em_hi = em_correct(r_hi.get("prediction", ""), r_hi.get("gold", []))
            if em_lo != em_hi:
                continue
        noise = r_hi.get("noise_ratio")
        if noise is None:
            noise = r_lo.get("noise_ratio")
        yield q, bool(h_lo), bool(h_hi), noise


# ── H1: McNemar on paired faithfulness flips (C1 vs C3), EM-preserved subset ──

def analyze_h1_mcnemar(idx, report_lines) -> List[Dict]:
    rows = []
    report_lines.append("H1 – Asymmetric Degradation: McNemar on FP16→INT4 faithfulness flips")
    report_lines.append("     (EM-preserved subset; b=faithful→hallucinated, c=hallucinated→faithful)")
    report_lines.append("-" * 70)

    pooled = {"a": 0, "b": 0, "c": 0, "d": 0}
    for (ds, k), byc in sorted(idx.items()):
        c1 = byc.get("C1")
        c3 = byc.get("C3")
        if not c1 or not c3:
            continue
        a = b = c = d = 0
        for _q, h1, h3, _noise in _paired_hall(c1, c3, require_em_preserved=True):
            if not h1 and not h3:
                a += 1
            elif not h1 and h3:
                b += 1          # quantization MADE it hallucinate (degradation)
            elif h1 and not h3:
                c += 1          # quantization fixed a hallucination
            else:
                d += 1
        n = a + b + c + d
        for key, val in zip("abcd", (a, b, c, d)):
            pooled[key] += val
        if n == 0:
            report_lines.append(f"  {ds:12s} K={k}: no EM-preserved paired examples with HHEM flags")
            continue
        res = mcnemar([[a, b], [c, d]], exact=(b + c) < 25, correction=True)
        pval = float(res.pvalue)
        supported = (b > c) and (pval < 0.05)
        rows.append({
            "dataset": ds, "k": k, "n_pairs": n,
            "both_faithful_a": a, "faithful_to_hall_b": b,
            "hall_to_faithful_c": c, "both_hall_d": d,
            "mcnemar_p": round(pval, 6), "h1_supported": supported,
        })
        report_lines.append(
            f"  {ds:12s} K={k}: n={n}  b={b} c={c}  p={pval:.4g}  "
            f"H1={'SUPPORTED' if supported else 'NOT supported'}"
        )

    pa, pb, pc, pd = pooled["a"], pooled["b"], pooled["c"], pooled["d"]
    pn = pa + pb + pc + pd
    if pn > 0:
        res = mcnemar([[pa, pb], [pc, pd]], exact=(pb + pc) < 25, correction=True)
        pval = float(res.pvalue)
        supported = (pb > pc) and (pval < 0.05)
        rows.append({
            "dataset": "POOLED", "k": "all", "n_pairs": pn,
            "both_faithful_a": pa, "faithful_to_hall_b": pb,
            "hall_to_faithful_c": pc, "both_hall_d": pd,
            "mcnemar_p": round(pval, 6), "h1_supported": supported,
        })
        report_lines.append(
            f"  {'POOLED':12s}     : n={pn}  b={pb} c={pc}  p={pval:.4g}  "
            f"H1={'SUPPORTED' if supported else 'NOT supported'}"
        )
    report_lines.append("")
    return rows


# ── A3: LLM-judge as a third faithfulness anchor (side-by-side + H1 robustness) ─

def _has_llm_judge(records: List[Dict]) -> bool:
    return any(r.get("llm_faithful") is not None for r in records)


def _paired_llm_faithful(c_lo: Dict[str, Dict], c_hi: Dict[str, Dict]):
    """Yield (query, hallucinated_lo, hallucinated_hi) from LLM-judge faithful flags.

    Mirrors _paired_hall but keys off the LLM judge's `llm_faithful` label, mapped
    to a 'hallucinated' boolean (hallucinated = not faithful) so the McNemar table
    has the same b=faithful→hallucinated / c=hallucinated→faithful orientation.
    """
    for q in set(c_lo) & set(c_hi):
        f_lo = c_lo[q].get("llm_faithful")
        f_hi = c_hi[q].get("llm_faithful")
        if f_lo is None or f_hi is None:
            continue
        yield q, (not bool(f_lo)), (not bool(f_hi))


def analyze_h1_llm_mcnemar(idx, report_lines) -> List[Dict]:
    """H1 robustness check: McNemar on FP16→INT4 LLM-judge faithfulness flips."""
    rows = []
    report_lines.append("H1 (robustness) – McNemar on FP16→INT4 LLM-JUDGE faithfulness flips")
    report_lines.append("     (b=faithful→hallucinated, c=hallucinated→faithful)")
    report_lines.append("-" * 70)

    pooled = {"a": 0, "b": 0, "c": 0, "d": 0}
    for (ds, k), byc in sorted(idx.items()):
        c1, c3 = byc.get("C1"), byc.get("C3")
        if not c1 or not c3:
            continue
        a = b = c = d = 0
        for _q, h1, h3 in _paired_llm_faithful(c1, c3):
            if not h1 and not h3:
                a += 1
            elif not h1 and h3:
                b += 1
            elif h1 and not h3:
                c += 1
            else:
                d += 1
        n = a + b + c + d
        for key, val in zip("abcd", (a, b, c, d)):
            pooled[key] += val
        if n == 0:
            report_lines.append(f"  {ds:12s} K={k}: no paired examples with LLM-judge labels")
            continue
        res = mcnemar([[a, b], [c, d]], exact=(b + c) < 25, correction=True)
        pval = float(res.pvalue)
        supported = (b > c) and (pval < 0.05)
        rows.append({
            "dataset": ds, "k": k, "n_pairs": n,
            "both_faithful_a": a, "faithful_to_hall_b": b,
            "hall_to_faithful_c": c, "both_hall_d": d,
            "mcnemar_p": round(pval, 6), "h1_supported": supported,
        })
        report_lines.append(
            f"  {ds:12s} K={k}: n={n}  b={b} c={c}  p={pval:.4g}  "
            f"H1={'SUPPORTED' if supported else 'NOT supported'}"
        )

    pa, pb, pc, pd = pooled["a"], pooled["b"], pooled["c"], pooled["d"]
    pn = pa + pb + pc + pd
    if pn > 0:
        res = mcnemar([[pa, pb], [pc, pd]], exact=(pb + pc) < 25, correction=True)
        pval = float(res.pvalue)
        supported = (pb > pc) and (pval < 0.05)
        rows.append({
            "dataset": "POOLED", "k": "all", "n_pairs": pn,
            "both_faithful_a": pa, "faithful_to_hall_b": pb,
            "hall_to_faithful_c": pc, "both_hall_d": pd,
            "mcnemar_p": round(pval, 6), "h1_supported": supported,
        })
        report_lines.append(
            f"  {'POOLED':12s}     : n={pn}  b={pb} c={pc}  p={pval:.4g}  "
            f"H1={'SUPPORTED' if supported else 'NOT supported'}"
        )
    report_lines.append("")
    return rows


def analyze_faithfulness_comparison(data, idx, report_lines) -> List[Dict]:
    """Side-by-side HHEM / NLI / LLM-judge faithfulness per dataset/k/condition.

    HHEM hallucination rate and NLI entailment come from the summary; the LLM-judge
    faithful/correct rates are aggregated from the per-example records. This is the
    three-anchor table the paper leads with given the weak HHEM↔NLI agreement.
    """
    rows = []
    report_lines.append("Faithfulness anchors – HHEM vs NLI vs LLM-judge (per dataset/K/condition)")
    report_lines.append("     HHEM=hallucination↓  NLI=entailment↑  LLMfaith=faithful↑  LLMcorr=correct↑")
    report_lines.append("-" * 70)

    # Aggregate LLM-judge per (dataset, k, condition) from records.
    llm_agg: Dict[Tuple, Dict[str, int]] = {}
    for (ds, k), byc in idx.items():
        for cond, qmap in byc.items():
            a = llm_agg.setdefault((ds, k, cond), {"n": 0, "faithful": 0, "correct": 0})
            for r in qmap.values():
                if r.get("llm_faithful") is None:
                    continue
                a["n"] += 1
                a["faithful"] += int(bool(r.get("llm_faithful")))
                a["correct"] += int(bool(r.get("llm_correct")))

    datasets   = sorted({r["dataset"] for r in data})
    k_values   = sorted({r["k"] for r in data})
    conditions = sorted({r["condition"] for r in data})
    for ds in datasets:
        for k in k_values:
            for cond in conditions:
                srow = get_row(data, ds, k, cond)
                if srow is None:
                    continue
                a = llm_agg.get((ds, k, cond), {"n": 0, "faithful": 0, "correct": 0})
                n = a["n"]
                llm_faith = round(a["faithful"] / n, 4) if n else None
                llm_corr  = round(a["correct"] / n, 4) if n else None
                hhem = srow.get("hallucination_rate", "N/A")
                nli  = srow.get("entailment_score", "N/A")
                rows.append({
                    "dataset": ds, "k": k, "condition": cond,
                    "hhem_hallucination": hhem, "nli_entailment": nli,
                    "llm_faithful_rate": llm_faith if llm_faith is not None else "N/A",
                    "llm_correct_rate": llm_corr if llm_corr is not None else "N/A",
                    "n_llm": n,
                })
                report_lines.append(
                    f"  {ds:10s} K={k} {cond}: HHEM={hhem}  NLI={nli}  "
                    f"LLMfaith={'N/A' if llm_faith is None else f'{llm_faith:.3f}'}  "
                    f"LLMcorr={'N/A' if llm_corr is None else f'{llm_corr:.3f}'}  (n={n})"
                )
    report_lines.append("")
    return rows


# ── H2: INT4−FP16 hallucination gap vs K (monotonic slope + paired CIs) ───────
# NOTE: the intended *mechanism* test (Fix S4/H2) is a regression of mean
# per-chunk attention JS-divergence (FP16 vs INT4) on K, which requires attention
# capture in the decode loop (output_attentions). That is a documented stretch
# goal; until it lands we test the defensible outcome-level claim: the paired
# INT4−FP16 hallucination gap grows with K (slope > 0), with bootstrap CIs.

def analyze_h2_slope(idx, report_lines) -> List[Dict]:
    rows = []
    report_lines.append("H2 – Multi-Chunk Amplification (paired INT4−FP16 hall gap vs K)")
    report_lines.append("     [outcome-level; attention-JS mechanism test is a documented stretch]")
    report_lines.append("-" * 70)

    datasets = sorted({ds for (ds, _k) in idx})
    for ds in datasets:
        ks, gaps = [], []
        for (d, k), byc in sorted(idx.items()):
            if d != ds:
                continue
            c1, c3 = byc.get("C1"), byc.get("C3")
            if not c1 or not c3:
                continue
            deltas = [int(h3) - int(h1)
                      for _q, h1, h3, _n in _paired_hall(c1, c3, require_em_preserved=False)]
            if not deltas:
                continue
            mean_gap = sum(deltas) / len(deltas)
            lo, hi = bootstrap_ci_delta(deltas)
            ks.append(k)
            gaps.append(mean_gap)
            rows.append({
                "dataset": ds, "k": k, "n_pairs": len(deltas),
                "mean_gap": round(mean_gap, 4),
                "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            })
            report_lines.append(
                f"  {ds:12s} K={k}: gap={mean_gap:+.4f}  95%CI=[{lo:+.4f},{hi:+.4f}]  "
                f"(n={len(deltas)})"
            )
        if len(set(ks)) >= 2:
            reg = linregress(ks, gaps)
            supported = (reg.slope > 0) and (reg.pvalue < 0.05)
            report_lines.append(
                f"  {ds:12s}  slope(gap~K)={reg.slope:+.4f}  p={reg.pvalue:.4g}  "
                f"H2={'SUPPORTED' if supported else 'NOT supported'}"
            )
    report_lines.append("")
    return rows


# ── H3: noise-slope within RGB only (regress paired gap on retrieved noise) ───

def analyze_h3_noise_slope(idx, report_lines) -> List[Dict]:
    rows = []
    report_lines.append("H3 – Task/Noise Complexity (RGB only: INT4−FP16 hall gap vs retrieved-noise ratio)")
    report_lines.append("-" * 70)

    noises: List[float] = []
    deltas: List[float] = []
    for (ds, k), byc in sorted(idx.items()):
        if "rgb" not in ds.lower():
            continue
        c1, c3 = byc.get("C1"), byc.get("C3")
        if not c1 or not c3:
            continue
        for _q, h1, h3, noise in _paired_hall(c1, c3, require_em_preserved=False):
            if noise is None:
                continue
            noises.append(float(noise))
            deltas.append(int(h3) - int(h1))

    if len(noises) < 3 or len(set(noises)) < 2:
        report_lines.append(
            f"  H3 NOT evaluated (need ≥3 RGB paired examples with ≥2 distinct noise "
            f"ratios; got n={len(noises)}, distinct={len(set(noises))})."
        )
        report_lines.append("")
        return rows

    reg = linregress(noises, deltas)
    supported = (reg.slope > 0) and (reg.pvalue < 0.05)
    rows.append({
        "n": len(noises),
        "slope": round(float(reg.slope), 4),
        "intercept": round(float(reg.intercept), 4),
        "r": round(float(reg.rvalue), 4),
        "p": round(float(reg.pvalue), 6),
        "h3_supported": supported,
    })
    report_lines.append(
        f"  n={len(noises)}  slope={reg.slope:+.4f}  r={reg.rvalue:+.3f}  "
        f"p={reg.pvalue:.4g}  H3={'SUPPORTED' if supported else 'NOT supported'}"
    )
    report_lines.append("")
    return rows


def main():
    # Make console output robust on non-UTF8 terminals (e.g. Windows cp1252):
    # the report contains arrows/symbols and is always written to report.txt in
    # UTF-8 regardless.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_json", type=str, required=True)
    parser.add_argument("--results_jsonl", type=str, default=None,
                        help="Raw per-example results JSONL from evaluate.py. Required "
                             "for H1 (McNemar), H3 (noise slope) and paired-delta CIs.")
    parser.add_argument("--output_dir",   type=str, default="analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = load_summary(args.summary_json)

    datasets   = sorted({r["dataset"]   for r in data})
    conditions = sorted({r["condition"] for r in data})
    k_values   = sorted({r["k"]         for r in data})

    report_lines = ["=" * 70,
                    "Precomputed-KV-Cache RAG – Quantization Faithfulness Analysis",
                    "=" * 70, ""]

    # ── Per-example hypothesis tests (require the raw JSONL) ──
    if args.results_jsonl and os.path.exists(args.results_jsonl):
        records = load_records(args.results_jsonl)
        idx = index_records(records)

        h1_rows = analyze_h1_mcnemar(idx, report_lines)
        write_csv(os.path.join(args.output_dir, "h1_mcnemar.csv"),
                  ["dataset", "k", "n_pairs", "both_faithful_a", "faithful_to_hall_b",
                   "hall_to_faithful_c", "both_hall_d", "mcnemar_p", "h1_supported"],
                  h1_rows)

        h2_rows = analyze_h2_slope(idx, report_lines)
        write_csv(os.path.join(args.output_dir, "h2_amplification.csv"),
                  ["dataset", "k", "n_pairs", "mean_gap", "ci_lo", "ci_hi"],
                  h2_rows)

        h3_rows = analyze_h3_noise_slope(idx, report_lines)
        write_csv(os.path.join(args.output_dir, "h3_noise_slope.csv"),
                  ["n", "slope", "intercept", "r", "p", "h3_supported"],
                  h3_rows)

        # A3: third faithfulness anchor — side-by-side HHEM/NLI/LLM-judge plus an
        # H1 robustness check on LLM-judge faithfulness flips (only when the
        # judged JSONL carries llm_faithful labels).
        if _has_llm_judge(records):
            cmp_rows = analyze_faithfulness_comparison(data, idx, report_lines)
            write_csv(os.path.join(args.output_dir, "faithfulness_anchors.csv"),
                      ["dataset", "k", "condition", "hhem_hallucination",
                       "nli_entailment", "llm_faithful_rate", "llm_correct_rate", "n_llm"],
                      cmp_rows)

            h1_llm_rows = analyze_h1_llm_mcnemar(idx, report_lines)
            write_csv(os.path.join(args.output_dir, "h1_mcnemar_llm.csv"),
                      ["dataset", "k", "n_pairs", "both_faithful_a", "faithful_to_hall_b",
                       "hall_to_faithful_c", "both_hall_d", "mcnemar_p", "h1_supported"],
                      h1_llm_rows)
        else:
            report_lines.append(
                "NOTE: no LLM-judge labels in the results JSONL — HHEM/NLI/LLM "
                "side-by-side and the LLM-judge H1 robustness check are SKIPPED. "
                "Run the `judge` stage (needs ANTHROPIC_API_KEY) to enable them."
            )
            report_lines.append("")
    else:
        report_lines.append(
            "WARNING: --results_jsonl not provided. H1 (McNemar), H3 (noise slope) "
            "and paired-delta CIs are SKIPPED. Pass the raw results JSONL to enable "
            "the per-example tests."
        )
        report_lines.append("")

    # ── Efficiency (aggregate, from summary) ──
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
                    "dataset": ds, "k": k, "condition": cond,
                    "avg_kv_mb": round(kv_mb, 3),
                    "avg_ttft_s": row.get("avg_ttft_s", "N/A"),
                    "avg_io_s": row.get("avg_io_s", "N/A"),
                })
    write_csv(os.path.join(args.output_dir, "efficiency.csv"),
              ["dataset", "k", "condition", "avg_kv_mb", "avg_ttft_s", "avg_io_s"],
              eff_rows)

    # ── Figure CSVs (aggregate) ──
    # Figure 1: relative F1 drop vs absolute pp hallucination change by precision.
    fig1 = []
    for cond in ["C1", "C2", "C3"]:
        f1_drops, faith_drops = [], []
        for ds in datasets:
            for k in k_values:
                base = get_row(data, ds, k, "C1")
                row = get_row(data, ds, k, cond)
                if base is None or row is None:
                    continue
                base_hall = safe_float(base["hallucination_rate"])
                row_hall = safe_float(row["hallucination_rate"])
                f1_drops.append(relative_drop(safe_float(base["F1"]), safe_float(row["F1"])))
                if not (math.isnan(base_hall) or math.isnan(row_hall)):
                    faith_drops.append((row_hall - base_hall) * 100)
        if f1_drops:
            fig1.append({
                "condition": cond,
                "avg_relative_f1_drop": round(sum(f1_drops) / len(f1_drops), 2),
                "avg_hall_delta_pp": round(sum(faith_drops) / len(faith_drops), 2)
                                     if faith_drops else "N/A",
            })
    write_csv(os.path.join(args.output_dir, "figure1_data.csv"),
              ["condition", "avg_relative_f1_drop", "avg_hall_delta_pp"], fig1)

    # Figure 2: INT4–FP16 hallucination gap vs K (per dataset)
    fig2 = []
    for ds in datasets:
        for k in k_values:
            r_fp16 = get_row(data, ds, k, "C1")
            r_int4 = get_row(data, ds, k, "C3")
            if r_fp16 and r_int4:
                gap = safe_float(r_int4["hallucination_rate"]) - safe_float(r_fp16["hallucination_rate"])
                if not math.isnan(gap):
                    fig2.append({"dataset": ds, "k": k, "delta_hall": round(gap, 4)})
    write_csv(os.path.join(args.output_dir, "figure2_data.csv"),
              ["dataset", "k", "delta_hall"], fig2)

    # Figure 3: storage size vs hallucination rate
    fig3 = []
    for ds in datasets:
        for k in k_values:
            for cond in ["C1", "C2", "C3"]:
                row = get_row(data, ds, k, cond)
                if row:
                    fig3.append({
                        "dataset": ds, "k": k, "condition": cond,
                        "kv_mb": round(row["avg_kv_bytes"] / 1e6, 3),
                        "hall_rate": safe_float(row["hallucination_rate"]),
                    })
    write_csv(os.path.join(args.output_dir, "figure3_data.csv"),
              ["dataset", "k", "condition", "kv_mb", "hall_rate"], fig3)

    # ── Write report ──
    report_path = os.path.join(args.output_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print("\n".join(report_lines))
    print(f"\nReport → {report_path}")
    print(f"Outputs → {args.output_dir}/")


if __name__ == "__main__":
    main()
