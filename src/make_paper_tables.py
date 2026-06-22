from __future__ import annotations
import os, sys, json, csv, argparse

CONDITION_ORDER = {"C0": 0, "C1": 1, "C2": 2, "C3": 3}

# A2: fallback labels per precompute_mode. The PRIMARY source of truth is the
# per-row `condition_label` written by evaluate.py (already mode-aware); these
# maps only fill in when an older summary lacks that field.
CONDITION_LABEL_STANDARD = {
    "C0": "Full-Context Oracle", "C1": "FP16 Precomputed-KV",
    "C2": "INT8 Precomputed-KV", "C3": "INT4 Precomputed-KV",
}
CONDITION_LABEL_REORDERED = {
    "C0": "Gold Oracle RAG", "C1": "FP16 TurboRAG",
    "C2": "INT8 TurboRAG", "C3": "INT4 TurboRAG",
}


def _condition_label_map(precompute_mode: str) -> dict:
    return CONDITION_LABEL_REORDERED if precompute_mode == "reordered" else CONDITION_LABEL_STANDARD


def _table_title(precompute_mode: str) -> str:
    if precompute_mode == "reordered":
        return "TurboRAG KV-Quantization — Faithfulness"
    return "Precomputed-KV-Cache RAG — Quantization Faithfulness"

# Master columns (key in summary row, header in the paper table, fmt).
# Fix 4: containment-EM is the PRIMARY accuracy metric; token-F1 is reported as a
# secondary signal only (strict token-F1 as primary is what produced the F1≈0.03
# artifact). Degeneracy + non-finite counts are surfaced so a reviewer can see at
# a glance whether a run is numerically valid.
COLUMNS = [
    ("dataset",            "Dataset",          "s"),
    ("k",                  "K",                "d"),
    ("condition_display",  "Condition",        "s"),
    ("contain_EM",         "ContainEM (prim.)", ".4f"),
    ("EM",                 "EM",               ".4f"),
    ("F1",                 "F1 (secondary)",   ".4f"),
    ("hallucination_rate", "Hallucination",    ".4f"),
    ("entailment_score",   "Entailment",       ".4f"),
    ("n_nonfinite",        "NonFinite",        "d"),
    ("degenerate_rate",    "Degen.",           ".4f"),
    ("kv_mb",              "KV Size (MB)",     ".3f"),
    ("avg_ttft_s",         "TTFT (s)",         ".4f"),
    ("avg_latency_s",      "Latency (s)",      ".4f"),
]


def _fmt(val, fmt):
    if val is None or val == "" or (isinstance(val, str) and val.upper() == "N/A"):
        return "N/A"
    try:
        if fmt == "s":
            return str(val)
        if fmt == "d":
            return str(int(val))
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return str(val)


def _infer_precompute_mode(data, run_cfg):
    """Single source of truth for the run's precompute mode.

    Prefer the run config, then the per-row field written by evaluate.py, else
    default to standard_causal.
    """
    if run_cfg and run_cfg.get("precompute_mode"):
        return run_cfg["precompute_mode"]
    for r in data:
        if r.get("precompute_mode"):
            return r["precompute_mode"]
    return "standard_causal"


def load_rows(summary_json, precompute_mode="standard_causal"):
    with open(summary_json, encoding="utf-8") as f:
        data = json.load(f)
    label_map = _condition_label_map(precompute_mode)
    rows = []
    for r in data:
        row = dict(r)
        row["kv_mb"] = (r.get("avg_kv_bytes", 0) or 0) / 1e6
        # A2: friendly, mode-correct label. Prefer evaluate.py's per-row label;
        # fall back to the mode map, then the bare condition code.
        row["condition_display"] = (
            r.get("condition_label")
            or label_map.get(r.get("condition"))
            or r.get("condition")
        )
        rows.append(row)
    rows.sort(key=lambda r: (str(r.get("dataset")), int(r.get("k", 0)),
                             CONDITION_ORDER.get(r.get("condition"), 99)))
    return rows


def write_master_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([h for _, h, _ in COLUMNS] + ["wiki_pages", "n_examples", "paired_n"])
        for r in rows:
            line = [_fmt(r.get(k), fmt) for k, _, fmt in COLUMNS]
            line += [r.get("wiki_pages", ""), r.get("n_examples", ""), r.get("paired_n", "")]
            w.writerow(line)


def write_markdown(path, rows, run_cfg, precompute_mode="standard_causal"):
    headers = [h for _, h, _ in COLUMNS]
    lines = []
    if run_cfg:
        lines.append(f"# {_table_title(precompute_mode)} — Main Results")
        lines.append("")
        lines.append(
            f"_model_: `{run_cfg.get('model','?')}` · "
            f"_wiki_pages_: {run_cfg.get('wiki_pages','?')} · "
            f"_datasets_: {', '.join(run_cfg.get('datasets', []))} · "
            f"_K_: {run_cfg.get('k_values')} · "
            f"_conditions_: {run_cfg.get('conditions')}"
        )
        lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        cells = [_fmt(r.get(k), fmt) for k, _, fmt in COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_latex(path, rows, precompute_mode="standard_causal"):
    headers = [h for _, h, _ in COLUMNS]
    col_spec = "ll" + "r" * (len(headers) - 2)
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\begin{tabular}{" + col_spec + "}", r"\toprule",
        " & ".join(headers) + r" \\", r"\midrule",
    ]
    prev_ds = None
    for r in rows:
        if prev_ds is not None and r.get("dataset") != prev_ds:
            lines.append(r"\midrule")
        prev_ds = r.get("dataset")
        cells = [_fmt(r.get(k), fmt) for k, _, fmt in COLUMNS]
        cells = [c.replace("_", r"\_") for c in cells]
        lines.append(" & ".join(cells) + r" \\")
    scheme = ("TurboRAG (reordered independent-attention cache)"
              if precompute_mode == "reordered"
              else "precomputed-KV-cache RAG (standard causal prefill)")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Offline KV-cache quantization in " + scheme + r": factual "
        r"accuracy (EM/F1) vs.\ faithfulness (hallucination/entailment) and "
        r"efficiency (KV size, TTFT, latency).}",
        r"\label{tab:main}", r"\end{table}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Build publication tables from summary JSON")
    parser.add_argument("--summary_json", type=str, required=True)
    parser.add_argument("--output_dir",   type=str, default="analysis")
    parser.add_argument("--config_json",  type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    run_cfg = None
    if args.config_json and os.path.exists(args.config_json):
        with open(args.config_json, encoding="utf-8") as f:
            run_cfg = json.load(f)

    # A2: derive the run's precompute mode once and let it drive both the per-row
    # condition labels and the table title/caption.
    with open(args.summary_json, encoding="utf-8") as f:
        precompute_mode = _infer_precompute_mode(json.load(f), run_cfg)
    rows = load_rows(args.summary_json, precompute_mode=precompute_mode)

    csv_path = os.path.join(args.output_dir, "paper_table.csv")
    md_path  = os.path.join(args.output_dir, "paper_table.md")
    tex_path = os.path.join(args.output_dir, "paper_table.tex")
    write_master_csv(csv_path, rows)
    write_markdown(md_path, rows, run_cfg, precompute_mode=precompute_mode)
    write_latex(tex_path, rows, precompute_mode=precompute_mode)
    print(f"[make_paper_tables] precompute_mode = {precompute_mode}")

    print(f"[make_paper_tables] {len(rows)} rows")
    print(f"[make_paper_tables] Master CSV -> {csv_path}")
    print(f"[make_paper_tables] Markdown   -> {md_path}")
    print(f"[make_paper_tables] LaTeX      -> {tex_path}")


if __name__ == "__main__":
    main()
