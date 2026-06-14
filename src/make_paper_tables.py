from __future__ import annotations
import os, sys, json, csv, argparse

CONDITION_ORDER = {"C0": 0, "C1": 1, "C2": 2, "C3": 3}
CONDITION_LABEL = {
    "C0": "Gold Oracle", "C1": "FP16 TurboRAG",
    "C2": "INT8 TurboRAG", "C3": "INT4 TurboRAG",
}

# Master columns (key in summary row, header in the paper table, fmt).
COLUMNS = [
    ("dataset",            "Dataset",        "s"),
    ("k",                  "K",              "d"),
    ("condition",          "Condition",      "s"),
    ("EM",                 "EM",             ".4f"),
    ("F1",                 "F1",             ".4f"),
    ("hallucination_rate", "Hallucination",  ".4f"),
    ("entailment_score",   "Entailment",     ".4f"),
    ("kv_mb",              "KV Size (MB)",   ".3f"),
    ("avg_ttft_s",         "TTFT (s)",       ".4f"),
    ("avg_latency_s",      "Latency (s)",    ".4f"),
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


def load_rows(summary_json):
    with open(summary_json, encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for r in data:
        row = dict(r)
        row["kv_mb"] = (r.get("avg_kv_bytes", 0) or 0) / 1e6
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


def write_markdown(path, rows, run_cfg):
    headers = [h for _, h, _ in COLUMNS]
    lines = []
    if run_cfg:
        lines.append(f"# TurboRAG KV-Quantization — Main Results")
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


def write_latex(path, rows):
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
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Offline KV-cache quantization in TurboRAG: factual accuracy "
        r"(EM/F1) vs.\ faithfulness (hallucination/entailment) and efficiency "
        r"(KV size, TTFT, latency).}",
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
    rows = load_rows(args.summary_json)

    run_cfg = None
    if args.config_json and os.path.exists(args.config_json):
        with open(args.config_json, encoding="utf-8") as f:
            run_cfg = json.load(f)

    csv_path = os.path.join(args.output_dir, "paper_table.csv")
    md_path  = os.path.join(args.output_dir, "paper_table.md")
    tex_path = os.path.join(args.output_dir, "paper_table.tex")
    write_master_csv(csv_path, rows)
    write_markdown(md_path, rows, run_cfg)
    write_latex(tex_path, rows)

    print(f"[make_paper_tables] {len(rows)} rows")
    print(f"[make_paper_tables] Master CSV -> {csv_path}")
    print(f"[make_paper_tables] Markdown   -> {md_path}")
    print(f"[make_paper_tables] LaTeX      -> {tex_path}")


if __name__ == "__main__":
    main()
