"""Optional LLM-as-judge scorer (Fix 4).

Post-hoc pass over a results_*.jsonl from evaluate.py. For each non-refusal
record it asks a Claude model two binary questions:

  * correct  – does the prediction answer the question (vs. the gold aliases)?
  * faithful – is the prediction supported by the retrieved context?

This is deliberately a SEPARATE, optional stage: it does not run inside the GPU
evaluation loop, so it can never destabilise the numerical-validity path, and it
is skipped entirely unless invoked. Token-F1 stays a secondary metric; this judge
and containment-EM are the accuracy signals to lead with in the paper.

Usage:
    export ANTHROPIC_API_KEY=...
    python src/llm_judge.py --results_jsonl results/.../results_XXXX.jsonl \
        --output_dir results/.../llm_judge [--model claude-haiku-4-5-20251001]

Outputs:
    <output_dir>/results_judged.jsonl   (input records + llm_correct/llm_faithful)
    <output_dir>/llm_judge_summary.csv  (per dataset/k/condition rates)
"""
from __future__ import annotations
import os, sys, json, csv, argparse, time, re, string

# Strong-but-cheap Claude judge by default; override with --model.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


# ── containment-EM (mirrors evaluate.batch_contain_em / analyze.em_correct) ───
# Used to anchor the LLM judge: we report how often the judge's `correct` label
# agrees with containment-EM on a sample, so the judge itself is validated rather
# than trusted blindly.

def _normalize(text: str) -> str:
    text = (text or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _contain_em(prediction: str, golds) -> bool:
    if isinstance(golds, str):
        golds = [golds]
    npred = _normalize(prediction)
    return any(_normalize(g) in npred for g in (golds or []) if g)

SYSTEM = (
    "You are a strict evaluation judge for a retrieval-augmented QA system. "
    "You will be given a question, the gold answer(s), the retrieved context, and "
    "a model's prediction. Reply with ONLY a compact JSON object: "
    '{"correct": true|false, "faithful": true|false}. '
    "correct = the prediction conveys the same answer as ANY gold answer "
    "(ignore phrasing/formatting). faithful = every factual claim in the "
    "prediction is supported by the retrieved context."
)


# Context cap (chars). The judge only needs enough context to verify a short
# answer; 2000 chars (~500 tokens) keeps per-call cost low without starving the
# faithfulness check. Lower it further to cut cost; raise it if answers cite long
# passages.
CONTEXT_CHAR_CAP = 2000

# Approximate Anthropic pricing ($ per 1M tokens) for the cost estimate. Keyed by
# substring of the model id; falls back to Haiku rates.
_PRICES = {
    "haiku":  (1.0, 5.0),
    "sonnet": (3.0, 15.0),
    "opus":   (5.0, 25.0),
}


def _rates_for(model: str):
    for key, rates in _PRICES.items():
        if key in (model or "").lower():
            return rates
    return _PRICES["haiku"]


def _build_user_prompt(rec: dict) -> str:
    gold = rec.get("gold", "")
    if isinstance(gold, list):
        gold = " | ".join(str(g) for g in gold)
    context = (rec.get("context") or "")[:CONTEXT_CHAR_CAP]
    return (
        f"Question: {rec.get('query','')}\n\n"
        f"Gold answer(s): {gold}\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"Model prediction: {rec.get('prediction','')}\n\n"
        'Respond with ONLY the JSON object: {"correct": ..., "faithful": ...}'
    )


def _judge_one(client, model: str, rec: dict, max_retries: int = 3):
    prompt = _build_user_prompt(rec)
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=64,
                system=SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
            start, end = text.find("{"), text.rfind("}")
            obj = json.loads(text[start:end + 1])
            return bool(obj.get("correct")), bool(obj.get("faithful"))
        except Exception as e:                       # noqa: BLE001
            if attempt == max_retries - 1:
                print(f"[llm_judge] giving up on a record: {e}", file=sys.stderr)
                return None, None
            time.sleep(2 ** attempt)
    return None, None


def main():
    ap = argparse.ArgumentParser(description="Optional Claude LLM-as-judge scorer")
    ap.add_argument("--results_jsonl", required=True)
    ap.add_argument("--output_dir", default="llm_judge")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max_records", type=int, default=-1,
                    help="Global cap on records scored (cost control); -1 = no cap.")
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="Only score these conditions (e.g. C1 C3). Default: all. "
                         "Focusing on the H1 pair (C1, C3) roughly halves cost.")
    ap.add_argument("--max_per_group", type=int, default=0,
                    help="Cap records scored per (dataset, k, condition) for a "
                         "balanced sample; 0 = no per-group cap.")
    ap.add_argument("--skip_refusals", action="store_true", default=True)
    ap.add_argument("--validate_n", type=int, default=50,
                    help="Anchor the judge: report judge↔containment-EM agreement on "
                         "the first N scored records (0 disables).")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print the cost estimate and exit without calling the API.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    records = []
    with open(args.results_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # ── Decide which records to score (cost control) ──────────────────────────
    # Skip refusals, restrict to the requested conditions, take a balanced sample
    # per (dataset, k, condition), and honour the global cap. This is a budget
    # gate: every selected record is one paid API call.
    cond_filter = set(args.conditions) if args.conditions else None
    group_counts: dict = {}
    to_score = set()          # indices into `records`
    for i, rec in enumerate(records):
        if args.skip_refusals and rec.get("is_refusal"):
            continue
        if cond_filter is not None and rec.get("condition") not in cond_filter:
            continue
        key = (rec.get("dataset"), rec.get("k"), rec.get("condition"))
        if args.max_per_group and group_counts.get(key, 0) >= args.max_per_group:
            continue
        if args.max_records >= 0 and len(to_score) >= args.max_records:
            break
        group_counts[key] = group_counts.get(key, 0) + 1
        to_score.add(i)

    # ── Cost estimate (printed before spending a cent) ────────────────────────
    in_rate, out_rate = _rates_for(args.model)
    est_in_tokens = 0
    SYSTEM_TOKENS = len(SYSTEM) // 4
    for i in to_score:
        est_in_tokens += SYSTEM_TOKENS + len(_build_user_prompt(records[i])) // 4
    est_out_tokens = len(to_score) * 30        # compact JSON reply
    est_cost = est_in_tokens / 1e6 * in_rate + est_out_tokens / 1e6 * out_rate
    print(f"[llm_judge] model={args.model}  records_to_score={len(to_score)}  "
          f"conditions={sorted(cond_filter) if cond_filter else 'all'}  "
          f"max_per_group={args.max_per_group or 'none'}")
    print(f"[llm_judge] cost estimate ~ ${est_cost:.2f} "
          f"(~{est_in_tokens/1e3:.0f}K in @ ${in_rate}/1M, "
          f"~{est_out_tokens/1e3:.0f}K out @ ${out_rate}/1M)")

    if args.dry_run:
        print("[llm_judge] --dry_run set; not calling the API.")
        return

    try:
        import anthropic
    except ImportError:
        sys.exit("[llm_judge] `pip install anthropic` to use the LLM judge.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("[llm_judge] set ANTHROPIC_API_KEY to use the LLM judge.")
    client = anthropic.Anthropic()

    judged = []
    n_scored = 0
    for i, rec in enumerate(records):
        if i not in to_score:
            judged.append(rec)
            continue
        correct, faithful = _judge_one(client, args.model, rec)
        rec = dict(rec)
        rec["llm_correct"] = correct
        rec["llm_faithful"] = faithful
        judged.append(rec)
        n_scored += 1

    out_jsonl = os.path.join(args.output_dir, "results_judged.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in judged:
            f.write(json.dumps(rec) + "\n")

    # Aggregate rates per (dataset, k, condition).
    agg = {}
    for rec in judged:
        if rec.get("llm_correct") is None:
            continue
        key = (rec["dataset"], rec["k"], rec["condition"])
        a = agg.setdefault(key, {"n": 0, "correct": 0, "faithful": 0})
        a["n"] += 1
        a["correct"] += int(bool(rec["llm_correct"]))
        a["faithful"] += int(bool(rec["llm_faithful"]))

    summary_csv = os.path.join(args.output_dir, "llm_judge_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "k", "condition", "n", "llm_correct_rate", "llm_faithful_rate"])
        for (ds, k, cond), a in sorted(agg.items()):
            n = max(a["n"], 1)
            w.writerow([ds, k, cond, a["n"],
                        round(a["correct"] / n, 4), round(a["faithful"] / n, 4)])

    # ── Validate the judge against containment-EM (A3 anchor) ──
    # The judge's `correct` label is only trustworthy if it tracks an objective
    # signal. We report agreement with containment-EM on a sample so the paper can
    # cite the judge as an anchored third metric rather than an unvetted oracle.
    if args.validate_n and args.validate_n > 0:
        sample = [r for r in judged if r.get("llm_correct") is not None][:args.validate_n]
        n_val = len(sample)
        n_agree = sum(
            1 for r in sample
            if bool(r["llm_correct"]) == _contain_em(r.get("prediction", ""), r.get("gold", []))
        )
        agreement = (n_agree / n_val) if n_val else float("nan")
        validation = {
            "n_validated":        n_val,
            "n_agree":            n_agree,
            "agreement":          round(agreement, 4) if n_val else None,
            "metric":             "llm_correct_vs_containment_EM",
            "model":              args.model,
        }
        val_path = os.path.join(args.output_dir, "judge_validation.json")
        with open(val_path, "w", encoding="utf-8") as f:
            json.dump(validation, f, indent=2)
        print(f"[llm_judge] judge↔containment-EM agreement on {n_val} examples: "
              f"{'N/A' if not n_val else f'{agreement:.3f}'}")
        print(f"[llm_judge] validation JSON → {val_path}")

    print(f"[llm_judge] scored {n_scored} records with {args.model}")
    print(f"[llm_judge] judged JSONL → {out_jsonl}")
    print(f"[llm_judge] summary CSV  → {summary_csv}")


if __name__ == "__main__":
    main()
