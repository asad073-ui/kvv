
from __future__ import annotations
import re
import string
from collections import Counter
from typing import List, Optional, Tuple

import torch



# Text normalisation (shared by EM + F1)


def _normalize(text: str) -> str:
    """Lower-case, strip punctuation and articles, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())



# Exact Match


def exact_match(prediction: str, ground_truth: str) -> int:
    return int(_normalize(prediction) == _normalize(ground_truth))


def batch_em(predictions: List[str], ground_truths: List[str]) -> float:
    assert len(predictions) == len(ground_truths)
    if not predictions:
        return 0.0
    return sum(exact_match(p, g) for p, g in zip(predictions, ground_truths)) / len(predictions)



# Token-level F1


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens  = _normalize(prediction).split()
    gold_tokens  = _normalize(ground_truth).split()
    # FIX-F1DIV: guard against empty token lists after normalisation
    if not pred_tokens or not gold_tokens:
        return 0.0
    common       = Counter(pred_tokens) & Counter(gold_tokens)
    num_same     = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def batch_f1(predictions: List[str], ground_truths: List[str]) -> float:
    assert len(predictions) == len(ground_truths)
    if not predictions:
        return 0.0
    return sum(token_f1(p, g) for p, g in zip(predictions, ground_truths)) / len(predictions)



# Long-context windowing (A1: stop silently truncating the NLI/HHEM context)
#
# HHEM-2.1-Open (flan-t5-base) and DeBERTa-NLI both have a ~512-token window.  At
# K=3/K=5 the concatenated context overflows it, so the scorer silently sees only
# the head (the "522 > 512" warning).  Instead of dropping the tail we split the
# context into ≤512-token windows, score the answer against each, and take the
# MOST-SUPPORTING (max) window — the standard long-input entailment workaround.


def _char_windows(text: str, window_chars: int, stride_chars: int) -> List[str]:
    """Split text into overlapping char windows (≈4 chars/token heuristic)."""
    if not text:
        return [""]
    if len(text) <= window_chars:
        return [text]
    windows: List[str] = []
    start = 0
    while start < len(text):
        windows.append(text[start:start + window_chars])
        if start + window_chars >= len(text):
            break
        start += stride_chars
    return windows


def estimate_token_len(text: str, tokenizer=None) -> int:
    """Token length of text; uses tokenizer when given, else a ~4 chars/token heuristic."""
    if tokenizer is not None:
        return len(tokenizer.encode(text or "", add_special_tokens=False))
    return len(text or "") // 4


def count_contexts_over_limit(contexts: List[str], tokenizer=None, limit_tokens: int = 512) -> int:
    """Count contexts whose token length exceeds limit_tokens (truncation exposure)."""
    return sum(1 for c in contexts if estimate_token_len(c, tokenizer) > limit_tokens)


# HHEM-2.1-Open hallucination scorer


class HHEMScorer:
    """
    Wrapper around Vectara's HHEM-2.1-Open model.
    https://huggingface.co/vectara/hallucination_evaluation_model

    HHEM-2.1-Open is T5-based. AutoTokenizer fails because the model uses a
    custom HHEMv2Config that is not in the transformers registry. The model
    exposes a custom predict(pairs) method that handles tokenisation internally
    using a bundled prompt template and the flan-t5-base tokenizer — so we
    never need to load a tokenizer ourselves.

    Output: float in [0, 1] where 1 = fully faithful, 0 = hallucinated.
    """

    MODEL_ID = "vectara/hallucination_evaluation_model"

    def __init__(self, device: Optional[torch.device] = None):
        from transformers import AutoModelForSequenceClassification
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_ID,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

    def score(self, context: str, answer: str) -> float:
        """Returns faithfulness probability (1 = faithful, 0 = hallucinated)."""
        scores = self.model.predict([(context, answer)])
        return float(scores[0])

    def batch_score(
        self,
        contexts: List[str],
        answers:  List[str],
        batch_size: int = 8,
    ) -> List[float]:
        pairs = list(zip(contexts, answers))
        results = []
        for i in range(0, len(pairs), batch_size):
            scores = self.model.predict(pairs[i:i + batch_size])
            results.extend(scores.tolist())
        return results

    def batch_score_windowed(
        self,
        contexts: List[str],
        answers:  List[str],
        batch_size: int = 8,
        window_chars: int = 1600,   # ≈400 tokens of context, leaving room for the answer
        stride_chars: int = 1200,
    ) -> List[float]:
        """Max faithfulness over ≤512-token windows of each context (A1).

        Avoids the silent tail-drop when a K=3/K=5 context overflows the flan-t5
        512-token window: we score the answer against each overlapping window of
        the context and keep the most-supporting (max) score.
        """
        results: List[float] = []
        for ctx, ans in zip(contexts, answers):
            windows = _char_windows(ctx, window_chars, stride_chars)
            pairs = [(w, ans) for w in windows]
            per_window: List[float] = []
            for i in range(0, len(pairs), batch_size):
                scores = self.model.predict(pairs[i:i + batch_size])
                per_window.extend(scores.tolist())
            results.append(max(per_window) if per_window else 0.0)
        return results



# DeBERTa-v3-large NLI scorer


class DeBERTaNLIScorer:
    """
    Wrapper around cross-encoder/nli-deberta-v3-large.
    https://huggingface.co/cross-encoder/nli-deberta-v3-large

    Returns (entailment_prob, neutral_prob, contradiction_prob).
    Primary signal: entailment_prob (context entails answer).
    """

    MODEL_ID = "cross-encoder/nli-deberta-v3-large"

    def __init__(self, device: Optional[torch.device] = None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer as AT
        self.device    = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AT.from_pretrained(self.MODEL_ID)
        self.model     = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.float32,
        ).to(self.device)
        self.model.eval()
        # Label mapping: cross-encoder/nli-deberta-v3-large uses
        # id2label = {0: 'contradiction', 1: 'entailment', 2: 'neutral'}
        self._label_names = self.model.config.id2label  # may vary by ckpt

    @torch.no_grad()
    def score(self, context: str, answer: str) -> Tuple[float, float, float]:
        """
        Returns (entailment, neutral, contradiction) probabilities.
        """
        inputs  = self.tokenizer(
            context, answer,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        logits  = self.model(**inputs).logits
        probs   = torch.softmax(logits, dim=-1)[0].tolist()
        # Map to (entailment, neutral, contradiction) regardless of label order
        label_map = {v.lower(): i for i, v in self._label_names.items()}
        e = probs[label_map.get("entailment", 1)]
        n = probs[label_map.get("neutral",    2)]
        c = probs[label_map.get("contradiction", 0)]
        return e, n, c

    @torch.no_grad()
    def batch_score(
        self,
        contexts:  List[str],
        answers:   List[str],
        batch_size: int = 8,
    ) -> List[Tuple[float, float, float]]:
        results = []
        for i in range(0, len(contexts), batch_size):
            ctxs = contexts[i:i+batch_size]
            ans  = answers[i:i+batch_size]
            inputs = self.tokenizer(
                ctxs, ans,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)
            logits = self.model(**inputs).logits
            probs  = torch.softmax(logits, dim=-1)
            label_map = {v.lower(): j for j, v in self._label_names.items()}
            e_idx = label_map.get("entailment", 1)
            n_idx = label_map.get("neutral",    2)
            c_idx = label_map.get("contradiction", 0)
            for row in probs.tolist():
                results.append((row[e_idx], row[n_idx], row[c_idx]))
        return results

    @torch.no_grad()
    def batch_score_windowed(
        self,
        contexts:  List[str],
        answers:   List[str],
        batch_size: int = 8,
        window_tokens: int = 430,   # leaves ~80 tokens of the 512 budget for the answer
        stride_tokens: int = 320,
    ) -> List[Tuple[float, float, float]]:
        """Max-entailment over ≤512-token windows of each context (A1).

        Tokenises the context with the model's own tokenizer, splits it into
        overlapping ≤window_tokens windows, scores the answer against each, and
        returns the (entailment, neutral, contradiction) triple from the window
        with the highest entailment.  Removes the silent K-dependent tail-drop.
        """
        results: List[Tuple[float, float, float]] = []
        for ctx, ans in zip(contexts, answers):
            ctx_ids = self.tokenizer.encode(ctx or "", add_special_tokens=False)
            if len(ctx_ids) <= window_tokens:
                windows = [ctx or ""]
            else:
                windows = []
                start = 0
                while start < len(ctx_ids):
                    win_ids = ctx_ids[start:start + window_tokens]
                    windows.append(self.tokenizer.decode(win_ids, skip_special_tokens=True))
                    if start + window_tokens >= len(ctx_ids):
                        break
                    start += stride_tokens
            triples = self.batch_score(windows, [ans] * len(windows), batch_size)
            best = max(triples, key=lambda t: t[0]) if triples else (0.0, 0.0, 1.0)
            results.append(best)
        return results



# Aggregation helpers


def hallucination_rate(faithfulness_scores: List[float], threshold: float = 0.5) -> float:
    """Fraction of samples where faithfulness < threshold (= hallucinated)."""
    if not faithfulness_scores:
        return 0.0
    return sum(s < threshold for s in faithfulness_scores) / len(faithfulness_scores)


def mean_entailment(nli_scores: List[Tuple[float, float, float]]) -> float:
    """Average entailment probability across all examples."""
    if not nli_scores:
        return 0.0
    return sum(e for e, _, _ in nli_scores) / len(nli_scores)


def hallucination_rate_per_chunk(
    hhem_scorer,
    chunk_texts_list: List[List[str]],
    answers: List[str],
    batch_size: int = 8,
    threshold: float = 0.5,
) -> Tuple[List[float], List[bool]]:
    
    faith_scores = []
    for chunks, answer in zip(chunk_texts_list, answers):
        if not chunks:
            faith_scores.append(0.0)
            continue
        trimmed_chunks = [c[:1600] for c in chunks]
        pairs = [(c, answer) for c in trimmed_chunks]
        per_chunk: List[float] = []
        for i in range(0, len(pairs), batch_size):
            scores = hhem_scorer.model.predict(pairs[i:i + batch_size])
            per_chunk.extend(scores.tolist())
        faith_scores.append(max(per_chunk))
    hallucinated = [s < threshold for s in faith_scores]
    return faith_scores, hallucinated


def entailment_per_chunk(
    nli_scorer,
    chunk_texts_list: List[List[str]],
    answers: List[str],
    batch_size: int = 8,
) -> List[Tuple[float, float, float]]:
    """Max-entailment across chunks per example.

    Scores each (chunk, answer) pair with the NLI model and returns the
    (entailment, neutral, contradiction) triple that has the highest entailment
    probability. Avoids the K-dependent truncation artifact.
    """
    results = []
    for chunks, answer in zip(chunk_texts_list, answers):
        if not chunks:
            results.append((0.0, 0.0, 1.0))
            continue
        trimmed = [c[:800] for c in chunks]
        per_chunk = nli_scorer.batch_score(trimmed, [answer] * len(trimmed), batch_size)
        best = max(per_chunk, key=lambda t: t[0])
        results.append(best)
    return results
