"""
metrics.py  –  Evaluation metrics for the TurboRAG quantization study.

Implements:
  - Exact Match  (EM)
  - Token-level F1
  - HHEM-2.1-Open hallucination scoring  (primary faithfulness signal)
  - DeBERTa-large-v3-NLI entailment scoring  (secondary validation signal)

All metrics follow the methodology described in the refined research idea
(Stages 6, 7, and 8).
"""

from __future__ import annotations
import re
import string
from collections import Counter
from typing import List, Optional, Tuple

import torch


# ──────────────────────────────────────────────────────────────────────────────
# Text normalisation (shared by EM + F1)
# ──────────────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lower-case, strip punctuation and articles, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


# ──────────────────────────────────────────────────────────────────────────────
# Exact Match
# ──────────────────────────────────────────────────────────────────────────────

def exact_match(prediction: str, ground_truth: str) -> int:
    return int(_normalize(prediction) == _normalize(ground_truth))


def batch_em(predictions: List[str], ground_truths: List[str]) -> float:
    assert len(predictions) == len(ground_truths)
    if not predictions:
        return 0.0
    return sum(exact_match(p, g) for p, g in zip(predictions, ground_truths)) / len(predictions)


# ──────────────────────────────────────────────────────────────────────────────
# Token-level F1
# ──────────────────────────────────────────────────────────────────────────────

def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens  = _normalize(prediction).split()
    gold_tokens  = _normalize(ground_truth).split()
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


# ──────────────────────────────────────────────────────────────────────────────
# HHEM-2.1-Open hallucination scorer
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# DeBERTa-v3-large NLI scorer
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ──────────────────────────────────────────────────────────────────────────────

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
