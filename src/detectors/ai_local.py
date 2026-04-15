"""Local AI-generated text detector.

Honest design notes:

* We initially planned to use a multilingual classifier (`yaya36095/xlm-roberta-text-detector`)
  but probing it on real samples showed it returns ~100% AI for every input — useless.
  No similarly-sized open-source detector calibrated for Russian text seems to exist at
  the time of writing.
* Therefore the local layer is built on **statistical signals** that work in any language
  with a small causal LM:
    1. **Perplexity** of the text under a small reference LM (ruGPT3-small for ru,
       distilgpt2 for en). LLM-generated prose is usually more predictable, i.e. lower
       PPL relative to a baseline. Strong signal but not absolute.
    2. **Burstiness** — std-dev of sentence lengths in tokens. Human writing is bursty
       (short, long, short); LLM writing tends to be uniform.
    3. **Per-sentence breakdown** — we score each sentence under the same LM so the
       MCP tool can highlight the most "predictable" passages.
* We blend these into a single `ai_probability` with a transparent heuristic and surface
  every raw signal so callers can override our judgement.
* For high-stakes use the client should pass a Sapling key in headers and request
  `mode='deep'`, which cross-checks via a properly trained classifier.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev

import torch
from razdel import sentenize as razdel_sentenize
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import settings
from ..language import Language

# Empirically chosen baselines from probing the reference LMs on a handful of
# clearly-human and clearly-AI samples. These are rules of thumb, not learned
# thresholds — see PLAN.md "Stage 5".
_BASELINES = {
    "ru": {"ppl_low": 10.0, "ppl_high": 30.0, "model_dir": "ai-forever__rugpt3small_based_on_gpt2"},
    "en": {"ppl_low": 15.0, "ppl_high": 45.0, "model_dir": "distilgpt2"},
}
_BURSTINESS_LOW = 3.0  # words; below this we consider the rhythm "flat"
_MAX_TOKENS = 510  # leaves room for special tokens in 512-cap models
_MAX_SENTENCES_FOR_BREAKDOWN = 60  # cap CPU cost per request


@dataclass(slots=True)
class SuspiciousSentence:
    text: str
    perplexity: float
    char_start: int
    char_end: int


@dataclass(slots=True)
class AIAnalysis:
    language: Language
    ai_probability: float
    confidence: str  # "low" | "medium" | "high"
    perplexity: float | None
    perplexity_baseline: tuple[float, float] | None
    burstiness: float
    sentence_count: int
    avg_sentence_length: float
    suspicious_sentences: list[SuspiciousSentence] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # Populated only in mode='deep' when an external provider replied. Each
    # entry: {provider, ai_probability, error}. The top-level `ai_probability`
    # is overridden by the deep provider when one is available.
    external_sources: list[dict] = field(default_factory=list)
    local_heuristic_probability: float | None = None


class LocalAIDetector:
    """Lazy-loaded, language-routed perplexity + burstiness detector."""

    def __init__(self, models_dir: Path | None = None) -> None:
        self.models_dir = models_dir or settings.models_dir
        self._lms: dict[str, tuple[AutoTokenizer, AutoModelForCausalLM]] = {}

    def _load_lm(self, language: Language) -> tuple[AutoTokenizer, AutoModelForCausalLM] | None:
        key = language if language in _BASELINES else "en"
        if key in self._lms:
            return self._lms[key]
        path = self.models_dir / _BASELINES[key]["model_dir"]
        if not path.exists():
            return None
        tok = AutoTokenizer.from_pretrained(path)
        m = AutoModelForCausalLM.from_pretrained(path)
        m.eval()
        self._lms[key] = (tok, m)
        return self._lms[key]

    async def analyze(self, text: str, language: Language) -> AIAnalysis:
        return await asyncio.to_thread(self._analyze_sync, text, language)

    def _analyze_sync(self, text: str, language: Language) -> AIAnalysis:
        sentences = _split_sentences(text, language)
        sent_lens = [len(s["text"].split()) for s in sentences if s["text"].strip()]
        burstiness = stdev(sent_lens) if len(sent_lens) >= 2 else 0.0
        avg_len = mean(sent_lens) if sent_lens else 0.0
        notes: list[str] = []

        lm = self._load_lm(language)
        if lm is None:
            notes.append(f"no perplexity model available for language={language}")
            return AIAnalysis(
                language=language,
                ai_probability=0.5,
                confidence="low",
                perplexity=None,
                perplexity_baseline=None,
                burstiness=burstiness,
                sentence_count=len(sent_lens),
                avg_sentence_length=avg_len,
                notes=notes,
            )

        tok, model = lm
        ppl = _perplexity(text, tok, model)

        # Per-sentence perplexities for highlight surfacing.
        suspicious = _suspicious_sentences(sentences, tok, model)

        baseline_key = language if language in _BASELINES else "en"
        baseline = (_BASELINES[baseline_key]["ppl_low"], _BASELINES[baseline_key]["ppl_high"])
        ai_prob, confidence = _blend_probability(
            ppl=ppl, baseline=baseline, burstiness=burstiness, n_words=sum(sent_lens), notes=notes
        )

        return AIAnalysis(
            language=language,
            ai_probability=ai_prob,
            confidence=confidence,
            perplexity=ppl,
            perplexity_baseline=baseline,
            burstiness=burstiness,
            sentence_count=len(sent_lens),
            avg_sentence_length=avg_len,
            suspicious_sentences=suspicious,
            notes=notes,
        )


def _split_sentences(text: str, language: Language) -> list[dict]:
    """Return [{text, start, end}]. Razdel handles RU; for EN we still use razdel
    (it gives sane results for Latin too), with a regex fallback for the degenerate case."""
    try:
        spans = list(razdel_sentenize(text))
        if spans:
            return [{"text": s.text, "start": s.start, "stop": s.stop} for s in spans]
    except Exception:
        pass
    # Fallback: split on punctuation
    import re

    out: list[dict] = []
    cursor = 0
    for chunk in re.split(r"(?<=[.!?])\s+", text):
        if not chunk:
            continue
        start = text.find(chunk, cursor)
        if start < 0:
            start = cursor
        out.append({"text": chunk, "start": start, "stop": start + len(chunk)})
        cursor = start + len(chunk)
    return out


def _perplexity(text: str, tok, model) -> float:
    """Sliding-window perplexity capped at MAX_TOKENS per window."""
    ids = tok.encode(text, add_special_tokens=False)
    if not ids:
        return float("nan")
    chunks = [ids[i : i + _MAX_TOKENS] for i in range(0, len(ids), _MAX_TOKENS)]
    losses: list[float] = []
    weights: list[int] = []
    for chunk in chunks:
        inp = torch.tensor([chunk])
        with torch.no_grad():
            out = model(inp, labels=inp)
        losses.append(float(out.loss.item()))
        weights.append(len(chunk))
    weighted_loss = sum(
        loss_value * w for loss_value, w in zip(losses, weights, strict=True)
    ) / sum(weights)
    return math.exp(weighted_loss)


def _suspicious_sentences(sentences: list[dict], tok, model) -> list[SuspiciousSentence]:
    pool = [s for s in sentences if len(s["text"].split()) >= 4][:_MAX_SENTENCES_FOR_BREAKDOWN]
    scored: list[SuspiciousSentence] = []
    for s in pool:
        try:
            p = _perplexity(s["text"], tok, model)
        except Exception:
            continue
        if math.isnan(p) or math.isinf(p):
            continue
        scored.append(
            SuspiciousSentence(
                text=s["text"], perplexity=p, char_start=s["start"], char_end=s["stop"]
            )
        )
    scored.sort(key=lambda x: x.perplexity)  # most predictable first
    return scored[:5]


def _blend_probability(
    *, ppl: float, baseline: tuple[float, float], burstiness: float, n_words: int, notes: list
) -> tuple[float, str]:
    """Heuristic blend. Honest and explainable beats opaque and wrong."""
    ppl_low, ppl_high = baseline
    score = 0.5  # neutral prior

    # Perplexity signal: text *much* more predictable than typical → push toward AI.
    if ppl <= ppl_low * 0.7:
        score += 0.35
        notes.append(f"perplexity {ppl:.1f} far below baseline {ppl_low}–{ppl_high}")
    elif ppl <= ppl_low:
        score += 0.15
        notes.append(f"perplexity {ppl:.1f} at low end of baseline")
    elif ppl >= ppl_high * 1.5:
        score -= 0.20
        notes.append(f"perplexity {ppl:.1f} unusually high → likely human or noisy text")
    elif ppl >= ppl_high:
        score -= 0.05

    # Burstiness signal: very flat sentence rhythm → push toward AI.
    if burstiness <= _BURSTINESS_LOW:
        score += 0.10
        notes.append(f"burstiness {burstiness:.1f} below threshold {_BURSTINESS_LOW}")
    elif burstiness >= _BURSTINESS_LOW * 3:
        score -= 0.05

    score = max(0.0, min(1.0, score))

    if n_words < 30:
        confidence = "low"
        notes.append("very short text — confidence reduced")
    elif n_words < 120:
        confidence = "medium"
    else:
        confidence = "high"

    return score, confidence
