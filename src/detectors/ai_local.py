"""Local AI-generated text detector.

Runs three independent signals on CPU and combines them:

1. **Classifier** — `xlm-roberta` fine-tuned for AI/human classification.
   Single forward pass, gives a probability per chunk.
2. **Perplexity** — sliding-window LM scoring with a small causal model
   (ruGPT3-small for ru, distilgpt2 for en). Lower PPL ≈ more "predictable",
   weakly correlated with AI authorship.
3. **Burstiness** — std-dev of sentence lengths in tokens. Human prose is
   bursty; LLM prose tends to be uniform.

Signals are surfaced individually so the MCP tool can show them to the user
and let them judge — we don't pretend any single number is ground truth.
See PLAN.md "Stage 1" for the implementation plan.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..language import Language


@dataclass(slots=True)
class AIAnalysis:
    ai_probability: float
    perplexity: float | None
    burstiness: float
    language: Language
    suspicious_sentences: list[dict]
    notes: list[str]


class LocalAIDetector:
    """TODO Stage 1: load models lazily and implement `analyze`."""

    def __init__(self) -> None:
        self._classifier = None
        self._ppl_ru = None
        self._ppl_en = None

    async def analyze(self, text: str, language: Language) -> AIAnalysis:
        raise NotImplementedError("Stage 1: implement classifier + PPL + burstiness pipeline")
