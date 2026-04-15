"""Plagiarism detection via shingling + web search + embedding similarity.

Pipeline:
  1. Split into sentences (`razdel` for ru, `nltk` for en)
  2. Generate 6–10 word shingles, drop boilerplate, pick the most
     "characteristic" N (rare n-grams)
  3. Search each in quotes via Serper.dev (parallel, async)
  4. Fetch candidate URLs with `trafilatura`
  5. Compute exact-shingle overlap and paraphrase similarity (embeddings)
  6. Aggregate into per-fragment matches with source URLs

See PLAN.md "Stage 2" for details.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import RequestCredentials
from ..language import Language


@dataclass(slots=True)
class Match:
    quote: str
    source_url: str
    similarity: float
    kind: str  # "exact" | "paraphrase"


@dataclass(slots=True)
class PlagiarismAnalysis:
    match_percentage: float
    matches: list[Match] = field(default_factory=list)
    skipped_reason: str | None = None  # populated when no Serper key given


class PlagiarismDetector:
    async def analyze(
        self,
        text: str,
        language: Language,
        creds: RequestCredentials,
    ) -> PlagiarismAnalysis:
        if not creds.serper_key:
            return PlagiarismAnalysis(
                match_percentage=0.0,
                skipped_reason="no Serper API key in request headers",
            )
        raise NotImplementedError("Stage 2: implement shingle + search + similarity")
