"""External AI-detection APIs used by `mode='deep'`.

These adapters are called only when the client passes a matching key in
request headers (or the server has one as a fallback). Each returns the
same shape so the pipeline can blend it with the local heuristic.

Honest note on Sapling's scoring: for short single-sentence inputs the
top-level `score` sometimes disagrees with the per-sentence scores
(e.g. 0.04 vs 0.9999 on the same AI-generated text). We defensively take
`max(top_level_score, mean(sentence_scores))` so a clear per-sentence
signal isn't lost to short-text calibration quirks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean

import httpx

SAPLING_URL = "https://api.sapling.ai/api/v1/aidetect"
_SAPLING_TIMEOUT = 30.0
_SAPLING_MAX_CHARS = 50_000  # API hard limit per request; long inputs are chunked


@dataclass(slots=True)
class ExternalResult:
    provider: str
    ai_probability: float
    raw: dict = field(default_factory=dict)
    error: str | None = None


class SaplingDetector:
    """Sapling AI Detector API client (stateless, per-call auth)."""

    async def analyze(self, text: str, key: str) -> ExternalResult:
        if not key:
            return ExternalResult(provider="sapling", ai_probability=0.0, error="missing api key")

        try:
            async with httpx.AsyncClient(timeout=_SAPLING_TIMEOUT) as client:
                chunks = _split_for_sapling(text)
                scores: list[float] = []
                raws: list[dict] = []
                for chunk in chunks:
                    resp = await client.post(
                        SAPLING_URL,
                        json={"key": key, "text": chunk, "sent_scores": True},
                        headers={"Content-Type": "application/json"},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    raws.append(data)
                    scores.append(_robust_score(data))
        except httpx.HTTPStatusError as e:
            return ExternalResult(
                provider="sapling",
                ai_probability=0.0,
                error=f"http {e.response.status_code}: {e.response.text[:200]}",
            )
        except Exception as e:
            return ExternalResult(
                provider="sapling", ai_probability=0.0, error=f"{type(e).__name__}: {e}"
            )

        # Aggregate across chunks: average is good enough since chunks are equal length.
        overall = mean(scores) if scores else 0.0
        return ExternalResult(
            provider="sapling",
            ai_probability=overall,
            raw={"chunks": raws, "chunk_scores": scores},
        )


def _robust_score(data: dict) -> float:
    """Take max(top-level score, mean of sentence scores).

    Sapling's top-level `score` disagrees with per-sentence scores on short
    inputs; we defensively combine them so a clear per-sentence signal isn't
    discarded by a quirky aggregate.
    """
    top = float(data.get("score", 0.0) or 0.0)
    sents = data.get("sentence_scores") or []
    sent_mean = mean(float(s.get("score", 0.0) or 0.0) for s in sents) if sents else top
    return max(top, sent_mean)


def _split_for_sapling(text: str) -> list[str]:
    if len(text) <= _SAPLING_MAX_CHARS:
        return [text]
    chunks = []
    for i in range(0, len(text), _SAPLING_MAX_CHARS):
        chunks.append(text[i : i + _SAPLING_MAX_CHARS])
    return chunks


class GPTZeroDetector:
    """Placeholder — GPTZero's API is paid-only (min $45/mo for 300K words), so
    this adapter is not wired into the deep-mode blend. Kept for future opt-in."""

    async def analyze(self, text: str, key: str) -> ExternalResult:
        return ExternalResult(
            provider="gptzero",
            ai_probability=0.0,
            error="not implemented — paid-only service, opt-in later",
        )
