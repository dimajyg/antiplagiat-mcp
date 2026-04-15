"""Optional external AI-detection APIs (Sapling, GPTZero) for `mode='deep'`.

These are only called when the client provides a key in their request headers
and explicitly asks for deep analysis. Each adapter returns the same shape so
the pipeline can blend it with local signals.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ExternalResult:
    provider: str
    ai_probability: float
    raw: dict


class SaplingDetector:
    async def analyze(self, text: str, key: str) -> ExternalResult:
        raise NotImplementedError("Stage 3: wire up Sapling /api/v1/aidetect")


class GPTZeroDetector:
    async def analyze(self, text: str, key: str) -> ExternalResult:
        raise NotImplementedError("Stage 3: wire up GPTZero /v2/predict/text")
