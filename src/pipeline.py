"""Top-level orchestration: language detect → AI + plagiarism → blend → cache."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Literal

from . import language as lang_mod
from .cache import Cache, content_hash
from .config import RequestCredentials, settings
from .detectors.ai_local import AIAnalysis, LocalAIDetector
from .detectors.external import SaplingDetector
from .detectors.plagiarism import PlagiarismAnalysis, PlagiarismDetector

Mode = Literal["fast", "deep"]


@dataclass(slots=True)
class FullAnalysis:
    hash: str
    language: str
    ai: AIAnalysis | None
    plagiarism: PlagiarismAnalysis | None
    summary: str

    def to_mcp(self) -> dict:
        return {
            "hash": self.hash,
            "language": self.language,
            "ai": _safe_asdict(self.ai),
            "plagiarism": _safe_asdict(self.plagiarism),
            "summary": self.summary,
        }


def _safe_asdict(obj: Any) -> Any:
    if obj is None:
        return None
    if is_dataclass(obj):
        return asdict(obj)
    return obj


class Pipeline:
    def __init__(self) -> None:
        self.cache = Cache(settings.cache_db)
        self.ai = LocalAIDetector()
        self.plagiarism = PlagiarismDetector()
        self.sapling = SaplingDetector()

    async def analyze(
        self,
        text: str,
        creds: RequestCredentials,
        mode: Mode = "fast",
        check_ai: bool = True,
        check_plagiarism: bool = True,
    ) -> FullAnalysis:
        h = content_hash(text)
        language = lang_mod.detect(text)

        ai_result = await self.ai.analyze(text, language) if check_ai else None
        plag_result = (
            await self.plagiarism.analyze(text, language, creds) if check_plagiarism else None
        )

        if ai_result is not None and mode == "deep" and creds.sapling_key:
            external = await self.sapling.analyze(text, creds.sapling_key)
            ai_result.external_sources.append(
                {
                    "provider": external.provider,
                    "ai_probability": external.ai_probability,
                    "error": external.error,
                }
            )
            if external.error is None:
                ai_result.local_heuristic_probability = ai_result.ai_probability
                ai_result.ai_probability = external.ai_probability
                ai_result.confidence = "high"  # trained classifier trumps heuristic
                ai_result.notes.append(
                    f"deep mode: ai_probability overridden by Sapling "
                    f"({external.ai_probability:.2f}); local heuristic was "
                    f"{ai_result.local_heuristic_probability:.2f}"
                )
            else:
                ai_result.notes.append(f"deep mode: Sapling failed — {external.error}")

        result = FullAnalysis(
            hash=h,
            language=language,
            ai=ai_result,
            plagiarism=plag_result,
            summary=_summarise(ai_result, plag_result),
        )
        return result


def _summarise(ai: AIAnalysis | None, plag: PlagiarismAnalysis | None) -> str:
    parts = []
    if ai is not None:
        parts.append(f"AI probability: {ai.ai_probability:.0%} ({ai.confidence})")
    if plag is not None:
        if plag.skipped_reason:
            parts.append(f"Plagiarism: skipped ({plag.skipped_reason})")
        else:
            parts.append(f"Plagiarism: {plag.match_percentage:.0%}")
    return " | ".join(parts) or "no checks selected"
