"""Top-level orchestration: language detect → AI + plagiarism → blend → cache."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Literal

from . import language as lang_mod
from .cache import Cache, content_hash
from .config import RequestCredentials, settings
from .detectors.ai_local import AIAnalysis, LocalAIDetector
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
