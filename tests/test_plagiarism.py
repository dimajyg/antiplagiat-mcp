"""Unit tests for plagiarism helpers — no network involved."""

from __future__ import annotations

import asyncio

from src.config import RequestCredentials, settings
from src.detectors.plagiarism import (
    Match,
    PlagiarismDetector,
    _aggregate_coverage,
    _build_shingles,
)


def test_shingles_extracted_from_russian_text():
    text = (
        "Современные методы машинного обучения позволяют детектировать "
        "сгенерированный искусственным интеллектом текст с приемлемой точностью."
    )
    shingles = _build_shingles(text, "ru")
    assert shingles, "expected at least one characteristic shingle"
    assert all(len(s.split()) == 7 for s in shingles)


def test_shingles_skip_short_text():
    assert _build_shingles("Слишком короткий текст.", "ru") == []


def test_aggregate_coverage_caps_at_one():
    matches = [
        Match(quote="word " * 50, source_url="x", source_title="", similarity=1.0, kind="exact")
    ]
    assert _aggregate_coverage(matches, "word " * 5) == 1.0


def test_aggregate_coverage_zero_with_no_matches():
    assert _aggregate_coverage([], "any text here") == 0.0


def test_no_serper_key_returns_skipped():
    detector = PlagiarismDetector()
    creds = RequestCredentials(openrouter_key="", serper_key="", sapling_key="", gptzero_key="")
    result = asyncio.run(detector.analyze("a" * 200, "ru", creds))
    assert result.skipped_reason is not None
    assert result.match_percentage == 0.0


def test_no_serper_key_uses_settings_fallback_when_present(monkeypatch):
    # Ensure that the per-request creds, not server settings, gate the path.
    monkeypatch.setattr(settings, "serper_api_key", "fake-default")
    detector = PlagiarismDetector()
    creds = RequestCredentials(openrouter_key="", serper_key="", sapling_key="", gptzero_key="")
    result = asyncio.run(detector.analyze("a" * 200, "ru", creds))
    assert result.skipped_reason is not None
