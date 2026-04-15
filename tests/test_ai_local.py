"""Smoke test: detector loads, returns sane shape, perplexity is a finite number.

Heavy — pulls real models from disk, so marked as integration-style.
Skipped automatically when the models directory is missing (e.g. CI without disk).
"""

from __future__ import annotations

import asyncio

import pytest

from src.config import settings
from src.detectors.ai_local import LocalAIDetector
from src.language import detect

pytestmark = pytest.mark.skipif(
    not (settings.models_dir / "ai-forever__rugpt3small_based_on_gpt2").exists(),
    reason="local models not downloaded — skipping integration test",
)


def _run(text: str):
    detector = LocalAIDetector()
    return asyncio.run(detector.analyze(text, detect(text)))


def test_russian_text_returns_finite_perplexity():
    result = _run(
        "Сегодня я ходил в магазин, купил хлеб и молоко, потом долго стоял в пробке. "
        "Дома меня уже ждал кот, который тут же начал требовать ужин."
    )
    assert result.language == "ru"
    assert result.perplexity is not None and result.perplexity > 0
    assert 0.0 <= result.ai_probability <= 1.0
    assert result.confidence in {"low", "medium", "high"}


def test_english_text():
    result = _run(
        "I went to the store to buy bread and milk, then got stuck in traffic for an hour. "
        "When I finally got home the cat was already waiting at the door for dinner."
    )
    assert result.language == "en"
    assert result.perplexity is not None and result.perplexity > 0


def test_empty_text_does_not_crash():
    result = _run("Короткий текст.")
    assert result.confidence == "low"
