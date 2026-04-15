"""Unit tests for external detector adapters — no real network."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import httpx

from src.detectors.external import SaplingDetector, _robust_score


def test_robust_score_prefers_sentence_mean_when_higher():
    data = {
        "score": 0.04,
        "sentence_scores": [{"score": 0.99}, {"score": 0.98}],
    }
    assert _robust_score(data) > 0.95


def test_robust_score_prefers_top_when_sentences_empty():
    assert _robust_score({"score": 0.73, "sentence_scores": []}) == 0.73


def test_sapling_empty_key_returns_error():
    result = asyncio.run(SaplingDetector().analyze("hello", ""))
    assert result.error == "missing api key"
    assert result.ai_probability == 0.0


def test_sapling_happy_path(monkeypatch):
    async def fake_post(self, url, **kwargs):
        req = httpx.Request("POST", url)
        return httpx.Response(
            200,
            json={
                "score": 0.04,
                "sentence_scores": [{"score": 0.99, "sentence": "foo"}],
                "text": "foo",
            },
            request=req,
        )

    with patch.object(httpx.AsyncClient, "post", new=fake_post):
        result = asyncio.run(SaplingDetector().analyze("foo", "dummy-key"))
    assert result.error is None
    assert result.ai_probability > 0.9
    assert result.provider == "sapling"


def test_sapling_http_error_wrapped():
    async def fake_post(self, url, **kwargs):
        req = httpx.Request("POST", url)
        return httpx.Response(401, text='{"error":"invalid key"}', request=req)

    with patch.object(httpx.AsyncClient, "post", new=fake_post):
        result = asyncio.run(SaplingDetector().analyze("foo", "dummy-key"))
    assert result.error is not None
    assert "401" in result.error
