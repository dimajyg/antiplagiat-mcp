"""Embedding provider with OpenRouter primary and a local fallback.

The client passes their own OpenRouter key per request. If absent, we fall
back to a local `multilingual-e5-small` model so the tool still works for
plagiarism similarity even without paid credentials.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import httpx

from .config import RequestCredentials, settings

OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"


class EmbeddingProvider:
    def __init__(self) -> None:
        self._local_model = None  # lazy
        self._client = httpx.AsyncClient(timeout=30.0)

    async def embed(self, texts: Sequence[str], creds: RequestCredentials) -> list[list[float]]:
        if creds.openrouter_key:
            try:
                return await self._openrouter(texts, creds.openrouter_key)
            except Exception:
                pass  # graceful fallback
        return await asyncio.to_thread(self._local, list(texts))

    async def _openrouter(self, texts: Sequence[str], key: str) -> list[list[float]]:
        resp = await self._client.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": settings.openrouter_embedding_model, "input": list(texts)},
        )
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["data"]]

    def _local(self, texts: list[str]) -> list[list[float]]:
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer

            self._local_model = SentenceTransformer("intfloat/multilingual-e5-small")
        prefixed = [f"query: {t}" for t in texts]
        vectors = self._local_model.encode(prefixed, normalize_embeddings=True)
        return vectors.tolist()

    async def aclose(self) -> None:
        await self._client.aclose()
