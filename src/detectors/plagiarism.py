"""Plagiarism detection: shingle → web search → fetch → similarity.

Pipeline:
  1. Sentence-split the input (razdel for both RU and EN — works fine for Latin)
  2. Build 7-word shingles with rare-word weighting, dedup, take ~6 of them
  3. Search each shingle in quotes via Serper.dev (parallel httpx)
  4. Fetch top candidate URLs and extract main content with trafilatura
  5. Score each fetched document:
       * exact-shingle hits (literal substring)
       * sentence-level paraphrase similarity (cosine on embeddings)
  6. Aggregate into match_percentage + ranked matches list

The whole layer is gated on the client passing `X-Serper-Key` — otherwise we
return early with `skipped_reason`. Serper has a free tier of 2500 queries,
so the user gets ~400 plagiarism checks without paying anything.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field

import httpx

from ..config import RequestCredentials
from ..embeddings import EmbeddingProvider
from ..language import Language
from .ai_local import _split_sentences

SERPER_URL = "https://google.serper.dev/search"
SHINGLE_WORDS = 7
MAX_SHINGLES = 6
MIN_SHINGLE_RARE_WORDS = 2
TOP_RESULTS_PER_QUERY = 4
FETCH_TIMEOUT = 8.0
MAX_FETCH_CONCURRENCY = 6
PARAPHRASE_THRESHOLD = 0.85

# Stopwords kept tiny on purpose — full lists aren't needed because shingle
# selection only counts "rare-enough" words, not "stopword-free" ones.
_RU_STOP = {
    "и",
    "в",
    "во",
    "не",
    "что",
    "он",
    "на",
    "я",
    "с",
    "со",
    "как",
    "а",
    "то",
    "все",
    "она",
    "так",
    "его",
    "но",
    "да",
    "ты",
    "к",
    "у",
    "же",
    "вы",
    "за",
    "бы",
    "по",
    "только",
    "ее",
    "мне",
    "было",
    "вот",
    "от",
    "меня",
    "еще",
    "нет",
    "о",
    "из",
    "ему",
    "теперь",
    "когда",
    "даже",
    "ну",
    "вдруг",
    "ли",
    "если",
    "уже",
    "или",
    "ни",
    "быть",
    "был",
    "него",
    "до",
    "вас",
    "нибудь",
    "опять",
    "уж",
    "вам",
    "ведь",
    "там",
    "потом",
    "себя",
    "ничего",
    "ей",
    "может",
    "они",
    "тут",
    "где",
    "есть",
    "надо",
    "ней",
    "для",
    "мы",
    "тебя",
    "их",
    "чем",
    "была",
    "сам",
    "чтоб",
    "без",
    "будто",
    "чего",
    "раз",
    "тоже",
    "себе",
    "под",
    "будет",
    "ж",
    "тогда",
    "кто",
    "этот",
    "того",
    "потому",
    "этого",
    "какой",
    "ним",
    "здесь",
    "этом",
    "один",
    "почти",
    "мой",
    "тем",
    "чтобы",
    "нее",
    "кажется",
    "сейчас",
    "были",
    "куда",
    "зачем",
    "всех",
    "никогда",
    "можно",
    "при",
    "наконец",
    "два",
    "об",
    "другой",
    "хоть",
    "после",
    "над",
    "больше",
    "тот",
    "через",
    "эти",
    "нас",
    "про",
    "всего",
    "них",
    "какая",
    "много",
    "разве",
    "три",
    "эту",
    "моя",
    "впрочем",
    "хорошо",
    "свою",
    "этой",
    "перед",
    "иногда",
    "лучше",
    "чуть",
    "том",
    "нельзя",
    "такой",
    "им",
    "более",
    "всегда",
    "конечно",
    "всю",
    "между",
}
_EN_STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "than",
    "that",
    "this",
    "these",
    "those",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "should",
    "could",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "my",
    "your",
    "his",
    "her",
    "its",
    "our",
    "their",
    "as",
    "not",
    "no",
    "so",
}


@dataclass(slots=True)
class Match:
    quote: str
    source_url: str
    source_title: str
    similarity: float
    kind: str  # "exact" | "paraphrase"


@dataclass(slots=True)
class PlagiarismAnalysis:
    match_percentage: float
    matches: list[Match] = field(default_factory=list)
    shingles_searched: int = 0
    sources_fetched: int = 0
    skipped_reason: str | None = None


class PlagiarismDetector:
    def __init__(self, embeddings: EmbeddingProvider | None = None) -> None:
        self.embeddings = embeddings or EmbeddingProvider()

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
        if len(text.split()) < SHINGLE_WORDS * 2:
            return PlagiarismAnalysis(
                match_percentage=0.0, skipped_reason="text too short for shingle search"
            )

        shingles = _build_shingles(text, language)
        if not shingles:
            return PlagiarismAnalysis(
                match_percentage=0.0, skipped_reason="no characteristic shingles found"
            )

        async with httpx.AsyncClient(timeout=FETCH_TIMEOUT) as client:
            search_tasks = [_search_shingle(client, s, creds.serper_key) for s in shingles]
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

            url_to_shingle: dict[str, str] = {}
            url_to_title: dict[str, str] = {}
            for shingle, hits in zip(shingles, search_results, strict=True):
                if isinstance(hits, Exception):
                    continue
                for hit in hits[:TOP_RESULTS_PER_QUERY]:
                    url = hit.get("link")
                    if not url:
                        continue
                    url_to_shingle.setdefault(url, shingle)
                    url_to_title.setdefault(url, hit.get("title", ""))

            sem = asyncio.Semaphore(MAX_FETCH_CONCURRENCY)
            fetch_tasks = [_fetch_text(client, url, sem) for url in url_to_shingle]
            fetched = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        matches: list[Match] = []
        for url, doc in zip(url_to_shingle, fetched, strict=True):
            if isinstance(doc, Exception) or not doc:
                continue
            shingle = url_to_shingle[url]
            title = url_to_title[url]
            if shingle.lower() in doc.lower():
                matches.append(
                    Match(
                        quote=shingle,
                        source_url=url,
                        source_title=title,
                        similarity=1.0,
                        kind="exact",
                    )
                )

        # Paraphrase pass: only run when we actually have candidate documents
        # AND the client has an OpenRouter key (or we accept the local-model fallback)
        if url_to_shingle:
            paraphrase_matches = await self._paraphrase_pass(
                text=text,
                language=language,
                docs={
                    url: d
                    for url, d in zip(url_to_shingle, fetched, strict=True)
                    if isinstance(d, str) and d
                },
                titles=url_to_title,
                creds=creds,
            )
            # Avoid double-counting an exact match as also paraphrase
            seen_urls = {m.source_url for m in matches}
            for m in paraphrase_matches:
                if m.source_url not in seen_urls:
                    matches.append(m)

        match_percentage = _aggregate_coverage(matches, text)

        return PlagiarismAnalysis(
            match_percentage=match_percentage,
            matches=matches,
            shingles_searched=len(shingles),
            sources_fetched=sum(1 for d in fetched if isinstance(d, str) and d),
        )

    async def _paraphrase_pass(
        self,
        *,
        text: str,
        language: Language,
        docs: dict[str, str],
        titles: dict[str, str],
        creds: RequestCredentials,
    ) -> list[Match]:
        if not docs:
            return []
        input_sentences = [
            s["text"] for s in _split_sentences(text, language) if len(s["text"].split()) >= 5
        ]
        if not input_sentences:
            return []
        # Embed the input sentences once
        try:
            input_vecs = await self.embeddings.embed(input_sentences, creds)
        except Exception:
            return []
        results: list[Match] = []
        for url, doc in docs.items():
            doc_sentences = [
                s["text"] for s in _split_sentences(doc, language) if len(s["text"].split()) >= 5
            ][:30]  # cap per document to control embedding cost
            if not doc_sentences:
                continue
            try:
                doc_vecs = await self.embeddings.embed(doc_sentences, creds)
            except Exception:
                continue
            best = _best_pair_similarity(input_vecs, input_sentences, doc_vecs, doc_sentences)
            if best and best[2] >= PARAPHRASE_THRESHOLD:
                quote, _doc_match, sim = best
                results.append(
                    Match(
                        quote=quote,
                        source_url=url,
                        source_title=titles.get(url, ""),
                        similarity=sim,
                        kind="paraphrase",
                    )
                )
        return results


# -------- helpers (module-level so they're easy to unit-test) --------


def _build_shingles(text: str, language: Language) -> list[str]:
    stop = _RU_STOP if language == "ru" else _EN_STOP
    words = re.findall(r"[\w'-]+", text, flags=re.UNICODE)
    if len(words) < SHINGLE_WORDS:
        return []
    seen: set[str] = set()
    candidates: list[tuple[int, str]] = []
    for i in range(len(words) - SHINGLE_WORDS + 1):
        chunk = words[i : i + SHINGLE_WORDS]
        rare = sum(1 for w in chunk if w.lower() not in stop and len(w) >= 4)
        if rare < MIN_SHINGLE_RARE_WORDS:
            continue
        joined = " ".join(chunk)
        key = joined.lower()
        if key in seen:
            continue
        seen.add(key)
        candidates.append((rare, joined))
    candidates.sort(key=lambda x: -x[0])
    # Spread out so we don't pick all from the same paragraph
    chosen: list[str] = []
    spacing = max(1, len(candidates) // (MAX_SHINGLES * 2)) if candidates else 1
    for idx, (_score, sh) in enumerate(candidates):
        if idx % spacing == 0:
            chosen.append(sh)
        if len(chosen) >= MAX_SHINGLES:
            break
    return chosen


async def _search_shingle(client: httpx.AsyncClient, shingle: str, key: str) -> list[dict]:
    payload = {"q": f'"{shingle}"', "num": TOP_RESULTS_PER_QUERY}
    resp = await client.post(
        SERPER_URL,
        json=payload,
        headers={"X-API-KEY": key, "Content-Type": "application/json"},
    )
    resp.raise_for_status()
    return resp.json().get("organic", [])


async def _fetch_text(client: httpx.AsyncClient, url: str, sem: asyncio.Semaphore) -> str:
    import trafilatura

    async with sem:
        try:
            resp = await client.get(url, follow_redirects=True, timeout=FETCH_TIMEOUT)
            if resp.status_code != 200:
                return ""
            extracted = trafilatura.extract(resp.text) or ""
            return extracted
        except Exception:
            return ""


def _best_pair_similarity(
    qv: list[list[float]], qs: list[str], dv: list[list[float]], ds: list[str]
) -> tuple[str, str, float] | None:
    import numpy as np

    if not qv or not dv:
        return None
    Q = np.array(qv)
    D = np.array(dv)
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
    D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
    sim = Q @ D.T
    i, j = np.unravel_index(int(np.argmax(sim)), sim.shape)
    return qs[i], ds[j], float(sim[i, j])


def _aggregate_coverage(matches: list[Match], text: str) -> float:
    if not matches or not text.strip():
        return 0.0
    total_words = max(1, len(text.split()))
    covered = 0
    seen_quotes: set[str] = set()
    for m in matches:
        key = m.quote.lower()
        if key in seen_quotes:
            continue
        seen_quotes.add(key)
        covered += len(m.quote.split())
    return min(1.0, covered / total_words)
