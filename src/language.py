"""Lightweight language detection. RU vs EN routing for downstream models."""

from __future__ import annotations

from typing import Literal

Language = Literal["ru", "en", "other"]


def detect(text: str) -> Language:
    """Return primary language of `text`. Uses cyrillic ratio as a fast heuristic.

    For mixed or short texts we fall back to `langdetect` which is more accurate
    but ~10x slower. Anything that isn't ru/en is bucketed as "other" and
    handled with the multilingual model path.
    """
    if not text:
        return "other"

    cyr = sum(1 for ch in text if "\u0400" <= ch <= "\u04ff")
    lat = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    total = cyr + lat
    if total < 20:
        return _langdetect_fallback(text)

    if cyr / total > 0.6:
        return "ru"
    if lat / total > 0.8:
        return "en"
    return _langdetect_fallback(text)


def _langdetect_fallback(text: str) -> Language:
    try:
        from langdetect import DetectorFactory
        from langdetect import detect as ld

        DetectorFactory.seed = 0
        code = ld(text)
    except Exception:
        return "other"
    if code == "ru":
        return "ru"
    if code == "en":
        return "en"
    return "other"
