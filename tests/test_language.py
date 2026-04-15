from src.language import detect


def test_russian():
    assert detect("Это длинный текст на русском языке для проверки детектора.") == "ru"


def test_english():
    assert detect("This is a fairly long English sentence used to test the detector.") == "en"


def test_short_or_mixed_returns_something():
    result = detect("hi")
    assert result in {"ru", "en", "other"}
