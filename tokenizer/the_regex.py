#Construisons notre propre regex.
import regex
import pytest


REGEX_PATTERN = r""" ?(?i:jusqu|lorsqu|puisqu|quoiqu|qu|[cdjlmnst])['’](?=\p{L})|[^\r\n\p{L}\p{N}]?\p{L}+(?:['’]\p{L}+)*|\p{N}{1}|[ \xa0\u202f]?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""


FR_CASES = [
    ("L'été, j'ai vu 12345 astéroïdes\xa0!",
     ["L'", "été", ",", " j'", "ai", " vu", " ", "1", "2", "3", "4", "5", " astéroïdes", "\xa0!"]),
    ("Qu’il vienne « demain » à 18h30.",
     ["Qu’", "il", " vienne", " «", " demain", " »", " à", " ", "1", "8", "h", "3", "0", "."]),
    ("def f(x):\n    return x",
     ["def", " f", "(x", "):\n", "   ", " return", " x"]),
    ("aujourd'hui",
     ["aujourd'hui"]),
    ("jusqu'à",
     ["jusqu'", "à"])
]


@pytest.mark.parametrize("text, expected", FR_CASES)
def test_pattern(text, expected):
    assert regex.findall(REGEX_PATTERN, text) == expected


@pytest.mark.parametrize("text", [text for text, _ in FR_CASES])
def test_pattern_lossless(text):
    assert "".join(regex.findall(REGEX_PATTERN, text)) == text