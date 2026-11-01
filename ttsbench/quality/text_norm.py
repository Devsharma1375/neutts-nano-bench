"""Normalise text before computing word error rate.

The ASR model writes "25" where the corpus says "twenty-five" (or the reverse), and
formatting differences like that are not speech errors. Both sides go through the same
normalisation, so only genuinely wrong or missing words count.
"""
from __future__ import annotations

import re
import unicodedata

from num2words import num2words

_ABBREV = {
    "dr": "doctor", "mr": "mister", "mrs": "missus", "ms": "miss", "st": "saint", "vs": "versus",
    "etc": "et cetera", "e.g": "for example", "i.e": "that is", "no": "number",
}
_ORDINAL = re.compile(r"\b(\d+)(st|nd|rd|th)\b")
_MONEY = re.compile(r"\$(\d+(?:\.\d+)?)")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s?%")
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
_TIME = re.compile(r"\b(\d{1,2}):(\d{2})\b")


def _num(m: re.Match) -> str:
    s = m.group(0)
    try:
        value = float(s) if "." in s else int(s)
        if isinstance(value, int) and 1100 <= value <= 2099:  # years read as "twenty twenty-six"
            return num2words(value, to="year")
        return num2words(value)
    except (ValueError, OverflowError):
        return s


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = _TIME.sub(lambda m: f"{num2words(int(m.group(1)))} {'' if m.group(2) == '00' else num2words(int(m.group(2)))}", text)
    text = _MONEY.sub(lambda m: f"{m.group(1)} dollars", text)
    text = _PERCENT.sub(lambda m: f"{m.group(1)} percent", text)
    text = _ORDINAL.sub(lambda m: num2words(int(m.group(1)), to="ordinal"), text)
    text = re.sub(r"(\d)([a-z])", r"\1 \2", text)  # "10a.m" and "10 a.m." are the same words
    text = re.sub(r"\b([ap])\.\s?m\b\.?", r"\1m", text)  # a.m. / a.m / a. m. -> am
    text = re.sub(r"\b(" + "|".join(re.escape(k) for k in _ABBREV) + r")\.", lambda m: _ABBREV[m.group(1)], text)
    text = _NUMBER.sub(_num, text)
    text = re.sub(r"[-–—/]", " ", text)
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"'(?![a-z])|(?<![a-z])'", " ", text)
    return re.sub(r"\s+", " ", text).strip()
