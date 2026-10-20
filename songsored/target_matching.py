"""Find candidate target occurrences in supplied, already prepared text.

Extracted from the neighbor-passage-v1 workflow: src/lyrics.py normalization,
scripts/full_song_evaluation.py tokenize, and scripts/compare_genius_full_songs.py
occurrences. No fetching, audio access, timestamp estimation or label approval.
The extraction adds rejection of target phrases that tokenize to nothing.
"""
from __future__ import annotations

import re
import unicodedata


ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
TOKEN_RE = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)


def strip_diacritics(text: str) -> str:
    """Remove Arabic diacritics using the original matching rules."""
    return ARABIC_DIACRITICS_RE.sub("", text)


def normalise_arabic_text(text: str) -> str:
    """Apply the original spelling and whitespace normalization."""
    text = unicodedata.normalize("NFC", text)
    text = strip_diacritics(text)
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    text = re.sub(r"\s+", " ", text).strip()
    return unicodedata.normalize("NFC", text)


def tokenize(text: str) -> list[str]:
    """Normalize and split text with the frozen workflow's token expression."""
    return TOKEN_RE.findall(normalise_arabic_text(text).lower())


def occurrences(
    words: list[str],
    specs: list[tuple[str, str, str, str]],
) -> list[dict]:
    """Match (phrase, language, severity, reason) definitions exactly by tokens.

    Pass words produced by tokenize(). Token ranges are zero-based with an
    exclusive end. Specs must list spelling variants explicitly; no stemming,
    phonetic matching or family inference is performed. Duplicate or overlapping
    definitions can produce duplicate or overlapping candidates, as upstream.
    """
    found = []
    for phrase, language, severity, reason in specs:
        parts = tokenize(phrase)
        if not parts:
            raise ValueError("A target phrase must contain at least one token.")
        for start in range(len(words) - len(parts) + 1):
            if words[start : start + len(parts)] == parts:
                end = start + len(parts)
                found.append({
                    "id": f"w{start}-{end}",
                    "token_start": start,
                    "token_end": end,
                    "target": phrase,
                    "labels": [phrase],
                    "language": language,
                    "severity": severity,
                    "policy_status": "run-specific candidate",
                    "policy_reason": reason,
                })
    return sorted(found, key=lambda candidate: candidate["token_start"])
