"""Invented text only: candidate matching is not audio-presence validation."""
import pytest

from songsored.target_matching import normalise_arabic_text, occurrences, tokenize


def test_normalization_and_tokenization():
    assert normalise_arabic_text("  إِلَى\n أَلْف  ") == "الي الف"
    assert tokenize("ZEBI, 9lawik! _gap_") == ["zebi", "9lawik", "_gap_"]


def test_exact_forms_repetitions_and_text_order():
    words = tokenize("zebi zebbi ZEBI 9lawik")
    specs = [
        ("9lawik", "ar", "hard", "Explicit test target"),
        ("zebi", "ar", "hard", "Explicit test target"),
    ]
    hits = occurrences(words, specs)
    assert [(h["target"], h["token_start"], h["token_end"]) for h in hits] == [
        ("zebi", 0, 1), ("zebi", 2, 3), ("9lawik", 3, 4),
    ]
    assert hits[0]["policy_status"] == "run-specific candidate"
    assert "start" not in hits[0] and "end" not in hits[0]


def test_phrase_requires_adjacent_tokens():
    words = tokenize("red blue red _gap_ blue")
    hits = occurrences(words, [("red blue", "en", "review", "Invented example")])
    assert len(hits) == 1
    assert hits[0]["id"] == "w0-2"
    assert words[hits[0]["token_start"]:hits[0]["token_end"]] == ["red", "blue"]


def test_rejects_empty_target():
    with pytest.raises(ValueError, match="at least one token"):
        occurrences(tokenize("red blue"), [("!!!", "en", "review", "Invalid target")])
