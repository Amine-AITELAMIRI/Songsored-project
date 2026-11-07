"""Check keyword timing with generated CTC probabilities, without audio."""

import itertools

import numpy as np
import pytest

from songsored.ctc_keyword_timing import frames_to_seconds, locate_keyword_frames
from songsored.ctc_keyword_verifier import CTCKeywordVerifier


# I use "example" and "examples" here because these tests only check the CTC logic.
VOCABULARY = {char: i for i, char in enumerate("_|examplsq~")}
RULE = CTCKeywordVerifier(VOCABULARY, ["example", "examples"], blank=0, delimiter=1)


def onehot(text):
    return np.eye(len(VOCABULARY))[[VOCABULARY[char] for char in text]]


@pytest.mark.parametrize("text, expected", [
    ("__ee_xampless_", (2, 12)),
    ("examples", (0, 8)),
    ("|example|", (1, 8)),
    ("qexample|examples", (9, 17)),
    ("exampleq|example", (9, 16)),
    ("example|examples", (0, 7)),
])
def test_letter_emissions_and_whole_word_boundaries(text, expected):
    assert locate_keyword_frames(onehot(text), RULE) == expected


def test_repeat_separated_by_blank_and_ignored_token():
    vocab = {"o": 0, "|": 1, "_": 2, "~": 3}
    rule = CTCKeywordVerifier(vocab, ["oo"], blank=2, delimiter=1, ignored=[3])
    for text in ("oo_o", "oo~o"):
        p = np.eye(4)[[vocab[char] for char in text]]
        assert locate_keyword_frames(p, rule) == (0, 4)


def first_target_span(path):
    """Collapse a small test alphabet and find the first ab or abc."""
    words = [[]]
    previous = 0
    for frame, token in enumerate(path):
        if token != previous and token != 0:
            if token == 1:
                words.append([])
            else:
                words[-1].append((token, frame))
        previous = token
    for word in words:
        if [token for token, _ in word] in ([2, 3], [2, 3, 4]):
            return word[0][1], word[-1][1] + 1
    return None


def test_best_accepted_path_matches_exhaustive_search():
    # A short alphabet lets me check every path without a huge test.
    rule = CTCKeywordVerifier({char: i for i, char in enumerate("_|abcq")}, ["ab", "abc"], 0, 1)
    rng = np.random.default_rng(7)
    for _ in range(10):
        p = rng.dirichlet(np.ones(6), size=4)
        best_score = -1
        expected = None
        for path in itertools.product(range(6), repeat=4):
            span = first_target_span(path)
            score = np.prod([p[frame, token] for frame, token in enumerate(path)])
            if span is not None and score > best_score:
                best_score, expected = score, span
        assert locate_keyword_frames(p, rule) == expected


def test_localization_is_not_a_presence_check():
    p = onehot("________")
    assert RULE.probability(p) == 0
    # Flooring zeros still permits a very unlikely target path. Gate on score first.
    assert locate_keyword_frames(p, RULE) is not None


def test_no_path_fits_empty_or_one_frame_input():
    assert locate_keyword_frames(np.empty((0, len(VOCABULARY))), RULE) is None
    assert locate_keyword_frames(onehot("e"), RULE) is None


def test_frame_conversion_and_padded_tail():
    assert frames_to_seconds((10, 15), 0.02, offset_s=3) == pytest.approx((3.2, 3.3))
    assert frames_to_seconds((10, 15), 0.02, offset_s=3, end_s=3.25) == pytest.approx((3.2, 3.25))
    assert frames_to_seconds((10, 15), 0.02, offset_s=3, end_s=3.1) is None


@pytest.mark.parametrize("p", [
    np.ones((2, len(VOCABULARY))), np.full((2, len(VOCABULARY)), np.nan), np.zeros((2, 3)),
    np.array([[-1, 2] + [0] * (len(VOCABULARY) - 2)]), np.ones(len(VOCABULARY)),
])
def test_invalid_probabilities(p):
    with pytest.raises(ValueError):
        locate_keyword_frames(p, RULE)


@pytest.mark.parametrize("span, spacing, offset, end", [
    ((-1, 2), .02, 0, None), ((2, 2), .02, 0, None),
    ((1.5, 2), .02, 0, None), ((0, 2), 0, 0, None),
    ((0, 2), .02, -1, None), ((0, 2), .02, 3, 2),
    ((0, 2), float("nan"), 0, None), ((0, 2), .02, 0, float("inf")),
])
def test_invalid_time_mapping(span, spacing, offset, end):
    with pytest.raises(ValueError):
        frames_to_seconds(span, spacing, offset, end)
