"""Check CTC keyword scoring with generated character probabilities."""

import itertools

import numpy as np
import pytest

from songsored.ctc_keyword_verifier import CTCKeywordVerifier


# I use "example" and "examples" here because these tests only check the CTC logic.
VOCABULARY = {char: i for i, char in enumerate("_|examplsq~")}


def onehot(text):
    return np.eye(len(VOCABULARY))[[VOCABULARY[char] for char in text]]


def verifier():
    return CTCKeywordVerifier(VOCABULARY, ["example", "examples"], blank=0, delimiter=1, ignored=[VOCABULARY["~"]])


def exhaustive_probability(probabilities, allowed):
    """Enumerate a small alphabet: blank, delimiter, a, b, c, q and ignored."""
    total = 0.0
    for path in itertools.product(range(probabilities.shape[1]), repeat=len(probabilities)):
        previous = 0
        words = [[]]
        for frame, token in enumerate(path):
            if token != 0 and token != previous:
                if token == 1:
                    words.append([])
                elif token != 6:
                    words[-1].append((token, allowed[frame]))
            previous = token
        if any(
            [token for token, _ in word] in ([2, 3], [2, 3, 4])
            and all(allow for _, allow in word)
            for word in words
        ):
            total += np.prod([probabilities[frame, token] for frame, token in enumerate(path)])
    return total


def test_probability_matches_every_possible_ctc_path():
    # A short alphabet lets me check every path without a huge test.
    rule = CTCKeywordVerifier({char: i for i, char in enumerate("_|abcq~")}, ["ab", "abc"], 0, 1, ignored=[6])
    rng = np.random.default_rng(42)
    probabilities = rng.dirichlet(np.ones(7), size=4)
    for allowed in (np.ones(4, dtype=bool), np.array([False, True, True, False])):
        expected = exhaustive_probability(probabilities, allowed)
        assert rule.probability(probabilities, allowed) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("text", ["example", "examples", "|example|", "ex~ample", "eexxaammpplleess", "q|examples|q", "example___"])
def test_complete_words_are_accepted(text):
    assert verifier().probability(onehot(text)) == 1


@pytest.mark.parametrize("text", ["exampleq", "qexample", "ex_a_amples", "ex|amples", "__", "q", "e", "amples"])
def test_substrings_and_incomplete_words_are_rejected(text):
    assert verifier().probability(onehot(text)) == 0


def test_blank_and_ignored_tokens_affect_ctc_repeats():
    repeated = CTCKeywordVerifier(VOCABULARY, ["ee"], blank=0, delimiter=1, ignored=[VOCABULARY["~"]])
    assert repeated.probability(onehot("ee")) == 0
    assert repeated.probability(onehot("e_e")) == 1
    assert repeated.probability(onehot("e~e")) == 1


def test_allowed_mask_restricts_new_target_letters():
    probabilities = onehot("q|examples|q")
    allowed = np.zeros(len(probabilities), dtype=bool)
    assert verifier().probability(probabilities, allowed) == 0
    allowed[2:10] = True
    assert verifier().probability(probabilities, allowed) == 1
    # The final delimiter confirms the word even outside the letter mask.
    allowed[3] = False
    assert verifier().probability(probabilities, allowed) == 0


def test_empty_input_and_finite_score_endpoints():
    assert verifier().probability(np.zeros((0, len(VOCABULARY)))) == 0
    for text in ("examples", "q"):
        result = verifier().score(onehot(text))
        assert result["keyword_mass"] + result["alternative_mass"] == 1
        assert np.isfinite(result["log_odds"])


def test_invalid_probabilities_and_configuration():
    for probabilities in (np.ones((3, len(VOCABULARY))), np.full((3, len(VOCABULARY)), np.nan), np.zeros((3, 4))):
        with pytest.raises(ValueError):
            verifier().probability(probabilities)
    with pytest.raises(ValueError):
        verifier().probability(onehot("examples"), [1] * 8)
    for targets in ([], [""], ["?"]):
        with pytest.raises(ValueError):
            CTCKeywordVerifier(VOCABULARY, targets, blank=0, delimiter=1)
    with pytest.raises(ValueError):
        CTCKeywordVerifier(VOCABULARY, ["example"], blank=0, delimiter=0)
