"""Sum the probability of CTC paths containing a complete target word."""

from __future__ import annotations

import math

import numpy as np


class CTCKeywordVerifier:
    """Score character probabilities from an acoustic model.

    Targets must use the model's alphabet. The delimiter separates words.
    The start and end of the input also count as word boundaries.
    """

    def __init__(self, vocabulary, targets, blank, delimiter, ignored=()):
        if sorted(vocabulary.values()) != list(range(len(vocabulary))):
            raise ValueError("Expected contiguous unique token IDs.")
        self.vocab = dict(vocabulary)
        self.targets = set(targets)
        self.blank = blank
        self.delimiter = delimiter
        self.ignored = set(ignored)

        if not self.targets or any(not target for target in self.targets):
            raise ValueError("Targets must contain at least one nonempty word.")
        if blank == delimiter or blank in self.ignored or delimiter in self.ignored:
            raise ValueError("Conflicting special token IDs.")
        special_ids = {blank, delimiter, *self.ignored}
        if not special_ids <= set(vocabulary.values()):
            raise ValueError("Unknown special token IDs.")
        for target in self.targets:
            if any(char not in vocabulary or vocabulary[char] in special_ids for char in target):
                raise ValueError("Target contains an unsupported character.")

        self.chars = {token_id: char for char, token_id in vocabulary.items()}
        prefixes = sorted(
            {target[:i] for target in self.targets for i in range(len(target) + 1)},
            key=lambda prefix: (len(prefix), prefix),
        )
        # Track a target prefix, an unrelated word, or a completed target.
        self.states = prefixes + [None, True]
        self.bad = len(prefixes)
        self.found = self.bad + 1
        self.lookup = {prefix: i for i, prefix in enumerate(prefixes)}
        self.terminal = [self.lookup[target] for target in self.targets] + [self.found]
        self.transitions = {allow: self._transitions(allow) for allow in (False, True)}

    def _transitions(self, allow):
        """Choose the next word state when a new CTC character is emitted."""
        result = np.empty((len(self.states), len(self.vocab)), dtype=np.int64)
        for state, prefix in enumerate(self.states):
            for token_id, char in self.chars.items():
                if state == self.found:
                    next_state = state
                elif token_id in self.ignored or token_id == self.blank:
                    next_state = state
                elif token_id == self.delimiter:
                    next_state = self.found if prefix in self.targets else 0
                elif state == self.bad:
                    next_state = self.bad
                elif allow:
                    next_state = self.lookup.get(prefix + char, self.bad)
                else:
                    next_state = self.bad
                result[state, token_id] = next_state
        return result

    def probability(self, probabilities, allowed=None):
        """Return the total probability of paths containing a target word.

        Input shape is (frames, vocabulary_size), with each row summing to one.
        An optional boolean mask restricts where new target letters can occur.
        Repeated CTC labels collapse before ignored characters are removed.
        """
        p = np.asarray(probabilities, dtype=np.float64)
        if (
            p.ndim != 2
            or p.shape[1] != len(self.vocab)
            or not np.isfinite(p).all()
            or (p < 0).any()
        ):
            raise ValueError("Invalid probabilities.")
        if not np.allclose(p.sum(1), 1, atol=1e-6, rtol=1e-6):
            raise ValueError("Each probability row must sum to one.")
        if allowed is None:
            allowed = np.ones(len(p), dtype=bool)
        allowed = np.asarray(allowed)
        if allowed.shape != (len(p),) or allowed.dtype != np.bool_:
            raise ValueError("Expected one boolean mask value per frame.")

        state_count = len(self.states)
        token_count = len(self.vocab)
        # Keep probability for each word state and previous CTC token.
        alpha = np.zeros((state_count, token_count))
        alpha[0, self.blank] = 1
        columns = np.arange(token_count)
        nonblank = columns != self.blank

        for frame, allow in zip(p, allowed):
            frame = frame / frame.sum()
            total = alpha.sum(1)
            out = np.zeros_like(alpha)

            # Blank emits nothing. Repeating a nonblank token adds no character.
            out[:, self.blank] = total * frame[self.blank]
            out[:, nonblank] = alpha[:, nonblank] * frame[nonblank]

            # A different nonblank token emits a new character.
            emitted = np.maximum(total[:, None] - alpha, 0) * frame[None, :]
            emitted[:, self.blank] = 0
            destinations = self.transitions[bool(allow)] * token_count + columns[None, :]
            out += np.bincount(
                destinations.ravel(),
                weights=emitted.ravel(),
                minlength=state_count * token_count,
            ).reshape(state_count, token_count)
            alpha = out

        if not np.isclose(alpha.sum(), 1, atol=1e-10):
            raise ArithmeticError("CTC probability mass was not conserved.")
        return float(np.clip(alpha[self.terminal].sum(), 0, 1))

    def score(self, probabilities, allowed=None):
        """Return keyword mass, its complement, and finite log-odds.

        These are acoustic-model scores. They are not calibrated probabilities
        that a listener would hear a target word.
        """
        mass = self.probability(probabilities, allowed)
        log_odds = math.log(max(mass, 1e-300)) - math.log(max(1 - mass, 1e-300))
        return {
            "keyword_mass": mass,
            "alternative_mass": 1 - mass,
            "log_odds": log_odds,
        }
