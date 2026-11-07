"""Locate a target on the best target-containing CTC path."""

from __future__ import annotations

import math

import numpy as np

from .ctc_keyword_verifier import CTCKeywordVerifier


def locate_keyword_frames(probabilities, rule: CTCKeywordVerifier):
    """Return (first frame, exclusive end), or None if no path can fit.

    Call this only after the keyword score passes a chosen cutoff. The search
    requires a target, even if its probability is tiny. Like the original
    detector, it floors probabilities at 1e-300 before taking their logs.

    Return the first target on the best accepted path. Its interval spans the
    first emissions of its letters, not necessarily the full pronunciation.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    if (
        p.ndim != 2
        or p.shape[1] != len(rule.vocab)
        or not np.isfinite(p).all()
        or (p < 0).any()
        or not np.allclose(p.sum(1), 1, atol=1e-6, rtol=1e-6)
    ):
        raise ValueError("Expected one probability row per frame.")
    if not len(p):
        return None

    state_count = len(rule.states)
    token_count = len(rule.vocab)
    size = state_count * token_count

    # Each position tracks a word state and the previous CTC token.
    destinations = np.empty((size, token_count), dtype=int)
    for previous in range(size):
        state, last = divmod(previous, token_count)
        for token in range(token_count):
            next_state = state
            if token not in (last, rule.blank):
                next_state = rule.transitions[True][state, token]
            destinations[previous, token] = next_state * token_count + token

    incoming = [
        np.flatnonzero(destinations[:, position % token_count] == position)
        for position in range(size)
    ]
    best = np.full(size, -np.inf)
    best[rule.lookup[""] * token_count + rule.blank] = 0
    back = np.zeros((len(p), size), dtype=int)

    # Keep the best predecessor, rather than summing paths as the scorer does.
    for frame_index, frame in enumerate(p):
        next_best = np.full(size, -np.inf)
        for position, sources in enumerate(incoming):
            if len(sources):
                source = sources[np.argmax(best[sources])]
                back[frame_index, position] = source
                next_best[position] = best[source]
        best = next_best + np.tile(np.log(np.maximum(frame, 1e-300)), state_count)

    finals = [
        state * token_count + token
        for state in rule.terminal
        for token in range(token_count)
    ]
    position = finals[np.argmax(best[finals])]
    if not np.isfinite(best[position]):
        return None

    # Walk backward through the saved choices to recover the token sequence.
    path = []
    for frame_index in reversed(range(len(p))):
        path.append(position % token_count)
        position = back[frame_index, position]
    path.reverse()

    previous = rule.blank
    word, frames = [], []
    # An extra delimiter lets a target finish at the end of the input.
    for frame_index, token in enumerate(path + [rule.delimiter]):
        if token != previous and token != rule.blank:
            if token == rule.delimiter:
                if "".join(word) in rule.targets:
                    return frames[0], frames[-1] + 1
                word, frames = [], []
            elif token not in rule.ignored:
                word.append(rule.chars[token])
                frames.append(frame_index)
        previous = token
    raise AssertionError("Missing target on accepted path.")


def frames_to_seconds(span, frame_seconds, offset_s=0.0, end_s=None):
    """Map a frame interval to song time, clipping any padded tail.

    Supply the acoustic model's frame spacing; the retained detector uses
    0.02 seconds. end_s is the end of real audio in song coordinates.
    Return None if the interval lies entirely in padding.
    """
    start, end = span
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in span):
        raise ValueError("Frame indices must be integers.")
    if not 0 <= start < end:
        raise ValueError("Expected a nonempty frame interval.")
    if not math.isfinite(frame_seconds) or frame_seconds <= 0:
        raise ValueError("Frame spacing must be positive and finite.")
    if not math.isfinite(offset_s) or offset_s < 0:
        raise ValueError("Offset must be nonnegative and finite.")
    if end_s is not None and (not math.isfinite(end_s) or end_s < offset_s):
        raise ValueError("Audio end must be finite and at or after the offset.")

    start_s = offset_s + start * frame_seconds
    stop_s = offset_s + end * frame_seconds
    if end_s is not None:
        stop_s = min(stop_s, end_s)
    return (start_s, stop_s) if start_s < stop_s else None
