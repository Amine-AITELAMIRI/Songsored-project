"""Conservative lyric and alignment guards for provisional negative clips.

Screening is not proof that a target is absent from the audio."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
import re

from songsored.target_matching import TOKEN_RE, normalise_arabic_text

EXTRA_SCREEN = re.compile(
    r"(?:fuck|fuk|mufuck|bitch|niqu|nke7|9e7ab|97ab|foufoun|schnek|"
    r"7awik|neghtas|tize|kharaw|trammi|da3ar|in3el|shlag)"
)

EXTRA_EXACT = {"ker", "kerr", "zok", "saute", "khchi", "koulibalys", "ass"}


def tokens(text):
    return TOKEN_RE.findall(normalise_arabic_text(text).lower())


def forbidden_spans(words, policy, focused):
    """Conservative candidate screening only; this does not expand positive labels."""
    aliases = {
        tuple(tokens(form))
        for group in focused["families"].values()
        for form in group["forms"]
    }
    for target in policy["targets"]:
        for alias in target.get("aliases", []):
            if isinstance(alias, str):
                aliases.add(tuple(tokens(alias)))
            elif alias.get("status", "confirmed") == "confirmed":
                aliases.add(tuple(tokens(alias["text"])))

    # Exclude obvious foreign profanity even if outside the focused positive task.
    common_aliases = (
        "fuck fucking fucker bitch shit nigger nigga motherfucker pussy "
        "pute putain salope nique puta zebbi zbi"
    )
    aliases.update(tuple(tokens(word)) for word in common_aliases.split())

    found = []
    for alias in sorted(aliases):
        if not alias:
            continue
        for start in range(len(words) - len(alias) + 1):
            if tuple(words[start : start + len(alias)]) == alias:
                found.append((start, start + len(alias)))

    for index, word in enumerate(words):
        if EXTRA_SCREEN.search(word) or word in EXTRA_EXACT:
            found.append((index, index + 1))

    return sorted(set(found))


def overlaps(a, b, spans):
    return any(a < high and b > low for low, high in spans)


def finite_span(span):
    return all(
        type(span.get(key)) in (float, int) and math.isfinite(span[key])
        for key in ("start", "end")
    ) and 0 <= span["start"] < span["end"]


def clean_anchor(words, align, i, forbidden_indices):
    """Require nearby mapped lyric words, with no gaps/profanity or disorder."""
    if i < 2 or i + 2 >= len(words):
        return False
    neighbors = range(i - 2, i + 3)
    if any(
        neighbor in forbidden_indices or words[neighbor] == "_gap_"
        for neighbor in neighbors
    ):
        return False
    spans = [align.get(str(neighbor), {}) for neighbor in neighbors]
    if not all(
        finite_span(span) and 0.04 <= span["end"] - span["start"] <= 1
        for span in spans
    ):
        return False
    return all(
        earlier["end"] <= later["start"] + 0.04
        and later["start"] - earlier["end"] <= 0.8
        for earlier, later in zip(spans, spans[1:])
    )


def pick_pools(pools, count, desired):
    """Spread examples across recordings; no more than three per recording."""
    picked = defaultdict(list)
    categories = Counter()
    order = sorted(pools, key=lambda recording_id: hashlib.sha256(
        recording_id.encode()
    ).hexdigest())
    for round_number in range(3):
        for recording_id in order:
            if sum(map(len, picked.values())) >= count:
                return picked
            available = [
                candidate
                for candidate in pools[recording_id]
                if all(
                    abs(candidate["start"] - previous["start"]) >= 5
                    for previous in picked[recording_id]
                )
            ]
            if not available:
                continue
            category_order = sorted(
                desired,
                key=lambda category: -(desired[category] - categories[category]),
            )
            preferred = next(
                (
                    category
                    for category in category_order
                    if categories[category] < desired[category]
                    and any(
                        candidate["category"] == category
                        for candidate in available
                    )
                ),
                None,
            )
            usable = (
                [
                    candidate
                    for candidate in available
                    if candidate["category"] == preferred
                ]
                if preferred
                else available
            )
            selected = min(
                usable,
                key=lambda candidate: hashlib.sha256(
                    (
                        f"auto-negative-v1:{recording_id}:{round_number}:"
                        f"{candidate['anchor_token']}"
                    ).encode()
                ).hexdigest(),
            )
            picked[recording_id].append(selected)
            categories[selected["category"]] += 1
    return picked
