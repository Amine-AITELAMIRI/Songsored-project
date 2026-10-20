"""Recording grouping from the historical reviewed-data protocol.

The public wrapper requires human-reviewed rows. Automatic labels instead
use explicit training roles; existing held-out assignments are never redrawn."""

from __future__ import annotations

import numpy as np

FAMILIES = ["9lawi", "zeb", "fuck"]

SPLITS = ["train", "val", "test"]


def assign_recordings(rows, seed=20261015, trials=20000):
    """Select a seeded group split using label counts only, never model results."""
    if any(row.get("human_reviewed") is not True for row in rows):
        raise ValueError("Historical splitting requires human-reviewed labels only")
    if any(row.get("split") not in (None, "unassigned") for row in rows):
        raise ValueError("Do not redraw existing recording assignments")
    return _assign_reviewed_recordings(rows, seed, trials)


def _assign_reviewed_recordings(rows, seed=20261015, trials=20000):
    recording_ids = sorted({row["recording_id"] for row in rows})
    recording_count = len(recording_ids)
    if recording_count < 18:
        raise ValueError("Too few recordings for this split protocol")

    recording_index = {
        recording_id: index for index, recording_id in enumerate(recording_ids)
    }
    features = np.zeros((recording_count, 6))
    family_presence = np.zeros((recording_count, 3))

    for row in rows:
        index = recording_index[row["recording_id"]]
        if row["label"] == 1:
            family_index = FAMILIES.index(row["family"])
            features[index, family_index] += 1
            features[index, 5] += 1
            family_presence[index, family_index] = 1
        elif row["label"] == 0:
            features[index, 3] += 1
            features[index, 4] += row.get("category") == "similar_sound"
        else:
            raise ValueError("Unreviewed row cannot be split")

    validation_count = round(recording_count * 0.15)
    test_count = round(recording_count * 0.15)
    split_sizes = [
        recording_count - validation_count - test_count,
        validation_count,
        test_count,
    ]
    fractions = np.array([0.7, 0.15, 0.15])
    desired = fractions[:, None] * features.sum(0)[None, :]

    rng = np.random.default_rng(seed)
    best_groups = None
    best_score = float("inf")
    for _ in range(trials):
        permutation = rng.permutation(recording_count)
        groups = np.split(permutation, np.cumsum(split_sizes)[:-1])
        counts = np.stack([features[group].sum(0) for group in groups])
        family_recordings = np.stack(
            [family_presence[group].sum(0) for group in groups]
        )

        if (
            (family_recordings < np.array([[6], [2], [2]])).any()
            or (counts[1:, :3] < 3).any()
            or (counts[:, 3] < 1).any()
        ):
            continue

        score = float(
            np.square((counts - desired) / np.maximum(desired, 1)).sum()
        )
        if score < best_score:
            best_score = score
            best_groups = [group.copy() for group in groups]

    if best_groups is None:
        raise ValueError("No feasible recording split; collect more examples")

    assignments = {
        recording_ids[index]: split
        for split, group in zip(SPLITS, best_groups)
        for index in group
    }
    protocol = {
        "seed": seed,
        "search_trials": trials,
        "objective": best_score,
        "requested_fractions": dict(zip(SPLITS, fractions.tolist())),
        "recording_counts": dict(zip(SPLITS, split_sizes)),
        "method": (
            "Seeded group assignment; minimize relative deviations in "
            "target-family, class and similar-sound-negative counts. "
            "No model performance used."
        ),
        "minimum_family_recordings": {"train": 6, "val": 2, "test": 2},
        "minimum_family_clips_val_test": 3,
    }
    return assignments, protocol


def validate_roles(recordings, roster):
    """Check the whole input roster before any recording is opened.

    The roster must include existing held-out recordings and their source hashes.
    Only train rows are admitted here. A reserve remains a reserve, not validation.
    """
    roles = {"train", "val", "test", "reserve"}
    seen_hashes = {}
    for recording_id, entry in roster.items():
        if entry.get("role") not in roles:
            raise ValueError(f"Invalid recording role: {recording_id}")
        sha = entry.get("source_sha256")
        if not isinstance(sha, str) or len(sha) != 64 or any(
            char not in "0123456789abcdef" for char in sha
        ):
            raise ValueError("Each roster entry needs a lowercase source SHA-256")
        if sha in seen_hashes:
            raise ValueError("Duplicate recording content in roster")
        seen_hashes[sha] = recording_id

    seen = set()
    for recording in recordings:
        recording_id = recording["recording_id"]
        if recording_id in seen or recording_id not in roster:
            raise ValueError("Duplicate or unknown recording ID")
        seen.add(recording_id)
        entry = roster[recording_id]
        if entry["role"] != "train":
            raise ValueError("Builder accepts training recordings only; preserve reserves")
        if recording.get("source_sha256") != entry["source_sha256"]:
            raise ValueError("Source hash does not match the recording roster")
