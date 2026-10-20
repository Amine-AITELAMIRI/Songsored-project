"""Frozen neighbor-passage-v1 CTC alignment and boundary selection.

Extracted from the existing experiments. Models and prepared phonetic rows
are supplied explicitly; no model or recording is loaded on import."""

from __future__ import annotations

import copy
import re

import numpy as np

def normalize_words(words, primary):
    import uroman

    romanizer = uroman.Uroman()
    rows = []
    digit_map = str.maketrans(
        {"2": "'", "3": "a", "5": "kh", "7": "h", "8": "gh", "9": "q"}
    )

    for word in words:
        is_arabic = bool(re.search("[\u0600-\u06ff]", word))
        has_arabizi_digits = (
            not is_arabic
            and not word.isdigit()
            and any(character.isdigit() for character in word)
        )
        transformed = word.translate(digit_map) if has_arabizi_digits else word
        romanized = re.sub(
            "[^a-z']",
            "",
            romanizer.romanize_string(transformed).lower(),
        )
        rows.append(
            {
                "original": word,
                "mms": romanized,
                "native": word,
                "language": "ar" if is_arabic else ("fr" if primary == "ar" else primary),
                "heuristic_arabizi": transformed != word,
            }
        )

    return rows


def generate_emissions(wave, model, processor, engine, sr=16000):
    import torch

    chunks = []
    times = []
    for core_start in range(0, len(wave), 20 * sr):
        core_end = min(len(wave), core_start + 20 * sr)
        clip_start = max(0, core_start - 2 * sr)
        clip_end = min(len(wave), core_end + 2 * sr)
        clip = wave[clip_start:clip_end]

        with torch.inference_mode():
            if processor:
                emission = model(
                    **processor(
                        clip,
                        sampling_rate=sr,
                        return_tensors="pt",
                    )
                ).logits[0].log_softmax(-1)
            else:
                emission = model(torch.from_numpy(clip)[None])[0][0]
                if engine != "mms":
                    emission = emission.log_softmax(-1)

        edges = np.linspace(clip_start / sr, clip_end / sr, len(emission) + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        keep = (centers >= core_start / sr) & (centers < core_end / sr)
        chunks.append(emission.cpu()[keep])
        times.extend(zip(edges[:-1][keep], edges[1:][keep]))
        print(engine, "audio seconds", round(core_end / sr), flush=True)

    return torch.cat(chunks), np.asarray(times)


def align_transcript(emission, rows, dictionary, blank, engine, language, times):
    import torch
    import torchaudio.functional as F

    star = emission.shape[-1]
    transcript = [star]
    token_positions = {}
    for row_index, row in enumerate(rows):
        word = row["mms"] if engine == "mms" else row["native"]
        supported = (
            bool(word)
            and all(character in dictionary for character in word)
            and (engine == "mms" or row["language"] == language)
        )
        if not supported:
            if transcript[-1] != star:
                transcript.append(star)
            continue

        token_start = len(transcript)
        transcript.extend(dictionary[character] for character in word)
        token_positions[row_index] = (token_start, len(transcript))

    if transcript[-1] != star:
        transcript.append(star)
    if not token_positions:
        return {}

    emission_with_star = torch.cat(
        [emission, emission.new_zeros((len(emission), 1))],
        dim=-1,
    )
    path, scores = F.forced_align(
        emission_with_star[None].contiguous(),
        torch.tensor([transcript], dtype=torch.int32),
        blank=blank,
    )
    spans = F.merge_tokens(path[0], scores[0].exp())
    if len(spans) != len(transcript):
        raise ValueError("Transcript alignment lost token identity")

    return {
        row_index: {
            "start": float(times[spans[start].start, 0]),
            "end": float(times[spans[end - 1].end - 1, 1]),
        }
        for row_index, (start, end) in token_positions.items()
    }


def span_for(indices, occurrence):
    spans = [
        indices.get(index)
        for index in range(occurrence["token_start"], occurrence["token_end"])
    ]
    if not all(spans):
        return {
            "status": "unavailable",
            "reason": "Unsupported or missing transcript tokens",
        }

    start = spans[0]["start"]
    end = spans[-1]["end"]
    if not 0.04 <= end - start <= 2.5:
        return {
            "status": "unavailable",
            "reason": "Collapsed or implausibly long alignment",
            "start": start,
            "end": end,
        }
    return {"status": "ok", "start": round(start, 4), "end": round(end, 4)}


def passage(rows, indices, c, duration):
    target_start, target_end = c["token_start"], c["token_end"]
    left = []
    right = []

    for index in range(target_start - 1, -1, -1):
        if rows[index]["original"] == "_gap_":
            break
        left.append(index)
        if len(left) == 4:
            break

    for index in range(target_end, len(rows)):
        if rows[index]["original"] == "_gap_":
            break
        right.append(index)
        if len(right) == 4:
            break

    context = sorted(left + right)
    timed = [(index, indices[index]) for index in context if index in indices]
    if not timed:
        return {"status": "unavailable", "reason": "No surrounding word anchors"}

    usable = [
        (index, span)
        for index, span in timed
        if 0.02 <= span["end"] - span["start"] <= 2.5
    ]
    if not usable:
        return {
            "status": "unavailable",
            "reason": "No plausible surrounding word anchors",
        }

    # Keep the passage bounds independent of the target alignment.
    start = min(span["start"] for _, span in usable)
    end = max(span["end"] for _, span in usable)
    if not any(index < target_start for index, _ in usable):
        start = max(0, start - 1.5)
    if not any(index >= target_end for index, _ in usable):
        end = min(duration, end + 1.5)

    warnings = []
    if len(usable) != len(timed):
        warnings.append("An adjacent lyric word has an implausible duration")

    # Check continuity on each side without treating the target as a gap.
    sides = [
        [span for index, span in timed if index < target_start],
        [span for index, span in timed if index >= target_end],
    ]
    for side in sides:
        if any(
            later["start"] - earlier["end"] > 1.5
            for earlier, later in zip(side, side[1:])
        ):
            warnings.append("Surrounding lyrics contain a large internal time gap")

    if end - start > 15:
        return {
            "status": "unavailable",
            "reason": "Surrounding passage exceeds 15 seconds",
            "warnings": warnings,
        }

    return {
        "status": "ok",
        "start": start,
        "end": end,
        "lo": min(context + [target_start]),
        "hi": max(context + [target_end - 1]) + 1,
        "anchors": [{"token": index, **span} for index, span in timed],
        "warnings": warnings,
        "left_neighbor": indices.get(target_start - 1),
        "right_neighbor": indices.get(target_end),
    }


def character_support(blob, row_subset, aligned, target):
    """Mean peak posterior per target character; diagnostic, not probability of correctness."""
    scores = []
    for index in range(target["token_start"], target["token_end"]):
        span = aligned.get(index)
        if not span:
            return 0.0

        text = (
            row_subset[index]["mms"]
            if blob["engine"] == "mms"
            else row_subset[index]["native"]
        )
        character_ids = [
            blob["dictionary"][character]
            for character in text
            if character in blob["dictionary"]
        ]
        in_span = (blob["times"][:, 1] > span["start"]) & (
            blob["times"][:, 0] < span["end"]
        )
        if not character_ids or not in_span.any():
            return 0.0

        values = (
            blob["emission"][in_span][:, character_ids]
            .exp()
            .max(dim=0)
            .values
        )
        scores.extend(values.tolist())

    return float(np.mean(scores)) if scores else 0.0


def align_local(blob, rows, c, p, duration):
    if p["status"] != "ok":
        return {"status": "unavailable", "reason": p["reason"]}

    subset = rows[p["lo"] : p["hi"]]
    target = {
        "token_start": c["token_start"] - p["lo"],
        "token_end": c["token_end"] - p["lo"],
    }
    variants = []
    for margin in [0.35, 0.70, 1.05]:
        start = max(0, p["start"] - margin)
        end = min(duration, p["end"] + margin)
        in_window = (blob["times"][:, 1] > start) & (
            blob["times"][:, 0] < end
        )
        try:
            indices = align_transcript(
                blob["emission"][in_window],
                subset,
                blob["dictionary"],
                blob["blank"],
                blob["engine"],
                blob["language"],
                blob["times"][in_window],
            )
            prediction = span_for(indices, target)
            if prediction["status"] == "ok":
                prediction["character_support"] = character_support(
                    blob, subset, indices, target
                )
        except (ValueError, RuntimeError) as exc:
            prediction = {"status": "unavailable", "reason": str(exc)}
        variants.append({"margin": margin, **prediction})

    center = copy.deepcopy(variants[1])
    center.pop("margin")
    all_usable = all(variant["status"] == "ok" for variant in variants)
    spread = (
        max(
            max(variant[key] for variant in variants)
            - min(variant[key] for variant in variants)
            for key in ["start", "end"]
        )
        if all_usable
        else None
    )
    center.update(
        stable=all_usable and spread <= 0.12,
        endpoint_spread=spread,
        variants=variants,
    )
    return center


def choose(candidates, p, duration):
    evidence = {"passage": p, "support_threshold": 0.10}

    def reject(reason):
        return {"status": "unresolved", "reason": reason, "evidence": evidence}

    if p["status"] != "ok":
        return reject(p["reason"])
    if p["warnings"]:
        return reject("Passage anchors are unreliable: " + "; ".join(p["warnings"]))
    if any(candidate["status"] != "ok" for candidate in candidates.values()):
        return reject("A local model could not return a usable span")
    if any(not candidate["stable"] for candidate in candidates.values()):
        return reject("Changing passage margins changes the target boundary")
    if any(candidate["character_support"] < 0.10 for candidate in candidates.values()):
        return reject("Low acoustic character support; word presence requires review")

    mms = candidates["mms"]
    native = candidates["native"]
    endpoint_disagreement = max(
        abs(mms[key] - native[key]) for key in ["start", "end"]
    )
    overlap_gap = max(mms["start"], native["start"]) - min(
        mms["end"], native["end"]
    )
    if endpoint_disagreement > 0.30 or overlap_gap > 0.08:
        return reject("Local proposals still disagree about the word location")

    proposal_start = min(mms["start"], native["start"])
    proposal_end = max(mms["end"], native["end"])
    left = p.get("left_neighbor")
    right = p.get("right_neighbor")
    floor = max(0, (left["start"] + left["end"]) / 2) if left else 0
    ceiling = min(duration, (right["start"] + right["end"]) / 2) if right else duration
    if floor > proposal_start or ceiling < proposal_end:
        return reject("Target overlaps the center of a neighboring lyric word")

    start = max(floor, proposal_start - 0.06)
    end = min(ceiling, proposal_end + 0.06)
    if end - start > 2.5:
        return reject("Refined envelope is implausibly long")

    return {
        "status": "ok",
        "start": round(start, 4),
        "end": round(end, 4),
        "route": "passage_envelope",
        "reason": "Stable local agreement with a bounded 60 ms boundary envelope",
        "evidence": evidence,
    }
