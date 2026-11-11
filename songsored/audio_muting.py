"""Combine selected detections and mute a copy of the original audio."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf


def combine_detections(*groups):
    """Keep detections from either branch and merge touching or overlapping spans.

    Each event needs start and end in seconds on the original song timeline.
    Apply each detector's calibrated cutoff before calling this function.
    Scores from different detectors are not averaged or compared here.
    """
    spans = []
    for group in groups:
        for event in group:
            start, end = float(event["start"]), float(event["end"])
            if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                raise ValueError("Expected a finite, nonempty detection interval.")
            spans.append((start, end))

    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1]["end"]:
            merged[-1]["end"] = max(end, merged[-1]["end"])
        else:
            merged.append({"start": start, "end": end})
    return merged


def mute_intervals(audio, sample_rate, events):
    """Return a muted copy and its sample intervals, with exclusive ends.

    Mono input has shape (samples,); multichannel input has shape
    (samples, channels). Every channel is muted over the same intervals.
    """
    audio = np.asarray(audio)
    if (
        audio.ndim not in (1, 2)
        or not audio.size
        or audio.dtype.kind not in "fiu"
        or not np.isfinite(audio).all()
    ):
        raise ValueError("Expected nonempty, finite mono or multichannel audio.")
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, (int, np.integer))
        or sample_rate <= 0
    ):
        raise ValueError("Sample rate must be a positive integer.")

    spans = []
    for event in combine_detections(events):
        left = max(0, math.floor(event["start"] * sample_rate))
        right = min(len(audio), math.ceil(event["end"] * sample_rate))
        if left < right:
            # Rounding can make separate time intervals touch at sample level.
            if spans and left <= spans[-1][1]:
                spans[-1][1] = max(right, spans[-1][1])
            else:
                spans.append([left, right])

    result = audio.copy()
    for left, right in spans:
        result[left:right] = 0
    return result, spans


def render_muted_audio(source, destination, event_path, *detection_groups):
    """Write a new muted WAV and JSON event list from already selected detections.

    No models run here. Audio keeps its sample rate and channels. No padding,
    fades or bleeps are added. Existing files cannot be replaced.
    """
    source, destination, event_path = map(Path, (source, destination, event_path))
    if destination.suffix.lower() != ".wav":
        raise ValueError("Choose a .wav output.")
    if len({path.resolve() for path in (source, destination, event_path)}) != 3:
        raise ValueError("Source, audio output and event list must have distinct paths.")
    if destination.exists() or event_path.exists():
        raise FileExistsError("Choose new output paths.")

    events = combine_detections(*detection_groups)
    audio, sample_rate = sf.read(source, dtype="float64", always_2d=True)
    muted, spans = mute_intervals(audio, sample_rate, events)
    record = {
        "sample_rate": sample_rate,
        "channels": audio.shape[1],
        "frames": len(audio),
        "detections": events,
        "mute_sample_intervals": spans,
    }
    subtype = sf.info(source).subtype
    if subtype not in ("PCM_U8", "PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"):
        subtype = "FLOAT"

    # Finish both files before exposing them. Linking refuses to overwrite a file.
    with (
        tempfile.NamedTemporaryFile(dir=destination.parent) as audio_file,
        tempfile.NamedTemporaryFile(dir=event_path.parent) as events_file,
    ):
        sf.write(audio_file.name, muted, sample_rate, format="WAV", subtype=subtype)
        events_file.write((json.dumps(record, indent=2, allow_nan=False) + "\n").encode())
        events_file.flush()
        os.link(audio_file.name, destination)
        try:
            os.link(events_file.name, event_path)
        except OSError:
            destination.unlink()
            raise
    return record
