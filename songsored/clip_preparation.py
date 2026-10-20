"""Reproduce one-second full-mix previews and bind them to review metadata.

Extracted from the existing review lab; the web server is not included."""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path

import soundfile as sf


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def item_hash(item):
    serialized = json.dumps(item, sort_keys=True).encode()
    return hashlib.sha256(serialized).hexdigest()


def candidate(folder, identifier):
    candidates = read(folder / "queue.json")["candidates"]
    item = next(
        (candidate for candidate in candidates if candidate["id"] == identifier),
        None,
    )
    if item is None:
        raise ValueError("Unknown candidate")
    return item


def preview(folder, payload):
    item = candidate(folder, payload.get("id"))
    start, end = payload.get("start"), payload.get("end")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in (start, end)
    ):
        raise ValueError("Enter finite numeric boundaries")
    if (
        not item["context_start"] <= start < end <= item["context_end"]
        or end - start > 1
    ):
        raise ValueError("Choose a target span of at most one second inside the context")

    path = folder / item["context_path"]
    if not path.resolve().is_relative_to(folder.resolve()):
        raise ValueError("Context path must remain inside the queue")
    if digest(path) != item["context_sha256"]:
        raise ValueError("Context audio changed; rebuild this candidate")
    audio, sr = sf.read(path, dtype='float32')
    if audio.ndim != 1 or sr != 22050 or len(audio) < sr:
        raise ValueError("Expected mono 22050 Hz context of at least one second")

    midpoint = (start + end) / 2
    first = max(
        0,
        min(round((midpoint - item["context_start"] - 0.5) * sr), len(audio) - sr),
    )
    crop_start = item["context_start"] + first / sr
    if not crop_start <= start < end <= crop_start + 1:
        raise ValueError("Target does not fit in the one-second crop")

    buffer = io.BytesIO()
    sf.write(buffer, audio[first : first + sr], sr, format="WAV", subtype="PCM_16")
    content = buffer.getvalue()
    sha = hashlib.sha256(content).hexdigest()
    name = sha + ".wav"
    (folder / "previews").mkdir(exist_ok=True)
    (folder / "previews" / name).write_bytes(content)
    return dict(
        id=item["id"],
        start=start,
        end=end,
        crop_start=crop_start,
        crop_end=crop_start + 1,
        clip_sha256=sha,
        item_sha256=item_hash(item),
        audio_url="/preview/" + name,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare a preview; this does not approve it.")
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    args = parser.parse_args()
    result = preview(args.queue, dict(id=args.id, start=args.start, end=args.end))
    print(json.dumps(result, indent=2))
