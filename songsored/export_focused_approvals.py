"""Export exact approved preview bytes to a separate, unsplit collection.

This never changes listening decisions, existing datasets, or training splits.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import tempfile
import re

import soundfile as sf


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def export(queue_dir: Path, destination: Path):
    queue_dir, destination = queue_dir.resolve(), destination.resolve()
    if destination.exists():
        raise ValueError("Destination already exists; preserve the previous export")

    queue_bytes = (queue_dir / "queue.json").read_bytes()
    review_bytes = (queue_dir / "reviews.json").read_bytes()
    candidates = json.loads(queue_bytes)["candidates"]
    reviews = json.loads(review_bytes)
    prepared = []
    sources = {}
    identifiers = set()

    for item in candidates:
        identifier = item["id"]
        if identifier in identifiers or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
            raise ValueError("Duplicate or unsafe candidate ID")
        identifiers.add(identifier)

        review = reviews.get(identifier, {})
        if review.get("decision") != "good":
            continue

        serialized_item = json.dumps(item, sort_keys=True).encode()
        expected_item = hashlib.sha256(serialized_item).hexdigest()
        if (
            review.get("item_sha256") != expected_item
            or item["source_type"] != "full_mix"
        ):
            raise ValueError(f'Stale or non-full-mix approval: {identifier}')

        context = queue_dir / item["context_path"]
        if not context.resolve().is_relative_to(queue_dir):
            raise ValueError("Context path must remain inside the queue")
        if not re.fullmatch(r"[0-9a-f]{64}", review.get("clip_sha256", "")):
            raise ValueError("Invalid preview hash")
        clip = queue_dir / "previews" / (review["clip_sha256"] + ".wav")
        if (
            digest(context) != item["context_sha256"]
            or digest(clip) != review["clip_sha256"]
        ):
            raise ValueError(f'Audio changed: {identifier}')

        source = Path(item["source_path"])
        source_key = str(source)
        if source_key not in sources:
            sources[source_key] = digest(source)
        actual_source_hash = sources[source_key]
        if actual_source_hash != item["source_sha256"]:
            raise ValueError(f'Source changed: {identifier}')

        info = sf.info(clip)
        actual_format = (info.samplerate, info.channels, info.frames, info.subtype)
        expected_format = (22050, 1, 22050, "PCM_16")
        if actual_format != expected_format:
            raise ValueError(f'Unexpected preview format: {identifier}')

        start, end = review["start"], review["end"]
        if not all(type(value) in (int, float) and math.isfinite(value) for value in (
            start, end, review["crop_start"], review["crop_end"]
        )):
            raise ValueError("Review boundaries must be finite numbers")
        if (
            not item["context_start"] <= start < end <= item["context_end"]
            or end - start > 1
        ):
            raise ValueError(f'Invalid reviewed span: {identifier}')

        audio, sr = sf.read(context, dtype='float32')
        if audio.ndim != 1 or sr != 22050 or len(audio) < sr:
            raise ValueError("Expected mono 22050 Hz context of at least one second")
        midpoint = (start + end) / 2
        first = max(
            0,
            min(
                round((midpoint - item["context_start"] - 0.5) * sr),
                len(audio) - sr,
            ),
        )
        crop_start = item["context_start"] + first / sr
        if not crop_start <= start < end <= crop_start + 1:
            raise ValueError(f'Target outside crop: {identifier}')
        if (
            abs(crop_start - review["crop_start"]) > 1e-8
            or abs(review["crop_end"] - crop_start - 1) > 1e-8
        ):
            raise ValueError(f'Crop coordinates changed: {identifier}')

        buffer = io.BytesIO()
        sf.write(
            buffer,
            audio[first : first + sr],
            sr,
            format="WAV",
            subtype="PCM_16",
        )
        if hashlib.sha256(buffer.getvalue()).hexdigest() != review["clip_sha256"]:
            raise ValueError(f'Preview does not reproduce: {identifier}')

        row = {
            "id": identifier,
            "path": f"clips/{identifier}.wav",
            "label": 1,
            "target": item["target"],
            "family": item["family"],
            "split": "unassigned",
            "recording_id": item.get("recording_id", "sha256-" + item["acquisition_sha256"]),
            "human_reviewed": True,
            "title": item["title"],
            "start": review["crop_start"],
            "end": review["crop_end"],
            "target_start": start,
            "target_end": end,
            "sha256": review["clip_sha256"],
            "source_path": str(source),
            "source_sha256": actual_source_hash,
            "acquisition_sha256": item["acquisition_sha256"],
            "source_url": item["source_url"],
            "lyric_url": item["lyric_url"],
            "review": review,
            "candidate": item,
        }
        if item.get("recording_role") is not None:
            if item["recording_role"] != "train":
                raise ValueError("This builder queue must retain its training assignment")
            row.update(split="train", training_only=True, evaluation_eligible=False)
        prepared.append((clip, row))

    if not prepared:
        raise ValueError('No approved clips')

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=destination.name + ".tmp-",
            dir=destination.parent,
        )
    )
    try:
        (stage / "clips").mkdir()
        for clip, row in prepared:
            exported_clip = stage / row["path"]
            shutil.copyfile(clip, exported_clip)
            if digest(exported_clip) != row["sha256"]:
                raise ValueError("Export copy failed verification")

        (stage / "source_queue.json").write_bytes(queue_bytes)
        (stage / "source_reviews.json").write_bytes(review_bytes)
        split_status = "train" if all(row["split"] == "train" for _, row in prepared) else "unassigned"
        metadata = {
            "version": 1,
            "task": "reviewed focused-target positive collection",
            "sample_rate": 22050,
            "clip_samples": 22050,
            "split_status": split_status,
            "note": (
                "Exact reviewed full-mix previews. Assign recording-separated "
                "splits and add reviewed negatives before training."
            ),
            "clips": [row for _, row in prepared],
        }
        (stage / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
        fields = ["path", "label", "recording_id", "split", "target", "family", "sha256"]
        with (stage / "collection.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(metadata["clips"])

        recordings = {row["recording_id"] for _, row in prepared}
        families = sorted({row["family"] for _, row in prepared})
        receipt = {
            "count": len(prepared),
            "recordings": len(recordings),
            "family_counts": {
                family: sum(row["family"] == family for _, row in prepared)
                for family in families
            },
            "queue_sha256": hashlib.sha256(queue_bytes).hexdigest(),
            "reviews_sha256": hashlib.sha256(review_bytes).hexdigest(),
            "exact_preview_bytes_verified": True,
            "existing_datasets_modified": False,
            "split": split_status,
        }
        (stage / "export_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        destination.mkdir()
        for child in stage.iterdir():
            child.rename(destination / child.name)
        stage.rmdir()
    except BaseException:
        shutil.rmtree(stage)
        raise
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.queue, args.destination), indent=2))
