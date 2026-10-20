"""Build provisional training clips from supplied audio and prepared lyric rows.

This is new orchestration around the reviewed frozen alignment/crop helpers.
It does not fetch lyrics, solve native phonetic preparation, or certify labels.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import re
import shutil
import tempfile

import librosa
import numpy as np
import soundfile as sf

from .alignment_models import digest, load_model
from .candidate_routing import route_frozen_selection
from .clip_preparation import preview
from .ctc_alignment import align_local, align_transcript, choose, generate_emissions, passage
from .negative_candidates import clean_anchor, finite_span, forbidden_spans, overlaps, pick_pools
from .recording_split import validate_roles
from .target_matching import occurrences, tokenize


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_input(document):
    """Require explicit roles, prepared spellings and unambiguous family matches."""
    recordings = document["recordings"]
    roster = document["roster"]
    if not recordings:
        raise ValueError("Supply at least one training recording")
    validate_roles(recordings, roster)
    families = document["families"]
    if not families:
        raise ValueError("At least one target family is required")
    forms = set()
    for family, group in families.items():
        if not family or not group["forms"] or group["language"] not in {"ar", "fr", "es", "en"}:
            raise ValueError("Families need forms and an explicit native language")
        for form in group["forms"]:
            normalized = tuple(tokenize(form))
            if not normalized or normalized in forms:
                raise ValueError("Target forms must be nonempty and unique after normalization")
            forms.add(normalized)
    for recording in recordings:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", recording["recording_id"]):
            raise ValueError("Use simple recording IDs without path characters")
        if recording.get("source_type") != "full_mix":
            raise ValueError("Export input must be the full mix")
        if not recording["rows"]:
            raise ValueError("Prepared lyric rows are required")
        for row in recording["rows"]:
            original = row["original"]
            if tokenize(original) != [original]:
                raise ValueError("Each original row must be one normalized token")
            if not all(isinstance(row.get(key), str) for key in ("mms", "native", "language")):
                raise ValueError("Supply MMS spelling, native spelling and language for each row")
    return recordings


def acoustic_alignment(wave, rows, languages, *, allow_download=False):
    """Compute emissions serially, releasing each acoustic model before the next."""
    blobs = {}
    for engine, language in [("mms", "ar")] + [("native", lang) for lang in sorted(languages)]:
        model, processor, dictionary, blank, identity = load_model(
            engine, language, allow_download=allow_download
        )
        emission, times = generate_emissions(wave, model, processor, engine)
        blobs[(engine, language)] = dict(
            engine=engine, language=language, emission=emission, times=times,
            dictionary=dictionary, blank=blank, identity=identity,
        )
        del model, processor
        gc.collect()
    mms = blobs[("mms", "ar")]
    indices = align_transcript(
        mms["emission"], rows, mms["dictionary"], mms["blank"], "mms", "ar", mms["times"]
    )
    return blobs, indices


def propose(rows, targets, blobs, indices, duration):
    cases = []
    for target in targets:
        context = passage(rows, indices, target, duration)
        candidates = {}
        for engine in ("mms", "native"):
            language = "ar" if engine == "mms" else target["language"]
            candidates[engine] = align_local(
                blobs[(engine, language)], rows, target, context, duration
            )
        selection = choose(candidates, context, duration)
        cases.append({**target, "candidates": candidates, "selection": selection})
    # Separate occurrences cannot share/reverse selected time spans. Reject both.
    conflicts = set()
    for left_index, left in enumerate(cases):
        if left["selection"]["status"] != "ok":
            continue
        for right_index in range(left_index + 1, len(cases)):
            right = cases[right_index]
            if right["selection"]["status"] == "ok" and (
                left["token_end"] > right["token_start"]
                or left["selection"]["end"] > right["selection"]["start"]
            ):
                conflicts.update((left_index, right_index))
    for index in conflicts:
        cases[index]["selection"] = {
            "status": "unresolved", "reason": "Occurrence order or overlap conflict"
        }
    return cases


def negative_pool(words, indices, families, policy, duration, positive_windows):
    """Skip uncertain exclusions; require mapped clean neighbors and a 2 s guard."""
    forbidden = forbidden_spans(words, policy, {"families": families})
    forbidden_indices = {i for start, end in forbidden for i in range(start, end)}
    # A missing forbidden/gap anchor leaves its location unknown: no negatives.
    if "_gap_" in words or any(
        index not in indices or not finite_span(indices[index])
        for index in forbidden_indices
    ):
        return []
    guards = [(indices[i]["start"] - 2, indices[i]["end"] + 2) for i in forbidden_indices]
    guards.extend((start - 2, end + 2) for start, end in positive_windows)
    string_indices = {str(index): span for index, span in indices.items()}
    pool = []
    for index in range(len(words)):
        if not clean_anchor(words, string_indices, index, forbidden_indices):
            continue
        span = indices[index]
        start = (span["start"] + span["end"]) / 2 - .5
        end = start + 1
        if start < 0 or end > duration or overlaps(start, end, guards):
            continue
        pool.append(dict(start=start, end=end, anchor_token=index, category="ordinary_vocals"))
    return pool


def build_dataset(manifest, output, *, negatives_per_recording=0, allow_download=False,
                  alignment_backend=None):
    """Write a new training collection and low-confidence review queue.

    alignment_backend is an injection point for synthetic tests, not a CLI shortcut
    for bypassing the acoustic models. The CLI always uses the actual aligners.
    """
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Output exists; choose a new collection directory")
    if type(negatives_per_recording) is not int or not 0 <= negatives_per_recording <= 3:
        raise ValueError("Request between zero and three negatives per recording")
    document = json.loads(manifest.read_text())
    recordings = validate_input(document)
    # Validate every admitted source before any model or audio decoder is used.
    sources = {}
    acoustic_sources = {}
    for recording in recordings:
        path = (manifest.parent / recording["audio"]).resolve()
        if digest(path) != recording["source_sha256"]:
            raise ValueError("Source bytes differ from the admitted recording hash")
        sources[recording["recording_id"]] = path
        if "acoustic_audio" in recording:
            acoustic_path = (manifest.parent / recording["acoustic_audio"]).resolve()
            if digest(acoustic_path) != recording.get("acoustic_sha256"):
                raise ValueError("Acoustic input hash differs from its prepared manifest")
            acoustic_sources[recording["recording_id"]] = acoustic_path
    families = document["families"]
    specs = [(form, group["language"], "target", family)
             for family, group in families.items() for form in group["forms"]]
    backend = alignment_backend or acoustic_alignment
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".songsored-build-", dir=output.parent))
    try:
        (stage / "contexts").mkdir()
        (stage / "clips").mkdir()
        queue, exported, diagnostics, identities = [], [], [], {}
        seen_clip_hashes = set()
        for recording in recordings:
            rid = recording["recording_id"]
            source = sources[rid]
            full_mix, sample_rate = librosa.load(source, sr=22050, mono=True)
            if len(full_mix) < sample_rate or not np.isfinite(full_mix).all():
                raise ValueError("Full mix must contain at least one second of finite audio")
            duration = len(full_mix) / sample_rate
            context_path = stage / "contexts" / f"{rid}.wav"
            sf.write(context_path, full_mix, sample_rate, subtype="PCM_16")
            context_sha = digest(context_path)
            rows = recording["rows"]
            words = [row["original"] for row in rows]
            targets = occurrences(words, specs)
            wave = librosa.resample(full_mix, orig_sr=sample_rate, target_sr=16000)
            if rid in acoustic_sources:
                wave, acoustic_rate = sf.read(acoustic_sources[rid], dtype="float32")
                if acoustic_rate != 16000 or wave.ndim != 1 or not np.isfinite(wave).all():
                    raise ValueError("Prepared acoustic audio must be finite mono 16 kHz")
                if abs(len(wave) / 16000 - duration) > .001:
                    raise ValueError("Acoustic input and full mix must share the same timeline")
            languages = {target["language"] for target in targets}
            blobs, indices = backend(wave, rows, languages, allow_download=allow_download)
            cases = propose(rows, targets, blobs, indices, duration)
            identities[rid] = {f"{engine}:{lang}": blob["identity"]
                               for (engine, lang), blob in blobs.items()}
            identities[rid]["acoustic_input"] = (
                dict(kind="supplied_prepared_audio", sha256=recording["acoustic_sha256"])
                if rid in acoustic_sources
                else dict(kind="resampled_full_mix", sha256=recording["source_sha256"])
            )
            positive_windows = []
            all_items = []
            for case in cases:
                selected = case["selection"]
                route = route_frozen_selection(case, duration)
                item = dict(
                    id=f"{rid}-{case['id']}", recording_id=rid, recording_role="train",
                    target=case["target"],
                    family=case["policy_reason"], title=recording.get("title", rid),
                    source_path=str(source), source_sha256=recording["source_sha256"],
                    acquisition_sha256=recording["source_sha256"], source_type="full_mix",
                    source_url="", lyric_url="", context_path=f"contexts/{rid}.wav",
                    context_sha256=context_sha, context_start=0., context_end=duration,
                    selection=selected, candidates=case["candidates"], routing=route,
                )
                all_items.append(item)
                if selected["status"] == "ok":
                    positive_windows.append((selected["start"], selected["end"]))
                for candidate in case["candidates"].values():
                    if finite_span(candidate):
                        positive_windows.append((candidate["start"], candidate["end"]))
                diagnostics.append(dict(id=item["id"], routing=route, selection=selected))
            pool = negative_pool(words, indices, families, document.get("exclusion_policy", {"targets": []}),
                                 duration, positive_windows)
            picks = pick_pools({rid: pool}, negatives_per_recording,
                               {"ordinary_vocals": negatives_per_recording})
            for negative in picks[rid]:
                center = negative["start"] + .5
                all_items.append(dict(
                    id=f"{rid}-negative-{negative['anchor_token']}", recording_id=rid,
                    target="", family="none", context_path=f"contexts/{rid}.wav",
                    context_sha256=context_sha, context_start=0., context_end=duration,
                    source_sha256=recording["source_sha256"],
                    # A tiny anchor interval centers the same one-second crop.
                    selection=dict(status="ok", start=center - .01, end=center + .01),
                    routing=dict(route="auto_candidate", confidence_calibrated=False),
                    negative=True,
                ))
            write_json(stage / "queue.json", {"candidates": all_items})
            for item in all_items:
                if item["routing"]["route"] != "auto_candidate":
                    queue.append(item)
                    continue
                selected = item["selection"]
                try:
                    result = preview(stage, dict(id=item["id"], start=selected["start"], end=selected["end"]))
                except ValueError as exc:
                    if not item.get("negative"):
                        item["routing"] = dict(route="review_required", reasons=[str(exc)], confidence_calibrated=False)
                        queue.append(item)
                    continue
                sha = result["clip_sha256"]
                if sha in seen_clip_hashes:
                    continue
                seen_clip_hashes.add(sha)
                filename = f"clips/{item['id']}.wav"
                shutil.copyfile(stage / "previews" / f"{sha}.wav", stage / filename)
                exported.append(dict(
                    id=item["id"], path=filename, label=int(not item.get("negative", False)),
                    recording_id=rid, family=item["family"], split="train", sha256=sha,
                    source_sha256=recording["source_sha256"], human_reviewed=False,
                    label_source="automatic_unreviewed", training_only=True, evaluation_eligible=False,
                    start=result["crop_start"], end=result["crop_end"],
                ))
            if digest(source) != recording["source_sha256"]:
                raise ValueError("Source changed during processing")
            if rid in acoustic_sources and digest(acoustic_sources[rid]) != recording["acoustic_sha256"]:
                raise ValueError("Acoustic input changed during processing")
        write_json(stage / "queue.json", {"candidates": queue})
        write_json(stage / "reviews.json", {})
        write_json(stage / "metadata.json", {"items": exported})
        fields = ["path", "label", "recording_id", "split"]
        with (stage / "train_manifest.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(exported)
        write_json(stage / "receipt.json", dict(
            protocol="neighbor-passage-v1-public-adapter", manifest_sha256=digest(manifest),
            models=identities, exported=len(exported), review_required=len(queue),
            automatic_positives=sum(row["label"] for row in exported),
            automatic_negatives=sum(row["label"] == 0 for row in exported),
            confidence_calibrated=False, diagnostics=diagnostics,
        ))
        # mkdir is the no-clobber reservation; do not replace a raced output folder.
        output.mkdir()
        for child in stage.iterdir():
            child.rename(output / child.name)
        return output
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negatives-per-recording", type=int, default=0)
    parser.add_argument("--download-models", action="store_true")
    args = parser.parse_args()
    output = build_dataset(args.manifest, args.output,
                           negatives_per_recording=args.negatives_per_recording,
                           allow_download=args.download_models)
    print((output / "receipt.json").read_text())


if __name__ == "__main__":
    main()
