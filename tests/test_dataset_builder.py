"""Synthetic CTC scores and generated audio, never song data or real weights."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

pytest.importorskip("torchaudio")
from songsored.alignment_models import digest, load_model
from songsored.build_dataset import build_dataset, negative_pool, propose, validate_input
from songsored.clip_preparation import preview
from songsored.ctc_alignment import align_transcript
from songsored.export_focused_approvals import export
from songsored.recording_split import assign_recordings, validate_roles


def example(tmp_path):
    rng = np.random.default_rng(12)
    audio = tmp_path / "generated.wav"
    sf.write(audio, rng.normal(0, .05, 20 * 22050).astype(np.float32), 22050)
    sha = digest(audio)
    rows = [dict(original=letter, mms=letter, native=letter, language="en")
            for letter in "abcdefghijklmno"]
    document = dict(
        recordings=[dict(recording_id="generated", audio=audio.name, source_type="full_mix",
                         source_sha256=sha, rows=rows)],
        roster={"generated": dict(role="train", source_sha256=sha)},
        families={"example": dict(forms=["c", "j"], language="en")},
    )
    manifest = tmp_path / "prepared.json"
    manifest.write_text(json.dumps(document))
    return manifest, document


def synthetic_backend(wave, rows, languages, **kwargs):
    # Peaks represent deliberately supplied alignment evidence, not recognition.
    emission = torch.full((1000, 16), -12.)
    emission[:, 0] = 2.
    for i in range(15):
        start = 50 + i * 50
        emission[start:start + 10, i + 1] = 12.
    emission = emission.log_softmax(-1)
    times = np.column_stack((np.arange(1000) / 50, np.arange(1, 1001) / 50))
    dictionary = {letter: i + 1 for i, letter in enumerate("abcdefghijklmno")}
    common = dict(emission=emission, times=times, dictionary=dictionary, blank=0,
                  identity={"name": "synthetic test emissions, not a pretrained model"})
    mms = dict(common, engine="mms", language="ar")
    native = dict(common, engine="native", language="en")
    # Deliberately unsupported target c should go to review, not become label 0.
    native["dictionary"] = {key: value for key, value in dictionary.items() if key != "c"}
    indices = align_transcript(emission, rows, dictionary, 0, "mms", "ar", times)
    return {("mms", "ar"): mms, ("native", "en"): native}, indices


def test_ctc_to_crop_and_review_then_exact_approved_export(tmp_path):
    manifest, document = example(tmp_path)
    destination = build_dataset(manifest, tmp_path / "built", negatives_per_recording=2,
                                alignment_backend=synthetic_backend)
    receipt = json.loads((destination / "receipt.json").read_text())
    assert receipt["automatic_positives"] == 1
    assert receipt["review_required"] == 1
    assert receipt["automatic_negatives"] >= 1
    items = json.loads((destination / "metadata.json").read_text())["items"]
    for item in items:
        assert item["split"] == "train" and item["training_only"]
        assert not item["human_reviewed"] and not item["evaluation_eligible"]
        info = sf.info(destination / item["path"])
        assert (info.frames, info.channels, info.samplerate) == (22050, 1, 22050)
        assert digest(destination / item["path"]) == item["sha256"]
    queue = json.loads((destination / "queue.json").read_text())["candidates"]
    assert queue[0]["target"] == "c"
    review = preview(destination, dict(id=queue[0]["id"], start=3., end=3.2))
    review["decision"] = "good"
    (destination / "reviews.json").write_text(json.dumps({review["id"]: review}))
    reviewed = tmp_path / "approved"
    result = export(destination, reviewed)
    assert result["count"] == 1
    row = json.loads((reviewed / "dataset.json").read_text())["clips"][0]
    assert row["human_reviewed"] and row["recording_id"] == "generated"
    assert row["split"] == "train"
    assert (reviewed / row["path"]).read_bytes() == (
        destination / "previews" / f"{review['clip_sha256']}.wav"
    ).read_bytes()
    with pytest.raises(ValueError, match="exists"):
        export(destination, reviewed)


@pytest.mark.parametrize("role", ["val", "test", "reserve"])
def test_held_out_rejected_before_any_file_or_model_access(tmp_path, monkeypatch, role):
    manifest, document = example(tmp_path)
    document["roster"]["generated"]["role"] = role
    document["recordings"][0]["audio"] = "must-not-open.wav"
    manifest.write_text(json.dumps(document))
    monkeypatch.setattr("songsored.build_dataset.digest", lambda path: pytest.fail("opened source"))
    with pytest.raises(ValueError, match="training recordings only"):
        build_dataset(manifest, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_roster_detects_duplicate_bytes_and_unknown_identity(tmp_path):
    _, document = example(tmp_path)
    roster = document["roster"]
    roster["another-name"] = {**roster["generated"], "role": "test"}
    with pytest.raises(ValueError, match="Duplicate recording content"):
        validate_input(document)
    del roster["another-name"]
    document["recordings"][0]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not match"):
        validate_input(document)


def test_no_overwrite_and_no_partial_export_on_failed_alignment(tmp_path):
    manifest, _ = example(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep").write_text("keep")
    with pytest.raises(FileExistsError):
        build_dataset(manifest, output)
    assert (output / "keep").read_text() == "keep"
    def failed(*args, **kwargs):
        raise RuntimeError("alignment failed")
    with pytest.raises(RuntimeError, match="alignment failed"):
        build_dataset(manifest, tmp_path / "failed", alignment_backend=failed)
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".songsored-build-*"))


def test_wrong_source_hash_fails_before_acoustics(tmp_path):
    manifest, document = example(tmp_path)
    (tmp_path / "generated.wav").write_bytes(b"changed")
    def forbidden(*args, **kwargs):
        pytest.fail("must not align changed audio")
    with pytest.raises(ValueError, match="Source bytes differ"):
        build_dataset(manifest, tmp_path / "out", alignment_backend=forbidden)


def test_missing_forbidden_timing_disables_negatives():
    words = list("abcdefghijk")
    indices = {i: dict(start=i + 1., end=i + 1.2) for i in range(len(words))}
    families = {"example": dict(forms=["c"])}
    assert negative_pool(words, indices, families, {"targets": []}, 20, [])
    del indices[2]
    assert negative_pool(words, indices, families, {"targets": []}, 20, []) == []


def test_historical_split_rejects_automatic_or_preassigned_labels():
    with pytest.raises(ValueError, match="human-reviewed"):
        assign_recordings([dict(human_reviewed=False)])
    with pytest.raises(ValueError, match="redraw"):
        assign_recordings([dict(human_reviewed=True, split="test")])


def test_models_never_download_implicitly(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    with pytest.raises(FileNotFoundError, match="Missing cached aligner"):
        load_model("mms", "ar")
    with pytest.raises(FileNotFoundError, match="Missing cached aligner"):
        load_model("native", "fr")


def test_stale_review_is_not_exported(tmp_path):
    manifest, _ = example(tmp_path)
    output = build_dataset(manifest, tmp_path / "built", alignment_backend=synthetic_backend)
    item = json.loads((output / "queue.json").read_text())["candidates"][0]
    review = preview(output, dict(id=item["id"], start=3., end=3.2))
    review.update(decision="good", item_sha256="0" * 64)
    (output / "reviews.json").write_text(json.dumps({item["id"]: review}))
    with pytest.raises(ValueError, match="Stale"):
        export(output, tmp_path / "invalid")


def test_conflicting_occurrence_order_requires_review(monkeypatch):
    def local(blob, rows, target, context, duration):
        return dict(status="ok", start=2., end=2.3, stable=True,
                    character_support=.9, endpoint_spread=0.)
    monkeypatch.setattr("songsored.build_dataset.align_local", local)
    monkeypatch.setattr("songsored.build_dataset.passage", lambda *args: dict(status="ok", warnings=[]))
    blobs = {("mms", "ar"): {}, ("native", "en"): {}}
    targets = [dict(token_start=i, token_end=i + 1, language="en") for i in [1, 4]]
    cases = propose([], targets, blobs, {}, 10)
    assert all(case["selection"]["status"] == "unresolved" for case in cases)


def test_supplied_acoustic_timeline_must_match_full_mix(tmp_path):
    manifest, document = example(tmp_path)
    acoustic = tmp_path / "short-stem.wav"
    sf.write(acoustic, np.zeros(16000), 16000)
    document["recordings"][0].update(acoustic_audio=acoustic.name, acoustic_sha256=digest(acoustic))
    manifest.write_text(json.dumps(document))
    def forbidden(*args, **kwargs):
        pytest.fail("must not align mismatched timelines")
    with pytest.raises(ValueError, match="same timeline"):
        build_dataset(manifest, tmp_path / "out", alignment_backend=forbidden)


def test_review_with_nonfinite_crop_coordinates_is_rejected(tmp_path):
    manifest, _ = example(tmp_path)
    output = build_dataset(manifest, tmp_path / "built", alignment_backend=synthetic_backend)
    item = json.loads((output / "queue.json").read_text())["candidates"][0]
    review = preview(output, dict(id=item["id"], start=3., end=3.2))
    review.update(decision="good", crop_start=float("nan"))
    (output / "reviews.json").write_text(json.dumps({item["id"]: review}))
    with pytest.raises(ValueError, match="finite"):
        export(output, tmp_path / "invalid")
