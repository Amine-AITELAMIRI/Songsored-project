"""Route the original mix and its vocal stem through two configured detectors."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .audio_muting import combine_detections, render_muted_audio


def load_analysis_audio(path, res_type):
    """Return mono 16 kHz audio and the duration before resampling."""
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError("Expected nonempty, finite audio.")
    duration = len(audio) / sample_rate
    wave = librosa.resample(
        audio.mean(1), orig_sr=sample_rate, target_sr=16000, res_type=res_type,
    )
    if not np.isfinite(wave).all():
        raise ValueError("Resampling produced nonfinite audio.")
    return wave.astype(np.float32, copy=False), duration


def detect_song(source, vocals, *, ctc_detector, vocal_detector):
    """Run both branches and return their selected intervals and their union.

    Each supplied detector takes (mono_16khz_audio, duration_seconds) and returns
    events with start/end in song seconds. It must already apply its calibrated
    cutoff, persistence rules and any trained score correction.

    vocals must be the aligned, untrimmed stem of source. Checking duration can
    catch a cropped stem, but cannot establish its identity or synchronization.
    Model loading, separation and trained weights are not provided here.
    """
    source, vocals = Path(source), Path(vocals)
    if source.resolve() == vocals.resolve():
        raise ValueError("Supply the original mix and its separate vocal stem.")

    # Keep the resampling methods used by the private CTC and vocal paths.
    mix, duration = load_analysis_audio(source, "polyphase")
    voice, vocal_duration = load_analysis_audio(vocals, "soxr_hq")
    if abs(duration - vocal_duration) > 1 / 16000 + 1e-9:
        raise ValueError("The vocal stem must cover the same timeline as the mix.")

    # Each branch sees only its own audio, not lyrics or the other branch's scores.
    ctc_events = list(ctc_detector(mix, duration))
    vocal_events = list(vocal_detector(voice, duration))
    combined = combine_detections(ctc_events, vocal_events)
    return {"ctc": ctc_events, "vocals": vocal_events, "combined": combined}


def censor_song(source, vocals, destination, event_path, *, ctc_detector, vocal_detector):
    """Run both branches, then mute the original mix using their selected events.

    Return branch events and the render record. The JSON written by the renderer
    contains the combined events and exact mute samples, not branch scores.
    """
    paths = [Path(path) for path in (source, vocals, destination, event_path)]
    if len({path.resolve() for path in paths}) != 4:
        raise ValueError("Inputs and outputs must all have distinct paths.")
    if paths[2].suffix.lower() != ".wav":
        raise ValueError("Choose a .wav output.")
    if paths[2].exists() or paths[3].exists():
        raise FileExistsError("Choose new output paths.")

    detections = detect_song(
        source, vocals, ctc_detector=ctc_detector, vocal_detector=vocal_detector,
    )
    # Render the original mix, keeping its sample rate and channels.
    record = render_muted_audio(
        source, destination, event_path, detections["ctc"], detections["vocals"],
    )
    return {"branches": detections, "render": record}
