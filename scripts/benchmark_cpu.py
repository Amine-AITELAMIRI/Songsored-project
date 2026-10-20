"""Measure warm CPU cost of the global-mean path using generated audio.

Real pretrained XLS-R, random task head and identity feature statistics.
Not a trained classifier demo or a real-time/full-song benchmark.
"""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np
import soundfile as sf
import torch

from songsored.inference import DIM, MODEL_ID, REVISION, make_encoder
from songsored.speech_classifier import SpeechFeatureClassifier
from songsored.speech_features import load_speech_waveform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download-encoder", action="store_true")
    args = parser.parse_args()
    if args.runs < 2 or args.warmup < 1:
        parser.error("Use at least two measured runs and one warm-up.")
    if args.output.exists():
        parser.error("Output exists; choose a new path.")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    start = time.perf_counter()
    encoder = make_encoder(args.download_encoder)
    load_seconds = time.perf_counter() - start
    head = SpeechFeatureClassifier(DIM).eval()
    mean, scale = torch.zeros(DIM), torch.ones(DIM)
    def forward(path):
        x = encoder.encode(load_speech_waveform(path))
        with torch.no_grad():
            return head(((x - mean) / scale).unsqueeze(0)).sigmoid()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "generated.wav"
        wave = (.1 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)).astype("float32")
        sf.write(path, wave, 16000, subtype="FLOAT")
        for _ in range(args.warmup):
            forward(path)
        times = []
        for _ in range(args.runs):
            start = time.perf_counter_ns()
            output = forward(path)
            times.append((time.perf_counter_ns() - start) / 1e6)
            assert output.shape == (1, 1) and torch.isfinite(output).all()
    # ru_maxrss is bytes on macOS, KiB on Linux; includes imports/model loading.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    result = {
        "hardware": {"cpu": cpu, "architecture": platform.machine(), "os": platform.platform()},
        "python": platform.python_version(),
        "versions": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "numpy", "librosa", "soundfile"]},
        "model_id": MODEL_ID, "revision": REVISION, "pooling": "global mean",
        "input": "generated 440 Hz sine, amplitude 0.1, mono float WAV, 16000 Hz, 1 second",
        "batch_size": 1, "torch_threads": 1, "torch_interop_threads": 1,
        "warmup": args.warmup, "runs": args.runs, "encoder_load_seconds": load_seconds,
        "method": "perf_counter_ns wall time: warm WAV decode + preprocessing + encoder + pooling + normalization + linear head + sigmoid; model loading excluded",
        "latency_ms": {"median": statistics.median(times), "p95_linear": float(np.percentile(times,95)), "min": min(times), "max": max(times), "samples": times},
        "process_peak_rss_mib": peak_bytes / 1024**2,
        "memory_method": "resource.ru_maxrss lifetime high-water mark; includes interpreter, imports, encoder loading and warm-up; not tensor memory or incremental inference memory",
        "limits": "Pretrained encoder with random task head and identity feature statistics. Generated audio only. No accuracy, streaming, sustained-load, power, or real-time claim. OS caching and other apps can affect results.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k:result[k] for k in ["hardware", "latency_ms", "process_peak_rss_mib"]}, indent=2))


if __name__ == "__main__":
    main()
