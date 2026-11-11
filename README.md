# Songsored

**In progress**

The project's still ongoing, so I'm sharing what I've learned so far here. The main development repository is private. This repo now includes the classifier and dataset-builder code alongside the experiments, results, and what I'm working on next.

The idea started with something simple: I wanted to listen to rap when my parents were around without worrying about explicit lyrics coming on. Songsored is my attempt to make that possible for Moroccan rap by finding and eventually censoring selected words.

I started by checking one-second clips. In the private prototype, I can now give it a whole song and get back a copy with the detected words muted. So far I've focused on one word family, and it still misses words.

Getting the data has been a big part of the challenge. Darija speech resources exist, but I needed reliable, task-specific examples from songs, especially profanity and slurs. I ended up building a focused dataset for this task.

Speech-to-text support for Darija also doesn't guarantee reliable rap transcription. Words can be stretched, shortened, or pronounced differently in music, while instruments, overlapping voices, and Darija/French code-switching add to the difficulty. Here's what I've tried so far.

## Run the code

I've included the CNN, the speech detectors, the dataset builder, and the code that combines detections and mutes audio here. The dataset and trained weights are still private. You can run the checks below without them, but you won't get trained word predictions.

### 1. Install

Use Python 3.11 or 3.12. I checked this on Apple Silicon with Python 3.11. From the repo folder, create a virtual environment and install the package with its test, training, and alignment dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install '.[test,train,alignment]'
```

### 2. Run the tests

```bash
python -m pytest -q
```

The tests use generated audio and tensors to check the preprocessing, training code, dataset builder, sliding windows, temporal detection, CTC scoring and timing, and audio muting. They don't download pretrained models or measure detection accuracy.

### 3. Try a small example

```bash
python -m songsored.cnn_inference --smoke
python -m songsored.inference --smoke
```

There are two smoke checks. The CNN one runs a generated tone through the model. The speech-head one starts with a generated feature vector, so it skips XLS-R. Both use random classifier weights. Any prediction from these checks is meaningless, but they let you check that the code runs.

I've put the training and dataset-builder instructions in the [running guide](docs/running-the-code.md). It also shows how to try the real XLS-R encoder and use your own compatible checkpoint to make predictions. The later ordered-pooling and shifted-training speech experiments are still in the private repo.

The [CTC runner](songsored/ctc_detector.py) has an extra dependency:

```bash
python -m pip install '.[ctc]'
```

It loads a local OmniASR ONNX model, token file, and trained output adapter. These files aren't included. Installing the extra alone doesn't give you a trained detector.

The [combined runner](songsored/combined_detector.py) takes the original song, an aligned vocal stem, and both configured detectors. It writes a muted WAV and an event list. It doesn't run vocal separation itself. The code is here, but the private checkpoints and calibrated settings are still needed to reproduce my detections.

Reproducing the validation predictions requires the private data and trained weights.

## First, I needed examples

I used lyrics and alignment models to propose word locations, then listened to clips in a review lab. My implementation work includes the dataset loader, feature extraction, CNN, training loop, and later the frozen speech-feature wrapper and classification head.

I separated training and validation by recording and kept automatic labels in training. Labels still need care: a confident alignment doesn't guarantee an audible word or a correct crop.

[More on building the dataset and resolving alignment disagreements](docs/dataset-builder.md).

## Starting with a CNN, then trying speech features

I started with a small CNN on log-mel spectrograms, then tried frozen XLS-R-300M features with a linear head. Only the head was trained.

| Model | Validation BCE ↓ | Accuracy ↑ |
|---|---:|---:|
| CNN with global averaging | 0.620 | 62.9% |
| Frozen XLS-R + linear head | 0.554 | 75.1% |

Both used **1,297 training clips / 449 recordings** and **41 validation clips / 8 other recordings**. These are five-seed means: 24,600 Adam updates per run, batch size 16, and training-only normalization. For each run, I kept the checkpoint with the lowest validation loss (binary cross-entropy, or BCE).

Speech features helped in every seed. But **75.1% comes from repeatedly used validation data, not an independent test**. Equal updates also don't mean equal compute.

[The classification experiments, including the approaches I didn't keep](docs/classification-experiments.md).

## Keeping more timing information looked promising

I then tried four ordered pooling bins instead of averaging all frames. With expanded training data (**1,428 clips / 487 recordings**), accuracy rose from **68.9% to 79.7%** on a **70-clip / 26-recording development panel**. That panel had already informed development; the original 41 validation clips still selected checkpoints. This isn't directly comparable to the earlier table.

## Then I moved the word inside the window

I shifted the crops to put words earlier or later. On the same **25 eligible positive clips / 21 recordings**, ordered-pooling recall fell from **85.6% at the original position to 22.4% at 200 ms earlier**.

![Recall at different word positions, averaged over five seeds.](assets/position_sensitivity.png)

Training with eligible shifted crops raised recall at that earlier position to **52.8%**, but lowered original-position recall to **75.2%**. It helped some positions and hurt others; it didn't solve the problem.

Recropping changes surrounding audio too, and the target boundaries are approximate. This positive-only diagnostic doesn't measure false alarms.

[The ordered-pooling and position-sensitivity experiments in more detail](docs/temporal-structure.md).

## Moving beyond clips

I added a [sliding-window runner](songsored/sliding_window.py) that moves the CNN through audio one second at a time, advancing by 100 ms. It returns a score for each window. Those timestamps describe the window, not the word inside it. Incomplete final windows are skipped.

## Keeping scores over time

I added a [temporal head](songsored/speech_residual_detector.py) over frozen XLS-R features. The [vocal runner](songsored/vocal_detector.py) uses four-second windows and keeps scores from the middle two seconds. It also loads a learned score correction from the false-trigger work. Running it needs aligned vocals, compatible checkpoints, and a calibrated cutoff; those aren't included here.

## Looking for the sound sequence with CTC

The other branch uses adapted OmniASR CTC on three-second windows of the original mix. I [score the target character sequences](songsored/ctc_keyword_verifier.py), then [locate them in time](songsored/ctc_keyword_timing.py). It doesn't need supplied lyrics. The runner returns at most one detection per window, so that remains a limitation.

## Connecting detections to playback

I combine intervals from either branch, merge overlaps, and [mute those samples](songsored/audio_muting.py) in a copy of the original song. The original stays intact. Keeping either branch's detections can recover different words, but it can also add false alarms.

## What I want to test next

Next I want to compare the combined detector with fixed settings on development songs. I'll count located words and background triggers separately. Touching a label isn't enough to show that the mute actually obscures the word.

## Check the numbers

The aggregate CSVs reproduce these numbers and the plot, not training or audio labels. Audio, lyrics, clip identities, weights, and the rest of the private development source are excluded. [Settings and dataset counts](evidence/protocol.json).

```bash
python3 scripts/reproduce_results.py --check
```

To redraw the plot with Python 3.11+:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-figures.txt
.venv/bin/python scripts/render_figures.py
```

## Tools I used

The keyword branch uses Meta's OmniASR CTC model through a local ONNX export, plus my output adapter and keyword scoring code. ONNX Runtime runs that export on CPU.

PyTorch, torchaudio, Transformers, and librosa; Meta's [XLS-R](https://huggingface.co/facebook/wav2vec2-xls-r-300m) for frozen features; [MMS](https://docs.pytorch.org/audio/2.7.0/generated/torchaudio.pipelines.MMS_FA.html) and language-specific CTC models, including [Jonatas Grosman's Arabic model](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-arabic), for alignment; [Demucs](https://github.com/facebookresearch/demucs) for vocal separation; Genius as a lyric source; Matplotlib for the plot. The pretrained models are external work. MMS weights have a noncommercial license.

## A first CPU cost check

I measured the global-mean path on an **Apple M5, macOS 26.5, Python 3.11**, with one CPU thread and batch size 1. For a generated one-second WAV, 5 warm-ups followed by 30 runs gave **87.6 ms median / 248.8 ms p95**. Lifetime peak process RSS was **1,675.7 MiB**, including imports and model loading.

Timing includes WAV loading, preprocessing, the actual pretrained encoder, pooling and a random linear head with identity normalization. It excludes model loading. This is a warm synthetic-input cost check, **not trained prediction quality or real-time performance**. Other applications were not isolated; the spread matters. [Raw timings, versions and method](evidence/cpu_cost.json).

```bash
python scripts/benchmark_cpu.py --output cpu_cost_local.json --download-encoder
```

## Code and model credits

The core classifier and alignment functions come from the experiment code, with readability edits and public package imports. The inference commands and builder orchestration are new. The adapter adds explicit inputs, cache-only model loading and recording/provenance guards. The speech feature revision is pinned to the reported global-mean experiment. The offline tests check mechanics, not new detection performance. See the [running notes](docs/running-the-code.md#credits-and-licenses) for the aligners and their separate licenses.

[XLS-R's model card](https://huggingface.co/facebook/wav2vec2-xls-r-300m/blob/1a640f32ac3e39899438a2931f9924c02f080a54/README.md) declares Apache-2.0. Dependencies are installed separately: [PyTorch](https://github.com/pytorch/pytorch/blob/v2.7.1/LICENSE) (BSD-style with third-party notices), [Transformers](https://github.com/huggingface/transformers/blob/v4.57.6/LICENSE) (Apache-2.0), [librosa](https://github.com/librosa/librosa/blob/0.10.2.post1/LICENSE.md) (ISC), [NumPy](https://github.com/numpy/numpy/blob/v2.0.2/LICENSE.txt) and [SoundFile](https://github.com/bastibe/python-soundfile/blob/0.13.1/LICENSE) (BSD-style). SoundFile uses [libsndfile](https://github.com/libsndfile/libsndfile/blob/master/COPYING), licensed LGPL-2.1-or-later. Their licenses do not establish redistribution rights for the song data or task-trained weights. The installed resampling dependency, soxr, also declares LGPL-2.1-or-later. Those data and weights remain excluded.

## License

My original code and documentation in this repository are available under the [MIT License](LICENSE). External libraries and pretrained models retain their own licenses, including the noncommercial terms for MMS and VoxPopuli weights. MIT does not grant rights to the songs, lyrics, private datasets or model weights used in the experiments.
