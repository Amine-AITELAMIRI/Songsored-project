# Songsored

**In progress**

The project's still ongoing, so I'm sharing what I've learned so far here. Development happens in a private repository, and this log follows the experiments, the results, and what I'm working on next.

The idea started with something simple: I wanted to listen to rap when my parents were around without worrying about explicit lyrics coming on. Songsored is my attempt to make that possible for Moroccan rap by finding and eventually censoring selected words.

Right now, I can run a classifier that checks whether a one-second clip contains a target word. Full-song detection and censoring are still missing.

Getting the data has been a big part of the challenge. Darija speech resources exist, but I needed reliable, task-specific examples from songs, especially profanity and slurs. I ended up building a focused dataset for this task.

Speech-to-text support for Darija also doesn't guarantee reliable rap transcription. Words can be stretched, shortened, or pronounced differently in music, while instruments, overlapping voices, and Darija/French code-switching add to the difficulty. Here's what I've tried so far.

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

Recropping changes surrounding audio too, and the target boundaries are approximate. This positive-only diagnostic doesn't measure false alarms.

[The ordered-pooling and position-sensitivity experiments in more detail](docs/temporal-structure.md).

## What I want to test next

Train with eligible shifted crops and check whether that helps at different positions without losing the centered-clip performance. Keep the original validation set for checkpoint selection.

## Check the numbers

The aggregate files reproduce the figures and summaries, not training or audio labels. Audio, lyrics, clip identities, weights, and the development source remain private. [Settings and split counts](evidence/protocol.json).

```bash
python3 scripts/reproduce_results.py --check
```

To redraw the figures with Python 3.11+:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-figures.txt
.venv/bin/python scripts/render_figures.py
```

## Tools I used

PyTorch, torchaudio, Transformers, and librosa; Meta's [XLS-R](https://huggingface.co/facebook/wav2vec2-xls-r-300m) for frozen features; [MMS](https://docs.pytorch.org/audio/2.7.0/generated/torchaudio.pipelines.MMS_FA.html) and language-specific CTC models, including [Jonatas Grosman's Arabic model](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-arabic), for alignment; [Demucs](https://github.com/facebookresearch/demucs) for vocal separation; Genius as a lyric source; Matplotlib for the plot. The pretrained models are external work. MMS weights have a noncommercial license.
