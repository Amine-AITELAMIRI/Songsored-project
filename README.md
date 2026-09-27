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

## Four ordered bins

I divided the frozen speech encoder's frame sequence into four consecutive groups and averaged each group separately. Concatenating those averages gives the linear head a coarse beginning-to-end representation. The encoder remained frozen.

This also makes the head larger. The experiment changes both temporal information and classifier capacity, so any improvement cannot be attributed to word order alone.

The training set had expanded to 1,428 clips from 487 recordings. The original 41-clip validation set still selected checkpoints. A 70-clip development panel from 26 recordings was used to inspect the selected models; it had already informed development and was not an independent test.

| Representation | Mean accuracy on the centered development panel |
|---|---:|
| Global averaging | 68.9% |
| Four ordered bins | 79.7% |

These are five-seed means with the same 24,600-update budget. They are not directly comparable to the earlier CNN-versus-speech table because the training data and reporting panel changed.

## Next

The ordered model looks better on the centered clips, but a full-song detector would encounter words at different positions. Shift the windows and check whether that improvement survives.

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
