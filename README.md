# Songsored

**In progress**

The project's still ongoing, so I'm sharing what I've learned so far here. Development happens in a private repository, and this log follows the experiments, the results, and what I'm working on next.

The idea started with something simple: I wanted to listen to rap when my parents were around without worrying about explicit lyrics coming on. Songsored is my attempt to make that possible for Moroccan rap by finding and eventually censoring selected words.

Right now, I can run a classifier that checks whether a one-second clip contains a target word. Full-song detection and censoring are still missing.

Getting the data has been a big part of the challenge. Darija speech resources exist, but I needed reliable, task-specific examples from songs, especially profanity and slurs. I ended up building a focused dataset for this task.

Speech-to-text support for Darija also doesn't guarantee reliable rap transcription. Words can be stretched, shortened, or pronounced differently in music, while instruments, overlapping voices, and Darija/French code-switching add to the difficulty. Here's what I've tried so far.

## First, I needed examples

I used lyrics and alignment models to propose word locations, then listened to clips in a local review lab. Recording-separated splits keep clips from the same song together; automatic labels stay in training.

[More on building the dataset and resolving alignment disagreements](docs/dataset-builder.md).

## First: get the CNN working

I built the log-mel features, dataset loader, small CNN, and training loop. After the tiny-subset check, I tried longer runs and compared seeds.

![Early CNN training and validation loss, with and without normalization.](assets/cnn_learning_curves.png)

*189 training clips / 37 recordings; 41 validation clips / 8 other recordings. Mean and shaded sample SD across five seeds. Adam, learning rate 0.001, batch size 16.*

- I kept training-only normalization. Compare the first 100 epochs for an equal-budget view.
- Continuing to 300 epochs lowered training loss, but validation flattened and later rose. Longer training didn't remove the gap.
- Weight decay at `1e-4` and frequency masking up to 8 bands didn't give a consistent improvement in their separate trials. I kept zero decay and masking disabled.

## Next

Check the dataset's word coverage and recording diversity, then compare representations on the expanded data. The training loss falling doesn't by itself explain the validation errors.

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

PyTorch and librosa for the classifier and audio features; torchaudio for alignment; [MMS](https://docs.pytorch.org/audio/2.7.0/generated/torchaudio.pipelines.MMS_FA.html), [Qwen3 ForcedAligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B), and language-specific CTC models including [Jonatas Grosman's Arabic model](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-arabic); [Demucs](https://github.com/facebookresearch/demucs) for vocal separation; Genius as a lyric source; Matplotlib for figures. The pretrained models are external work. MMS weights have a noncommercial license.
