# Running the code

I've included the original log-mel CNN, the frozen speech classifier with global averaging, and the dataset builder. The later ordered-pooling and shifted-training speech experiments are described in the log, but their code is still in the private repo. The dataset and trained task weights are private too.

## Start with the checks

Use Python 3.11 or 3.12. From the repo folder, with your virtual environment active:

```bash
python -m pip install '.[test,train,alignment]'
python -m pytest -q
python -m songsored.cnn_inference --smoke
python -m songsored.inference --smoke
```

The tests generate their own audio and tensors. For the speech wrapper, they use a small replacement backbone to check freezing, pooling, and input handling. The builder tests run torchaudio's CTC alignment on synthetic scores.

Both smoke commands use random classifier weights. They let you check that the code runs, but they don't tell you how well it detects words.

## Predict from a clip

You'll need your own compatible trained checkpoint and an audio clip of up to one second:

```bash
python -m songsored.cnn_inference --audio local-data/clip.wav --checkpoint local-runs/cnn.pt
python -m songsored.inference --audio local-data/clip.wav --checkpoint local-runs/head.pt
```

Both commands run on CPU. Short clips get silence added at the end. Longer clips are rejected. The prediction says whether a target word is present, without identifying the word or its timing.

For speech inference, the XLS-R encoder must already be cached. Add `--download-encoder` to allow the download. To try the real encoder without a trained head, run `python -m songsored.inference --smoke-encoder --download-encoder`. That uses generated audio and a random head, so its predictions still aren't meaningful.

Training accepts a device argument. I checked this public code on CPU.

## Train the CNN

The training function is `songsored.train.train(...)`. It reads a CSV with `path,label,recording_id,split` columns. Paths are relative to the manifest, labels are 0 or 1, and each recording must stay in one split. Include both training and validation rows.

For the CNN baseline I kept, the settings were:

- One mean and standard deviation fitted on the training log-mel features.
- Zero weight decay and frequency masking disabled.
- Pooling size `(1, 1)`.

Set `normalize_inputs=True` yourself, since normalization is off by default. Check the other function arguments before starting and use a new output path. The README comparison used a separate protocol to match optimizer updates across models. Calling `train()` with its defaults won't reproduce that table.

## Train the speech head

Use the same CSV format, keeping recordings separate between training and validation. First cache the speech features, then train the head:

```bash
python -m songsored.build_speech_feature_cache \
  --manifest local-data/train_val.csv --output local-runs/features.pt
python -m songsored.train_speech_classifier \
  --cache local-runs/features.pt --run-dir local-runs/head-seed42 --seed 42
```

The encoder must already be cached, or you can add `--download-encoder` to the cache-building command. The cache builder uses a pinned encoder revision, averages the frame features, and rejects test rows.

The head runner fits normalization separately for each feature using training embeddings only. It uses an unweighted binary loss and keeps the checkpoint with the lowest validation loss. The defaults are 24,600 updates, batch size 16, and validation every 82 updates. Reproducing the reported results still requires the private data.

## Build candidate clips

The builder starts from your own audio and **prepared text**. It doesn't fetch songs or lyrics, check artist identity, or reliably turn arbitrary Darija spellings into the text each aligner needs. I've connected the existing alignment functions to the crop and routing rules in this public entry point.

### Prepare the inputs

Start with a local JSON manifest:

```json
{
  "families": {"example": {"forms": ["hello"], "language": "en"}},
  "roster": {
    "recording-a": {"role": "train", "source_sha256": "REPLACE_WITH_FILE_SHA256"}
  },
  "recordings": [{
    "recording_id": "recording-a",
    "audio": "my-recording.wav",
    "source_type": "full_mix",
    "source_sha256": "REPLACE_WITH_FILE_SHA256",
    "rows": [
      {"original": "we", "mms": "we", "native": "we", "language": "en"},
      {"original": "say", "mms": "say", "native": "say", "language": "en"},
      {"original": "hello", "mms": "hello", "native": "hello", "language": "en"},
      {"original": "now", "mms": "now", "native": "now", "language": "en"}
    ]
  }]
}
```

There's no recording attached to this example. Replace it with audio you have permission to use and matching text. Each `original` field holds one normalized token. The `mms` and `native` fields hold its prepared spelling for each aligner. The native loader supports `ar`, `fr`, `es`, and `en`. Target matching works on exact tokens, so list spelling variants explicitly.

Put **all existing recording assignments and hashes** in `roster`, including `val`, `test`, and `reserve`. Only recordings with the `train` role belong in `recordings`, which is the list to process. The builder rejects held-out roles before opening their audio. It also rejects the same file hash under different IDs.

You still need to check recording identities and versions yourself. Two encodings of the same song can have different hashes.

### If you have separated vocals

I also used separated vocals in the alignment experiments. You can pass an existing stem through `acoustic_audio` and `acoustic_sha256`. It must be mono 16 kHz and have the same start time and duration as the full mix.

The builder checks duration and hashes, but you need to check that the stem is synchronized. It doesn't run Demucs. Without a stem, it aligns the resampled full mix. Exported training clips always come from the full mix.

### Run the builder

```bash
python -m songsored.build_dataset \
  --manifest local-data/prepared.json --output local-runs/collection-001 \
  --negatives-per-recording 2
```

Models run one at a time on CPU. Their weights must already be cached. If you need to download them, check the licenses below and add `--download-models`. Use a new output directory.

The files it writes are:

- `train_manifest.csv`, `metadata.json`, and `clips/`: automatic training examples that haven't been reviewed. Don't use them as trusted validation examples.
- `queue.json` and `contexts/`: low-confidence positives to listen to. The contexts are mono copies of the full recordings. The private web review lab isn't included here.
- `receipt.json`: input hash, model identities, routing diagnostics, and export counts. The agreement and support scores aren't calibrated confidence estimates.

For negatives, the builder looks for aligned words with clean neighbors and avoids the target/profanity regions. If forbidden words have missing timestamps, or the transcript has gaps, it skips negatives for that recording. It picks at most three spaced negatives per recording. These guards reduce obvious mistakes, but automatic negatives can still be wrong.

### Review a queued positive

Choose the word boundaries you want to check and make a one-second preview:

```bash
python -m songsored.clip_preparation \
  --queue local-runs/collection-001 --id recording-a-w2-3 --start 2.0 --end 2.3
```

Listen to `previews/<clip_sha256>.wav`. If you approve it, copy the returned JSON object into `reviews.json`, keyed by its `id`, and add `"decision": "good"`. Keep the hashes and crop coordinates as returned.

Then export the approved clips:

```bash
python -m songsored.export_focused_approvals \
  --queue local-runs/collection-001 --destination local-runs/reviewed-001
```

The exporter checks the review receipt against the queue, source audio, and exact preview bytes. Rejected or undecided examples stay out. A training recording stays in training after review.

The older `assign_recordings` helper is for a new collection where everything is human-reviewed and no splits have been assigned yet. Don't use it to reshuffle existing assignments. The builder exports a training-only CSV, so you'll need a separate reviewed validation set when preparing classifier inputs.

Keep these outputs out of Git. They contain audio, prepared text, recording identities, and local paths. The repository's tests only use generated inputs.

## Credits and licenses

These are the models and libraries I used. Each has its own license:

- [MMS_FA](https://docs.pytorch.org/audio/2.7.0/generated/torchaudio.pipelines.MMS_FA.html) weights: CC-BY-NC 4.0.
- [VoxPopuli French](https://docs.pytorch.org/audio/2.7.0/generated/torchaudio.pipelines.VOXPOPULI_ASR_BASE_10K_FR.html) and [Spanish](https://docs.pytorch.org/audio/2.7.0/generated/torchaudio.pipelines.VOXPOPULI_ASR_BASE_10K_ES.html) weights: CC-BY-NC 4.0.
- [Arabic XLSR-53 model card](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-arabic/blob/af46c2d8531b8dcbb5e23b952f739b372c2e5d2d/README.md): Apache-2.0. I use this for Arabic alignment. The speech classifier uses XLS-R-300M.
- [English wav2vec 2.0 bundle](https://docs.pytorch.org/audio/2.7.0/generated/torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.html) weights: MIT.
- [torchaudio](https://github.com/pytorch/audio/blob/v2.7.1/LICENSE) code: BSD-style. Its model weights have separate licenses.
- This project uses the universal romanizer software 'uroman' written by Ulf Hermjakob, USC Information Sciences Institute (2015-2020). [License](https://github.com/isi-nlp/uroman/blob/master/LICENSE.txt). Paper: Hermjakob, May and Knight (2018), *Out-of-the-box universal romanization tool uroman*.
- [tqdm](https://github.com/tqdm/tqdm/blob/master/LICENCE): MPL-2.0 and MIT. Matplotlib uses its [own license](https://matplotlib.org/stable/project/license.html).

My code and documentation use [MIT](../LICENSE). The model weights aren't included here, and their licenses still apply if you download them. They don't give permission to share the songs, lyrics, or dataset.
