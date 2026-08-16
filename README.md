# Songsored

**In progress**

The project's still ongoing, so I'm sharing what I've learned so far here. Development happens in a private repository, and this log follows the experiments, the results, and what I'm working on next.

The idea started with something simple: I wanted to listen to rap when my parents were around without worrying about explicit lyrics coming on. Songsored is my attempt to make that possible for Moroccan rap by finding and eventually censoring selected words.

## First, I need examples

I need reliable examples from songs before I can train a detector. Rap makes this awkward: words can be stretched or shortened, Darija and French can appear together, and the voice sits inside a music mix.

Lyrics give me a starting point: find the target words in the text, then estimate where they occur in the audio. The timestamps still need checking before I can use them as training examples.

From there, the initial learning task is small: does a one-second clip contain a target word? Finding every occurrence across a full song and applying the censoring are later steps.

## A lab for checking word locations

I built a local listening lab around the timing code. It takes a vocal recording, lyrics in sung order, and a list of target words. For each occurrence, I can:

- Play the proposed word span on its own or with nearby audio.
- Move the start and end handles if the crop cuts into the word or includes too much around it.
- Mark it correct, incorrect, or unsure, and leave a note.

The lab saves the estimates and my review decisions. Its summary separates automatic times I accepted, times I corrected, and missed words I located manually. A word I found by moving the handles shouldn't count as a successful automatic estimate.

There are two different boundaries to think about here. The word span marks the beginning and end of the word itself. The eventual one-second training clip can include neighboring audio, as long as the complete target fits inside it.

This gives me a way to inspect the timing errors. It doesn't yet tell me how reliable the method is across songs, and the lab itself doesn't train a detector or censor the audio.

## Comparing aligners

I compared three approaches on 12 confirmed word spans from two songs. Each method received a broad vocal excerpt around the target, without my exact timestamps. This checks alignment within a known region; it doesn't test finding a word anywhere in a song.

| Approach | What I found by listening |
| --- | --- |
| [Qwen3 ForcedAligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B), given only the target text | Poor proposals in this setup. Supplying an incomplete transcript makes this a limited control, not a general verdict on Qwen. |
| Meta's multilingual [MMS aligner](https://docs.pytorch.org/audio/2.7.0/generated/torchaudio.pipelines.MMS_FA.html) | Useful boundaries on most examples. |
| Language-specific CTC models, including [Jonatas Grosman's Arabic model](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-arabic) | Also useful, but sometimes placed the same word differently. |

These are pretrained models from other projects. My work here was the comparison, review lab, and logic for choosing between proposals. The CTC runs allowed untranscribed surrounding audio; the Qwen control used the target text alone.

My own cursor placements weren't always precise either. I started using listening verdicts as the main check, with differences from my manual timestamps kept as a diagnostic.

## What to do when the models disagree?

First I tried scoring both proposed locations with both CTC models. I also varied the padding around each location to see whether the preference held. All three disagreements stayed unresolved: the models didn't consistently prefer the same proposal.

Then I tried including the surrounding lyrics. Instead of scoring just the target word, I scored the phrase in order, constraining the target to each proposed location. The phrase text was manually prepared; the scorer didn't receive my listening decisions or cursor positions.

- All three phrase-based choices matched the saved listening judgments.
- Nine agreement cases were carried forward unchanged. They weren't new context tests.
- The method selected an existing span. It didn't refine the boundaries or establish that they were more precise.

That gives me something to try on new recordings. It's still a small development experiment using previously reviewed examples, not held-out accuracy. The scores aren't word-presence probabilities, and the eight rejected-region controls weren't context-tested. I haven't established a false-positive rate or complete full-song coverage.

## From word proposals to training examples

The next lab runs on complete recordings with untimed lyrics, without my reviewed timestamps or known target regions as inputs.

The first draft transcripts produced no exact target matches. That didn't mean the songs were clean: a missing word or an unrecognized spelling could stop a candidate from reaching alignment. Published lyrics were a better starting point, but still needed preparation.

Later boundary experiments used neighboring lyrics and checked whether proposals stayed stable across audio windows. Recovered candidates went back into the listening lab. Some words were already censored in the source recording, so even a plausible location wasn't enough to call a clip positive.

For the dataset, I'm using these rules:

- **Positive:** a configured target family is audible inside the clip.
- **Negative:** no configured target is present. This doesn't mean no offensive language of any kind.
- **Automatic labels:** provisional, even when aligners agree. The confidence checks aren't calibrated probabilities.
- **Splits:** keep clips from the same recording together, and automatic labels in training only. Artists can still overlap.

The builder is useful for collecting examples. It still depends on lyric coverage, spellings, and the quality of the proposed boundaries.

## Next

Build a small CNN and training loop for one-second target-word classification, starting with reviewed examples.
