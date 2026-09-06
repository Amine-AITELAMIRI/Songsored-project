# Building the dataset

[Back to the project log](../README.md)

I needed examples of target words in Moroccan rap before I could train a detector. The builder uses lyrics to find candidates, proposes timestamps, and lets me listen to the clips in a local lab.

## Getting the boundaries right

The first comparison used known regions from two songs:

- The Qwen target-only control did poorly on these examples. This was a small, partial-transcript experiment, so it says little about other uses of the model.
- MMS and language-specific CTC gave more useful proposals, but sometimes disagreed.
- My manually placed timestamps were sometimes off too. I started judging crops by listening, with cursor differences kept as a diagnostic.

I tried scoring both candidate locations with both models. All three disagreements stayed unresolved.

Adding surrounding lyrics helped: the three choices matched the saved listening judgments. Nine agreement cases were carried forward. Only the three disagreements received the context test, and the phrases were manually prepared. The scorer selected an existing span; it didn't prove that the boundary was more precise.

## Moving to full recordings

The next lab searched complete recordings using untimed lyrics, without my reviewed timestamps as inputs.

The first draft transcripts produced no exact target matches. The songs weren't necessarily clean: the transcript and spelling rules could miss a word before alignment even started. Published lyrics were a better starting point, but still needed preparation.

Later experiments used passage context and stable proposals across windows. Recovered cases went back to listening review. Some words were already censored in the recording, which exposed another problem: a plausible timestamp doesn't guarantee an audible training example.

## Making training clips

- **Word span versus crop:** the span marks the word; the one-second training crop can include neighboring audio. The complete target needs to fit inside it.
- **Positive labels:** presence of a configured target family. This isn't a detector for all profanity.
- **Negatives:** no configured target present. Other offensive words can still occur.
- **Automatic labels:** provisional, even when the models agree. Confidence gates use stability and boundary checks, not calibrated probabilities.
- **Splits:** group by source recording and keep automatic labels in training. This does not separate artists.

Collection can use reviewed clips and provisional automatic candidates. [The settings file](../evidence/protocol.json) gives the counts for each reported experiment.

## Still uncertain

Lyrics can be incomplete, spelling variants can escape the rules, and repeated choruses add little recording diversity. Forced alignment can fit supplied text without proving the word was spoken.

The builder helps collect examples. It doesn't establish exhaustive full-song detection. These alignment observations come from private development reports and listening reviews; the public scripts don't reproduce those judgments.

Models and tools are credited in the [README](../README.md#tools-i-used). Audio, lyrics, and identifiable review records remain private.
