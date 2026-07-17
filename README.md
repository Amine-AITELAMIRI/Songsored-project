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

## Next

Compare alignment methods on the reviewed examples and listen to where their boundaries disagree.
