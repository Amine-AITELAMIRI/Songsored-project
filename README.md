# Songsored

**In progress**

The project's still ongoing, so I'm sharing what I've learned so far here. Development happens in a private repository, and this log follows the experiments, the results, and what I'm working on next.

The idea started with something simple: I wanted to listen to rap when my parents were around without worrying about explicit lyrics coming on. Songsored is my attempt to make that possible for Moroccan rap by finding and eventually censoring selected words.

## Where I'm starting

I need reliable examples from songs before I can train a detector. Rap makes this awkward: words can be stretched or shortened, Darija and French can appear together, and the voice sits inside a music mix.

The first task is to use lyrics to find candidate words and check where they occur in the audio. I want to be able to listen to the proposed clips and correct the boundaries.

From there, the initial learning task is small: does a one-second clip contain a target word? Finding every occurrence across a full song and applying the censoring are later steps.

## Next

Build a local listening lab so I can check the word locations before turning them into training examples.
