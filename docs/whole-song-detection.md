# Trying it on whole songs

[Back to the project log](../README.md)

With the one-second clips, I already knew roughly where to listen. On a whole song, the detector has to find the word and decide which parts to leave alone.

## What I kept

I use Demucs to separate the vocals, then run frozen XLS-R features through a small temporal head. It scores the vocals every 20 ms. Some ordinary words sound like a target when the end of one word runs into the next. I listened to 12 false triggers and used them to train a small correction to those scores.

I also run an adapted OmniASR CTC model on the original mix. It looks for the target sound sequence and can pick up words the vocal detector misses. I combine the intervals from both and mute those parts of a copy of the original song. Lyrics helped me build examples. When I run it on a song, it only gets the audio.

## What it found

I chose the cutoff on calibration recordings, allowing up to one trigger per ten minutes of screened background. Then I kept the settings fixed and scanned 36 other development songs.

| Split | Songs | Labeled words found | Triggers in screened background | Mutes in unlabeled audio |
|---|---:|---:|---:|---:|
| Calibration | 128 | 23 of 32 | 12 in 133.23 min | 29 |
| Development | 36 | 15 of 22 | 2 in 19.27 min | 19 |

Screened background means passages marked as ordinary audio. The other 19 development mutes landed outside both those passages and the labeled words. Some could be real targets that the labels missed, so I still can't say how often the system would mute an ordinary word across a whole song.

I counted a word as found if a predicted mute overlapped its label by at least 40 ms. A short mute can be enough when you listen, but overlap alone doesn't tell me whether the word was still recognizable. My goal is to obscure at least 80% of audible targets with at most one unnecessary mute per ten minutes. I haven't shown that yet. These songs have been used during development. The test set is still untouched.

[Whole-song counts](../evidence/whole_song_results.csv)
