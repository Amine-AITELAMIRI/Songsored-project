"""Scan the original mix with a local OmniASR CTC export and output adapter."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from .ctc_keyword_timing import frames_to_seconds, locate_keyword_frames
from .ctc_keyword_verifier import CTCKeywordVerifier


def read_tokens(path):
    """Read the export's token-ID pairs, preserving a literal space token."""
    tokens = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        token, separator, number = line.rpartition(" ")
        if not separator or not token or not number.isdigit() or token in tokens:
            raise ValueError("Invalid or repeated token in the token file.")
        tokens[token] = int(number)
    if not tokens or len(set(tokens.values())) != len(tokens):
        raise ValueError("Expected unique token IDs.")
    if set(tokens.values()) != set(range(len(tokens))):
        raise ValueError("Expected contiguous token IDs starting at zero.")
    if "<blank>" in tokens and tokens["<blank>"] != 0:
        raise ValueError("This export must use token zero for CTC blank.")
    # The private OmniASR export treats ID zero as blank, regardless of its name.
    tokens["<blank>"] = 0
    return tokens


class CTCDetector:
    """Return at most one target interval per three-second window.

    Targets must be written in the acoustic model's alphabet. Supplying a new
    target does not train the model to recognize it. Use matching local export,
    token table, adapter and calibration settings; shapes alone cannot verify
    their provenance. The retained export has a 20 ms frame spacing.
    """

    def __init__(self, session, token_ids, checkpoint, targets, *, threshold):
        if not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("Supply a calibrated threshold in (0, 1].")
        if checkpoint.get("model") != "omni":
            raise ValueError("Expected an OmniASR output-adapter checkpoint.")
        alphabet = checkpoint["alphabet"]
        if (
            not isinstance(alphabet, list) or not alphabet
            or any(not isinstance(char, str) or not char for char in alphabet)
            or len(set(alphabet)) != len(alphabet)
            or "<blank>" not in alphabet or " " not in alphabet
            or any(char not in token_ids for char in alphabet)
        ):
            raise ValueError("The adapter alphabet must match the token table.")
        if token_ids.get("<blank>") != 0:
            raise ValueError("This export must use token zero for CTC blank.")
        if any(type(value) is not int or value < 0 for value in token_ids.values()):
            raise ValueError("Token IDs must be nonnegative integers.")
        self.token_count = max(token_ids.values()) + 1
        if set(token_ids.values()) != set(range(self.token_count)):
            raise ValueError("Expected contiguous token IDs.")
        self.columns = [token_ids[char] for char in alphabet]
        if len(set(self.columns)) != len(self.columns) or len(self.columns) >= self.token_count:
            raise ValueError("Expected distinct adapter columns and remaining background tokens.")
        if isinstance(targets, str):
            raise ValueError("Supply a list of target words.")
        targets = list(targets)
        if not targets or any(not isinstance(word, str) or not word or " " in word for word in targets):
            raise ValueError("Supply nonempty individual target words.")
        characters = list(dict.fromkeys("".join(targets)))
        if any(char not in alphabet for char in characters):
            raise ValueError("A target character is missing from the trained adapter alphabet.")

        # Keep blank, word boundary and target letters. Everything else is OTHER.
        symbols = ["<blank>", " ", *characters, "<other>"]
        vocabulary = {char: index for index, char in enumerate(symbols)}
        self.rule = CTCKeywordVerifier(vocabulary, targets, blank=0, delimiter=1)
        self.kept = [alphabet.index(char) for char in symbols[:-1]]
        width = len(alphabet) + 1
        self.adapter = torch.nn.Linear(width, width)
        self.adapter.load_state_dict(checkpoint["layer"], strict=True)
        if not all(torch.isfinite(p).all() for p in self.adapter.parameters()):
            raise ValueError("Adapter parameters must be finite.")
        self.adapter.eval().requires_grad_(False)
        inputs = session.get_inputs()
        if len(inputs) != 1:
            raise ValueError("Expected an export with one waveform input.")
        self.session = session
        self.input_name = inputs[0].name
        self.threshold = threshold

    @classmethod
    def from_files(cls, model_path, tokens_path, adapter_path, targets, *, threshold):
        """Load trusted local files on CPU. No weights are bundled or downloaded."""
        import onnxruntime as ort

        tokens = read_tokens(tokens_path)
        checkpoint = torch.load(adapter_path, map_location="cpu", weights_only=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        return cls(session, tokens, checkpoint, targets, threshold=threshold)

    @torch.inference_mode()
    def frame_probabilities(self, window):
        """Run one padded three-second window and return reduced character probabilities."""
        if (
            not isinstance(window, np.ndarray) or window.shape != (48000,)
            or not np.issubdtype(window.dtype, np.floating) or not np.isfinite(window).all()
        ):
            raise ValueError("Expected a finite three-second mono waveform at 16 kHz.")
        x = window.astype(np.float32, copy=False)
        x = (x - x.mean()) / np.sqrt(x.var() + 1e-5)
        if not np.isfinite(x).all():
            raise ValueError("Waveform normalization is nonfinite.")
        output = self.session.run(None, {self.input_name: x[None]})[0]
        if (
            output.ndim != 3 or output.shape[0] != 1 or not output.shape[1]
            or output.shape[2] != self.token_count or output.dtype != np.float32
            or not np.isfinite(output).all()
        ):
            raise ValueError("Unexpected acoustic-model logits.")
        z = torch.from_numpy(output[0])
        rest = torch.ones(self.token_count, dtype=torch.bool)
        rest[self.columns] = False
        grouped = torch.cat([z[:, self.columns], z[:, rest].logsumexp(1, keepdim=True)], 1)

        features = (grouped - grouped.mean(-1, keepdim=True)).clamp(-20, 20) / 5
        self.adapter.eval()
        logits = grouped + self.adapter(features)
        if not torch.isfinite(logits).all():
            raise ValueError("Adapted logits are nonfinite.")
        kept = logits.double().softmax(-1)[:, self.kept].numpy()
        return np.column_stack([kept, np.maximum(0, 1 - kept.sum(1))])

    def __call__(self, audio, duration):
        """Scan non-overlapping windows; reject exact silence and clip the padded tail."""
        if (
            not isinstance(audio, np.ndarray) or audio.ndim != 1 or not audio.size
            or not np.issubdtype(audio.dtype, np.floating) or not np.isfinite(audio).all()
        ):
            raise ValueError("Expected nonempty, finite mono floating-point audio at 16 kHz.")
        if not math.isfinite(duration) or duration <= 0 or abs(len(audio) / 16000 - duration) > 1 / 16000 + 1e-9:
            raise ValueError("Duration must match the 16 kHz waveform.")

        events = []
        for offset in range(0, len(audio), 48000):
            block = audio[offset:offset + 48000]
            if not np.any(block):
                continue
            end = min(duration, (offset + len(block)) / 16000)
            probabilities = self.frame_probabilities(np.pad(block, (0, 48000 - len(block))))
            score = self.rule.probability(probabilities)
            if score < self.threshold:
                continue
            frames = locate_keyword_frames(probabilities, self.rule)
            if frames is None:
                continue
            interval = frames_to_seconds(frames, .02, offset_s=offset / 16000, end_s=end)
            if interval is not None:
                start, stop = interval
                events.append({"start": start, "end": stop, "score": score})
        return events
