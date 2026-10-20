"""Load the external CTC models used by the alignment experiments.

Weights are not distributed here. Cache-only by default; see builder docs
for the separate noncommercial MMS weight license."""

from __future__ import annotations

import hashlib
from pathlib import Path


def digest(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_model(engine, language, *, allow_download=False):
    import torch
    import torchaudio

    processor = None
    if engine not in {"mms", "native"} or language not in {"ar", "fr", "es", "en"}:
        raise ValueError("Expected MMS/native and one of ar, fr, es, en")

    def require_cached(filename):
        checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / filename
        if not allow_download and not checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing cached aligner: {filename}. "
                "Download explicitly with --download-models after checking its license."
            )

    if engine == "mms":
        bundle = torchaudio.pipelines.MMS_FA
        require_cached(Path(bundle._path).name)
        model = bundle.get_model(with_star=False).eval()
        dictionary = bundle.get_dict(star=None)
        blank = 0
        identity = {
            "name": "torchaudio.pipelines.MMS_FA",
            "checkpoint_sha256": digest(
                Path(torch.hub.get_dir()) / "checkpoints/model.pt"
            ),
        }
    elif language == "ar":
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        name = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
        revision = "af46c2d8531b8dcbb5e23b952f739b372c2e5d2d"
        processor = Wav2Vec2Processor.from_pretrained(
            name, revision=revision, local_files_only=not allow_download
        )
        model = Wav2Vec2ForCTC.from_pretrained(
            name,
            revision=revision,
            use_safetensors=False,
            local_files_only=not allow_download,
        ).eval()
        dictionary = processor.tokenizer.get_vocab()
        blank = model.config.pad_token_id
        identity = {"name": name, "revision": revision}
    else:
        pipeline_names = {
            "fr": "VOXPOPULI_ASR_BASE_10K_FR",
            "es": "VOXPOPULI_ASR_BASE_10K_ES",
            "en": "WAV2VEC2_ASR_BASE_960H",
        }
        name = pipeline_names[language]
        bundle = getattr(torchaudio.pipelines, name)
        require_cached(Path(bundle._path).name)
        model = bundle.get_model().eval()
        dictionary = {
            character.lower(): index
            for index, character in enumerate(bundle.get_labels())
        }
        blank = 0
        checkpoint = (
            Path(torch.hub.get_dir())
            / "checkpoints"
            / Path(bundle._path).name
        )
        identity = {
            "name": name,
            "checkpoint_sha256": digest(checkpoint),
        }

    return model, processor, dictionary, blank, identity
