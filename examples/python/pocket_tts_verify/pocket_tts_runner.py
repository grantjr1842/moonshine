"""Thin wrapper around ``pocket_tts.TTSModel`` for the verification workflow.

The wrapper exists for three reasons:

1. **Lazy import.** Loading PocketTTS pulls in torch + several hundred MB
   of model weights. We want the rest of the workflow (and the ``--help``
   screen) to work without it.
2. **PCM conversion.** PocketTTS returns a ``torch.Tensor`` of float
   samples; Moonshine (and our wav writer) want a plain list of
   ``float`` and an int sample rate.
3. **A stable internal interface** so the workflow's main loop is
   decoupled from the upstream PocketTTS Python API.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

# All heavy imports are deferred to inside ``_load_pocket_tts_model``.
_pocket_tts_model = None


def _load_pocket_tts_model(
    language: str,
    *,
    temperature: float,
    lsd_decode_steps: int,
    eos_threshold: float,
    quantize: bool = False,
):
    """Import ``pocket_tts`` and load the model.

    Raises ``ImportError`` with a friendly message if the package
    isn't installed. The model is cached module-wide so repeated calls
    are free.
    """
    global _pocket_tts_model
    if _pocket_tts_model is not None:
        return _pocket_tts_model
    try:
        from pocket_tts import TTSModel
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PocketTTS is required for this workflow. "
            "Install with `pip install pocket-tts`. "
            f"(Original error: {exc!r})"
        ) from exc

    print(
        f"  loading PocketTTS model: language={language!r} "
        f"temp={temperature} lsd_steps={lsd_decode_steps}",
        file=sys.stderr,
    )
    print(
        "  (first call downloads ~600 MB of weights from HuggingFace)",
        file=sys.stderr,
    )
    _pocket_tts_model = TTSModel.load_model(
        language=language,
        temp=temperature,
        lsd_decode_steps=lsd_decode_steps,
        eos_threshold=eos_threshold,
        quantize=quantize,
    )
    return _pocket_tts_model


def _resolve_voice_state(model, voice: str) -> dict:
    """Resolve a voice name or path into a PocketTTS model state dict.

    PocketTTS's ``get_state_for_audio_prompt`` accepts a built-in voice
    name (e.g. ``"alba"``), a path to a wav, a HuggingFace ``hf://…``
    reference, or a path to a previously exported ``.safetensors``
    voice state.
    """
    # Built-in voice name — PocketTTS treats short strings as voice names.
    # A path with a slash or ending in .wav / .safetensors is treated
    # as a file.
    looks_like_path = (
        "/" in voice
        or "\\" in voice
        or voice.endswith(".wav")
        or voice.endswith(".safetensors")
    )
    if looks_like_path and Path(voice).exists():
        voice_path = Path(voice)
        if voice_path.suffix == ".safetensors":
            # Use safetensors.torch.load directly.
            from safetensors.torch import load_file
            return load_file(str(voice_path))
        return model.get_state_for_audio_prompt(str(voice_path))
    # Built-in voice name.
    return model.get_state_for_audio_prompt(voice)


def synthesize_sentence(
    text: str,
    *,
    language: str,
    voice: str,
    temperature: float,
    lsd_decode_steps: int,
    eos_threshold: float,
) -> Tuple[List[float], int]:
    """Synthesise one sentence.

    Returns ``(pcm_samples, sample_rate)``. ``pcm_samples`` is a
    flat ``list[float]`` in ``[-1.0, 1.0]`` (the same shape Moonshine
    expects). ``sample_rate`` is whatever PocketTTS reports for the
    loaded model (typically 24 kHz).
    """
    if not text.strip():
        return [], 0
    model = _load_pocket_tts_model(
        language,
        temperature=temperature,
        lsd_decode_steps=lsd_decode_steps,
        eos_threshold=eos_threshold,
    )
    state = _resolve_voice_state(model, voice)
    audio_tensor = model.generate_audio(state, text)
    # ``audio_tensor`` is a 1-D torch.Tensor of float32 samples.
    # ``.tolist()`` works on a torch tensor without us needing to
    # import torch directly (the conversion to Python float happens
    # inside the tensor's C++ binding).
    samples = audio_tensor.detach().cpu().float().tolist()
    sample_rate = int(model.sample_rate)
    return samples, sample_rate


def list_languages() -> List[str]:
    """Return the language codes supported by PocketTTS.

    Mirrors the upstream ``TTSModel.load_model`` signature. Updated
    against pocket-tts 0.2.x — adjust if a newer release adds codes.
    """
    return [
        "english_2026-01",
        "english_2026-04",
        "english",
        "french_24l",
        "german_24l",
        "portuguese_24l",
        "italian_24l",
        "spanish_24l",
    ]


def list_built_in_voices() -> List[str]:
    """The voices shipped with ``kyutai/tts-voices`` (English 2026-04)."""
    return [
        "alba",
        "marius",
        "javert",
        "jean",
        "fantine",
        "cosette",
        "eponine",
        "azelma",
    ]


def save_wav(path: Path, samples: List[float], sample_rate: int) -> None:
    """Write a 16-bit PCM mono WAV file.

    Uses Python's stdlib ``wave`` module so we don't need scipy just
    for the writer. The samples are clipped to ``[-1, 1]`` and
    converted to int16.
    """
    import struct
    import wave

    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert to int16 PCM.
    clipped = [max(-1.0, min(1.0, s)) for s in samples]
    pcm = b"".join(
        struct.pack("<h", int(round(s * 32767))) for s in clipped
    )
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
