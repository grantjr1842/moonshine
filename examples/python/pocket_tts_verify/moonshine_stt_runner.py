"""Moonshine STT runner for the PocketTTS verification workflow.

Wraps :class:`moonshine_voice.Transcriber` with a one-call interface
that takes a PCM buffer and returns the recognised text. Uses the
same ``transcribe_without_streaming`` API as the
``examples/python/stt/01_offline_transcribe.py`` worked example.

The runner is deliberately minimal: it doesn't expose streaming,
events, or word timestamps. Those are documented in ``stt/`` for
users who want them; for batch round-trip evaluation, the simpler
API is easier to read and reason about.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

# All Moonshine imports are deferred so this file is importable in
# environments that only have PocketTTS installed.
_moonshine_transcriber = None
_moonshine_model_arch = None


def _load_transcriber(
    language: str,
    model_arch: Optional[int],
    options: dict,
):
    """Import ``moonshine_voice`` and load the STT model once."""
    global _moonshine_transcriber, _moonshine_model_arch
    if _moonshine_transcriber is not None:
        return _moonshine_transcriber, _moonshine_model_arch
    try:
        from moonshine_voice import Transcriber, get_model_for_language
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "moonshine_voice is required for this workflow. "
            "Install with `pip install moonshine-voice`. "
            f"(Original error: {exc!r})"
        ) from exc

    model_path, arch = get_model_for_language(language, model_arch)
    # Empty dict means "no options"; the C API treats None and {}
    # identically.
    _moonshine_transcriber = Transcriber(
        model_path=model_path,
        model_arch=arch,
        options=options or None,
    )
    _moonshine_model_arch = arch
    return _moonshine_transcriber, _moonshine_model_arch


@dataclass
class TranscriptionResult:
    """The recogniser's view of one wav.

    Attributes
    ----------
    text:
        Full transcript (all completed lines joined with spaces).
    line_texts:
        Per-line text, in time order. Empty for short single-segment audio.
    duration_sec:
        Audio duration in seconds (as reported by PocketTTS's sample rate
        and the sample count).
    latency_ms:
        Wall-clock time spent in the C library, in milliseconds.
    num_lines:
        How many completed lines the recogniser emitted.
    """

    text: str
    line_texts: List[str]
    duration_sec: float
    latency_ms: int
    num_lines: int


def transcribe(
    samples: List[float],
    sample_rate: int,
    *,
    language: str = "en",
    model_arch: Optional[int] = None,
    options: Optional[dict] = None,
) -> TranscriptionResult:
    """Transcribe a single PCM buffer.

    ``samples`` is a list of float in ``[-1.0, 1.0]`` at the given
    sample rate. PocketTTS outputs at 24 kHz; Moonshine's C API
    resamples to 16 kHz internally, so we just pass the rate through.
    """
    if not samples:
        return TranscriptionResult(
            text="", line_texts=[], duration_sec=0.0,
            latency_ms=0, num_lines=0,
        )
    transcriber, _ = _load_transcriber(language, model_arch, options or {})
    start = time.perf_counter()
    transcript = transcriber.transcribe_without_streaming(
        samples, sample_rate=sample_rate, flags=0
    )
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    line_texts = [line.text for line in transcript.lines]
    full_text = " ".join(t for t in line_texts if t)
    duration_sec = len(samples) / float(sample_rate) if sample_rate else 0.0
    return TranscriptionResult(
        text=full_text,
        line_texts=line_texts,
        duration_sec=duration_sec,
        latency_ms=elapsed_ms,
        num_lines=len(transcript.lines),
    )
