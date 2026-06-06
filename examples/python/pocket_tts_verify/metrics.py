"""Pure metrics functions for the PocketTTS round-trip evaluator.

Everything in here is dependency-light and easy to unit-test. The
heavier dependencies (``jiwer``, ``whisper.normalizers``) are imported
lazily so the module is usable in environments that only have the
Moonshine package installed.
"""

from __future__ import annotations

import math
from typing import Sequence


# ---------------------------------------------------------------------------
# Lazy imports for the heavy deps.
# ---------------------------------------------------------------------------

_jiwer = None
_whisper_normalizer_unavailable = False
_whisper_normalizer = None


def _get_jiwer():
    """Return the imported ``jiwer`` module, importing it on first use."""
    global _jiwer
    if _jiwer is None:
        try:
            import jiwer  # noqa: WPS433 — lazy import is intentional
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "jiwer is required for WER/CER computation. "
                "Install with `pip install jiwer`."
            ) from exc
        _jiwer = jiwer
    return _jiwer


def _get_english_normalizer():
    """Return the Whisper EnglishTextNormalizer, or None if unavailable.

    The cache is split into two globals: ``_whisper_normalizer`` holds
    the normalizer instance (or None if the import failed), and
    ``_whisper_normalizer_unavailable`` is a separate sentinel so the
    success and failure paths don't share a single nullable variable.
    """
    global _whisper_normalizer, _whisper_normalizer_unavailable
    if _whisper_normalizer is None and not _whisper_normalizer_unavailable:
        try:
            from whisper.normalizers import EnglishTextNormalizer
        except ImportError:
            _whisper_normalizer_unavailable = True
        else:
            _whisper_normalizer = EnglishTextNormalizer()
    return _whisper_normalizer


# ---------------------------------------------------------------------------
# WER / CER
# ---------------------------------------------------------------------------


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate, 0.0–1.0+. Higher is worse.

    Uses ``jiwer.wer`` after light normalisation. Returns 1.0 for an
    empty reference (no words to align), matching the jiwer default.
    """
    if not reference:
        return 1.0 if hypothesis else 0.0
    jiwer = _get_jiwer()
    return float(jiwer.wer(reference, hypothesis))


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate, 0.0–1.0+. Higher is worse."""
    if not reference:
        return 1.0 if hypothesis else 0.0
    jiwer = _get_jiwer()
    return float(jiwer.cer(reference, hypothesis))


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------


def normalise(text: str, language: str) -> str:
    """Normalise text for fair WER/CER comparison.

    For English locales, applies the Whisper EnglishTextNormalizer
    (lowercase, expand contractions, strip punctuation, etc). For
    other languages, applies a light pass — strip whitespace, NFC
    unicode, no lowercasing (most non-Latin scripts are case-less or
    have meaningful case).
    """
    if text is None:
        return ""
    text = text.strip()
    if not text:
        return ""
    if language.startswith("en"):
        normalizer = _get_english_normalizer()
        if normalizer is not None:
            return normalizer(text)
    # Light fallback: NFC + collapse internal whitespace.
    import unicodedata

    return " ".join(unicodedata.normalize("NFC", text).split())


def exact_match(reference: str, hypothesis: str, language: str) -> bool:
    """True iff the normalised reference and hypothesis are byte-equal."""
    return normalise(reference, language) == normalise(hypothesis, language)


# ---------------------------------------------------------------------------
# Audio-level stats
# ---------------------------------------------------------------------------


def audio_stats(
    samples: Sequence[float],
    sample_rate: int,
    *,
    silence_threshold: float = 1e-3,
) -> dict:
    """Cheap audio-level sanity checks for a synthesized wav.

    Returns a dict with duration, peak, RMS, and silence ratio. These
    are useful for catching the "TTS produced silence" failure mode
    (often caused by a bad voice embedding or a misconfigured
    PocketTTS language), which would otherwise show up as a
    spuriously perfect 0% WER — the recogniser heard nothing and
    produced nothing.
    """
    if not samples:
        return {
            "duration_sec": 0.0,
            "peak_amplitude": 0.0,
            "rms": 0.0,
            "silence_ratio": 1.0,
            "num_samples": 0,
        }
    n = len(samples)
    duration = n / float(sample_rate) if sample_rate > 0 else 0.0
    peak = max(abs(s) for s in samples)
    if peak <= 0.0:
        rms = 0.0
        silence_ratio = 1.0
    else:
        rms = math.sqrt(sum(s * s for s in samples) / n)
        # silence = absolute sample below threshold, OR NaN-like
        silent = sum(1 for s in samples if abs(s) < silence_threshold)
        silence_ratio = silent / n
    return {
        "duration_sec": duration,
        "peak_amplitude": float(peak),
        "rms": float(rms),
        "silence_ratio": float(silence_ratio),
        "num_samples": n,
    }


def is_audio_silent(
    samples: Sequence[float],
    *,
    silence_threshold: float = 1e-3,
    min_duration_sec: float = 0.05,
    sample_rate: int = 16000,
) -> bool:
    """True if the audio is effectively silent (no signal above the threshold)."""
    stats = audio_stats(
        samples, sample_rate, silence_threshold=silence_threshold
    )
    if stats["duration_sec"] < min_duration_sec:
        return True
    if stats["peak_amplitude"] < silence_threshold:
        return True
    if stats["silence_ratio"] > 0.95:
        return True
    return False
