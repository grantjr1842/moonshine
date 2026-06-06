"""Shared helpers for the Moonshine STT example scripts.

This module centralises the patterns that every example needs:

  * ``make_argparser`` — a base :class:`argparse.ArgumentParser` with the
    flags the examples all converge on (``--language``, ``--model-arch``,
    ``--embedding-model``, ``--quantization``, ``--threshold``, ``--mic``,
    ``--quiet``, ``--debug``).
  * ``default_wav_path`` / ``require_wav_path`` — locate the bundled test
    audio (``test-assets/two_cities.wav``).
  * ``load_stt_model`` / ``load_embedding_model`` — wrap the C-loading
    helpers in ``moonshine_voice.download``.
  * ``TranscriptPrinter`` — the canonical carriage-return-overwrite
    :class:`TranscriptEventListener` from ``mic_transcription.py``.
  * ``format_line`` — pretty-printer for every field on
    :class:`TranscriptLine`.
  * ``chunk_iter`` — yield successive chunks of a long PCM array, used to
    feed file audio into a streaming transcriber.

The examples themselves import from this module; nothing in ``common``
imports from the examples, so the dependency graph stays a clean tree.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, List, Optional, Tuple

from moonshine_voice import (
    ModelArch,
    TranscriptEventListener,
    TranscriptLine,
    get_model_for_language,
    load_wav_file,
)

if TYPE_CHECKING:
    # Imported only for type checkers so that --help still works on machines
    # where moonshine_voice is not installed.
    from moonshine_voice import Transcriber


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------


# The repo's test-assets directory (not the same as the package's bundled
# assets path, which only contains the tiny English model files).
_TEST_ASSETS_DIR = Path(__file__).resolve().parents[3] / "test-assets"


def default_wav_path() -> Path:
    """Path to the bundled ``two_cities.wav`` used by the offline demos."""
    return _TEST_ASSETS_DIR / "two_cities.wav"


def fallback_wav_path() -> Path:
    """Path to the 16 kHz pre-resampled copy, useful for benchmarking."""
    return _TEST_ASSETS_DIR / "two_cities_16k.wav"


def require_wav_path(cli_arg: Optional[Path]) -> Path:
    """Return ``cli_arg`` if given, else the default WAV.

    Raises ``FileNotFoundError`` with a friendly message if neither exists.
    """
    if cli_arg is not None:
        if not cli_arg.exists():
            raise FileNotFoundError(
                f"Specified WAV file does not exist: {cli_arg}\n"
                f"Hint: leave --wav-path empty to use the bundled sample at "
                f"{default_wav_path()}"
            )
        return cli_arg
    if not default_wav_path().exists():
        raise FileNotFoundError(
            f"Default WAV file not found at {default_wav_path()}.\n"
            f"Run from the repository root, or pass --wav-path."
        )
    return default_wav_path()


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def make_argparser(
    description: str,
    *,
    include_mic: bool = False,
    include_embedding: bool = False,
) -> argparse.ArgumentParser:
    """Build a standard argparse for the example scripts.

    The returned parser pre-defines the flags most examples need. Examples
    can call ``parser.add_argument(...)`` to add their own.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="BCP-47 language tag (default: en). Use --language foo to see "
        "the list of supported languages.",
    )
    parser.add_argument(
        "--model-arch",
        type=int,
        default=None,
        help="Override the architecture constant (see ModelArch). Default: "
        "auto-pick the best available for --language.",
    )
    parser.add_argument(
        "--wav-path",
        type=Path,
        default=None,
        help="Input WAV file. Default: test-assets/two_cities.wav.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-line printing — only the final summary is shown.",
    )
    parser.add_argument(
        "--no-speaker-ids",
        action="store_true",
        help="Don't print speaker prefixes.",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Request word-level timestamps (requires an attention-decoder "
        "variant of the model; falls back gracefully if unavailable).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose timing and option dumps on stderr.",
    )
    parser.add_argument(
        "--options",
        type=str,
        default=None,
        help="Raw C API options string, e.g. "
        '"vad_threshold=0.6,max_tokens_per_second=13.0". See README.',
    )

    if include_mic:
        parser.add_argument(
            "--mic",
            action="store_true",
            help="Read audio from the system microphone instead of a WAV file.",
        )
        parser.add_argument(
            "--device",
            type=str,
            default=None,
            help="sounddevice input device name or index (only with --mic).",
        )
        parser.add_argument(
            "--samplerate",
            type=int,
            default=16000,
            help="Requested input sample rate in Hz (only with --mic). The "
            "library resamples to 16 kHz internally.",
        )

    if include_embedding:
        parser.add_argument(
            "--embedding-model",
            type=str,
            default="embeddinggemma-300m",
            help="Sentence-embedding model for IntentRecognizer.",
        )
        parser.add_argument(
            "--quantization",
            type=str,
            default="q4",
            help="Embedding model quantization: q4, q8, fp16, fp32, q4f16.",
        )
        parser.add_argument(
            "--threshold",
            type=float,
            default=0.8,
            help="Similarity threshold (0-1) for intent matching.",
        )

    return parser


def parse_options_string(s: Optional[str]) -> dict:
    """Parse ``--options=key=value,key=value`` into a dict of strings.

    The C API consumes option values as strings, so the dict values are
    left as strings. Numeric thresholds are passed verbatim — the C side
    parses them.
    """
    if not s:
        return {}
    out: dict = {}
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(
                f"--options fragment must look like key=value, got {chunk!r}"
            )
        k, v = chunk.split("=", 1)
        out[k.strip()] = v.strip()
    return out


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


@dataclass
class LoadedSttModel:
    """Bundle of (Transcriber, architecture) the example scripts use a lot."""

    transcriber: "Transcriber"
    model_arch: ModelArch


def load_stt_model(
    language: str = "en",
    model_arch: Optional[int] = None,
    options: Optional[dict] = None,
    spelling_model_path: Optional[str] = None,
) -> Tuple["Transcriber", ModelArch]:
    """Resolve and load an STT model for the requested language.

    This is a thin wrapper around :func:`moonshine_voice.get_model_for_language`
    that handles the case where ``options`` is ``None`` (the C API needs an
    empty dict, not ``None``).
    """
    # Lazy import so that ``--help`` works before the C library is loaded.
    from moonshine_voice import Transcriber

    model_path, arch = get_model_for_language(language, model_arch)
    if options is None:
        options = {}
    t = Transcriber(
        model_path=model_path,
        model_arch=arch,
        options=options or None,
        spelling_model_path=spelling_model_path,
    )
    return t, arch


# ---------------------------------------------------------------------------
# Listener: pretty-print lines as they're produced
# ---------------------------------------------------------------------------


class TranscriptPrinter(TranscriptEventListener):
    """Carriage-return-overwrite printer for in-progress transcript lines.

    For each ``LineCompleted`` event it prints a clean line. For
    ``LineTextChanged`` it overwrites the current line so a terminal
    user sees the text growing in place.

    Mirrors the pattern used in ``examples/python/mic_transcription.py``
    and the other examples — factored out so the example scripts can share
    it.
    """

    def __init__(
        self,
        *,
        quiet: bool = False,
        show_speaker: bool = True,
        show_words: bool = False,
        prefix: str = "",
    ):
        self.quiet = quiet
        self.show_speaker = show_speaker
        self.show_words = show_words
        self.prefix = prefix
        self._last_len = 0
        self._completed_count = 0

    def _overwrite(self, text: str) -> None:
        print(f"\r{text}", end="", flush=True)
        if len(text) < self._last_len:
            # Pad with spaces to wipe the previous (longer) text.
            print(" " * (self._last_len - len(text)), end="", flush=True)
        self._last_len = len(text)

    def _speaker_prefix(self, line: TranscriptLine) -> str:
        if not self.show_speaker or not line.has_speaker_id:
            return ""
        return f"Speaker #{line.speaker_index}: "

    def on_line_started(self, event) -> None:
        # The event itself is unused; we just reset the carriage-return
        # tracking so the next on_line_text_changed starts fresh.
        del event
        self._last_len = 0

    def on_line_text_changed(self, event) -> None:
        if self.quiet:
            return
        line = event.line
        text = f"{self.prefix}{self._speaker_prefix(line)}{line.text}"
        self._overwrite(text)

    def on_line_completed(self, event) -> None:
        self._completed_count += 1
        if self.quiet:
            return
        line = event.line
        text = f"{self.prefix}{self._speaker_prefix(line)}{line.text}"
        # Wipe any in-progress line first.
        if self._last_len:
            print(f"\r{' ' * self._last_len}\r", end="", flush=True)
        print(text, flush=True)
        self._last_len = 0

        if self.show_words and line.words:
            for w in line.words:
                print(
                    f"   └─ {w.start:6.2f}s → {w.end:6.2f}s  "
                    f"conf={w.confidence:.2f}  {w.word!r}",
                    flush=True,
                )


# ---------------------------------------------------------------------------
# Pretty printing for the offline / one-shot path
# ---------------------------------------------------------------------------


def format_line(
    line: TranscriptLine,
    *,
    show_speaker: bool = True,
    show_words: bool = False,
    show_audio_len: bool = False,
) -> str:
    """Format a ``TranscriptLine`` as a one-line human-readable summary.

    All fields on the line are included; callers can suppress the noisier
    ones. Used by the offline (one-shot) examples where there's no
    carriage-return overwriting to worry about.
    """
    bits: List[str] = []
    if show_speaker and line.has_speaker_id:
        bits.append(f"Speaker #{line.speaker_index}")
    bits.append(
        f"[{line.start_time:6.2f}s → "
        f"{line.start_time + line.duration:6.2f}s]"
    )
    bits.append(f"{line.text!r}")
    if line.last_transcription_latency_ms:
        bits.append(f"({line.last_transcription_latency_ms} ms)")
    if line.line_id:
        bits.append(f"id={line.line_id}")
    if show_audio_len and line.audio_data:
        bits.append(f"audio={len(line.audio_data)} samples")
    out = " ".join(bits)
    if show_words and line.words:
        out += "\n" + "\n".join(
            f"   └─ {w.start:6.2f}s → {w.end:6.2f}s  "
            f"conf={w.confidence:.2f}  {w.word!r}"
            for w in line.words
        )
    return out


# ---------------------------------------------------------------------------
# Streaming helper: feed a WAV in chunks through a Transcriber
# ---------------------------------------------------------------------------


def chunk_iter(
    audio: List[float],
    sample_rate: int,
    chunk_duration: float = 0.1,
) -> Iterator[List[float]]:
    """Yield successive PCM chunks of ``chunk_duration`` seconds.

    The library's `add_audio` accepts any chunk size, but 100 ms is a good
    default — it matches the streaming model's 80 ms stride plus a small
    margin and is short enough that the implicit ``update_transcription``
    cadence in ``Stream.add_audio`` fires a few times per second.
    """
    chunk_size = max(1, int(chunk_duration * sample_rate))
    for i in range(0, len(audio), chunk_size):
        yield audio[i : i + chunk_size]


def stream_wav_to_transcriber(
    transcriber: "Transcriber",
    wav_path: Path,
    *,
    chunk_duration: float = 0.1,
) -> None:
    """Feed a WAV file into ``transcriber`` in 100 ms chunks.

    Calls ``start()`` / ``stop()`` on the default stream. The library
    handles resampling and VAD segmentation. Listeners attached to the
    default stream receive ``LineStarted``/``LineUpdated``/etc. events.
    """
    audio, sample_rate = load_wav_file(wav_path)
    transcriber.start()
    try:
        for chunk in chunk_iter(audio, sample_rate, chunk_duration):
            transcriber.add_audio(chunk, sample_rate)
    finally:
        transcriber.stop()


# ---------------------------------------------------------------------------
# Misc small helpers
# ---------------------------------------------------------------------------


def errprint(*args, **kwargs) -> None:
    """Print to stderr without buffering surprises."""
    print(*args, file=sys.stderr, **kwargs)


def hr(label: str = "", char: str = "─", width: int = 60) -> None:
    """Print a section divider, used between examples / phases."""
    if label:
        pad = max(1, (width - len(label) - 2) // 2)
        print(char * pad + f" {label} " + char * pad, file=sys.stderr)
    else:
        print(char * width, file=sys.stderr)


def first_n(iterable: Iterable, n: int) -> List:
    """Materialise the first ``n`` items of an iterable (handy in REPL demos)."""
    out: List = []
    it = iter(iterable)
    for _ in range(n):
        try:
            out.append(next(it))
        except StopIteration:
            break
    return out
