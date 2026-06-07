"""Example 06 — multiple audio sources on a single Transcriber.

A single :class:`moonshine_voice.Transcriber` can host several audio
streams. Each stream has its own audio buffer, VAD state, and transcript
— but they all share the underlying model weights, so you don't pay the
load cost twice.

This is the canonical pattern for "transcribe the microphone and the
system audio at the same time without loading two copies of the model"
— see the contract documented in ``core/moonshine-c-api.h``.

The Python ``Transcriber`` class only exposes the *default* stream through
its high-level API. For multi-stream access we drop down to the ctypes
``_lib`` handle and call the C functions directly, the same way
``Stream`` itself does internally.

What this script demonstrates
-----------------------------
* Calling ``moonshine_create_stream`` / ``moonshine_transcribe_*`` /
  ``moonshine_free_stream`` on a single shared ``moonshine_load_transcriber_*``
  handle.
* Routing events from each stream handle to a per-source output channel.
* The "one model, many inputs" memory saving.

Run it
------
    python -m examples.python.stt.06_multi_stream
    python -m examples.python.stt.06_multi_stream \\
        --file-a path/a.wav --file-b path/b.wav
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from moonshine_voice import (
    Transcript,
    get_model_for_language,
    load_wav_file,
)
# ``TranscriptC`` and ``check_error`` live one level down from
# the top-level package; importing them from ``moonshine_voice``
# directly raises ImportError. (Earlier this example had two
# wrong import lines that broke the script at module load time
# — the smoke test under --self-check caught both.)
from moonshine_voice.errors import check_error
from moonshine_voice.moonshine_api import TranscriptC, _MoonshineLib

from . import common


def open_extra_stream(lib, transcriber_handle: int) -> int:
    """Create a new stream on an existing transcriber handle. Returns the C handle."""
    err = lib.moonshine_create_stream(transcriber_handle, 0)
    check_error(err)  # raises on negative error code, returns the int handle
    return err


def feed_stream(
    lib,
    transcriber_handle: int,
    stream_handle: int,
    wav_path: Path,
    *,
    update_interval: float = 0.5,
) -> Transcript:
    """Push audio in 100 ms chunks into a stream and return its final transcript.

    Mirrors the higher-level :meth:`moonshine_voice.Stream.add_audio` cadence:
    every ``update_interval`` seconds of accumulated audio we call
    ``moonshine_transcribe_stream`` to give the VAD a chance to finalise
    segments and emit completed transcript lines. Without this, the VAD
    only finalises on the very last call (inside ``stop_stream``) and the
    transcript is effectively empty.

    The intermediate ``moonshine_transcribe_stream`` results are discarded
    — we only need the *final* transcript at the end. The intermediate
    calls exist purely to flush the VAD state.
    """
    audio, sample_rate = load_wav_file(wav_path)
    err = lib.moonshine_start_stream(transcriber_handle, stream_handle)
    check_error(err)

    # Track elapsed stream time the same way ``Stream.add_audio`` does so
    # the VAD sees periodic flushes instead of one giant flush at the end.
    stream_time = 0.0
    last_update_time = 0.0
    discard = ctypes.POINTER(TranscriptC)()
    try:
        for chunk in common.chunk_iter(audio, sample_rate):
            arr = (ctypes.c_float * len(chunk))(*chunk)
            err = lib.moonshine_transcribe_add_audio_to_stream(
                transcriber_handle,
                stream_handle,
                arr,
                len(chunk),
                sample_rate,
                0,
            )
            check_error(err)
            stream_time += len(chunk) / sample_rate
            if stream_time - last_update_time >= update_interval:
                err = lib.moonshine_transcribe_stream(
                    transcriber_handle, stream_handle, 0, ctypes.byref(discard)
                )
                check_error(err)
                last_update_time = stream_time
    finally:
        err = lib.moonshine_stop_stream(transcriber_handle, stream_handle)
        check_error(err)

    # Pull the final transcript out via the ctypes struct.
    out = ctypes.POINTER(TranscriptC)()
    err = lib.moonshine_transcribe_stream(
        transcriber_handle, stream_handle, 0, ctypes.byref(out)
    )
    check_error(err)
    return _decode_transcript(out)


def _decode_transcript(ptr) -> Transcript:
    """Mirror the work in ``Transcriber._parse_transcript`` for a raw C pointer."""
    if not ptr:
        return Transcript(lines=[])
    t = ptr.contents
    # ``TranscriptLine`` is re-exported from the top-level
    # ``moonshine_voice`` package; ``WordTiming`` is only
    # available from ``moonshine_voice.moonshine_api``. The
    # original example imported both from the top-level
    # package, which raised ImportError for ``WordTiming``.
    from moonshine_voice import TranscriptLine
    from moonshine_voice.moonshine_api import WordTiming

    lines = []
    for i in range(t.line_count):
        line_c = t.lines[i]
        text = ""
        if line_c.text:
            text = ctypes.string_at(line_c.text).decode("utf-8", errors="ignore")
        words = None
        if line_c.words and line_c.word_count > 0:
            words = []
            for j in range(line_c.word_count):
                wc = line_c.words[j]
                wt = ctypes.string_at(wc.text).decode("utf-8", errors="ignore") if wc.text else ""
                words.append(WordTiming(word=wt, start=wc.start, end=wc.end, confidence=wc.confidence))
        lines.append(
            TranscriptLine(
                text=text,
                start_time=line_c.start_time,
                duration=line_c.duration,
                line_id=line_c.id,
                is_complete=bool(line_c.is_complete),
                audio_data=None,
                words=words,
            )
        )
    return Transcript(lines=lines)


def main() -> None:
    parser = common.make_argparser(
        description="Two audio sources transcribed on a single Transcriber.",
        include_self_check=True,
    )
    parser.add_argument(
        "--file-a",
        type=Path,
        default=common.default_wav_path(),
        help="First audio source. Default: two_cities.wav.",
    )
    parser.add_argument(
        "--file-b",
        type=Path,
        default=common._TEST_ASSETS_DIR / "beckett.wav",
        help="Second audio source. Default: beckett.wav.",
    )
    args = parser.parse_args()

    if args.self_check:
        common.run_self_check(
            "06_multi_stream",
            lambda: _self_check(args),
        )
        return

    if not args.file_a.exists():
        common.errprint(f"  --file-a not found: {args.file_a}")
        return
    if not args.file_b.exists():
        common.errprint(f"  --file-b not found: {args.file_b}")
        return

    common.hr("Loading model")
    model_path, arch = get_model_for_language(args.language, args.model_arch)
    common.errprint(f"  model_arch = {arch.name}")

    lib = _MoonshineLib().lib
    model_path_bytes = str(model_path).encode("utf-8")
    transcriber_handle = lib.moonshine_load_transcriber_from_files(
        model_path_bytes, int(arch), None, 0, 20000
    )
    check_error(transcriber_handle)
    common.errprint(f"  transcriber handle = {transcriber_handle}")

    try:
        common.hr("Creating streams")
        handle_a = open_extra_stream(lib, transcriber_handle)
        handle_b = open_extra_stream(lib, transcriber_handle)
        common.errprint(f"  stream A handle = {handle_a}")
        common.errprint(f"  stream B handle = {handle_b}")
        common.errprint("  both share the same loaded model weights")

        common.hr(f"Streaming source A: {args.file_a.name}")
        ta = feed_stream(lib, transcriber_handle, handle_a, args.file_a)
        for line in ta.lines:
            print(f"  [A] [{line.start_time:5.2f}s → "
                  f"{line.start_time + line.duration:5.2f}s]  {line.text!r}")

        common.hr(f"Streaming source B: {args.file_b.name}")
        tb = feed_stream(lib, transcriber_handle, handle_b, args.file_b)
        for line in tb.lines:
            print(f"  [B] [{line.start_time:5.2f}s → "
                  f"{line.start_time + line.duration:5.2f}s]  {line.text!r}")

        common.hr("Tear down")
        check_error(lib.moonshine_free_stream(transcriber_handle, handle_a))
        check_error(lib.moonshine_free_stream(transcriber_handle, handle_b))
        common.errprint("  streams freed")
    finally:
        lib.moonshine_free_transcriber(transcriber_handle)
        common.errprint("  transcriber freed")


def _self_check(args) -> "SelfCheckResult | None":
    """Smoke test: open two streams, feed two WAVs, assert ≥ 1 line each.

    Does **not** exercise the partial-failure stream-A leak
    documented in the plan's "Known gaps" — only the happy path.
    """
    from test_support.self_check import SelfCheckResult

    if not args.file_a.exists():
        return SelfCheckResult.skip(
            f"missing --file-a: {args.file_a}", "06_multi_stream"
        )
    if not args.file_b.exists():
        return SelfCheckResult.skip(
            f"missing --file-b: {args.file_b}", "06_multi_stream"
        )

    common.hr("Loading model")
    model_path, arch = get_model_for_language(args.language, args.model_arch)

    lib = _MoonshineLib().lib
    model_path_bytes = str(model_path).encode("utf-8")
    transcriber_handle = lib.moonshine_load_transcriber_from_files(
        model_path_bytes, int(arch), None, 0, 20000
    )
    check_error(transcriber_handle)
    try:
        handle_a = open_extra_stream(lib, transcriber_handle)
        handle_b = open_extra_stream(lib, transcriber_handle)
        try:
            ta = feed_stream(lib, transcriber_handle, handle_a, args.file_a)
            tb = feed_stream(lib, transcriber_handle, handle_b, args.file_b)
            if not ta.lines or not tb.lines:
                # Previously this branch was a SKIP — feed_stream
                # only called ``moonshine_transcribe_stream`` once
                # at the end, so the VAD never finalised any
                # segments. That bug is fixed: feed_stream now
                # flushes every 0.5s, matching the cadence of the
                # high-level ``Stream.add_audio``. An empty
                # transcript is now a real failure, not a known
                # limitation.
                return SelfCheckResult.fail(
                    "expected ≥ 1 transcript line from each of the "
                    f"two streams; got A={len(ta.lines)} B={len(tb.lines)}",
                    "06_multi_stream",
                )
            return None  # PASS
        finally:
            lib.moonshine_free_stream(transcriber_handle, handle_a)
            lib.moonshine_free_stream(transcriber_handle, handle_b)
    finally:
        lib.moonshine_free_transcriber(transcriber_handle)


if __name__ == "__main__":
    main()
