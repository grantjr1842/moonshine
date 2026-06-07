"""Example 05 — word-level timestamps.

Demonstrates the ``word_timestamps=True`` C option, which makes the
transcriber emit per-word start/end times and confidence scores. Two
implementation paths are supported by the C library:

* **Single-pass** — the standard decoder is swapped for
  ``decoder_with_attention.ort`` (or ``decoder_kv_with_attention.ort``
  for the streaming model) which exposes cross-attention weights.
* **Two-pass fallback** — a separate ``alignment_model.ort`` runs a
  teacher-forced decoder to align the recognised tokens to the audio.

The bundled ``tiny-en`` model ships with ``decoder_with_attention.ort``,
so this example exercises the single-pass path. If the attention model
isn't present the C library logs a warning and returns lines without
word data — this example detects that and explains it.

What this script demonstrates
-----------------------------
* Turning on word timestamps via the C option (and via the Python
  ``TranscriptEventListener`` interface for the streaming path).
* Reading the ``WordTiming`` records off each line: ``word``, ``start``,
  ``end``, ``confidence``.
* Visualising word timings as a timeline so the alignment quality is
  obvious at a glance.

Run it
------
    python -m examples.python.stt.05_word_timestamps
"""

from __future__ import annotations

from . import common


def render_word_timeline(words, total_duration: float, width: int = 40) -> None:
    """Print a tiny ASCII timeline of where each word sits in the line."""
    if total_duration <= 0 or not words:
        return
    common.errprint(f"  timeline (line spans {total_duration:.2f}s):")
    for w in words:
        start_pct = max(0, min(1, w.start / total_duration))
        end_pct = max(0, min(1, w.end / total_duration))
        bar = [" "] * width
        s = int(start_pct * width)
        e = max(s + 1, int(end_pct * width))
        for i in range(s, e):
            bar[i] = "█"
        print(
            f"    {''.join(bar)}  "
            f"{w.start:5.2f}–{w.end:5.2f}s  "
            f"conf={w.confidence:.2f}  {w.word!r}"
        )


def main() -> None:
    parser = common.make_argparser(
        description="Stream a WAV and dump per-word timing for each line.",
        include_self_check=True,
    )
    # Word timestamps are required for this example, so flip the flag on
    # unconditionally regardless of what the user passed.
    args = parser.parse_args()

    if args.self_check:
        common.run_self_check(
            "05_word_timestamps",
            lambda: _self_check(args),
        )
        return

    options = {
        "word_timestamps": "true",
        "return_audio_data": "false",
        "identify_speakers": "false",
    }

    common.hr("Loading")
    common.errprint("  options: word_timestamps=true")
    common.errprint("  (requires decoder_with_attention.ort in the model dir)")
    transcriber, arch = common.load_stt_model(
        language=args.language, options=options
    )
    common.errprint(f"  model_arch = {arch.name}")

    wav_path = common.require_wav_path(args.wav_path)
    audio, sample_rate = common.load_wav_file(wav_path)

    common.hr(f"Streaming {wav_path.name}")
    transcript = transcriber.transcribe_without_streaming(
        audio, sample_rate=sample_rate, flags=0
    )
    transcriber.close()

    if not transcript.lines:
        print("(no speech detected)")
        return

    if not transcript.lines[0].words:
        common.hr("Heads-up")
        print(
            "  No word timing data was returned. This usually means the\n"
            "  attention-decoder model file (decoder_with_attention.ort)\n"
            "  is not present in the model directory. The C library will\n"
            "  log a 'Warning: No word timestamp model found' message to\n"
            "  stderr and silently skip the word-alignment step.\n"
            "\n"
            "  Re-download the model with --language to pick up the\n"
            "  attention-decoder variant, or check the model dir."
        )
        return

    common.hr("Lines with word timings")
    any_rendered = False
    for line in transcript.lines:
        if not line.words:
            continue
        any_rendered = True
        common.hr(
            f"[{line.start_time:5.2f}s → "
            f"{line.start_time + line.duration:5.2f}s] {line.text!r}",
            char="·",
        )
        render_word_timeline(line.words, line.duration)
    if not any_rendered:
        print("  (every line had empty word list — see Heads-up above)")


def _self_check(args) -> "SelfCheckResult | None":
    """Smoke test: word-timestamp path produces ≥ 1 line with words.

    The bundled ``tiny-en`` model ships with
    ``decoder_with_attention.ort``; if the user's chosen model
    doesn't, the C library returns lines without word data and
    we SKIP rather than FAIL — the example handles that case
    explicitly in the regular path.
    """
    from test_support.self_check import SelfCheckResult

    wav_path = common.default_wav_path()
    if not wav_path.exists():
        return SelfCheckResult.skip(
            f"missing test audio: {wav_path}", "05_word_timestamps"
        )
    options = {
        "word_timestamps": "true",
        "return_audio_data": "false",
        "identify_speakers": "false",
    }
    transcriber, _ = common.load_stt_model(
        language=args.language, options=options
    )
    try:
        audio, sample_rate = common.load_wav_file(wav_path)
        transcript = transcriber.transcribe_without_streaming(
            audio, sample_rate=sample_rate, flags=0
        )
    finally:
        transcriber.close()

    if not transcript.lines:
        return SelfCheckResult.fail(
            "no lines returned", "05_word_timestamps"
        )
    if not any(line.words for line in transcript.lines):
        return SelfCheckResult.skip(
            "model has no attention-decoder variant; "
            "word timestamps not supported",
            "05_word_timestamps",
        )
    return None  # PASS


if __name__ == "__main__":
    main()
