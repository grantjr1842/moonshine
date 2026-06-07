"""Example 02 — streaming transcription with an event listener.

Reads a WAV file in 100 ms chunks and feeds it through
:meth:`Transcriber.add_audio` to mimic a live audio source. A
:class:`TranscriptEventListener` subclass receives the four lifecycle
events (``on_line_started``, ``on_line_text_changed``, ``on_line_completed``,
``on_error``) for each speech segment.

What this script demonstrates
-----------------------------
* :meth:`Transcriber.start`, :meth:`Transcriber.add_audio`,
  :meth:`Transcriber.stop` — the streaming API.
* :class:`moonshine_voice.TranscriptEventListener` subclassing with
  the four event callbacks.
* :meth:`Stream.push_listener` / :meth:`Stream.pop_listener` — temporarily
  swap the active listener (here we push a "background task" listener that
  swallows events for a couple of chunks so the user can see the swap).
* :func:`moonshine_voice.Transcriber.update_transcription` — manual
  flush of the cached transcript with ``MOONSHINE_FLAG_FORCE_UPDATE``.

Run it
------
    python -m examples.python.stt.02_streaming_transcribe
"""

from __future__ import annotations

import time
from typing import List

from moonshine_voice import (
    LineCompleted,
    LineStarted,
    LineTextChanged,
    TranscriptEventListener,
)
from moonshine_voice.transcriber import MOONSHINE_FLAG_FORCE_UPDATE

from . import common


class LineCounterListener(TranscriptEventListener):
    """Counts events, demonstrates the per-event API surface."""

    def __init__(self) -> None:
        self.started = 0
        self.text_changes = 0
        self.completed = 0
        self.last_text: str = ""

    def on_line_started(self, event: LineStarted) -> None:
        self.started += 1
        common.errprint(
            f"  [event] line #{self.started} started at "
            f"{event.line.start_time:.2f}s (line_id={event.line.line_id})"
        )

    def on_line_text_changed(self, event: LineTextChanged) -> None:
        self.text_changes += 1
        self.last_text = event.line.text

    def on_line_completed(self, event: LineCompleted) -> None:
        self.completed += 1
        common.errprint(
            f"  [event] line #{event.line.line_id} completed at "
            f"{event.line.start_time + event.line.duration:.2f}s "
            f"({len(event.line.text)} chars, "
            f"{event.line.last_transcription_latency_ms} ms)"
        )

    def on_error(self, event) -> None:
        common.errprint(f"  [event] error: {event.error!r}")


def feed_wav_in_chunks(
    transcriber, audio: List[float], sample_rate: int
) -> None:
    """Feed PCM audio in 100 ms chunks, like a mic callback would."""
    chunk_count = 0
    for chunk in common.chunk_iter(audio, sample_rate, chunk_duration=0.1):
        transcriber.add_audio(chunk, sample_rate)
        chunk_count += 1
    common.errprint(f"  streamed {chunk_count} chunks of ~100 ms each")


def demonstrate_push_pop(transcriber) -> None:
    """Temporarily replace the active listener stack.

    ``push_listener`` saves the current listeners and installs a single new
    one. ``pop_listener`` restores the previous list. This is useful when a
    background task (a notification toast, a TTS reply) shouldn't update the
    main transcript display.
    """
    common.hr("push_listener / pop_listener")
    common.errprint("  → swapping to a 'silent' background listener for 3 chunks")

    swallowed: List[str] = []

    class BackgroundListener(TranscriptEventListener):
        def on_line_completed(self, event: LineCompleted) -> None:
            swallowed.append(event.line.text)

    transcriber.push_listener(BackgroundListener())
    # The transcriber's update_interval is 0.5 s, so we just sleep briefly
    # to let the stream emit zero-or-one events while the background
    # listener is in place.
    time.sleep(0.6)
    transcriber.pop_listener()
    common.errprint(f"  → swallowed {len(swallowed)} line(s) during the swap")
    common.errprint("  → main listener stack restored")


def demonstrate_force_update(transcriber) -> None:
    """Manually flush the transcript ignoring the 200 ms time-based cache."""
    common.hr("update_transcription(MOONSHINE_FLAG_FORCE_UPDATE)")
    transcript = transcriber.update_transcription(MOONSHINE_FLAG_FORCE_UPDATE)
    common.errprint(
        f"  forced update returned {len(transcript.lines)} lines "
        "(same as the cached version here, but useful when polling)"
    )


def main() -> None:
    parser = common.make_argparser(
        description="Streaming transcription with an event listener.",
        include_self_check=True,
    )
    args = parser.parse_args()

    if args.self_check:
        common.run_self_check(
            "02_streaming_transcribe",
            lambda: _self_check(args),
        )
        return

    wav_path = common.require_wav_path(args.wav_path)

    common.hr("Loading")
    transcriber, arch = common.load_stt_model(language=args.language)
    common.errprint(f"  model_arch = {arch.name}")

    # Listeners — both the printer and the counter — attach to the
    # default stream that ``start()`` will create.
    printer = common.TranscriptPrinter(
        quiet=args.quiet,
        show_speaker=not args.no_speaker_ids,
        show_words=args.word_timestamps,
    )
    counter = LineCounterListener()
    transcriber.add_listener(printer)
    transcriber.add_listener(counter)

    common.hr("Streaming")
    transcriber.start()
    try:
        audio, sample_rate = common.load_wav_file(wav_path)
        common.errprint(
            f"  {len(audio):,} samples @ {sample_rate} Hz from {wav_path}"
        )
        feed_wav_in_chunks(transcriber, audio, sample_rate)
        demonstrate_push_pop(transcriber)
        demonstrate_force_update(transcriber)
    finally:
        # stop() emits a final transcription event for any audio in
        # flight, which is critical for accurate last-line reporting.
        common.errprint("  stopping stream (final flush)…")
        transcriber.stop()

    common.hr("Event totals")
    print(f"  started        : {counter.started}")
    print(f"  text_changes   : {counter.text_changes}")
    print(f"  completed      : {counter.completed}")
    print(f"  last text      : {counter.last_text!r}")

    transcriber.close()


def _self_check(args) -> "SelfCheckResult | None":
    """Smoke test: stream two_cities.wav, assert ≥ 1 line completed."""
    from test_support.self_check import SelfCheckResult

    wav_path = common.default_wav_path()
    if not wav_path.exists():
        return SelfCheckResult.skip(
            f"missing test audio: {wav_path}", "02_streaming_transcribe"
        )
    transcriber, _arch = common.load_stt_model(language=args.language)
    try:
        counter = LineCounterListener()
        transcriber.add_listener(counter)
        transcriber.start()
        try:
            audio, sample_rate = common.load_wav_file(wav_path)
            feed_wav_in_chunks(transcriber, audio, sample_rate)
        finally:
            transcriber.stop()
        if counter.completed < 1:
            return SelfCheckResult.fail(
                f"no line.completed events (started={counter.started})",
                "02_streaming_transcribe",
            )
        return None  # PASS
    finally:
        transcriber.close()


if __name__ == "__main__":
    main()
