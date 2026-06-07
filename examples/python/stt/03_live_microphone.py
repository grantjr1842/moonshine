"""Example 03 — live microphone transcription with MicTranscriber.

Uses :class:`moonshine_voice.MicTranscriber` to attach to the system
microphone and stream audio into the recognizer in real time. The same
``TranscriptEventListener`` API works here — the difference is that audio
arrives asynchronously through a ``sounddevice`` callback instead of
``add_audio``.

What this script demonstrates
-----------------------------
* :class:`moonshine_voice.MicTranscriber` — wraps ``sounddevice`` and
  ``Transcriber`` so you don't have to manage the audio thread.
* Per-device capture (``--device``), including the **sample-rate
  fallback** path: when the device doesn't support 16 kHz natively
  (common on USB mics that lock to 44.1/48 kHz) the constructor
  detects the situation and falls back to the device's native rate.
  The C library resamples internally, so the model still sees 16 kHz.
* A :class:`TranscriptEventListener` that distinguishes between
  "in-progress" partial text and "completed" final text — the
  idiomatic way to drive a live terminal / web UI.

Run it
------
    python -m examples.python.stt.03_live_microphone           # file mode
    python -m examples.python.stt.03_live_microphone --mic    # live mic
    python -m examples.python.stt.03_live_microphone --mic --device "USB"
    python -m examples.python.stt.03_live_microphone --mic --samplerate 48000
"""

from __future__ import annotations

import time

from . import common


def main() -> None:
    parser = common.make_argparser(
        description="Transcribe live microphone input. Defaults to the "
        "bundled two_cities.wav if --mic is not given.",
        include_mic=True,
        include_self_check=True,
    )
    args = parser.parse_args()

    if args.self_check:
        # Force --mic mode under self-check; the fake sounddevice
        # shim supplies the audio.
        args.mic = True
        common.run_self_check(
            "03_live_microphone",
            lambda: _self_check(args),
        )
        return

    common.hr("Loading")
    transcriber, arch = common.load_stt_model(language=args.language)
    common.errprint(f"  model_arch = {arch.name}")

    printer = common.TranscriptPrinter(
        quiet=args.quiet,
        show_speaker=not args.no_speaker_ids,
        show_words=args.word_timestamps,
    )

    if args.mic:
        # Live capture: defer the sounddevice import to --mic mode so this
        # script remains importable on machines without audio hardware.
        from moonshine_voice import MicTranscriber

        common.hr("Microphone")
        # When --device isn't given MicTranscriber uses the system default.
        # When --samplerate is something the device doesn't support,
        # MicTranscriber queries the device's native rate and falls back.
        kwargs = dict(
            model_path=transcriber._model_path,  # reuse what load_stt_model picked
            model_arch=arch,
            samplerate=args.samplerate,
        )
        if args.device is not None:
            # Allow either int index or substring match.
            try:
                kwargs["device"] = int(args.device)
            except ValueError:
                kwargs["device"] = args.device
        mic = MicTranscriber(**kwargs)
        mic.add_listener(printer)

        common.errprint(
            f"  samplerate   = {mic._samplerate} Hz (the C API resamples "
            "to 16 kHz internally)"
        )
        common.errprint("  press Ctrl+C to stop")
        mic.start()
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            common.errprint("\n  stopping…")
        finally:
            mic.stop()
            mic.close()
    else:
        # File mode: same listener API, but we feed audio in 100 ms
        # chunks. Useful to confirm the listener wiring without a mic.
        common.hr("File mode")
        wav_path = common.require_wav_path(args.wav_path)
        transcriber.add_listener(printer)
        common.stream_wav_to_transcriber(transcriber, wav_path)
        common.errprint("  done — replay with --mic for live input.")

    transcriber.close()


def _self_check(args) -> "SelfCheckResult | None":
    """Smoke test: drive MicTranscriber with the fake mic for 10 s.

    The canned audio is 44 s of speech — long enough to fire
    multiple ``on_line_completed`` events. We bail after 10 s
    wall-clock to keep the test suite fast. Empirically, the
    first line completes around t=7 s with the bundled audio.
    """
    from test_support.self_check import SelfCheckResult
    from moonshine_voice import MicTranscriber

    transcriber, arch = common.load_stt_model(language=args.language)
    counter = _LineCounter()
    try:
        mic = MicTranscriber(
            model_path=transcriber._model_path,
            model_arch=arch,
            samplerate=args.samplerate,
        )
        mic.add_listener(counter)
        mic.start()
        try:
            time.sleep(10.0)
        finally:
            mic.stop()
            mic.close()
        if counter.completed < 1:
            return SelfCheckResult.fail(
                f"no on_line_completed events in 10 s of fake-mic audio "
                f"(started={counter.started})",
                "03_live_microphone",
            )
        return None  # PASS
    finally:
        transcriber.close()


class _LineCounter(common.TranscriptEventListener):
    """Counts line-completion events for the self-check assertion."""

    def __init__(self) -> None:
        self.started = 0
        self.completed = 0

    def on_line_started(self, event) -> None:
        self.started += 1

    def on_line_completed(self, event) -> None:
        self.completed += 1


if __name__ == "__main__":
    main()
