"""Example 07 — alphanumeric / spelling mode for passwords, codes, and serials.

Many voice interfaces need to accept input the user spells out: passwords,
license keys, Wi-Fi passphrases, model numbers. Moonshine handles this
with a two-layer pipeline:

1. A built-in **matcher** classifies the ASR text against a vocabulary
   table of letter words ("alpha", "aitch", "el", "ess", …), digit words
   ("one", "two", "niner"), NATO alphabet ("alpha", "bravo", …), and
   common punctuation / symbols ("at sign", "hash", "underscore").
2. A small **CNN** runs on the raw audio (1 second of 16 kHz waveform)
   and predicts a single character. The matcher + CNN predictions are
   fused — when they agree the line's text is replaced with the resolved
   character.

The library turns this on per-call with ``MOONSHINE_FLAG_SPELLING_MODE``.
The matcher runs even without the CNN (it can resolve "stop", "clear",
"delete" without invoking the model), so this example degrades gracefully
when the spelling model isn't downloaded.

What this script demonstrates
-----------------------------
* Loading the spelling CNN via the ``spelling_model_path`` kwarg on
  :class:`moonshine_voice.Transcriber`.
* :class:`moonshine_voice.AlphanumericListener` — a higher-level listener
  that wraps the matcher + C-side fusion and emits
  :class:`moonshine_voice.AlphanumericEvent` objects (CHARACTER, UNDO,
  CLEAR, STOPPED).
* The four event types: CHARACTER (one resolved char), UNDO (delete the
  last), CLEAR (wipe everything), STOPPED (final assembled string).
* The "capital H" / "at sign" / "underscore" modifiers.

Run it
------
    python -m examples.python.stt.07_spelling_mode
    python -m examples.python.stt.07_spelling_mode --mic
"""

from __future__ import annotations

from moonshine_voice import AlphanumericEvent, AlphanumericEventType

from . import common


def try_get_spelling_model(language: str):
    """Resolve the spelling CNN. Returns the path or None if unavailable."""
    try:
        from moonshine_voice import get_spelling_model_path

        return get_spelling_model_path(language)
    except Exception as exc:  # noqa: BLE001 — surface as warning, keep going
        common.errprint(f"  could not resolve spelling model: {exc!r}")
        return None


def make_event_logger():
    """Return a callback that pretty-prints every AlphanumericEvent."""
    assembled: list = []

    def on_event(event: AlphanumericEvent) -> None:
        if event.type is AlphanumericEventType.CHARACTER:
            assembled.append(event.character or "")
            common.errprint(
                f"  + char  {event.character!r:<8}  →  "
                f"assembled={''.join(assembled)!r}"
            )
        elif event.type is AlphanumericEventType.UNDO:
            if assembled:
                removed = assembled.pop()
                common.errprint(
                    f"  - undo  removed {removed!r:<8}  →  "
                    f"assembled={''.join(assembled)!r}"
                )
            else:
                common.errprint("  - undo  (nothing to remove)")
        elif event.type is AlphanumericEventType.CLEAR:
            assembled.clear()
            common.errprint("  × clear → assembled=''")
        elif event.type is AlphanumericEventType.STOPPED:
            common.errprint(
                f"  ■ stop  final text={event.text!r}"
            )
        else:
            common.errprint(f"  ? unrecognized: {event.text!r}")

    return on_event, assembled


def run_with_audio(transcriber, wav_path, *, label: str) -> None:
    """Stream a WAV and let the AlphanumericListener eat the events."""
    from moonshine_voice import AlphanumericListener

    on_event, assembled = make_event_logger()
    listener = AlphanumericListener(on_event)
    transcriber.add_listener(listener)

    common.hr(f"{label} → {wav_path.name}")
    audio, sample_rate = common.load_wav_file(wav_path)
    common.errprint(f"  {len(audio)/sample_rate:.2f}s of audio @ {sample_rate} Hz")
    common.stream_wav_to_transcriber(transcriber, wav_path)
    common.errprint(f"  assembled = {''.join(assembled)!r}")


def main() -> None:
    parser = common.make_argparser(
        description="Demonstrate alphanumeric / spelling mode for "
        "password and code input.",
        include_mic=True,
    )
    args = parser.parse_args()

    spelling_path = try_get_spelling_model(args.language)
    if spelling_path:
        common.errprint(
            f"  spelling model = {spelling_path}\n"
            f"  (CNN + matcher fusion will replace line.text with chars)"
        )
    else:
        common.errprint(
            "  spelling model not available; falling back to matcher-only\n"
            "  (still works for NATO / letter words / common modifiers)"
        )

    common.hr("Loading model")
    transcriber, arch = common.load_stt_model(
        language=args.language,
        spelling_model_path=spelling_path,
    )
    common.errprint(f"  model_arch = {arch.name}")

    if args.mic:
        # Live mic: the spelling mode flag is set per call by the
        # listener via MOONSHINE_FLAG_SPELLING_MODE.
        from moonshine_voice import MicTranscriber
        from moonshine_voice.transcriber import MOONSHINE_FLAG_SPELLING_MODE

        common.hr("Microphone")
        mic = MicTranscriber(
            model_path=transcriber._model_path,
            model_arch=arch,
            transcribe_flags=MOONSHINE_FLAG_SPELLING_MODE,
        )
        on_event, assembled = make_event_logger()
        from moonshine_voice import AlphanumericListener

        mic.add_listener(AlphanumericListener(on_event))
        common.errprint("  speak characters one at a time, e.g.:")
        common.errprint("    'capital H' 'e' 'l' 'l' 'o' 'at sign' "
                        "'one' 'two' 'three' 'stop'")
        common.errprint("  press Ctrl+C to stop")
        mic.start()
        try:
            import time

            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            common.errprint("\n  stopping…")
        finally:
            mic.stop()
            mic.close()
            common.errprint(f"  assembled = {''.join(assembled)!r}")
    else:
        # File mode: feed every alphanumeric sample we can find and
        # print the assembled string after each one.
        from moonshine_voice import AlphanumericListener

        # Reset the default stream between files so events don't carry
        # over. We re-attach the listener each time too.
        # The C library keeps the listener list across start/stop, so
        # we just push/pop.
        a_dir = common._TEST_ASSETS_DIR / "alphanumeric"
        if not a_dir.exists():
            common.errprint(f"  no test assets at {a_dir}; "
                            "re-run with --mic to dictate characters.")
        else:
            files = sorted(p for p in a_dir.iterdir() if p.is_file())
            for wav in files:
                transcriber.push_listener(
                    AlphanumericListener(make_event_logger()[0])
                )
                try:
                    run_with_audio(transcriber, wav, label=wav.parent.name)
                finally:
                    transcriber.pop_listener()

    transcriber.close()


if __name__ == "__main__":
    main()
