"""Example 10 — the full voice-agent stack wired together.

This is the canonical end-to-end setup:

    mic audio  →  MicTranscriber  →  IntentRecognizer
                                    →  DialogFlow  →  TextToSpeech
                                    →  (alphanumeric spelling CNN, on demand)

It demonstrates every integration point you'd hit building a real voice
assistant: spelling-mode toggling, mute-while-talking, success/error
beeps, intent fallthrough, and the global "cancel" / "start over"
handlers.

This example requires a working microphone **and** speaker (TTS plays
audio output). On a headless machine run only the
``--check-prereqs`` or ``--list-output-devices`` paths.

What this script demonstrates
-----------------------------
* Wiring :class:`MicTranscriber`, :class:`IntentRecognizer`,
  :class:`DialogFlow`, and :class:`TextToSpeech` together.
* Muting the mic while the assistant is speaking (avoids transcribing
  its own TTS playback).
* Toggling :data:`MOONSHINE_FLAG_SPELLING_MODE` on the live mic stream
  only while a ``SPELLED`` / ``DIGITS`` prompt is active.
* Auto-wired success / error beeps (``tts.play_success`` /
  ``tts.play_error``) — the runner picks them up automatically when a
  TTS object is provided.
* Picking a PortAudio output device (the common "TTS is silent" debug).

Run it
------
    python -m examples.python.stt.10_full_voice_agent --check-prereqs
    python -m examples.python.stt.10_full_voice_agent --list-output-devices
    python -m examples.python.stt.10_full_voice_agent --language en
"""

from __future__ import annotations

import time

from . import common


def list_output_devices() -> None:
    """Print the available PortAudio output devices and exit."""
    from moonshine_voice.tts import list_output_devices as _tts_list

    common.hr("PortAudio output devices")
    try:
        for line in _tts_list():
            print(f"  {line}")
    except Exception as e:
        common.errprint(f"  could not enumerate: {e!r}")


def check_prereqs() -> None:
    """Best-effort check that mic + speaker + a TTS model are reachable."""
    common.hr("Prereq check")
    try:
        import sounddevice  # noqa: F401

        common.errprint("  sounddevice     : OK")
    except ImportError:
        common.errprint("  sounddevice     : MISSING — pip install sounddevice")
    try:
        import numpy  # noqa: F401

        common.errprint("  numpy           : OK")
    except ImportError:
        common.errprint("  numpy           : MISSING — pip install numpy")

    from moonshine_voice import list_tts_languages
    common.errprint(f"  TTS languages   : {len(list_tts_languages())} available")
    list_output_devices()


def run_live_agent(args) -> None:
    """Stand up the full mic + intent + dialog + TTS stack."""
    from moonshine_voice import (
        DialogFlow,
        IntentRecognizer,
        MicTranscriber,
        TextToSpeech,
        get_model_for_language,
        get_embedding_model,
        get_spelling_model_path,
    )
    from moonshine_voice.transcriber import MOONSHINE_FLAG_SPELLING_MODE

    common.hr("Loading STT model")
    model_path, arch = get_model_for_language(args.language, args.model_arch)
    common.errprint(f"  model_arch = {arch.name}")

    common.hr("Loading embedding model")
    emb_path, emb_arch = get_embedding_model(args.embedding_model, args.quantization)
    recognizer = IntentRecognizer(
        model_path=emb_path,
        model_arch=emb_arch,
        model_variant=args.quantization,
        threshold=args.threshold,
    )

    common.hr("Loading spelling model (for password prompts)")
    spelling_path = None
    try:
        spelling_path = get_spelling_model_path(args.language)
        common.errprint(f"  spelling model = {spelling_path}")
    except Exception as e:
        common.errprint(f"  no spelling model: {e!r} (SPELLED mode uses matcher only)")

    common.hr("Loading TTS")
    tts = None if args.no_tts else TextToSpeech(
        language=args.language,
        debug=args.debug,
    )

    # ---- define a small flow library ------------------------------------
    def order_pizza(d):
        size = yield d.ask("What size — small, medium, or large?")
        style = yield d.ask(f"Got it, {size}. Margherita, pepperoni, or veggie?")
        ok = yield d.confirm(f"So that's a {size} {style}?")
        if ok:
            yield d.say(f"One {size} {style} coming up. Total fifteen dollars.")
        else:
            yield d.say("No problem, cancelled.")

    def set_password(d):
        # The captured string would be validated and stored in a real
        # app; we just show the success path here.
        password = yield d.ask(
            "Spell the new password one character at a time, "
            "and say 'done' when finished.",
            mode="spelled",  # dialog_flow.SPELLED
        )
        yield d.say(f"Got it, password of length {len(password)} updated.")

    # ---- wire everything -------------------------------------------------
    common.hr("Wiring")
    mic = MicTranscriber(
        model_path=model_path,
        model_arch=arch,
        spelling_model_path=spelling_path,
    )

    # Coerce numeric device string to int so the resolver picks it up
    # as an index rather than a name substring.
    out_device = args.output_device
    if isinstance(out_device, str) and out_device.strip().isdigit():
        out_device = int(out_device.strip())

    def mute(should_mute: bool) -> None:
        # Prevent the mic from transcribing our own TTS output.
        mic._should_listen = not should_mute

    def set_spelling_mode(active: bool) -> None:
        # Toggle the C++ spelling-CNN fusion path on the live stream.
        mic.set_transcribe_flags(MOONSHINE_FLAG_SPELLING_MODE if active else 0)

    def speak(text: str) -> None:
        common.errprint(f"  assistant: {text}")
        if tts is not None:
            tts.say(text)
            tts.wait()

    runner = DialogFlow(
        tts=tts,
        speak_fn=speak,
        intent_recognizer=recognizer,
        mute_fn=mute,
        spelling_mode_fn=set_spelling_mode if spelling_path else None,
    )
    runner.register_flow("order a pizza", order_pizza)
    runner.register_flow("set a new password", set_password)
    runner.register_global("cancel", lambda d: d.cancel())
    runner.register_global("start over", lambda d: d.restart())

    printer = common.TranscriptPrinter(
        quiet=args.quiet,
        show_speaker=not args.no_speaker_ids,
    )
    mic.add_listener(printer)
    mic.add_listener(recognizer)
    mic.add_listener(runner)

    common.hr("Listening")
    common.errprint("  try: 'order a pizza', 'set a new password', 'cancel'")
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
        recognizer.close()
        if tts is not None:
            tts.close()


def main() -> None:
    parser = common.make_argparser(
        description="Full voice-agent stack: STT + intent + dialog + TTS.",
        include_mic=True,
        include_embedding=True,
    )
    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Skip TTS — print prompts to stderr instead of speaking them.",
    )
    parser.add_argument(
        "--list-output-devices",
        action="store_true",
        help="List PortAudio output devices and exit.",
    )
    parser.add_argument(
        "--output-device",
        type=str,
        default=None,
        metavar="INDEX_OR_NAME",
        help="Pin TTS playback to a specific PortAudio device (integer index "
        "or case-insensitive name substring).",
    )
    parser.add_argument(
        "--check-prereqs",
        action="store_true",
        help="Verify sounddevice / numpy / TTS model are reachable, then exit.",
    )
    args = parser.parse_args()

    if args.list_output_devices:
        list_output_devices()
        return
    if args.check_prereqs:
        check_prereqs()
        return

    run_live_agent(args)


if __name__ == "__main__":
    main()
