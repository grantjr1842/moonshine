"""Example 08 — semantic intent recognition on top of STT.

:class:`moonshine_voice.IntentRecognizer` is a :class:`TranscriptEventListener`
that matches completed transcript lines against a set of registered
**canonical phrases** using cosine similarity over an embedding model.

The matching is *semantic* — "switch on the lights" matches "turn on the
lights" even though the words differ. No intent grammar to maintain, no
training step.

What this script demonstrates
-----------------------------
* Downloading / loading the embedding model (``embeddinggemma-300m`` by
  default, configurable via ``--embedding-model`` and ``--quantization``).
* Registering intents with :meth:`IntentRecognizer.register_intent` and
  a handler callback.
* Attaching the recognizer to a :class:`moonshine_voice.Transcriber` so
  completed lines are routed automatically.
* Using :meth:`IntentRecognizer.process_utterance` *standalone* (without
  a transcriber) — useful in tests and for routing pre-existing text.
* :meth:`IntentRecognizer.get_closest_intents` for top-N ranked matches,
  including the ``priority`` parameter that lets you break ties in favour
  of more important intents.
* Tuning the ``--threshold`` (similarity cutoff) and observing the
  trade-off between false positives and false negatives.

Run it
------
    python -m examples.python.stt.08_intent_recognizer
    python -m examples.python.stt.08_intent_recognizer --threshold 0.6
    python -m examples.python.stt.08_intent_recognizer --standalone
"""

from __future__ import annotations

import time

from moonshine_voice import (
    IntentRecognizer,
    LineCompleted,
    TranscriptEventListener,
)

from . import common


# ---------------------------------------------------------------------------
# Intent handlers
# ---------------------------------------------------------------------------


def on_lights_on(_trigger, utterance, similarity):
    common.errprint(f"  💡 LIGHTS ON     ({similarity:6.0%})  heard: {utterance!r}")


def on_lights_off(_trigger, utterance, similarity):
    common.errprint(f"  🌑 LIGHTS OFF    ({similarity:6.0%})  heard: {utterance!r}")


def on_weather(_trigger, utterance, similarity):
    common.errprint(f"  🌤️  WEATHER       ({similarity:6.0%})  heard: {utterance!r}")


def on_timer(_trigger, utterance, similarity):
    common.errprint(f"  ⏰ TIMER         ({similarity:6.0%})  heard: {utterance!r}")


def on_music_play(_trigger, utterance, similarity):
    common.errprint(f"  🎵 MUSIC PLAY    ({similarity:6.0%})  heard: {utterance!r}")


def on_music_stop(_trigger, utterance, similarity):
    common.errprint(f"  🔇 MUSIC STOP    ({similarity:6.0%})  heard: {utterance!r}")


# A catch-all "no match" listener so you can see when recognition fails.
class MissedUtteranceLogger(TranscriptEventListener):
    def __init__(self, recognizer: IntentRecognizer, threshold: float):
        self._recognizer = recognizer
        self._threshold = threshold

    def on_line_completed(self, event: LineCompleted) -> None:
        text = event.line.text
        # get_closest_intents returns matches above the threshold. If
        # the list is non-empty, the top one is what fired. If empty,
        # the recognizer's on_intent callback wasn't triggered and we
        # want to show that fact.
        matches = self._recognizer.get_closest_intents(
            text, tolerance_threshold=self._threshold
        )
        if not matches:
            common.errprint(f"  ❓ NO MATCH     ({self._threshold:.0%})  heard: {text!r}")
        else:
            top = matches[0]
            common.errprint(
                f"  → top: {top.canonical_phrase!r}  "
                f"sim={top.similarity:.2f}"
            )


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def build_intent_recognizer(args) -> IntentRecognizer:
    """Download (first run only) and load the embedding model."""
    common.errprint(
        f"  embedding model: {args.embedding_model} "
        f"(variant={args.quantization})"
    )
    common.errprint("  (first run downloads from download.moonshine.ai)")
    from moonshine_voice import EmbeddingModelArch, get_embedding_model

    emb_path, emb_arch = get_embedding_model(args.embedding_model, args.quantization)
    return IntentRecognizer(
        model_path=emb_path,
        model_arch=emb_arch,
        model_variant=args.quantization,
        threshold=args.threshold,
    )


def register_standard_intents(recognizer: IntentRecognizer) -> None:
    """Wire up the canonical phrases that ship with the README example."""
    recognizer.register_intent("turn on the lights", on_lights_on)
    recognizer.register_intent("turn off the lights", on_lights_off)
    recognizer.register_intent("what is the weather", on_weather)
    recognizer.register_intent("set a timer", on_timer)
    recognizer.register_intent("play some music", on_music_play)
    recognizer.register_intent("stop the music", on_music_stop)
    common.errprint(f"  registered {recognizer.intent_count()} intents")


# ---------------------------------------------------------------------------
# Standalone mode — drive process_utterance from canned text
# ---------------------------------------------------------------------------


def run_standalone_demo(recognizer: IntentRecognizer) -> None:
    """Drive the recognizer with a list of canned utterances.

    Useful for unit tests and for showing how ``get_closest_intents``
    ranks results. Each utterance is processed twice — once to fire
    the matching intent handler, and once with a very low threshold
    to see all top-N candidates.
    """
    common.hr("Standalone mode (no STT)")
    utterances = [
        "could you switch the lights on please",
        "turn the lights off now",
        "what's the weather like today",
        "set a timer for 5 minutes",
        "play my morning playlist",
        "pause the music",
        "what is the meaning of life",  # should be a miss
    ]
    common.errprint("  (top match in normal mode, then all candidates in "
                    "top-N mode)")
    for u in utterances:
        common.errprint(f"\n  utterance: {u!r}")
        recognizer.process_utterance(u)
        top_n = recognizer.get_closest_intents(u, tolerance_threshold=0.0)
        if top_n:
            preview = ", ".join(
                f"{m.canonical_phrase}={m.similarity:.2f}"
                for m in top_n[:3]
            )
            common.errprint(f"  top-N : {preview}")


# ---------------------------------------------------------------------------
# STT-driven mode — listen via MicTranscriber
# ---------------------------------------------------------------------------


def run_stt_demo(recognizer: IntentRecognizer, args) -> None:
    """Wire the recognizer to a live or file-driven transcriber."""
    common.hr("STT-driven mode")
    transcriber, arch = common.load_stt_model(language=args.language)
    common.errprint(f"  model_arch = {arch.name}")

    printer = common.TranscriptPrinter(
        quiet=args.quiet,
        show_speaker=not args.no_speaker_ids,
        show_words=args.word_timestamps,
    )
    transcriber.add_listener(printer)
    transcriber.add_listener(recognizer)
    transcriber.add_listener(
        MissedUtteranceLogger(recognizer, threshold=args.threshold)
    )

    if args.mic:
        from moonshine_voice import MicTranscriber

        mic = MicTranscriber(
            model_path=transcriber._model_path, model_arch=arch
        )
        # Forward all listeners we already registered.
        for listener in transcriber._default_stream._listeners:
            mic._should_listen = True
            mic.mic_stream.add_listener(listener)
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
        wav_path = common.require_wav_path(args.wav_path)
        common.stream_wav_to_transcriber(transcriber, wav_path)

    transcriber.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = common.make_argparser(
        description="Semantic intent recognition on top of STT. "
        "Two modes: --standalone (no audio) and STT-driven (file or mic).",
        include_mic=True,
        include_embedding=True,
    )
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Run a canned-utterance demo that doesn't need a microphone or "
        "a WAV file. Use this to see how the recognizer scores inputs.",
    )
    args = parser.parse_args()

    common.hr("Loading embedding model")
    recognizer = build_intent_recognizer(args)
    register_standard_intents(recognizer)

    try:
        if args.standalone:
            run_standalone_demo(recognizer)
        else:
            run_stt_demo(recognizer, args)
    finally:
        recognizer.close()


if __name__ == "__main__":
    main()
