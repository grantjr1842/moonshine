"""Comprehensive Moonshine Voice usage examples.

This package collects runnable, end-to-end examples for every layer of the
Moonshine Voice STT stack:

    01  offline_transcribe        — one-shot WAV → Transcript
    02  streaming_transcribe       — chunked WAV, event listener
    03  live_microphone            — MicTranscriber with --mic opt-in
    04  options_and_tuning         — every non-obvious C option explained
    05  word_timestamps            — word-level timing, attention decoder
    06  multi_stream               — one Transcriber, multiple audio sources
    07  spelling_mode              — AlphanumericListener for passwords / codes
    08  intent_recognizer          — semantic intent matching
    09  dialog_flow                — generator-based multi-turn conversations
    10  full_voice_agent           — STT + intent + dialog + TTS, mic in / TTS out
    11  schema_extraction          — typed post-processing of completed lines

All examples share a small ``common`` helper for argument parsing, model
loading, and the standard ``TranscriptPrinter`` listener. Examples that need
a microphone take a ``--mic`` flag; everything else defaults to the bundled
``test-assets/two_cities.wav`` so they run headless.

Run any example with ``python -m examples.python.stt.<module>`` from the
repo root, or with ``python <module>.py`` from this directory if the
``moonshine_voice`` package is installed.
"""
