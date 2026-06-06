"""Round-trip evaluation workflow for PocketTTS using Moonshine STT.

The workflow synthesises a list of ground-truth sentences with PocketTTS
(Kyutai's ~100M-parameter CPU TTS), feeds the resulting audio through
Moonshine's STT, and computes WER / CER plus a handful of audio-level
quality signals. The output is a directory of CSV, JSON, and Markdown
reports that summarises:

  * per-sentence pass/fail (configurable WER threshold, default 15%),
  * aggregate WER / CER / exact-match rate across the corpus,
  * audio-level stats (duration, peak, RMS, silence ratio) per wav,
  * a "worst N sentences" table to drive manual investigation.

The runner uses the same Moonshine Python classes as the rest of the
project (``Transcriber``, ``transcribe_without_streaming``,
``load_wav_file``) so it doubles as another worked example of the STT
API — see ``examples/python/stt/`` for the teaching material.

Layout:

    run.py                     CLI entry point
    pocket_tts_runner.py       wraps pocket_tts.TTSModel
    moonshine_stt_runner.py    wraps moonshine_voice.Transcriber
    metrics.py                 WER / CER / normalise / audio stats
    report.py                  CSV / JSON / Markdown writers
    corpora/english_smoke.txt  30-sentence smoke corpus
    test_metrics.py            unit tests for the metrics helpers
"""
