# PocketTTS round-trip verification with Moonshine STT

A standalone workflow that:

1. Synthesises a list of ground-truth sentences with **PocketTTS**
   (Kyutai's ~100M-parameter CPU TTS).
2. Transcribes the resulting audio back with **Moonshine's STT**.
3. Computes WER, CER, and audio-level quality signals.
4. Writes a per-sentence CSV, an aggregate JSON, and a Markdown report.

## Install

The workflow needs both PocketTTS and the Moonshine Voice Python
package. Both are pip-installable:

```bash
pip install pocket-tts
pip install moonshine-voice
python -m moonshine_voice.download --language en   # one-time model fetch
```

The first call to PocketTTS downloads the model weights from
HuggingFace (~600 MB). Moonshine's tiny model is bundled with the
wheel, so the English model is ready immediately.

## Run

```bash
python -m examples.python.pocket_tts_verify.run --config config.yaml.example
```

Useful overrides:

```bash
# Quick smoke test, first 5 sentences, only stdout output
python -m examples.python.pocket_tts_verify.run --limit 5 --quiet

# Use a different voice or language
python -m examples.python.pocket_tts_verify.run --voice alba --language english_2026-04

# Point at your own corpus file
python -m examples.python.pocket_tts_verify.run --corpus path/to/my_sentences.txt
```

## Outputs

After a successful run, the configured output directory contains:

* `synthesized/&lt;NNN&gt;.wav` — one wav per sentence (when
  `output.write_wav: true`).
* `results.csv` — one row per sentence with WER, CER, audio
  duration, transcription latency, and pass/fail.
* `results.json` — same data, plus per-line metadata
  (which words the recogniser got right, which it dropped, …).
* `report.md` — pretty Markdown report with summary stats, worst-N
  sentences, and a per-sentence detail table.

## Known limitations

* PocketTTS and Moonshine both download model weights on first use.
  First-run cost is several minutes; subsequent runs are quick.
* PocketTTS doesn't currently expose a sampling-rate parameter —
  output is fixed at 24 kHz mono. Moonshine resamples internally
  to 16 kHz, so this is transparent.
* The built-in smoke corpus is 30 English sentences — small on
  purpose so the smoke test finishes in under a minute on CPU.
  For statistically meaningful WER estimates, point the workflow
  at a larger corpus (e.g. 500+ sentences from `librispeech`).
