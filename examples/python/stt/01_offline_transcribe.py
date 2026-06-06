"""Example 01 — offline (one-shot) transcription of a WAV file.

This is the simplest possible use of the STT API: hand a ``Transcriber`` a
full audio buffer, get back a complete :class:`Transcript` whose lines are
all marked ``is_complete=True``.

What this script demonstrates
-----------------------------
* :func:`moonshine_voice.get_model_for_language` — picks the best (or
  requested) model architecture for a given language and returns a path the
  C API can load.
* :class:`moonshine_voice.Transcriber` construction with no options.
* :meth:`Transcriber.transcribe_without_streaming` — runs the VAD, the
  segmenter, and the ASR model in one synchronous call. Use this when you
  have a file or recording; reach for the streaming API (``add_audio``,
  ``add_listener``) when input is live.
* Every field on :class:`moonshine_voice.TranscriptLine` — text, timing,
  line id, speaker id, the raw audio buffer, and per-line latency.

Run it
------
    python -m examples.python.stt.01_offline_transcribe
    python -m examples.python.stt.01_offline_transcribe --language ja
    python -m examples.python.stt.01_offline_transcribe --wav-path ./clip.wav
"""

from __future__ import annotations

from moonshine_voice import Transcript

from . import common


def transcribe(wav_path, *, language: str = "en", show_words: bool = False):
    """Load the model, transcribe the WAV, and print every line in detail."""
    common.errprint(f"Loading model for language={language!r}…")
    transcriber, arch = common.load_stt_model(language=language)
    common.errprint(
        f"  model_arch = {arch.name} (constant={int(arch)})  "
        f"sample_rate = 16000 Hz internal"
    )

    common.errprint(f"Loading audio from {wav_path}…")
    audio, sample_rate = common.load_wav_file(wav_path)
    duration_sec = len(audio) / float(sample_rate)
    common.errprint(
        f"  {len(audio):,} samples @ {sample_rate} Hz "
        f"({duration_sec:.2f}s, mono float32)"
    )

    common.errprint("Running transcribe_without_streaming()…")
    transcript: Transcript = transcriber.transcribe_without_streaming(
        audio, sample_rate=sample_rate, flags=0
    )
    transcriber.close()

    common.hr("Transcript")
    if not transcript.lines:
        print("(no speech detected)")
        return transcript

    for line in transcript.lines:
        print(
            common.format_line(
                line,
                show_speaker=True,
                show_words=show_words,
                show_audio_len=True,
            )
        )

    # A few extra summary lines that are useful in the output but noisy
    # to compute for every line.
    total_audio = sum(len(l.audio_data or []) for l in transcript.lines)
    total_latency = sum(l.last_transcription_latency_ms for l in transcript.lines)
    common.hr("Summary")
    print(f"  lines         : {len(transcript.lines)}")
    print(f"  total audio   : {total_audio:,} samples")
    print(f"  sum latency   : {total_latency} ms")
    if transcript.lines:
        speakers = {
            l.speaker_index for l in transcript.lines if l.has_speaker_id
        }
        print(f"  speakers seen : {len(speakers)}")
    return transcript


def main() -> None:
    parser = common.make_argparser(
        description="One-shot transcription of a WAV file. No microphone "
        "involved; runs anywhere."
    )
    args = parser.parse_args()

    wav_path = common.require_wav_path(args.wav_path)
    transcribe(
        wav_path,
        language=args.language,
        show_words=args.word_timestamps,
    )


if __name__ == "__main__":
    main()
