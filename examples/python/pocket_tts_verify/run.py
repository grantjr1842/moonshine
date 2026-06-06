"""CLI entry point for the PocketTTS round-trip verification workflow.

Usage
-----
    python -m examples.python.pocket_tts_verify.run --config config.yaml
    python -m examples.python.pocket_tts_verify.run --limit 5 --quiet
    python -m examples.python.pocket_tts_verify.run --list-pocket-tts-languages

The runner is a thin orchestrator. The heavy lifting lives in:

  * ``pocket_tts_runner.synthesize_sentence`` — PocketTTS wrapper.
  * ``moonshine_stt_runner.transcribe`` — Moonshine STT wrapper.
  * ``metrics`` — WER, CER, normalise, audio stats.
  * ``report`` — CSV / JSON / Markdown writers.

The runner:

  1. Loads the YAML config (or applies CLI-flag overrides).
  2. Lazy-imports PocketTTS, loads the model once, caches the voice state.
  3. Lazy-imports Moonshine, loads the transcriber once.
  4. For each sentence in the corpus, synthesise → transcribe → score.
  5. Writes reports, prints a summary to stdout.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

# All heavy imports are deferred to inside the runners.
from . import metrics
from . import moonshine_stt_runner
from . import pocket_tts_runner
from . import report
from .report import SentenceResult


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PocketTtsConfig:
    language: str = "english_2026-04"
    voice: str = "alba"
    temperature: float = 0.7
    lsd_decode_steps: int = 1
    eos_threshold: float = -4.0


@dataclass
class MoonshineConfig:
    language: str = "en"
    model_arch: Optional[int] = 1
    options: dict = None  # type: ignore[assignment]


@dataclass
class CorpusConfig:
    source: str = ":smoke:"
    limit: Optional[int] = None
    skip_empty_audio: bool = True


@dataclass
class OutputConfig:
    directory: Path = Path("./pocket_tts_verify_report")
    write_wav: bool = True
    write_csv: bool = True
    write_json: bool = True
    write_markdown: bool = True
    wer_pass_threshold: float = 0.15


@dataclass
class Config:
    pocket_tts: PocketTtsConfig
    moonshine: MoonshineConfig
    corpus: CorpusConfig
    output: OutputConfig


def _coerce_moonshine_options(opts) -> dict:
    """Convert a YAML options block into a string-keyed dict.

    The C API parses every option as a string, so bool/int/float
    values are stringified. ``None`` is dropped.
    """
    if opts is None:
        return {}
    out: dict = {}
    for k, v in opts.items():
        if v is None:
            continue
        out[str(k)] = str(v)
    return out


def _config_from_dict(d: dict) -> Config:
    """Build a ``Config`` from a raw dict (e.g. parsed YAML)."""
    p = d.get("pocket_tts", {}) or {}
    m = d.get("moonshine", {}) or {}
    c = d.get("corpus", {}) or {}
    o = d.get("output", {}) or {}
    return Config(
        pocket_tts=PocketTtsConfig(**p),
        moonshine=MoonshineConfig(
            language=m.get("language", "en"),
            model_arch=m.get("model_arch", 1),
            options=_coerce_moonshine_options(m.get("options", {})),
        ),
        corpus=CorpusConfig(
            source=c.get("source", ":smoke:"),
            limit=c.get("limit"),
            skip_empty_audio=c.get("skip_empty_audio", True),
        ),
        output=OutputConfig(
            directory=Path(o.get("directory", "./pocket_tts_verify_report")),
            write_wav=o.get("write_wav", True),
            write_csv=o.get("write_csv", True),
            write_json=o.get("write_json", True),
            write_markdown=o.get("write_markdown", True),
            wer_pass_threshold=o.get("wer_pass_threshold", 0.15),
        ),
    )


def _yaml_loads(text: str) -> dict:
    """Parse a small YAML subset.

    We avoid the ``yaml`` dependency if the file uses only flat
    key-value / list / dict syntax (the case for our config). If
    PyYAML is available we use it; otherwise we fall back to a
    minimal hand-rolled parser that handles the subset we generate
    from ``config.yaml.example``.
    """
    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None
    if yaml is not None:
        return yaml.safe_load(text) or {}
    # Tiny fallback: only supports two-space indentation, top-level
    # keys, ``key: value`` pairs, ``key:`` with sub-dict of
    # ``- item`` lists. Good enough for our config.
    return _tiny_yaml(text)


def _tiny_yaml(text: str) -> dict:
    """Minimal YAML-ish parser.

    Only handles the subset produced by ``config.yaml.example``:
    top-level keys, two-space-indented sub-dicts, scalar values,
    no flow-style, no anchors, no multiline strings. The intention
    is that the example config can be parsed even if PyYAML isn't
    installed — installing PyYAML is recommended for real use.
    """
    root: dict = {}
    stack = [(root, -1)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        stripped = raw.lstrip()
        indent = len(raw) - len(stripped)
        # Pop stack until top is the parent for this indent.
        while stack and stack[-1][1] >= indent:
            stack.pop()
        parent = stack[-1][0] if stack else root
        if stripped.startswith("- "):
            # List item — append to the parent's last key.
            if not isinstance(parent, list):
                # Convert the last dict value to a list if needed.
                key = next(reversed(parent))
                if not isinstance(parent[key], list):
                    parent[key] = []
                parent = parent[key]
            parent.append(_parse_scalar(stripped[2:].strip()))
            continue
        if ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()
        if value == "":
            new: dict = {}
            if isinstance(parent, dict):
                parent[key] = new
            stack.append((new, indent))
        else:
            if isinstance(parent, dict):
                parent[key] = _parse_scalar(value)
    return root


def _parse_scalar(text: str):
    """Coerce a YAML scalar to a Python value (very minimal)."""
    if not text:
        return ""
    low = text.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "~"):
        return None
    try:
        if text.startswith("0x") or text.startswith("0X"):
            return int(text, 16)
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    # Strip surrounding quotes if any.
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    return text


def load_config(path: Optional[Path]) -> Config:
    """Load and parse the YAML config at ``path``.

    Returns a fully-defaulted ``Config`` if ``path`` is ``None``.
    """
    if path is None:
        return _config_from_dict({})
    with open(path) as f:
        raw = _yaml_loads(f.read())
    return _config_from_dict(raw)


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def _corpus_path(source: str) -> Path:
    """Resolve a corpus source spec to a file path.

    ``":smoke:"`` is the magic token that means the bundled
    ``corpora/english_smoke.txt``. Anything else is treated as a
    file path.
    """
    if source == ":smoke:":
        return Path(__file__).parent / "corpora" / "english_smoke.txt"
    return Path(source)


def load_corpus(source: str, limit: Optional[int]) -> List[str]:
    """Load sentences from a corpus file, one per line, ``#`` comments skipped."""
    path = _corpus_path(source)
    if not path.exists():
        raise FileNotFoundError(
            f"Corpus file not found: {path}. "
            f"Pass a different --corpus path or check the source config."
        )
    sentences: List[str] = []
    for raw in path.read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        sentences.append(s)
        if limit is not None and len(sentences) >= limit:
            break
    return sentences


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _run_one(
    text: str,
    index: int,
    cfg: Config,
) -> SentenceResult:
    """Run one sentence through TTS → STT → metrics."""
    out_dir = cfg.output.directory
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / "synthesized" / f"{index:04d}.wav"

    try:
        samples, sample_rate = pocket_tts_runner.synthesize_sentence(
            text,
            language=cfg.pocket_tts.language,
            voice=cfg.pocket_tts.voice,
            temperature=cfg.pocket_tts.temperature,
            lsd_decode_steps=cfg.pocket_tts.lsd_decode_steps,
            eos_threshold=cfg.pocket_tts.eos_threshold,
        )
    except Exception as exc:  # noqa: BLE001 — surface as a result row
        return SentenceResult(
            index=index, text=text, hypothesis="",
            wer=1.0, cer=1.0, exact_match=False, pass_=False,
            audio_duration_sec=0.0, audio_peak=0.0, audio_rms=0.0,
            audio_silence_ratio=1.0, transcription_latency_ms=0,
            num_lines=0, sample_rate=0, error=f"tts: {exc!r}",
        )

    audio = metrics.audio_stats(samples, sample_rate)
    if cfg.corpus.skip_empty_audio and metrics.is_audio_silent(
        samples, sample_rate=sample_rate
    ):
        return SentenceResult(
            index=index, text=text, hypothesis="",
            wer=1.0, cer=1.0, exact_match=False, pass_=False,
            audio_duration_sec=audio["duration_sec"],
            audio_peak=audio["peak_amplitude"],
            audio_rms=audio["rms"],
            audio_silence_ratio=audio["silence_ratio"],
            transcription_latency_ms=0, num_lines=0,
            sample_rate=sample_rate, error="silent_audio",
        )

    if cfg.output.write_wav:
        try:
            pocket_tts_runner.save_wav(wav_path, samples, sample_rate)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! could not write wav: {exc!r}", file=sys.stderr)

    try:
        stt = moonshine_stt_runner.transcribe(
            samples, sample_rate,
            language=cfg.moonshine.language,
            model_arch=cfg.moonshine.model_arch,
            options=cfg.moonshine.options,
        )
    except Exception as exc:  # noqa: BLE001
        return SentenceResult(
            index=index, text=text, hypothesis="",
            wer=1.0, cer=1.0, exact_match=False, pass_=False,
            audio_duration_sec=audio["duration_sec"],
            audio_peak=audio["peak_amplitude"],
            audio_rms=audio["rms"],
            audio_silence_ratio=audio["silence_ratio"],
            transcription_latency_ms=0, num_lines=0,
            sample_rate=sample_rate,
            wav_path=str(wav_path) if wav_path.exists() else None,
            error=f"stt: {exc!r}",
        )

    ref_n = metrics.normalise(text, cfg.moonshine.language)
    hyp_n = metrics.normalise(stt.text, cfg.moonshine.language)
    wer_v = metrics.wer(ref_n, hyp_n)
    cer_v = metrics.cer(ref_n, hyp_n)
    em = metrics.exact_match(text, stt.text, cfg.moonshine.language)
    pass_ = (
        wer_v <= cfg.output.wer_pass_threshold
        and not metrics.is_audio_silent(samples, sample_rate=sample_rate)
    )
    return SentenceResult(
        index=index, text=text, hypothesis=stt.text,
        wer=wer_v, cer=cer_v, exact_match=em, pass_=pass_,
        audio_duration_sec=audio["duration_sec"],
        audio_peak=audio["peak_amplitude"],
        audio_rms=audio["rms"],
        audio_silence_ratio=audio["silence_ratio"],
        transcription_latency_ms=stt.latency_ms,
        num_lines=stt.num_lines,
        sample_rate=sample_rate,
        wav_path=str(wav_path) if wav_path.exists() else None,
    )


def run(cfg: Config, *, quiet: bool = False) -> List[SentenceResult]:
    """Run the full evaluation and write reports. Returns the results."""
    sentences = load_corpus(cfg.corpus.source, cfg.corpus.limit)
    if not sentences:
        print("  (corpus is empty — nothing to do)", file=sys.stderr)
        return []

    print(
        f"  corpus           : {cfg.corpus.source}  "
        f"({len(sentences)} sentences)",
        file=sys.stderr,
    )
    print(
        f"  pocket_tts       : language={cfg.pocket_tts.language!r} "
        f"voice={cfg.pocket_tts.voice!r}",
        file=sys.stderr,
    )
    print(
        f"  moonshine stt    : language={cfg.moonshine.language!r} "
        f"arch={cfg.moonshine.model_arch}",
        file=sys.stderr,
    )
    print(
        f"  output dir       : {cfg.output.directory}",
        file=sys.stderr,
    )
    print("", file=sys.stderr)

    results: List[SentenceResult] = []
    for i, text in enumerate(sentences, start=1):
        if not quiet:
            print(
                f"  [{i:>3}/{len(sentences)}] {text!r}",
                file=sys.stderr,
            )
        r = _run_one(text, i, cfg)
        results.append(r)
        if not quiet:
            if r.error:
                print(f"      ERROR: {r.error}", file=sys.stderr)
            else:
                wer_str = f"WER {r.wer:6.2%}  CER {r.cer:6.2%}  "
                pass_str = "PASS" if r.pass_ else "FAIL"
                print(
                    f"      {wer_str} {pass_str}  "
                    f"hyp={r.hypothesis!r}",
                    file=sys.stderr,
                )

    # Write reports.
    out_dir = cfg.output.directory
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.output.write_csv:
        report.write_csv(results, out_dir / "results.csv")
    if cfg.output.write_json:
        report.write_json(
            results, out_dir / "results.json",
            config={
                "pocket_tts": cfg.pocket_tts.__dict__,
                "moonshine": {
                    "language": cfg.moonshine.language,
                    "model_arch": cfg.moonshine.model_arch,
                    "options": cfg.moonshine.options,
                },
                "corpus": cfg.corpus.__dict__,
                "output": {
                    "directory": str(cfg.output.directory),
                    "wer_pass_threshold": cfg.output.wer_pass_threshold,
                },
            },
        )
    if cfg.output.write_markdown:
        report.write_markdown(
            results, out_dir / "report.md",
            config={
                "pocket_tts": cfg.pocket_tts.__dict__,
                "moonshine": {
                    "language": cfg.moonshine.language,
                    "model_arch": cfg.moonshine.model_arch,
                },
                "corpus": cfg.corpus.__dict__,
                "output": {
                    "directory": str(cfg.output.directory),
                    "wer_pass_threshold": cfg.output.wer_pass_threshold,
                },
            },
        )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_pocket_tts_languages() -> None:
    print("PocketTTS supported languages (pocket-tts 0.2.x):")
    for lang in pocket_tts_runner.list_languages():
        print(f"  - {lang}")


def _print_pocket_tts_voices() -> None:
    print("PocketTTS built-in voices (English 2026-04):")
    for v in pocket_tts_runner.list_built_in_voices():
        print(f"  - {v}")
    print()
    print("Custom voices can be .wav files or .safetensors files produced")
    print("by `pocket-tts export-voice`.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[1] if __doc__ else "",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML config file. CLI flags override config values.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output.directory.",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Override corpus.source (':smoke:' or a path).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override corpus.limit.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Override pocket_tts.language.",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="Override pocket_tts.voice.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-line progress; only the final summary is shown.",
    )
    parser.add_argument(
        "--list-pocket-tts-languages",
        action="store_true",
        help="Print available PocketTTS language codes and exit.",
    )
    parser.add_argument(
        "--list-pocket-tts-voices",
        action="store_true",
        help="Print built-in voice names and exit.",
    )
    args = parser.parse_args(argv)

    if args.list_pocket_tts_languages:
        _print_pocket_tts_languages()
        return 0
    if args.list_pocket_tts_voices:
        _print_pocket_tts_voices()
        return 0

    cfg = load_config(args.config)

    # Apply CLI overrides.
    if args.output is not None:
        cfg.output.directory = args.output
    if args.corpus is not None:
        cfg.corpus.source = args.corpus
    if args.limit is not None:
        cfg.corpus.limit = args.limit
    if args.language is not None:
        cfg.pocket_tts.language = args.language
    if args.voice is not None:
        cfg.pocket_tts.voice = args.voice

    try:
        results = run(cfg, quiet=args.quiet)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    report.print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
