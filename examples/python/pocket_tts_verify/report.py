"""Report writers for the PocketTTS round-trip evaluator.

Three output formats are supported:

* **CSV** — one row per (sentence, run), with WER / CER / latency /
  pass-fail columns. Easy to import into a spreadsheet.
* **JSON** — same data plus per-line metadata (audio stats,
  transcript line texts). Machine-readable for downstream tooling.
* **Markdown** — pretty report with summary stats, worst-N
  sentences, and a per-sentence detail table. Drops into a
  GitHub issue, a PR description, or a CI artifact viewer.

All writers are pure functions that take a list of ``SentenceResult``
records and a path. They don't touch the global state.
"""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SentenceResult:
    """One row of the evaluation: the ground truth, the TTS audio,
    the STT hypothesis, and everything we measured about it."""

    index: int
    text: str                            # ground truth
    hypothesis: str                      # what STT heard
    wer: float
    cer: float
    exact_match: bool
    pass_: bool                         # ``pass_`` because ``pass`` is a keyword
    audio_duration_sec: float
    audio_peak: float
    audio_rms: float
    audio_silence_ratio: float
    transcription_latency_ms: int
    num_lines: int
    sample_rate: int
    error: Optional[str] = None          # set if synthesis or STT failed
    wav_path: Optional[str] = None       # set if we wrote the wav

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Rename ``pass_`` back to ``pass`` for the JSON consumers.
        d["pass"] = d.pop("pass_")
        return d


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


CSV_COLUMNS = [
    "index",
    "text",
    "hypothesis",
    "wer",
    "cer",
    "exact_match",
    "pass",
    "audio_duration_sec",
    "audio_peak",
    "audio_rms",
    "audio_silence_ratio",
    "transcription_latency_ms",
    "num_lines",
    "sample_rate",
    "error",
    "wav_path",
]


def write_csv(results: Sequence[SentenceResult], path: Path) -> None:
    """Write a CSV with one row per ``SentenceResult``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in results:
            row = r.to_dict()
            writer.writerow({c: row.get(c, "") for c in CSV_COLUMNS})


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def write_json(
    results: Sequence[SentenceResult],
    path: Path,
    *,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a JSON dump of all results + the resolved config."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config or {},
        "summary": summarise(results),
        "results": [r.to_dict() for r in results],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def write_markdown(
    results: Sequence[SentenceResult],
    path: Path,
    *,
    config: Optional[Dict[str, Any]] = None,
    worst_n: int = 5,
) -> None:
    """Write a human-readable Markdown report.

    The structure is:
      * Config block (which PocketTTS language/voice, which Moonshine arch)
      * Summary stats (mean / median WER, exact-match rate, pass rate)
      * Worst-N table
      * Full per-sentence detail table
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarise(results)
    lines: List[str] = []
    lines.append("# PocketTTS × Moonshine STT — round-trip report")
    lines.append("")

    if config:
        lines.append("## Config")
        lines.append("")
        for section, values in config.items():
            if not isinstance(values, dict):
                continue
            lines.append(f"### {section}")
            lines.append("")
            for k, v in values.items():
                lines.append(f"- **{k}**: `{v}`")
            lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **sentences evaluated**: {summary['count']}")
    lines.append(f"- **synthesis errors** : {summary['error_count']}")
    lines.append(f"- **pass rate**        : {summary['pass_rate']:.0%}  "
                 f"({summary['pass_count']} / {summary['count']})")
    lines.append(f"- **exact-match rate** : {summary['exact_match_rate']:.0%}")
    lines.append(f"- **mean WER**         : {summary['mean_wer']:.2%}")
    lines.append(f"- **median WER**       : {summary['median_wer']:.2%}")
    lines.append(f"- **mean CER**         : {summary['mean_cer']:.2%}")
    lines.append("")
    lines.append(
        f"Audio: mean duration "
        f"{summary['mean_audio_duration_sec']:.2f}s, mean RMS "
        f"{summary['mean_audio_rms']:.3f}."
    )
    lines.append("")

    if results:
        lines.append(f"## Worst {min(worst_n, len(results))} sentences (by WER)")
        lines.append("")
        lines.append("| # | WER | CER | text | hypothesis |")
        lines.append("|---|-----|-----|------|------------|")
        worst = sorted(results, key=lambda r: (-r.wer, r.index))[:worst_n]
        for r in worst:
            lines.append(
                f"| {r.index} | {r.wer:.0%} | {r.cer:.0%} | "
                f"{_md_escape(r.text)} | "
                f"{_md_escape(r.hypothesis or '∅')} |"
            )
        lines.append("")

    lines.append("## Per-sentence detail")
    lines.append("")
    lines.append("| # | pass | WER | CER | dur | hyp |")
    lines.append("|---|------|-----|-----|-----|-----|")
    for r in results:
        passed = "✓" if r.pass_ else ("✗" if r.error is None else "ERR")
        lines.append(
            f"| {r.index} | {passed} | {r.wer:.0%} | {r.cer:.0%} | "
            f"{r.audio_duration_sec:.2f}s | "
            f"{_md_escape(r.hypothesis or '∅')} |"
        )
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def _md_escape(text: str) -> str:
    """Light Markdown table escape — strip pipes, collapse newlines."""
    if not text:
        return ""
    return text.replace("|", "\\|").replace("\n", " ").replace("\r", " ")


# ---------------------------------------------------------------------------
# Aggregate stats
# ---------------------------------------------------------------------------


def _safe_median(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(values))


def summarise(results: Sequence[SentenceResult]) -> Dict[str, Any]:
    """Aggregate stats over a list of ``SentenceResult``s.

    Returns a dict suitable for embedding in a JSON / Markdown
    report. Missing values are reported as 0.0 or "" so the report
    never blows up on an empty corpus.
    """
    if not results:
        return {
            "count": 0,
            "error_count": 0,
            "pass_count": 0,
            "pass_rate": 0.0,
            "exact_match_rate": 0.0,
            "mean_wer": 0.0,
            "median_wer": 0.0,
            "mean_cer": 0.0,
            "median_cer": 0.0,
            "mean_audio_duration_sec": 0.0,
            "mean_audio_rms": 0.0,
        }
    wers = [r.wer for r in results]
    cers = [r.cer for r in results]
    durations = [r.audio_duration_sec for r in results]
    rms = [r.audio_rms for r in results]
    return {
        "count": len(results),
        "error_count": sum(1 for r in results if r.error is not None),
        "pass_count": sum(1 for r in results if r.pass_),
        "pass_rate": sum(1 for r in results if r.pass_) / len(results),
        "exact_match_rate": sum(1 for r in results if r.exact_match) / len(results),
        "mean_wer": _safe_mean(wers),
        "median_wer": _safe_median(wers),
        "mean_cer": _safe_mean(cers),
        "median_cer": _safe_median(cers),
        "mean_audio_duration_sec": _safe_mean(durations),
        "mean_audio_rms": _safe_mean(rms),
    }


# ---------------------------------------------------------------------------
# Stdout summary
# ---------------------------------------------------------------------------


def print_summary(results: Sequence[SentenceResult]) -> None:
    """Print a short one-screen summary to stdout."""
    s = summarise(results)
    print()
    print("=" * 60)
    print(f"PocketTTS × Moonshine STT — round-trip summary")
    print("=" * 60)
    print(f"  sentences        : {s['count']}")
    print(f"  synthesis errors : {s['error_count']}")
    print(f"  pass rate        : {s['pass_rate']:.0%} "
          f"({s['pass_count']} / {s['count']})")
    print(f"  exact-match rate : {s['exact_match_rate']:.0%}")
    print(f"  mean WER         : {s['mean_wer']:.2%}")
    print(f"  median WER       : {s['median_wer']:.2%}")
    print(f"  mean CER         : {s['mean_cer']:.2%}")
    print("=" * 60)
