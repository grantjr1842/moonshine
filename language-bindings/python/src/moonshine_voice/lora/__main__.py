"""CLI for ``python -m moonshine_voice.lora`` and ``moonshine-voice lora``.

Argparse lives here so ``--help`` does not import PyTorch. Training imports
happen only after the extra is confirmed present.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

VALID_GRAPHS = frozenset(
    {"frontend", "encoder", "adapter", "cross_kv", "decoder_kv"}
)


def build_parser(prog: Optional[str] = None) -> argparse.ArgumentParser:
    if prog is None:
        argv0 = sys.argv[0] if sys.argv else ""
        prog = argv0 if argv0.startswith("moonshine-voice") else "moonshine-voice lora"
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Train a LoRA adapter or full fine-tune for Moonshine Streaming. "
            "Requires pip install 'moonshine-voice[finetune]' "
            "(or the equivalent 'moonshine-voice[lora]')."
        )
    )
    data = parser.add_argument_group("data")
    data.add_argument(
        "--dataset",
        choices=["atcosim", "uwb_atcc"],
        default=None,
        help="built-in corpus. atcosim is speaker-disjoint headset ATC "
        "(phraseology). uwb_atcc is session-disjoint real VHF "
        "(CC BY-NC-SA 4.0, research only)",
    )
    data.add_argument(
        "--train-manifest",
        default=None,
        help="JSONL / JSON / TSV of {audio, text} rows for your own data",
    )
    data.add_argument(
        "--eval-manifest",
        default=None,
        help="held-out manifest scored when --eval is set",
    )
    data.add_argument(
        "--eval-dataset",
        choices=["atco2"],
        default=None,
        help="optional transfer canary. atco2 is ATCO2-test-set-1h and is "
        "never used for training",
    )
    data.add_argument(
        "--data-root",
        default=None,
        help="resolve relative audio paths against this directory "
        "(default: the manifest's parent)",
    )
    data.add_argument(
        "--text-mode",
        default="auto",
        choices=["auto", "none", "lower"],
        help="auto lowercases a corpus that is >90%% uppercase",
    )
    data.add_argument(
        "--replay-repo",
        default="moonshine-ai/yodas-en-replay",
        help="general-domain replay corpus (HF dataset id)",
    )
    data.add_argument(
        "--dataset-revision",
        default=None,
        help="immutable HF commit/tag for the selected training dataset",
    )
    data.add_argument(
        "--split-revision",
        default=None,
        help="immutable HF commit/tag for the published split definition",
    )
    data.add_argument(
        "--replay-revision",
        default=None,
        help="immutable HF commit/tag for the replay dataset",
    )
    data.add_argument(
        "--no-replay",
        action="store_true",
        help="train on in-domain audio only. Not recommended: the canary "
        "usually gets worse and in-domain WER rarely improves",
    )

    hours = parser.add_argument_group("hours")
    hours.add_argument(
        "--train-hours",
        type=float,
        default=None,
        help="in-domain hours to train on (default: 2.0 for built-in "
        "datasets, all of a custom manifest except the dev slice)",
    )
    hours.add_argument("--dev-hours", type=float, default=0.25)
    hours.add_argument("--replay-hours", type=float, default=6.0)
    hours.add_argument("--replay-dev-hours", type=float, default=0.2)
    hours.add_argument("--replay-ratio", type=float, default=0.5)

    model = parser.add_argument_group("model")
    model.add_argument(
        "--model",
        default="moonshine-ai/moonshine-streaming-medium",
        help="HF hub id or local save_pretrained directory",
    )
    model.add_argument(
        "--model-revision",
        default=None,
        help="immutable HF model commit/tag; recorded in summary.json",
    )
    model.add_argument(
        "--normalizer-revision",
        default=None,
        help="immutable HF revision for Whisper normalizer.json",
    )
    model.add_argument(
        "--eval-dataset-revision",
        default=None,
        help="immutable HF revision for the optional ATCO2 transfer set",
    )
    model.add_argument(
        "--canary-revision",
        default=None,
        help="immutable HF revision for the optional LibriSpeech canary",
    )
    model.add_argument(
        "--adapt",
        default="lora",
        choices=["lora", "full"],
        help="lora freezes the backbone (default). full unfreezes it; "
        "use for real radio, not a Colab T4",
    )
    model.add_argument(
        "--sites",
        default="decoder",
        choices=["decoder", "encoder", "both"],
        help="which self-attention stacks get LoRA (ignored when --adapt full)",
    )
    model.add_argument("--rank", type=int, default=8)
    model.add_argument("--alpha", type=float, default=None)
    model.add_argument(
        "--lr",
        type=float,
        default=None,
        help="default 1e-3 for decoder LoRA, 1e-4 when --sites "
        "includes the encoder, 1e-5 for --adapt full",
    )
    model.add_argument("--batch-size", type=int, default=8)
    model.add_argument("--max-steps", type=int, default=3000)
    model.add_argument("--eval-every", type=int, default=100)
    model.add_argument("--patience", type=int, default=4)
    model.add_argument("--warmup", type=int, default=100)
    model.add_argument("--seed", type=int, default=0)
    model.add_argument(
        "--device",
        default="auto",
        help="cuda, cpu, or auto (cuda when available)",
    )

    io = parser.add_argument_group("output")
    io.add_argument("--output-dir", "-o", default="lora_runs")
    io.add_argument("--work-dir", default="lora_work")
    io.add_argument(
        "--prepare-only",
        action="store_true",
        help="index the data and print hours/speakers, then exit",
    )
    io.add_argument(
        "--eval",
        action="store_true",
        help="score in-domain WER after training (ATCOSIM/UWB scored "
        "split, or --eval-manifest)",
    )
    io.add_argument("--eval-limit", type=int, default=None)
    io.add_argument(
        "--canary",
        action="store_true",
        help="also score LibriSpeech test-clean, so forgetting is visible",
    )
    io.add_argument("--canary-limit", type=int, default=None)

    export = parser.add_argument_group("export")
    export.add_argument(
        "--export",
        action="store_true",
        help="export a save_pretrained directory to the runtime's ONNX graphs "
        "instead of training. Use with --model and --output-dir",
    )
    export.add_argument(
        "--graphs",
        default="all",
        help="'all' or a comma-separated subset of "
        "frontend,encoder,adapter,cross_kv,decoder_kv. decoder-only LoRA "
        "only needs decoder_kv; --sites encoder|both and --adapt full "
        "need all",
    )
    export.add_argument(
        "--tokenizer-bin",
        default=None,
        help="tokenizer.bin to copy next to the exported graphs",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject combinations that would otherwise fail deep in a long run."""
    if args.export:
        graphs = {part.strip() for part in args.graphs.split(",") if part.strip()}
        if args.graphs != "all" and not graphs:
            parser.error("--graphs must name at least one graph")
        unknown = graphs - VALID_GRAPHS
        if unknown:
            parser.error(
                "unknown graph(s): "
                + ", ".join(sorted(unknown))
                + "; choose from "
                + ", ".join(sorted(VALID_GRAPHS))
            )
        if args.tokenizer_bin and not Path(args.tokenizer_bin).is_file():
            parser.error(f"--tokenizer-bin not found: {args.tokenizer_bin}")
        if args.graphs != "all":
            args.graphs = ",".join(sorted(graphs))
        return

    if (args.dataset is None) == (args.train_manifest is None):
        parser.error("choose exactly one of --dataset or --train-manifest")

    for name, value in (
        ("--rank", args.rank),
        ("--lr", args.lr),
        ("--batch-size", args.batch_size),
        ("--max-steps", args.max_steps),
        ("--eval-every", args.eval_every),
        ("--patience", args.patience),
    ):
        if value is None:
            continue
        if value <= 0:
            parser.error(f"{name} must be greater than zero")
    if args.alpha is not None and args.alpha <= 0:
        parser.error("--alpha must be greater than zero")
    for name, value in (
        ("--dev-hours", args.dev_hours),
        ("--replay-dev-hours", args.replay_dev_hours),
        ("--replay-hours", args.replay_hours),
        ("--warmup", args.warmup),
    ):
        if value is None:
            continue
        if value < 0:
            parser.error(f"{name} cannot be negative")
    if args.train_hours is not None and args.train_hours <= 0:
        parser.error("--train-hours must be greater than zero")
    if not 0 <= args.replay_ratio < 1:
        parser.error("--replay-ratio must be between 0 (inclusive) and 1 (exclusive)")
    for name, value in (
        ("--eval-limit", args.eval_limit),
        ("--canary-limit", args.canary_limit),
    ):
        if value is not None and value <= 0:
            parser.error(f"{name} must be greater than zero")


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    from moonshine_voice.lora._deps import require_lora_deps

    try:
        require_lora_deps(
            include_eval=args.eval or args.canary,
            include_export=args.export,
        )
    except ImportError as error:
        print(error, file=sys.stderr)
        return 1

    if args.export:
        from moonshine_voice.lora.export import main as export_main

        export_argv = [
            "--model",
            args.model,
            "--output-dir",
            args.output_dir,
            "--graphs",
            args.graphs,
        ]
        if args.tokenizer_bin:
            export_argv += ["--tokenizer-bin", args.tokenizer_bin]
        export_main(export_argv)
        return 0

    if args.dataset is None and args.train_manifest is None:
        parser.error("one of --dataset or --train-manifest is required")

    from moonshine_voice.lora.train import train_adapter

    train_adapter(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
