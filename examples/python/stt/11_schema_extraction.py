"""Example 11 — schema-based extraction of structured data from speech.

Most voice-interface applications don't just *display* the recognised
text — they want to *do something* with it: file it in a CRM, fill in a
form, dispatch an action. The structural pattern that surrounds this is
the same no matter what's downstream:

    1. A :class:`moonshine_voice.LineCompleted` event fires.
    2. The line text is normalised.
    3. The normalised text is matched against a **schema** describing
       the structured data you want to extract.
    4. The result is a typed object (or ``None`` if nothing matched).

This example defines a ``ContactInfo`` schema and a listener that
populates it from completed lines. The actual population logic is
deliberately lightweight — a few regular expressions and keyword
lookups — so the example has zero external dependencies. The point of
the example is the **shape** of the listener / extractor, not the
heuristics.

Where you'd plug in an LLM
--------------------------
The ``extract_with_llm`` function below is a stub that shows exactly
where an LLM call (Anthropic, OpenAI, local model, whatever) goes:

* The schema is serialised to a JSON-Schema description.
* The line text is sent with a system prompt that says "extract fields
  matching this schema, or return null."
* The LLM's response is parsed back into a ``ContactInfo``.

Three flags control whether the LLM path is used:
* ``--use-llm``  — opt in.
* ``--llm-backend`` — one of ``stub`` (default), ``anthropic``,
  ``openai``, ``ollama`` (the real backends are intentionally not
  implemented — this is a reference pattern).
* ``--llm-model`` — model name to pass through to the backend.

Run it
------
    python -m examples.python.stt.11_schema_extraction
    python -m examples.python.stt.11_schema_extraction --use-llm
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, List, Optional

from moonshine_voice import (
    LineCompleted,
    TranscriptEventListener,
)

from . import common


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass
class ContactInfo:
    """The structured record we want to extract from a single line.

    Every field is optional; ``None`` means "not detected in this
    utterance". A non-trivial extractor might accumulate fields across
    multiple lines (name from line 1, email from line 2) before
    emitting a complete record.
    """

    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    intent: Optional[str] = None  # "sales", "support", "billing", "other"
    raw_line: Optional[str] = None
    confidence: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def is_empty(self) -> bool:
        return not any(
            v for k, v in asdict(self).items()
            if k not in ("raw_line", "confidence")
        )


# ---------------------------------------------------------------------------
# The extractor — pure functions, easy to swap with an LLM.
# ---------------------------------------------------------------------------


_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
# Loose phone: 7+ digits possibly separated by spaces / dashes / parens.
_PHONE_RE = re.compile(
    r"\b\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"
)
_INTENT_KEYWORDS = {
    "sales": ["buy", "purchase", "pricing", "quote", "order", "demo"],
    "support": ["help", "broken", "error", "issue", "problem", "support"],
    "billing": ["invoice", "bill", "charge", "refund", "payment"],
}


def extract_with_regex(line_text: str) -> ContactInfo:
    """Lightweight, deterministic extractor — no external dependencies.

    Returns a ``ContactInfo`` with whatever fields could be inferred from
    regex matches and keyword lookups. The ``confidence`` field is
    0.0 (no signal) or 1.0 (at least one field matched).
    """
    info = ContactInfo(raw_line=line_text, confidence=0.0)
    if not line_text:
        return info

    if m := _EMAIL_RE.search(line_text):
        info.email = m.group(0)
    if m := _PHONE_RE.search(line_text):
        info.phone = m.group(0)

    lower = line_text.lower()
    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            info.intent = intent
            break

    # Name extraction: a sloppy "my name is X" / "this is X" heuristic
    # so the example has *some* path that fills the name field. Real
    # systems should use a proper NER model.
    if m := re.search(
        r"(?:my name is|i'?m|this is|it'?s)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        line_text,
    ):
        info.name = m.group(1)

    info.confidence = 1.0 if not info.is_empty() else 0.0
    return info


def extract_with_llm(
    line_text: str,
    *,
    schema: type,
    backend: str = "stub",
    model: str = "",
) -> Optional[ContactInfo]:
    """Stub: shows the integration point for a real LLM call.

    A real implementation would:
      1. Serialise ``schema`` to a JSON-Schema description.
      2. Send a chat completion request asking the model to extract
         fields matching the schema from ``line_text`` (or return null).
      3. Parse the structured response back into a ``ContactInfo``.

    The three backends below outline the call signatures. The actual
    SDK calls are intentionally omitted — install your own SDK and
    drop them in.
    """
    # ``schema`` and ``model`` are passed through to the real SDK
    # call. They're unused in the stub, so reference them to keep
    # the linter happy and document the contract.
    _ = (schema, model)
    if backend == "stub":
        common.errprint(
            "    (LLM stub: install an SDK and call it here; falling back "
            "to regex extraction)"
        )
        return extract_with_regex(line_text)

    if backend == "anthropic":
        common.errprint(
            "    (Anthropic path) would call messages.create() with a "
            "tool-use schema for ContactInfo"
        )
    elif backend == "openai":
        common.errprint(
            "    (OpenAI path) would call chat.completions.create() with "
            "response_format={type: json_schema, …}"
        )
    elif backend == "ollama":
        common.errprint(
            "    (Ollama path) would POST to /api/chat with format=json"
        )
    else:
        common.errprint(f"    (unknown backend {backend!r})")
    return extract_with_regex(line_text)

# Re-enable flake8/ruff's complaint: ``_ = (schema, model)`` is a
# legitimate use of the underscore for an explicit no-op reference.


# ---------------------------------------------------------------------------
# The listener that ties STT to the extractor
# ---------------------------------------------------------------------------


class SchemaExtractor(TranscriptEventListener):
    """Buffer completed lines, extract structured data, print JSON.

    In a real application the extracted object would be sent somewhere
    (a CRM, a queue, a follow-up TTS prompt). Here we just print it.
    """

    def __init__(
        self,
        *,
        extractor: Callable[[str], Optional[ContactInfo]] = extract_with_regex,
        min_confidence: float = 0.0,
    ):
        self._extractor = extractor
        self._min_confidence = min_confidence
        self._extracted: List[ContactInfo] = []

    def on_line_completed(self, event: LineCompleted) -> None:
        info = self._extractor(event.line.text)
        if info is None:
            return
        if info.confidence < self._min_confidence:
            return
        if info.is_empty():
            return
        self._extracted.append(info)
        common.errprint(f"  ✓ extracted from: {event.line.text!r}")
        print(info.to_json())
        print()

    @property
    def extracted(self) -> List[ContactInfo]:
        return list(self._extracted)


# ---------------------------------------------------------------------------
# Demo entry point
# ---------------------------------------------------------------------------


CANNED_UTTERANCES = [
    "Hi, my name is Jane Smith and I'd like to ask about pricing",
    "You can email me at jane.smith@example.com",
    "My phone number is 415-555-1234 if you need to call me back",
    "I'm having an issue with my account and need some support please",
    "I would like to purchase the enterprise plan",
    "Could you send me an invoice for last month",
]


def run_demo(args) -> None:
    """Drive the extractor from canned utterances, then from a real WAV."""
    common.hr("Schema")
    print(json.dumps(asdict(ContactInfo()), indent=2))
    print()

    # Build the extractor function — regex by default, LLM stub if asked.
    if args.use_llm:
        extractor = lambda text: extract_with_llm(
            text,
            schema=ContactInfo,
            backend=args.llm_backend,
            model=args.llm_model,
        )
    else:
        extractor = extract_with_regex

    common.hr(f"Canned utterances ({args.extractor})")
    listener = SchemaExtractor(extractor=extractor)
    for u in CANNED_UTTERANCES:
        # Bypass the STT pipeline — directly invoke the listener as if
        # the recogniser had produced this text.
        fake_event = LineCompleted(
            line=_FakeLine(text=u),
            stream_handle=0,
        )
        listener.on_line_completed(fake_event)

    common.hr(f"WAV: {args.wav_path or common.default_wav_path().name}")
    if not args.wav_path and not common.default_wav_path().exists():
        common.errprint("  (no bundled WAV; canned-utterance demo only)")
    else:
        wav_path = common.require_wav_path(args.wav_path)
        common.errprint(f"  feeding {wav_path}")
        transcriber, _ = common.load_stt_model(language=args.language)
        try:
            transcriber.add_listener(
                common.TranscriptPrinter(quiet=True, show_speaker=False)
            )
            transcriber.add_listener(listener)
            common.stream_wav_to_transcriber(transcriber, wav_path)
        finally:
            transcriber.close()

    common.hr(f"Total extracted: {len(listener.extracted)}")
    for i, info in enumerate(listener.extracted, 1):
        common.errprint(f"  {i}. name={info.name!r}  email={info.email!r}  "
                        f"phone={info.phone!r}  intent={info.intent!r}")


# A minimal shim so the canned-utterance path can fabricate events
# without a real STT pipeline.
@dataclass
class _FakeLine:
    text: str
    start_time: float = 0.0
    duration: float = 0.0
    line_id: int = 0
    is_complete: bool = True
    audio_data: Optional[List[float]] = None
    words: Optional[List[Any]] = None


def main() -> None:
    parser = common.make_argparser(
        description="Schema-based extraction of structured data from "
        "completed transcript lines."
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use the LLM extractor stub instead of the regex-based one.",
    )
    parser.add_argument(
        "--llm-backend",
        choices=("stub", "anthropic", "openai", "ollama"),
        default="stub",
        help="Which LLM backend to call (default: stub, which falls back "
        "to regex).",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="",
        help="Model name to pass to the LLM backend.",
    )
    parser.add_argument(
        "--extractor",
        choices=("regex", "llm"),
        default="regex",
        help="Which extractor to use in the demo (overrides --use-llm).",
    )
    args = parser.parse_args()
    if args.extractor == "llm":
        args.use_llm = True
    run_demo(args)


if __name__ == "__main__":
    main()
