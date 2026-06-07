"""Example 09 — multi-turn dialog flow (keyboard mode).

:class:`moonshine_voice.DialogFlow` lets you author multi-step,
branching voice conversations as ordinary Python generator functions. A
flow ``yield``s prompts (asks, confirms, chooses, says) and the runner
handles the user-input plumbing — for live audio via a Transcriber, or
for a chat-style keyboard exchange.

This example uses the **keyboard mode** so it runs anywhere — no
microphone, no TTS, no audio hardware. The runner's intent recognizer
is still doing real semantic matching, just on text instead of speech.

What this script demonstrates
-----------------------------
* The flow-author API: ``d.ask``, ``d.confirm``, ``d.choose``, ``d.say``.
* Input modes: ``FREE`` (default), ``SPELLED`` (character-by-character),
  ``DIGITS`` (digits-only spelled), ``PHRASE``.
* The bias-terms argument for narrow recognition.
* Global handlers (e.g. "cancel", "start over") that interrupt any
  active flow.
* Composing sub-flows with ``yield from``.

Run it
------
    python -m examples.python.stt.09_dialog_flow
    python -m examples.python.stt.09_dialog_flow --flow order-pizza
"""

from __future__ import annotations

from moonshine_voice import (
    DialogFlow,
    FREE,
    IntentRecognizer,
    SPELLED,
    spell_out,
)

from . import common


# ---------------------------------------------------------------------------
# Flow definitions — read like a script, branch with regular Python.
# ---------------------------------------------------------------------------


def order_pizza(d):
    """Slot-filling flow: size → style → confirm."""
    yield d.say("Welcome to the pizza order line.")

    size = yield d.ask(
        "What size would you like — small, medium, or large?",
        mode=FREE,
    )
    style = yield d.ask(
        f"Got it, {size}. What style — margherita, pepperoni, or veggie?",
    )

    if not (yield d.confirm(
        f"So that's a {size} {style}. Is that right?"
    )):
        yield d.say("No problem, let's start over.")
        return

    yield d.say(
        f"Great. One {size} {style} coming up. "
        f"Total is fifteen dollars."
    )


def set_wifi_password(d):
    """A flow that uses SPELLED mode for password input."""
    ssid = yield d.ask("What's the name of the wifi network?")
    if not (yield d.confirm(f"I heard {ssid}. Is that right?")):
        yield d.say("Okay, let's try again.")
        return

    password = yield d.ask(
        "Please spell the password one character at a time. "
        "Say 'done' when you've finished.",
        mode=SPELLED,
    )

    apply = yield d.confirm(
        f"I heard {spell_out(password)}. Apply these settings?"
    )
    if apply:
        yield d.say(f"Connecting to {ssid}… done.")
    else:
        yield d.say("Okay, nothing changed.")


def pick_from_list(d):
    """A flow that uses ``d.choose`` to pick from named options."""
    yield d.say("Let's pick a theme.")

    theme = yield d.choose(
        "Which theme?",
        options={
            "dark": ["dark", "black", "midnight"],
            "light": ["light", "white", "bright"],
            "auto": ["auto", "system", "default"],
        },
    )
    yield d.say(f"Theme set to {theme}.")


# ---------------------------------------------------------------------------
# Driver — interactive keyboard mode, no audio.
# ---------------------------------------------------------------------------


def run_keyboard(recognizer: IntentRecognizer, flow_name: str) -> None:
    """Drive a flow from the keyboard.

    The runner's intent recognizer still does real semantic matching on
    every user reply — "lights on" and "switch on the lights" both fire
    the same handler.
    """

    def speak(text: str) -> None:
        print(f"  assistant: {text}", flush=True)

    runner = DialogFlow(speak_fn=speak, intent_recognizer=recognizer)
    runner.register_flow("order a pizza", order_pizza)
    runner.register_flow("set the wifi password", set_wifi_password)
    runner.register_flow("pick a theme", pick_from_list)

    # Globals are always live, even while a flow is running.
    runner.register_global("cancel", lambda d: d.cancel())
    runner.register_global("start over", lambda d: d.restart())

    aliases = {
        "order-pizza": "order a pizza",
        "wifi": "set the wifi password",
        "list": "pick a theme",
    }
    trigger = aliases.get(flow_name, flow_name)
    print(f"  user:      {trigger}")
    runner.process_utterance(trigger)
    while runner.is_active:
        try:
            reply = input("  you>      ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            runner.cancel_active()
            break
        if not reply:
            continue
        runner.process_utterance(reply)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = common.make_argparser(
        description="Multi-turn dialog flow in keyboard mode (no audio).",
        include_embedding=True,
        include_self_check=True,
    )
    parser.add_argument(
        "--flow",
        choices=("order-pizza", "wifi", "list"),
        default="order-pizza",
        help="Which flow to run.",
    )
    args = parser.parse_args()

    if args.self_check:
        common.run_self_check(
            "09_dialog_flow",
            lambda: _self_check(args),
        )
        return

    common.hr("Loading embedding model")
    from moonshine_voice import get_embedding_model

    common.errprint(
        f"  {args.embedding_model} (variant={args.quantization})"
    )
    emb_path, emb_arch = get_embedding_model(args.embedding_model, args.quantization)
    recognizer = IntentRecognizer(
        model_path=emb_path,
        model_arch=emb_arch,
        model_variant=args.quantization,
        threshold=args.threshold,
    )
    try:
        run_keyboard(recognizer, args.flow)
    finally:
        recognizer.close()


def _self_check(args) -> "SelfCheckResult | None":
    """Smoke test: drive a canned dialog flow with stubbed input().

    Uses :func:`test_support.self_check.patch_builtins_input` to
    feed the runner a list of canned replies, so the example runs
    end-to-end without a real keyboard. Asserts the runner exits
    the active flow (i.e. ``runner.is_active`` is False at the
    end).
    """
    from test_support.self_check import SelfCheckResult, patch_builtins_input
    from moonshine_voice import DialogFlow, IntentRecognizer, get_embedding_model

    # Per-flow canned replies. Each list drives ``run_keyboard``'s
    # ``input()`` loop until ``runner.is_active`` becomes False.
    canned_replies = {
        "order-pizza": [
            "small", "margherita", "yes",
        ],
        "wifi": [
            "HomeWifi", "yes", "s e c r e t 1 2 3", "done", "yes", "yes",
        ],
        "list": ["dark", "yes"],
    }
    patch_builtins_input(canned_replies.get(args.flow, []))

    try:
        emb_path, emb_arch = get_embedding_model(
            args.embedding_model, args.quantization
        )
    except Exception as e:
        return SelfCheckResult.skip(
            f"embedding model unavailable: {e!r}",
            "09_dialog_flow",
        )
    recognizer = IntentRecognizer(
        model_path=emb_path,
        model_arch=emb_arch,
        model_variant=args.quantization,
        threshold=args.threshold,
    )
    try:
        # Replicate the body of run_keyboard but capture the
        # runner so we can inspect ``is_active`` at the end.
        # The flow functions are defined at module scope; access
        # them via globals() rather than an import (the module
        # filename starts with a digit so import-by-name is
        # awkward).
        g = globals()
        runner = DialogFlow(
            speak_fn=lambda text: None,  # swallow the prompts
            intent_recognizer=recognizer,
        )
        runner.register_flow("order a pizza", g["order_pizza"])
        runner.register_flow("set the wifi password", g["set_wifi_password"])
        runner.register_flow("pick a theme", g["pick_from_list"])
        runner.register_global("cancel", lambda d: d.cancel())
        runner.register_global("start over", lambda d: d.restart())

        aliases = {
            "order-pizza": "order a pizza",
            "wifi": "set the wifi password",
            "list": "pick a theme",
        }
        trigger = aliases.get(args.flow, args.flow)
        runner.process_utterance(trigger)
        # Drive replies; bail early if we run out.
        replies = iter(canned_replies.get(args.flow, []))
        while runner.is_active:
            try:
                reply = next(replies)
            except StopIteration:
                break
            runner.process_utterance(reply)

        if runner.is_active:
            return SelfCheckResult.fail(
                "flow did not complete within canned replies "
                f"({len(canned_replies.get(args.flow, []))} replies consumed)",
                "09_dialog_flow",
            )
        return None  # PASS
    finally:
        recognizer.close()


if __name__ == "__main__":
    main()
