"""Shared helpers for the example ``--self-check`` paths.

Each example that opts into self-checking follows this contract:

  * A ``--self-check`` flag is added to its argparse.
  * Under ``--self-check`` the example runs a short canned-audio
    smoke test instead of its normal interactive loop.
  * It emits **exactly one** final line on **stdout** in one of three
    forms::

        PASS: <name>
        FAIL: <short reason>
        SKIP: <reason>

  * It exits 0 / 1 / 77 (the autotools convention) to match.

The driver script (``scripts/test-python-examples.sh``) greps the
last line of stdout for ``^(PASS|FAIL|SKIP):`` and aggregates counts
across all examples. The helpers below wrap the boilerplate so each
example's ``--self-check`` block is three lines::

    from test_support.self_check import SelfCheckResult, report
    try:
        ...run the smoke test...
    except Exception as e:
        report(SelfCheckResult.fail(repr(e)))
    report(SelfCheckResult.pass_())

Diagnostic output during the test goes to **stderr**; the final
``PASS/FAIL/SKIP`` line is the only thing on stdout. This lets the
driver use simple line-level parsing without having to filter
progress messages.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class SelfCheckResult:
    """The verdict of a self-check run.

    Use the class methods to construct one — they keep the
    ``verdict`` field constrained to a known set so the driver can
    pattern-match without surprises.
    """

    verdict: str  # one of "PASS", "FAIL", "SKIP"
    name: str
    detail: Optional[str] = None

    @classmethod
    def pass_(cls, name: str = "") -> "SelfCheckResult":
        return cls(verdict="PASS", name=name)

    @classmethod
    def fail(cls, detail: str, name: str = "") -> "SelfCheckResult":
        return cls(verdict="FAIL", name=name, detail=detail)

    @classmethod
    def skip(cls, detail: str, name: str = "") -> "SelfCheckResult":
        return cls(verdict="SKIP", name=name, detail=detail)

    @property
    def line(self) -> str:
        """The single stdout line the driver parses.

        Form is ``VERDICT: <name>`` for PASS, and
        ``VERDICT: <detail>`` (when no name) or
        ``VERDICT: <name> (<detail>)`` for FAIL/SKIP.
        """
        if self.verdict == "PASS":
            return f"PASS: {self.name}" if self.name else "PASS"
        if not self.name:
            return f"{self.verdict}: {self.detail or ''}".rstrip(": ")
        if not self.detail:
            return f"{self.verdict}: {self.name}"
        return f"{self.verdict}: {self.name} ({self.detail})"

    @property
    def exit_code(self) -> int:
        return {"PASS": 0, "FAIL": 1, "SKIP": 77}[self.verdict]


def report(result: SelfCheckResult) -> None:
    """Print the result line to stdout and exit with the right code.

    The driver relies on this being the *only* stdout line. Anything
    else the example printed earlier (it shouldn't, by the contract)
    will be ignored by the driver — only the last line matters.
    """
    print(result.line, flush=True)
    raise SystemExit(result.exit_code)


def patch_builtins_input(answers):
    """Replace :func:`builtins.input` with a function that returns
    the next item from ``answers`` each call.

    Examples like ``09_dialog_flow.py`` use ``input()`` to gather
    keyboard replies. Under ``--self-check`` the driver has no
    keyboard, so we monkey-patch the builtin to replay a canned
    list. ``answers`` is a list; if exhausted, the patch returns
    the empty string (the existing call sites already treat empty
    replies as a no-op).

    Returns a function the caller can pass to ``atexit`` (or call
    manually) to restore the original ``input``.
    """
    import builtins
    import atexit

    original = builtins.input
    it = iter(answers)

    def fake(prompt: str = "") -> str:
        try:
            return next(it)
        except StopIteration:
            return ""

    builtins.input = fake

    def restore() -> None:
        builtins.input = original

    atexit.register(restore)
    return restore
