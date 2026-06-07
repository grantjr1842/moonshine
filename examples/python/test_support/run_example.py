"""``runpy``-based launcher for the example smoke-test driver.

The driver invokes each example as a Python module via::

    python -m test_support.run_example <dotted.module.name> --self-check [args]

This module:

1. Installs the fake sounddevice shim if
   ``MOONSHINE_SELF_CHECK=1`` is in the environment.
2. Uses :func:`runpy.run_module` with ``run_name="__main__"`` to
   execute the target module's ``if __name__ == "__main__":``
   block. Relative imports inside the target (``from . import
   common`` for the ``stt/`` suite) work because ``run_module``
   treats the target as a real package member rather than a
   free-standing script.

The driver prefers this over running the script as a plain file
because file-mode execution (``python script.py``) sets
``__name__`` to ``"__main__"`` and breaks relative imports.
"""

from __future__ import annotations

import os
import runpy
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print(
            "usage: python -m test_support.run_example <dotted.module> [args...]",
            file=sys.stderr,
        )
        return 2

    target = sys.argv[1]
    # Forward the remaining args to the target module's argparse.
    # ``runpy.run_module`` reads from ``sys.argv`` directly, so
    # we set ``sys.argv`` to ``[sys.argv[0], sys.argv[2:]]`` —
    # but only *for the duration* of the run. We restore
    # afterwards so callers / test harnesses see their own argv.
    saved_argv = sys.argv
    sys.argv = [target] + list(sys.argv[2:])
    try:
        runpy.run_module(target, run_name="__main__")
        return 0
    except SystemExit as e:
        return int(e.code) if e.code is not None else 0
    finally:
        sys.argv = saved_argv


# Install the fake shim on import. The installer is a no-op when
# ``MOONSHINE_SELF_CHECK`` isn't set, so importing this module
# from a developer prompt does nothing.
try:
    from test_support import _auto_install  # noqa: F401
except Exception:
    pass


if __name__ == "__main__":
    raise SystemExit(main())
