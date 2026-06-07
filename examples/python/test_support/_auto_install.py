"""Side-effect-on-import installer for the fake sounddevice shim.

This module exists so the test driver can prepend a one-liner to any
``python script.py`` invocation that installs the fake *before* the
target script imports ``sounddevice``. The driver invokes the example
as::

    python -c "import test_support._auto_install; \\
               import runpy; \\
               runpy.run_path('script.py', run_name='__main__')" \\
        --self-check

The install only fires when ``MOONSHINE_SELF_CHECK=1`` is in the
environment; otherwise the import is a no-op. This way, a developer's
interactive ``python script.py`` invocation behaves exactly as it
always did.
"""

import os
import sys


def _maybe_install() -> None:
    if os.environ.get("MOONSHINE_SELF_CHECK") != "1":
        return
    # The test_support package lives next to ``examples/python/``.
    # When this file is imported via ``python -c "..."`` the
    # directory containing it is on sys.path already (because we
    # got here by absolute import), but we double-check defensively.
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        from test_support.fake_sounddevice import install
    except Exception:
        return
    try:
        install()
    except Exception:
        return


_maybe_install()
