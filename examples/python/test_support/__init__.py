"""Test support for the Moonshine Voice example suite.

This package provides the shims and canned fixtures that let the
``examples/python/`` scripts be exercised end-to-end without real audio
hardware. It is **not** part of the published ``moonshine-voice``
wheel — it is used only by the smoke-test drivers under
``scripts/test-python-examples.sh``.

Modules
-------
``canned_audio``
    Load ``test-assets/two_cities.wav`` as a float32 mono buffer at
    16 kHz.

``fake_sounddevice``
    Replace :class:`sounddevice.InputStream` with a fake that drives
    the callback from a background thread reading the canned buffer.

``_auto_install``
    The ``PYTHONSTARTUP`` hook that calls ``fake_sounddevice.install()``
    when the ``MOONSHINE_SELF_CHECK`` environment variable is set.

``self_check``
    Helpers each example's ``--self-check`` path uses to print a
    uniform ``PASS:`` / ``FAIL:`` / ``SKIP:`` final line.
"""
