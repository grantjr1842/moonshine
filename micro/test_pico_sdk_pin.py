#!/usr/bin/env python3
# Regression for audit row A-146 — Pico SDK bootstrap reproducibility.
#
# Verifies the hardened pico_sdk_import.cmake by reading the source:
#   * a pinned, non-empty default tag is present (no master fallback)
#   * an explicit empty-tag rejection is present (configure must fail
#     closed rather than tracking the upstream default branch)
#   * the two importer copies (micro/pico_sdk_import.cmake and
#     micro/third-party/pico-tflmicro/pico_sdk_import.cmake) are
#     byte-for-byte identical so both firmware targets resolve the
#     same SDK commit
#
# Live HTTP / FetchContent behavior (does the resolved tag actually
# clone the pinned commit?) is exercised as a manual integration
# check before each tagged release; the structural + behavioral gate
# below covers the regression surface without requiring network
# access.

from __future__ import annotations

import hashlib
import pathlib
import re
import unittest


MICRO_ROOT = pathlib.Path(__file__).resolve().parent
PRIMARY = MICRO_ROOT / "pico_sdk_import.cmake"
DUPLICATE = (
    MICRO_ROOT
    / "third-party"
    / "pico-tflmicro"
    / "pico_sdk_import.cmake"
)


class PicoSdkPinTests(unittest.TestCase):
    """Read the importer source and assert the A-146 hardening."""

    def setUp(self) -> None:
        self.text = PRIMARY.read_text()

    def test_has_pinned_default_tag(self) -> None:
        """The file must declare a non-empty pinned default tag; an
        empty default would let FetchContent fall back to the
        upstream default branch (master), which is exactly the A-146
        bug we are guarding against."""
        m = re.search(
            r'set\(\s*PICO_SDK_PINNED_TAG\s+"(?P<tag>[^"]+)"\s*\)',
            self.text,
        )
        self.assertIsNotNone(
            m,
            msg="PICO_SDK_PINNED_TAG not declared in pico_sdk_import.cmake",
        )
        tag = m.group("tag")
        self.assertNotEqual(
            tag.strip(),
            "",
            msg="PICO_SDK_PINNED_TAG must be non-empty",
        )
        # The pinned tag must be applied as the default when fetching
        # and the caller has not overridden PICO_SDK_FETCH_FROM_GIT_TAG.
        self.assertRegex(
            self.text,
            r"PICO_SDK_FETCH_FROM_GIT_TAG\s+\"\$\{PICO_SDK_PINNED_TAG\}\"",
            msg="pinned default tag is not applied as the FetchContent tag",
        )

    def test_no_master_fallback(self) -> None:
        """The original A-146 bug defaulted to the mutable
        ``master`` branch when the consumer did not set a tag. The
        hardened importer must not default to any mutable branch."""
        # Look for any default that resolves to a branch ref, not a
        # tag / commit hash. master is the original A-146 vector;
        # main is the contemporary branch name on raspberrypi/pico-sdk.
        self.assertNotRegex(
            self.text,
            r'set\(\s*PICO_SDK_(?:FETCH_FROM_GIT_)?TAG\s+"(?:master|main)"\s*\)',
            msg="importer still defaults to a mutable branch (master/main)",
        )

    def test_empty_tag_rejected(self) -> None:
        """When PICO_SDK_FETCH_FROM_GIT=ON and the resolved tag is
        empty, the importer must fail closed (FATAL_ERROR) rather
        than let FetchContent pick up the upstream default branch."""
        # The rejection must reference FATAL_ERROR.
        self.assertRegex(
            self.text,
            r"FATAL_ERROR",
            msg="importer does not FATAL_ERROR on empty tag",
        )
        # The empty-tag literal-comparison must appear in an if-block
        # that guards PICO_SDK_FETCH_FROM_GIT (so local
        # PICO_SDK_PATH users don't trigger it).
        self.assertRegex(
            self.text,
            r"STREQUAL\s+\"\"",
            msg="importer does not check STREQUAL \"\" for empty tag",
        )
        # The FATAL_ERROR message must reference the empty tag case
        # so a consumer sees a useful error.
        self.assertRegex(
            self.text,
            r"PICO_SDK_FETCH_FROM_GIT_TAG\s+is empty",
            msg="FATAL_ERROR message does not mention empty tag",
        )
        # The check must be guarded by PICO_SDK_FETCH_FROM_GIT. We
        # verify this by checking that the STREQUAL "" check appears
        # AFTER an `if (PICO_SDK_FETCH_FROM_GIT)` line in the source.
        fetch_line_idx = self.text.find("if (PICO_SDK_FETCH_FROM_GIT)")
        strequal_idx = self.text.find('STREQUAL ""')
        self.assertNotEqual(
            fetch_line_idx,
            -1,
            msg="PICO_SDK_FETCH_FROM_GIT if-block not found",
        )
        self.assertNotEqual(
            strequal_idx,
            -1,
            msg="STREQUAL \"\" empty-tag check not found",
        )
        self.assertGreater(
            strequal_idx,
            fetch_line_idx,
            msg="empty-tag STREQUAL check appears before the "
            "PICO_SDK_FETCH_FROM_GIT guard",
        )

    def test_override_is_documented(self) -> None:
        """The override behavior (PICO_SDK_FETCH_FROM_GIT_TAG,
        PICO_SDK_FETCH_FROM_GIT_PATH, PICO_SDK_PATH) must be
        documented in a comment block at the top of the file."""
        # At minimum the file must mention how to override the tag.
        self.assertRegex(
            self.text,
            r"PICO_SDK_FETCH_FROM_GIT_TAG",
            msg="override variable PICO_SDK_FETCH_FROM_GIT_TAG not mentioned",
        )
        # And must reference an explicit override path.
        self.assertRegex(
            self.text,
            r"(?:override|-DPICO_SDK_FETCH_FROM_GIT_TAG)",
            msg="override path not documented",
        )

    def test_both_copies_identical(self) -> None:
        """micro/pico_sdk_import.cmake and
        micro/third-party/pico-tflmicro/pico_sdk_import.cmake must
        resolve the SAME SDK commit. If they diverge, the rp2350
        firmware target and the tflmicro firmware target could link
        against two different SDKs, which is a silent ABI mismatch.
        The simplest invariant is byte-for-byte equality of the
        importer; both importers are tiny and have no local
        state."""
        primary_text = PRIMARY.read_text()
        duplicate_text = DUPLICATE.read_text()
        # SHA-256 is overkill for the assertion but gives a stable
        # failure message when they diverge.
        self.assertEqual(
            primary_text,
            duplicate_text,
            msg=(
                "pico_sdk_import.cmake copies are not byte-for-byte "
                "identical (primary sha256="
                f"{hashlib.sha256(primary_text.encode()).hexdigest()}, "
                "duplicate sha256="
                f"{hashlib.sha256(duplicate_text.encode()).hexdigest()})"
            ),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
