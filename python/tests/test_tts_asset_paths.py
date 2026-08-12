import tempfile
import unittest
from pathlib import Path

from moonshine_voice.download import (
    _tts_asset_destination,
    cdn_url_for_tts_asset_key,
    is_downloadable_tts_asset_key,
)


class TtsAssetPathTests(unittest.TestCase):
    def test_asset_key_is_encoded_as_a_safe_relative_cdn_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.assertTrue(is_downloadable_tts_asset_key("en_us/dict.tsv"))
            self.assertTrue(
                cdn_url_for_tts_asset_key("en_us/dict.tsv").endswith("/en_us/dict.tsv")
            )
            self.assertEqual(
                _tts_asset_destination(root, "en_us/dict.tsv"),
                root.resolve() / "en_us" / "dict.tsv",
            )

    def test_asset_key_rejects_traversal_and_non_posix_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for key in (
                "../outside/file",
                "en_us/../../outside",
                "/tmp/file",
                "C:/tmp/file",
                "en_us\\file",
            ):
                with self.subTest(key=key):
                    with self.assertRaises(ValueError):
                        is_downloadable_tts_asset_key(key)
                    with self.assertRaises(ValueError):
                        cdn_url_for_tts_asset_key(key)
                    with self.assertRaises(ValueError):
                        _tts_asset_destination(root, key)

    def test_asset_key_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            root = base / "cache"
            root.mkdir()
            outside = base / "outside"
            outside.mkdir()
            (root / "en_us").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                _tts_asset_destination(root, "en_us/dict.tsv")

    def test_bare_override_label_is_not_downloadable(self):
        self.assertFalse(is_downloadable_tts_asset_key("custom-g2p-override"))


if __name__ == "__main__":
    unittest.main()
