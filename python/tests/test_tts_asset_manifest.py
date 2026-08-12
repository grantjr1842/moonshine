import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from moonshine_voice.download import (
    _download_tts_asset_keys,
    _download_model_components,
    load_model_asset_manifest,
    load_tts_asset_manifest,
)
from moonshine_voice.download_file import download_file


class TtsAssetManifestTests(unittest.TestCase):
    def test_manifest_validates_version_integrity_and_paths(self):
        digest = "a" * 64
        response = Mock()
        response.json.return_value = {
            "schema_version": 1,
            "assets": {"en_us/dict.tsv": {"sha256": digest, "size": 12}},
        }
        with patch("moonshine_voice.download.requests.get", return_value=response):
            self.assertEqual(
                load_tts_asset_manifest("https://cdn.example/manifest.json"),
                {"en_us/dict.tsv": {"sha256": digest, "size": 12}},
            )
        response.raise_for_status.assert_called_once()

    def test_manifest_rejects_unsafe_or_malformed_entries(self):
        cases = (
            {"../outside": {"sha256": "a" * 64, "size": 1}},
            {"en_us/dict.tsv": {"sha256": "bad", "size": 1}},
            {"en_us/dict.tsv": {"sha256": "a" * 64, "size": -1}},
        )
        for assets in cases:
            with self.subTest(assets=assets):
                response = Mock()
                response.json.return_value = {"schema_version": 1, "assets": assets}
                with patch("moonshine_voice.download.requests.get", return_value=response):
                    with self.assertRaises(ValueError):
                        load_tts_asset_manifest("https://cdn.example/manifest.json")

    def test_asset_download_requires_manifest_entry_and_forwards_budgets(self):
        digest = "b" * 64
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "moonshine_voice.download.load_tts_asset_manifest",
                return_value={"en_us/dict.tsv": {"sha256": digest, "size": 42}},
            ), patch("moonshine_voice.download.download_file") as mocked_download:
                _download_tts_asset_keys(
                    ["en_us/dict.tsv"], Path(temp_dir), show_progress=False
                )
            mocked_download.assert_called_once()
            kwargs = mocked_download.call_args.kwargs
            self.assertEqual(kwargs["expected_sha256"], digest)
            self.assertEqual(kwargs["expected_size"], 42)

    def test_cached_file_with_wrong_hash_is_replaced(self):
        payload = b"trusted asset"
        digest = hashlib.sha256(payload).hexdigest()

        class Response:
            status_code = 200
            headers = {"Content-Length": str(len(payload))}

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size=8192):
                yield payload

        with tempfile.TemporaryDirectory() as temp_dir:
            dest = Path(temp_dir) / "en_us" / "dict.tsv"
            dest.parent.mkdir()
            dest.write_bytes(b"tampered")
            with patch("moonshine_voice.download_file.requests.get", return_value=Response()):
                download_file(
                    "https://cdn.example/en_us/dict.tsv",
                    dest,
                    expected_sha256=digest,
                    expected_size=len(payload),
                    show_progress=False,
                )
            self.assertEqual(dest.read_bytes(), payload)

    def test_failed_refresh_keeps_previous_cache_entry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dest = Path(temp_dir) / "en_us" / "dict.tsv"
            dest.parent.mkdir()
            previous = b"previous cache entry"
            dest.write_bytes(previous)

            class FailedResponse:
                status_code = 503
                headers = {}

                def raise_for_status(self):
                    raise RuntimeError("temporary download failure")

            with patch(
                "moonshine_voice.download_file.requests.get",
                return_value=FailedResponse(),
            ):
                with self.assertRaises(RuntimeError):
                    download_file(
                        "https://cdn.example/en_us/dict.tsv",
                        dest,
                        expected_sha256="a" * 64,
                        expected_size=len(previous) + 1,
                        show_progress=False,
                    )
            self.assertEqual(dest.read_bytes(), previous)

    def test_model_manifest_is_required_and_forwards_component_integrity(self):
        digest = "c" * 64
        response = Mock()
        response.json.return_value = {
            "schema_version": 1,
            "assets": {"encoder_model.ort": {"sha256": digest, "size": 99}},
        }
        with patch("moonshine_voice.download.requests.get", return_value=response):
            self.assertEqual(
                load_model_asset_manifest("https://cdn.example/model/tiny"),
                {"encoder_model.ort": {"sha256": digest, "size": 99}},
            )
        with patch(
            "moonshine_voice.download.load_model_asset_manifest",
            return_value={"encoder_model.ort": {"sha256": digest, "size": 99}},
        ), patch("moonshine_voice.download.download_model") as mocked_download:
            _download_model_components(
                "https://cdn.example/model/tiny", "/tmp/cache", ["encoder_model.ort"]
            )
        mocked_download.assert_called_once_with(
            "https://cdn.example/model/tiny/encoder_model.ort",
            "/tmp/cache/encoder_model.ort",
            expected_sha256=digest,
            expected_size=99,
        )


if __name__ == "__main__":
    unittest.main()
