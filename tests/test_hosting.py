import hashlib
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from tourist_places.hosting import RELEASE_BASE, ensure_hosted_models
from tourist_places.model_catalog import MODEL_ORDER


class HostingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.cache = self.directory / "cache"
        self.manifest_path = self.directory / "models.json"
        self.payloads = {run: (run + "-weights").encode("ascii") for run in MODEL_ORDER}
        self.manifest = {"release": "streamlit-pilot-v1", "models": [
            {"run": run, "url": f"{RELEASE_BASE}/streamlit-pilot-v1/{run}.pt",
             "sha256": hashlib.sha256(self.payloads[run]).hexdigest(), "bytes": len(self.payloads[run]),
             "test_metrics": {"images": 209, "top1": 0.3}}
            for run in MODEL_ORDER
        ]}
        self.write_manifest()

    def write_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def response(self, request, timeout):
        self.assertEqual(timeout, 60)
        run = request.full_url.rsplit("/", 1)[-1].removesuffix(".pt")
        return io.BytesIO(self.payloads[run])

    def test_downloads_all_four_models_and_reuses_verified_cache(self):
        with patch("tourist_places.hosting.urllib.request.urlopen", side_effect=self.response) as download:
            self.assertEqual(ensure_hosted_models(self.manifest_path, self.cache), self.cache.resolve())
            self.assertEqual(download.call_count, 4)
        for run in MODEL_ORDER:
            self.assertEqual((self.cache / run / "best.pt").read_bytes(), self.payloads[run])
            self.assertEqual(json.loads((self.cache / run / "test_metrics.json").read_text()),
                             {"images": 209, "top1": 0.3})
        with patch("tourist_places.hosting.urllib.request.urlopen") as download:
            ensure_hosted_models(self.manifest_path, self.cache)
            download.assert_not_called()
        self.assertEqual(list(self.cache.rglob("*.part")), [])

    def test_same_size_corruption_is_replaced_only_after_verified_download(self):
        with patch("tourist_places.hosting.urllib.request.urlopen", side_effect=self.response):
            ensure_hosted_models(self.manifest_path, self.cache)
        target = self.cache / MODEL_ORDER[0] / "best.pt"
        target.write_bytes(b"x" * len(self.payloads[MODEL_ORDER[0]]))
        with patch("tourist_places.hosting.urllib.request.urlopen", side_effect=self.response) as download:
            ensure_hosted_models(self.manifest_path, self.cache)
            self.assertEqual(download.call_count, 1)
        self.assertEqual(target.read_bytes(), self.payloads[MODEL_ORDER[0]])

    def test_rejects_bad_digest_truncation_and_excess_bytes_and_cleans_own_partial(self):
        for body in (b"x" * len(self.payloads[MODEL_ORDER[0]]), b"short", self.payloads[MODEL_ORDER[0]] + b"!"):
            with self.subTest(body=body):
                first = self.cache / MODEL_ORDER[0]
                first.mkdir(parents=True, exist_ok=True)
                previous = first / "best.pt"
                previous.write_bytes(b"preserve this previous checkpoint")
                unrelated = first / "unrelated.part"
                unrelated.write_bytes(b"other work")
                with patch("tourist_places.hosting.urllib.request.urlopen", return_value=io.BytesIO(body)):
                    with self.assertRaises(RuntimeError):
                        ensure_hosted_models(self.manifest_path, self.cache)
                self.assertEqual(previous.read_bytes(), b"preserve this previous checkpoint")
                self.assertEqual(list(first.glob("*.part")), [unrelated])

    def test_network_failure_does_not_leave_a_partial_download(self):
        with patch("tourist_places.hosting.urllib.request.urlopen", side_effect=urllib.error.URLError("unavailable")):
            with self.assertRaises(RuntimeError):
                ensure_hosted_models(self.manifest_path, self.cache)
        self.assertEqual(list(self.cache.rglob("*.part")), [])

    def test_validates_all_entries_before_io_and_rejects_path_traversal(self):
        for bad_run in ("../outside", "C:/outside", MODEL_ORDER[0]):
            with self.subTest(run=bad_run):
                self.manifest["models"][-1]["run"] = bad_run
                self.write_manifest()
                with patch("tourist_places.hosting.urllib.request.urlopen") as download:
                    with self.assertRaises(ValueError):
                        ensure_hosted_models(self.manifest_path, self.cache)
                    download.assert_not_called()
                self.assertFalse(self.cache.exists())

    def test_rejects_foreign_or_modified_urls_invalid_size_and_invalid_digest(self):
        for key, value in (("url", "https://example.com/model.pt"),
                           ("url", self.manifest["models"][0]["url"] + "?query=1"),
                           ("bytes", True), ("bytes", 0), ("sha256", "not-a-hash")):
            with self.subTest(key=key, value=value):
                original = self.manifest["models"][0][key]
                self.manifest["models"][0][key] = value
                self.write_manifest()
                with patch("tourist_places.hosting.urllib.request.urlopen") as download:
                    with self.assertRaises(ValueError):
                        ensure_hosted_models(self.manifest_path, self.cache)
                    download.assert_not_called()
                self.manifest["models"][0][key] = original
                self.assertFalse(self.cache.exists())

    def test_rejects_missing_models_and_traversal_in_release_tag(self):
        self.manifest["release"] = "../outside"
        self.write_manifest()
        with self.assertRaises(ValueError):
            ensure_hosted_models(self.manifest_path, self.cache)
        self.manifest["release"] = "streamlit-pilot-v1"
        self.manifest["models"].pop()
        self.write_manifest()
        with self.assertRaises(ValueError):
            ensure_hosted_models(self.manifest_path, self.cache)
        self.assertFalse(self.cache.exists())


if __name__ == "__main__":
    unittest.main()
