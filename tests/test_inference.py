import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from streamlit.testing.v1 import AppTest

from tourist_places.inference import available_checkpoints, decode_upload, predict_candidates

PROJECT = Path(__file__).resolve().parents[1]


class UploadTests(unittest.TestCase):
    def test_decodes_exif_orientation_without_saving_upload(self):
        data = io.BytesIO()
        image = Image.new("RGB", (12, 24), "orange")
        exif = image.getexif()
        exif[274] = 6
        image.save(data, format="JPEG", exif=exif)
        decoded = decode_upload(data.getvalue())
        self.assertEqual(decoded.size, (24, 12))
        self.assertEqual(decoded.mode, "RGB")

    def test_rejects_corrupt_empty_and_oversized_uploads(self):
        for data in (b"", b"not a photo"):
            with self.assertRaises(ValueError):
                decode_upload(data)
        with patch("tourist_places.inference.MAX_UPLOAD_BYTES", 10):
            with self.assertRaisesRegex(ValueError, "20 MB"):
                decode_upload(b"x" * 11)

    def test_rejects_unsupported_image_format(self):
        stream = io.BytesIO()
        Image.new("RGB", (10, 10)).save(stream, format="GIF")
        with self.assertRaisesRegex(ValueError, "JPEG, PNG, or WebP"):
            decode_upload(stream.getvalue())

    def test_ranked_predictions_retain_saved_labels_and_cap_top_k(self):
        class FixedModel(torch.nn.Module):
            def forward(self, batch):
                return torch.tensor([[0.0, 2.0]]).expand(len(batch), -1)
        classes = [{"id": 0, "name": "TEQUILA", "types": ["town"]},
                   {"id": 1, "name": "TULUM", "types": ["beach", "town"]}]
        predictions = predict_candidates(FixedModel(), classes, Image.new("RGB", (256, 256)), torch.device("cpu"), 5)
        self.assertEqual(len(predictions), 2)
        self.assertEqual(predictions[0]["name"], "TULUM")
        self.assertEqual(predictions[0]["types"], ["beach", "town"])
        self.assertAlmostEqual(sum(r["score"] for r in predictions), 1, places=6)


class InterfaceTests(unittest.TestCase):
    def test_synthetic_smoke_models_are_not_offered_as_real_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for run in ("smoke_pretrained", "clean_pilot"):
                (root / run).mkdir()
                (root / run / "best.pt").touch()
            self.assertEqual([p.parent.name for p in available_checkpoints(root)], ["clean_pilot"])

    def test_missing_model_is_a_helpful_empty_state(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"TOURIST_MODEL_DIR": directory}):
            app = AppTest.from_file(str(PROJECT / "app.py")).run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("preparando" in item.value for item in app.info))
            self.assertTrue(app.button(key="analyze").disabled)

    def test_bad_checkpoint_produces_friendly_error_without_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "broken" / "best.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"broken model")
            with patch.dict(os.environ, {"TOURIST_MODEL_DIR": directory}):
                with self.assertLogs(level="ERROR"):
                    app = AppTest.from_file(str(PROJECT / "app.py")).run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("No se pudo abrir" in error.value for error in app.error))


if __name__ == "__main__":
    unittest.main()
