import base64
import io
import os
import string
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from streamlit.testing.v1 import AppTest

from tourist_places.i18n import MESSAGES, percentage, text
from tourist_places.inference import UploadError
from tourist_places.uploader import uploaded_bytes

PROJECT = Path(__file__).resolve().parents[1]


class FixedModel(torch.nn.Module):
    def forward(self, batch):
        return torch.tensor([[2.0, 1.0]]).expand(len(batch), -1)


class LanguageTests(unittest.TestCase):
    def test_all_messages_have_two_translations_and_matching_parameters(self):
        formatter = string.Formatter()
        for key, translations in MESSAGES.items():
            self.assertEqual(len(translations), 2, key)
            self.assertTrue(all(translations), key)
            placeholders = [{name for _, name, _, _ in formatter.parse(value) if name} for value in translations]
            self.assertEqual(*placeholders, key)
        self.assertEqual(percentage(.731, "es"), "73,1 %")
        self.assertEqual(percentage(.731, "en"), "73.1%")

    def test_english_direct_link_and_spanish_default(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"TOURIST_MODEL_DIR": directory}):
            app = AppTest.from_file(str(PROJECT / "app.py"))
            app.query_params["lang"] = "en"
            app.run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("Where could this be?" in item.value for item in app.markdown))
            app.segmented_control(key="language").set_value("es").run()
            self.assertEqual(app.query_params["lang"], "es")
            self.assertTrue(any("¿Dónde podría ser?" in item.value for item in app.markdown))

    def test_language_change_keeps_photo_prediction_score_and_model(self):
        stream = io.BytesIO()
        Image.new("RGB", (30, 20), "orange").save(stream, format="PNG")
        upload = {"name": "Mi viaje.png", "content": base64.b64encode(stream.getvalue()).decode("ascii")}
        classes = [{"id": 0, "name": "TULUM", "state": "QUINTANA ROO", "types": ["beach", "town"]},
                   {"id": 1, "name": "TEQUILA", "state": "JALISCO", "types": ["town"]}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recommended" / "best.pt"
            path.parent.mkdir()
            path.touch()
            with patch.dict(os.environ, {"TOURIST_MODEL_DIR": directory}), \
                    patch("tourist_places.learning.load_checkpoint", return_value=(FixedModel(), {"classes": classes})), \
                    patch("tourist_places.uploader.travel_photo_uploader", return_value=upload):
                app = AppTest.from_file(str(PROJECT / "app.py")).run(timeout=30)
                app.button(key="analyze").click().run()
                self.assertEqual(len(app.exception), 0)
                prediction = dict(app.session_state["prediction"])
                selected = app.selectbox(key="model_choice_es").value
                self.assertEqual(app.metric[0].value, "73,1 %")
                self.assertTrue(any("playa · pueblo" in item.value for item in app.caption))
                app.segmented_control(key="language").set_value("en").run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(app.session_state["prediction"], prediction)
                self.assertEqual(app.selectbox(key="model_choice_en").value, selected)
                self.assertEqual(app.selectbox(key="model_choice_en").options, ["Recommended"])
                self.assertEqual(app.metric[0].value, "73.1%")
                self.assertEqual(app.metric[0].label, "Closest match score")
                self.assertTrue(any("beach · town" in item.value for item in app.caption))
                self.assertEqual(list(app.dataframe[0].value.columns), ["Destination", "State", "Type"])
                app.segmented_control(key="language").set_value("es").run()
                self.assertEqual(app.session_state["prediction"], prediction)
                self.assertEqual(app.metric[0].value, "73,1 %")
                self.assertEqual(app.selectbox(key="model_choice_es").options, ["Recomendado"])
                self.assertEqual(list(app.dataframe[0].value.columns), ["Destino", "Estado", "Tipo"])

    def test_upload_errors_are_localized_after_switching_languages(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"TOURIST_MODEL_DIR": directory}), \
                patch("tourist_places.uploader.travel_photo_uploader", return_value={"error": "upload_size"}):
            app = AppTest.from_file(str(PROJECT / "app.py")).run(timeout=30)
            self.assertEqual(app.error[0].value, text("upload_size", "es"))
            app.segmented_control(key="language").set_value("en").run()
            self.assertEqual(app.error[0].value, text("upload_size", "en"))
            self.assertEqual(len(app.exception), 0)


class UploadTransportTests(unittest.TestCase):
    def test_binary_photo_is_not_truncated_or_reencoded(self):
        data = bytes(range(256)) * 10
        self.assertEqual(uploaded_bytes({"content": base64.b64encode(data).decode("ascii")}), data)

    def test_browser_limit_cannot_bypass_server_validation(self):
        for value in (None, {}, {"content": "%%%"}, {"content": 123}, {"content": ""}):
            with self.assertRaises(UploadError):
                uploaded_bytes(value)
        with patch("tourist_places.uploader.MAX_UPLOAD_BYTES", 10):
            with self.assertRaises(UploadError) as caught:
                uploaded_bytes({"content": base64.b64encode(b"x" * 11).decode("ascii")})
            self.assertEqual(caught.exception.code, "upload_size")


if __name__ == "__main__":
    unittest.main()
