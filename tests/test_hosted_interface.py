import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from streamlit.testing.v1 import AppTest

from tourist_places.i18n import text
from tourist_places.model_catalog import MODEL_ORDER

PROJECT = Path(__file__).resolve().parents[1]


def fake_checkpoint(path, device):
    return torch.nn.Identity(), {
        "classes": [{"id": 0, "name": "TULUM", "state": "QUINTANA ROO", "types": ["beach", "town"]},
                    {"id": 1, "name": "TEQUILA", "state": "JALISCO", "types": ["town"]}],
        "architecture": "legacy_cnn" if Path(path).parent.name == "legacy_pilot" else "efficientnet_b0",
        "config": {"epochs": 35, "warmup_epochs": 3, "no_pretrained": False, "update_batchnorm": False},
    }


class HostedInterfaceTests(unittest.TestCase):
    def test_hosted_success_shows_four_numbered_models_and_server_privacy_without_examples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for run in reversed(MODEL_ORDER):
                checkpoint = root / run / "best.pt"
                checkpoint.parent.mkdir()
                checkpoint.write_bytes(run.encode("ascii"))
            with patch.dict(os.environ, {"TOURIST_HOSTED": "1", "TOURIST_MODEL_DIR": directory}), \
                    patch("tourist_places.hosting.ensure_hosted_models", return_value=root) as prepare, \
                    patch("tourist_places.learning.load_checkpoint", side_effect=fake_checkpoint) as load, \
                    patch("tourist_places.uploader.travel_photo_uploader", return_value=None):
                app = AppTest.from_file(str(PROJECT / "app.py")).run(timeout=30)
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(len(app.error), 0)
                prepare.assert_called_once_with(PROJECT / "cloud" / "models.json", directory)
                load.assert_called_once()
                self.assertEqual(str(load.call_args.args[1]), "cpu")
                picker = app.selectbox(key="model_choice_es")
                self.assertEqual([option.split(".")[0] for option in picker.options], ["1", "2", "3", "4"])
                self.assertEqual(picker.value.parent.name, "stable_transfer_pilot")
                self.assertEqual(len(app.selectbox), 1)
                self.assertFalse(any(str(item.key).startswith("example_photo") for item in app.selectbox))
                self.assertIn(text("hosted_privacy", "es"), [item.value for item in app.sidebar.caption])
                self.assertNotIn(text("privacy", "es"), [item.value for item in app.caption])
                self.assertEqual(len(app.table[0].value), 4)

    def test_hosted_failure_supports_english_link_localized_retry_and_hides_model_picker(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"TOURIST_HOSTED": "1", "TOURIST_MODEL_DIR": directory}), \
                patch("tourist_places.hosting.ensure_hosted_models", side_effect=RuntimeError("private diagnostic")) as prepare, \
                patch("tourist_places.learning.load_checkpoint") as load, \
                self.assertLogs(level="ERROR"):
            app = AppTest.from_file(str(PROJECT / "app.py"))
            app.query_params["lang"] = "en"
            app.run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual([item.value for item in app.error], [text("hosted_error", "en")])
            self.assertEqual(len(app.selectbox), 0)
            self.assertEqual(app.segmented_control(key="language").value, "en")
            retry = next(button for button in app.button if button.label == text("retry_models", "en"))
            retry.click().run(timeout=30)
            self.assertGreaterEqual(prepare.call_count, 2)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual([item.value for item in app.error], [text("hosted_error", "en")])
            load.assert_not_called()
            app.segmented_control(key="language").set_value("es").run(timeout=30)
            self.assertEqual([item.value for item in app.error], [text("hosted_error", "es")])
            self.assertTrue(any(button.label == text("retry_models", "es") for button in app.button))
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.selectbox), 0)


if __name__ == "__main__":
    unittest.main()
