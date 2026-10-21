import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from streamlit.testing.v1 import AppTest

from tourist_places.i18n import model_name
from tourist_places.inference import available_checkpoints
from tourist_places.model_catalog import MODEL_ORDER, model_run_name

PROJECT = Path(__file__).resolve().parents[1]


def make_models(root, recommended=True):
    for index, name in enumerate(reversed(MODEL_ORDER)):
        checkpoint = root / name / "best.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(name.encode("ascii"))
        os.utime(checkpoint, (1000 + index, 1000 + index))
    if recommended:
        alias = root / "recommended" / "best.pt"
        alias.parent.mkdir()
        alias.write_bytes((root / "stable_transfer_pilot" / "best.pt").read_bytes())
        (alias.parent / "selection.json").write_text(json.dumps({
            "source_run": "stable_transfer_pilot",
            "checkpoint_sha256": hashlib.sha256(alias.read_bytes()).hexdigest(),
        }), encoding="utf-8")


class ModelCatalogTests(unittest.TestCase):
    def test_fixed_order_survives_changed_file_dates_and_extra_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_models(root, recommended=False)
            for name in ("z_new_run", "a_new_run", "smoke_pretrained"):
                (root / name).mkdir()
                (root / name / "best.pt").write_bytes(name.encode())
            expected = [*MODEL_ORDER, "a_new_run", "z_new_run"]
            self.assertEqual([path.parent.name for path in available_checkpoints(root)], expected)
            os.utime(root / "legacy_pilot" / "best.pt", None)
            self.assertEqual([path.parent.name for path in available_checkpoints(root)], expected)

    def test_verified_recommended_copy_is_the_same_option_in_both_languages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_models(root)
            paths = available_checkpoints(root)
            self.assertEqual(len(paths), 4)
            self.assertEqual(paths[0].parent.name, "recommended")
            self.assertEqual([model_run_name(path) for path in paths], list(MODEL_ORDER))
            self.assertEqual([model_name(path, "es").split(".")[0] for path in paths], ["1", "2", "3", "4"])
            self.assertIn("Recomendado", model_name(paths[0], "es"))
            self.assertIn("Recommended", model_name(paths[0], "en"))

    def test_distinct_checkpoint_is_not_hidden_by_stale_alias_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_models(root)
            self.assertEqual(len(available_checkpoints(root)), 4)
            (root / "stable_transfer_pilot" / "best.pt").write_bytes(b"a new stable checkpoint")
            self.assertEqual(len(available_checkpoints(root)), 5)
            self.assertEqual(model_run_name(root / "recommended" / "best.pt"), "recommended")

    def test_missing_or_invalid_selection_metadata_never_hides_a_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_models(root)
            selection = root / "recommended" / "selection.json"
            for content in ("[]", "not json", json.dumps({"source_run": "../outside"}), "null"):
                selection.write_text(content, encoding="utf-8")
                self.assertEqual(len(available_checkpoints(root)), 5)
            selection.unlink()
            self.assertEqual(len(available_checkpoints(root)), 5)

    def test_verified_alias_still_works_when_only_selected_copy_is_available(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_models(root)
            (root / "stable_transfer_pilot" / "best.pt").unlink()
            self.assertEqual([model_run_name(path) for path in available_checkpoints(root)], list(MODEL_ORDER))


class ModelInterfaceTests(unittest.TestCase):
    def test_numbered_main_picker_and_comparison_keep_the_selection_in_english(self):
        classes = [{"id": 0, "name": "TULUM", "state": "QUINTANA ROO", "types": ["beach", "town"]},
                   {"id": 1, "name": "TEQUILA", "state": "JALISCO", "types": ["town"]}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_models(root)

            def load_for_interface(path, device):
                name = Path(path).parent.name
                return torch.nn.Identity(), {
                    "classes": classes,
                    "architecture": "legacy_cnn" if name == "legacy_pilot" else "efficientnet_b0",
                    "config": {"epochs": 35, "warmup_epochs": 35 if name == "frozen_pilot" else 3,
                               "no_pretrained": False, "update_batchnorm": False},
                }

            with patch.dict(os.environ, {"TOURIST_MODEL_DIR": directory}), \
                    patch("tourist_places.learning.load_checkpoint", side_effect=load_for_interface):
                app = AppTest.from_file(str(PROJECT / "app.py")).run(timeout=30)
                self.assertEqual(len(app.exception), 0)
                picker = app.selectbox(key="model_choice_es")
                self.assertEqual(len(picker.options), 4)
                self.assertEqual(picker.value.parent.name, "recommended")
                self.assertEqual(len(app.sidebar.selectbox), 0)
                self.assertEqual(len(app.table[0].value), 4)
                frozen = root / "frozen_pilot" / "best.pt"
                picker.set_value(frozen).run()
                app.segmented_control(key="language").set_value("en").run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(app.selectbox(key="model_choice_en").value, frozen)
                self.assertEqual(app.selectbox(key="model_choice_en").options[1], "2. Frozen features")
                self.assertEqual(list(app.table[0].value.columns), ["Model", "What changes", "Correct first choice", "In the top 5"])
                self.assertEqual(app.table[0].value.iloc[0]["Model"], "1. Stable fine-tuning · Recommended")
                self.assertTrue(any("224 × 224" in item.value for item in app.markdown))
                legacy = root / "legacy_pilot" / "best.pt"
                app.selectbox(key="model_choice_en").set_value(legacy).run()
                app.segmented_control(key="language").set_value("es").run()
                self.assertEqual(app.selectbox(key="model_choice_es").value, legacy)
                self.assertTrue(any("128 × 128" in item.value for item in app.markdown))


if __name__ == "__main__":
    unittest.main()
