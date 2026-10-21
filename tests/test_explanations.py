import copy
import unittest

from tourist_places.explain import PILOT_MANIFEST_SHA256, explain_model


class ExplanationTests(unittest.TestCase):
    def test_architecture_changes_input_size_feature_stages_and_preprocessing(self):
        efficient = explain_model({"architecture": "efficientnet_b0", "classes": [0, 1],
                                   "config": {"no_pretrained": False}}, "en")
        legacy = explain_model({"architecture": "legacy_cnn", "classes": [0, 1]}, "en")
        self.assertIn("224 × 224", efficient["inference_mermaid"])
        self.assertIn("EfficientNet-B0", efficient["inference_mermaid"])
        self.assertIn("2 saved destinations", efficient["inference_mermaid"])
        self.assertIn("Up to 2 candidates", efficient["inference_mermaid"])
        self.assertIn("128 × 128", legacy["inference_mermaid"])
        self.assertIn("16 → 32 → 64", legacy["inference_mermaid"])
        self.assertIn("512-unit", legacy["inference_mermaid"])
        self.assertIn("without ImageNet normalization", legacy["inference_mermaid"])
        self.assertNotIn("224", legacy["inference_mermaid"])

    def test_frozen_fine_tuned_and_unrecorded_settings_are_distinguished(self):
        checkpoint = {"architecture": "efficientnet_b0", "classes": [0, 1],
                      "config": {"epochs": 35, "warmup_epochs": 35, "no_pretrained": False}}
        self.assertIn("Only the classifier learns", explain_model(checkpoint, "en")["training_text"])
        checkpoint["config"].update({"warmup_epochs": 3, "update_batchnorm": False})
        self.assertIn("3 epochs", explain_model(checkpoint, "en")["training_text"])
        self.assertIn("statistics stay fixed", explain_model(checkpoint, "en")["training_text"])
        checkpoint["config"].pop("update_batchnorm")
        self.assertIn("does not record", explain_model(checkpoint, "en")["training_text"])
        checkpoint["config"]["no_pretrained"] = True
        self.assertIn("learned from scratch", explain_model(checkpoint, "en")["build_text"])

    def test_pilot_counts_need_exact_split_identity_and_test_never_feeds_training(self):
        checkpoint = {"architecture": "efficientnet_b0", "classes": list(range(164)),
                      "manifest_sha256": PILOT_MANIFEST_SHA256}
        result = explain_model(checkpoint, "en")
        self.assertTrue(result["is_pilot"])
        self.assertIn("4,668", result["training_mermaid"])
        self.assertIn("957 training", result["training_mermaid"])
        self.assertIn("209 test", result["training_mermaid"])
        self.assertIn("H --> E", result["training_mermaid"])
        self.assertNotIn("H --> S", result["training_mermaid"])
        self.assertNotIn("H --> T", result["training_mermaid"])
        checkpoint["manifest_sha256"] = "another split"
        generic = explain_model(checkpoint, "en")
        self.assertFalse(generic["is_pilot"])
        self.assertNotIn("4,668", generic["training_mermaid"])
        checkpoint["manifest_sha256"] = PILOT_MANIFEST_SHA256
        checkpoint["classes"] = [0, 1]
        self.assertFalse(explain_model(checkpoint)["is_pilot"])

    def test_saved_epoch_and_metric_use_checkpoint_values_and_both_languages(self):
        checkpoint = {"epoch": 27, "classes": [0, 1], "val_metrics": {"macro_f1": .30979}}
        self.assertIn("27", explain_model(checkpoint, "es")["epoch_text"])
        self.assertIn("0,310", explain_model(checkpoint, "es")["epoch_text"])
        self.assertIn("0.310", explain_model(checkpoint, "en")["epoch_text"])
        self.assertIn("De tu foto", explain_model(checkpoint, "es")["inference_title"])
        self.assertIn("From your photo", explain_model(checkpoint, "en")["inference_title"])

    def test_missing_metadata_stays_honest_and_does_not_modify_the_checkpoint(self):
        for checkpoint in ({}, None, {"classes": [0, 1], "config": None}, {"architecture": "unknown"}):
            original = copy.deepcopy(checkpoint)
            result = explain_model(checkpoint, "en")
            self.assertFalse(result["is_pilot"])
            self.assertNotIn("224", result["inference_mermaid"])
            self.assertNotIn("4,668", result["training_mermaid"])
            self.assertIn("not identify", result["build_text"])
            self.assertEqual(checkpoint, original)
            self.assertTrue(result["inference_alt"])
            self.assertTrue(result["training_alt"])


if __name__ == "__main__":
    unittest.main()
