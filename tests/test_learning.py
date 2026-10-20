import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from tourist_places.learning import build_model, classification_metrics, load_checkpoint, read_prepared, run_epoch


class MetricTests(unittest.TestCase):
    def test_fine_tuning_can_preserve_or_update_batchnorm_statistics(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.features = torch.nn.Sequential(torch.nn.BatchNorm2d(3))
                self.classifier = torch.nn.Linear(3, 2)
            def forward(self, x):
                return self.classifier(self.features(x).mean((2, 3)))
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        batch = [(torch.randn(4,3,8,8), torch.tensor([0,1,0,1]))]
        criterion = torch.nn.CrossEntropyLoss()
        before = model.classifier.weight.detach().clone()
        run_epoch(model,batch,torch.device("cpu"),2,criterion,optimizer,scaler,amp=False,freeze_batchnorm=True)
        self.assertEqual(model.features[0].num_batches_tracked.item(), 0)
        self.assertFalse(torch.equal(before, model.classifier.weight))
        run_epoch(model,batch,torch.device("cpu"),2,criterion,optimizer,scaler,amp=False,freeze_batchnorm=False)
        self.assertEqual(model.features[0].num_batches_tracked.item(), 1)

    def test_legacy_baseline_restores_its_input_size_and_destination_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            model = build_model(3, architecture="legacy_cnn")
            path = Path(directory) / "legacy.pt"
            classes = [{"id": i, "name": name} for i,name in enumerate(["TULUM", "BACALAR", "TEQUILA"])]
            torch.save({"format_version": 1, "architecture": "legacy_cnn", "model_state": model.state_dict(), "classes": classes}, path)
            restored, metadata = load_checkpoint(path, torch.device("cpu"))
            self.assertEqual(restored.image_size, 128)
            self.assertFalse(restored.normalize_inputs)
            self.assertEqual(metadata["classes"], classes)
            self.assertEqual(tuple(restored(torch.zeros(2, 3, 128, 128)).shape), (2, 3))

    def test_macro_metrics_expose_failure_on_minority_destination(self):
        metrics = classification_metrics(torch.tensor([[9, 1], [1, 0]]), 11, 11, 22)
        self.assertAlmostEqual(metrics["top1"], 9 / 11)
        self.assertAlmostEqual(metrics["macro_recall"], 0.45)
        self.assertAlmostEqual(metrics["macro_f1"], 0.45)
        self.assertEqual(metrics["loss"], 2)

    def test_reused_group_cannot_cross_splits_even_with_updated_manifest_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for class_id in range(2):
                for split in ("train", "val", "test"):
                    path = root / f"{class_id}_{split}.jpg"
                    path.touch()
                    rows.append({"path": path.name, "class_id": class_id, "split": split,
                                 "group_id": "same-content" if class_id == 0 else path.stem})
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            metadata = {"format_version": 1, "source_root": str(root),
                        "classes": [{"id": 0}, {"id": 1}],
                        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
            (root / "metadata.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "duplicate group"):
                read_prepared(root)


if __name__ == "__main__":
    unittest.main()
