import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageOps

from tourist_places.audit import audit_legacy
from tourist_places.prepare import duplicate_groups, inspect_image, prepare


def make_image(path, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    image = Image.frombytes("RGB", (32, 32), rng.randbytes(32 * 32 * 3)).resize((128, 128))
    image.save(path)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "raw"
        for name, offset in [("TULUM", 0), ("BACALAR", 100)]:
            for index in range(6):
                make_image(self.source / "PLAYAS" / "QUINTANA_ROO" / name / "train" / f"{index}.png", offset + index)

    def prepare(self, output="prepared"):
        return prepare(self.source, self.root / output, min_images=6, hash_distance=-1)

    def test_merges_destination_types_and_removes_mirrored_duplicate(self):
        original = self.source / "PLAYAS/QUINTANA_ROO/TULUM/train/0.png"
        target = self.source / "PUEBLOS/QUINTANA_ROO/TULUM/validation/espejo.png"
        target.parent.mkdir(parents=True)
        with Image.open(original) as image:
            ImageOps.mirror(image).save(target)
        report = self.prepare()
        metadata = json.loads((self.root / "prepared/metadata.json").read_text())
        tulum = next(c for c in metadata["classes"] if c["name"] == "TULUM")
        self.assertEqual(tulum["types"], ["beach", "town"])
        self.assertEqual(report["included_independent_images"], 12)
        self.assertEqual(len(report["duplicate_groups"]), 1)
        with (self.root / "prepared/manifest.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len({r["group_id"] for r in rows}), len(rows))
        self.assertEqual({r["split"] for r in rows}, {"train", "val", "test"})
        for class_id in ("0", "1"):
            self.assertEqual({r["split"] for r in rows if r["class_id"] == class_id}, {"train", "val", "test"})

    def test_recompressed_mirror_stays_in_same_duplicate_group(self):
        path = self.source / "PLAYAS/QUINTANA_ROO/TULUM/train/0.png"
        mirror = self.root / "mirror.jpg"
        with Image.open(path) as image:
            ImageOps.mirror(image).save(mirror, quality=90)
        records = []
        for item in (path, mirror):
            digest, perceptual, _ = inspect_image(item)
            records.append({"content_hash": digest, "perceptual": perceptual})
        self.assertEqual(len(duplicate_groups(records, 4)), 1)

    def test_original_inventory_name_is_used_when_local_files_have_ids(self):
        old = self.source / "PLAYAS/QUINTANA_ROO/TULUM/train/0.png"
        original = old.with_name("zz_original_id.png")
        old.rename(original)
        with Image.open(original) as image:
            ImageOps.mirror(image).save(old)
        inventory = self.root / "inventory.json"
        inventory.write_text(json.dumps([
            {"path": str(original.relative_to(self.source)), "original_path": "tulum.jpg"},
            {"path": str(old.relative_to(self.source)), "original_path": "espejo.jpg"},
        ]))
        prepare(self.source, self.root / "prepared", min_images=6, hash_distance=-1, inventory_path=inventory)
        with (self.root / "prepared/manifest.csv").open(newline="") as stream:
            paths = {r["path"] for r in csv.DictReader(stream)}
        self.assertIn(original.relative_to(self.source).as_posix(), paths)
        self.assertNotIn(old.relative_to(self.source).as_posix(), paths)

    def test_conflicting_exact_content_stops_preparation(self):
        original = self.source / "PLAYAS/QUINTANA_ROO/TULUM/train/0.png"
        target = self.source / "PLAYAS/QUINTANA_ROO/BACALAR/validation/wrong.png"
        target.parent.mkdir(parents=True)
        target.write_bytes(original.read_bytes())
        with self.assertRaisesRegex(ValueError, "conflicting destination"):
            self.prepare()
        self.assertFalse((self.root / "prepared/manifest.csv").exists())
        report = json.loads((self.root / "prepared/preparation_report.json").read_text())
        self.assertEqual(len(report["exact_content_label_conflicts"]), 1)

    def test_sparse_classes_are_excluded_and_reported(self):
        make_image(self.source / "PUEBLOS/JALISCO/TEQUILA/train/0.png", 500)
        report = self.prepare()
        self.assertEqual(report["excluded_classes"], [{"destination": "JALISCO/TEQUILA", "files": 1, "unique_groups": 1}])

    def test_conflict_quarantine_removes_both_labels_without_touching_sources(self):
        original = self.source / "PLAYAS/QUINTANA_ROO/TULUM/train/0.png"
        target = self.source / "PLAYAS/QUINTANA_ROO/BACALAR/validation/wrong.png"
        target.parent.mkdir(parents=True)
        original_bytes = original.read_bytes()
        target.write_bytes(original_bytes)
        report = prepare(self.source, self.root / "prepared", min_images=3, hash_distance=-1, exclude_conflicts=True)
        self.assertEqual(report["excluded_conflicting_images"], 2)
        self.assertEqual(report["included_independent_images"], 11)
        self.assertEqual(original.read_bytes(), original_bytes)
        self.assertEqual(target.read_bytes(), original_bytes)

    def test_segment_index_finds_pairs_with_errors_in_different_hash_segments(self):
        records = [{"content_hash": str(i), "perceptual": (value, value)} for i,value in enumerate([0, (1<<2)|(1<<17)|(1<<35)|(1<<58), (1<<64)-1])]
        self.assertEqual(sorted(len(group) for group in duplicate_groups(records, 4)), [1, 2])

    def test_fixed_seed_repeats_split_and_existing_manifest_is_preserved(self):
        self.prepare("one")
        self.prepare("two")
        self.assertEqual((self.root / "one/manifest.csv").read_bytes(), (self.root / "two/manifest.csv").read_bytes())
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.prepare("one")

    def test_unreadable_image_is_reported(self):
        bad = self.source / "PLAYAS/QUINTANA_ROO/TULUM/train/bad.jpg"
        bad.write_bytes(b"not an image")
        report = self.prepare()
        self.assertEqual(len(report["unreadable_images"]), 1)
        self.assertEqual(report["included_independent_images"], 12)


class HistoricalAuditTests(unittest.TestCase):
    def test_detects_wrong_place_id_and_misspelled_header(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "labels.csv").write_text("category,subcategory,name,state\n1,31,TEPOTZOTLAN,MEXICO\n1,41,TAXCO,GUERRERO\n")
            (root / "train_labels.csv").write_text("filename,category,subcategory,name,state\ntaxco.jpg,1,41,TAXCO,GUERRERO\n")
            (root / "val_labels.csv").write_text("filename,category,subcategoty,name,state\ntaxco2.jpg,1,31,TAXCO,GUERRERO\n")
            report = audit_legacy(root)
            validation = report["splits"]["val"]
            self.assertTrue(validation["misspelled_subcategory_header"])
            self.assertEqual(validation["name_state_mismatches"][0]["mapping"]["name"], "TEPOTZOTLAN")
            self.assertEqual(validation["missing_classes"][0]["name"], "TAXCO")


if __name__ == "__main__":
    unittest.main()
