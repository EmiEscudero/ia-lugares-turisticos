"""Build destination labels and content-separated splits from the original Drive folders."""
import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
TYPES = {"PLAYAS": "beach", "PUEBLOS": "town"}


def normalize(value):
    plain = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Z0-9]+", " ", plain.upper()).strip()


def difference_hash(image):
    pixels = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).tobytes()
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return value


def inspect_image(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.load()
    mirrored = ImageOps.mirror(image)
    shape = f"{image.width}x{image.height}:".encode()
    digests = [hashlib.sha256(shape + img.tobytes()).hexdigest() for img in (image, mirrored)]
    return min(digests), (difference_hash(image), difference_hash(mirrored)), image.width * image.height


def discover(source, original_names=None):
    original_names = original_names or {}
    records, invalid, ignored = [], [], []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        relative = path.relative_to(source)
        parts = relative.parts
        category_index = next((i for i, part in enumerate(parts[:-1]) if normalize(part) in TYPES), None)
        if category_index is None or len(parts) < category_index + 4:
            ignored.append(relative.as_posix())
            continue
        state, place = normalize(parts[category_index + 1]), normalize(parts[category_index + 2])
        try:
            digest, perceptual, area = inspect_image(path)
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            invalid.append({"path": relative.as_posix(), "error": str(error)})
            continue
        records.append({"path": relative.as_posix(), "key": f"{state}/{place}",
                        "state": state, "name": place, "type": TYPES[normalize(parts[category_index])],
                        "original_name": original_names.get(relative.as_posix(), path.name),
                        "content_hash": digest, "perceptual": perceptual, "area": area})
    return records, invalid, ignored


def duplicate_groups(records, distance):
    """Group image families transitively, including re-encoded and mirrored photos.

    A hash within d bits must share one of d+1 disjoint hash segments. Buckets
    provide candidates; full Hamming distances still determine every match.
    """
    parents = list(range(len(records)))
    exact, buckets = {}, defaultdict(set)

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def segments(value):
        parts = distance + 1
        for part in range(parts):
            start, end = part * 64 // parts, (part + 1) * 64 // parts
            yield part, (value >> start) & ((1 << (end - start)) - 1)

    for i, left in enumerate(records):
        candidates = set()
        if left["content_hash"] in exact:
            candidates.add(exact[left["content_hash"]])
        if distance >= 0:
            for fingerprint in left["perceptual"]:
                for segment in segments(fingerprint):
                    candidates.update(buckets[segment])
        for j in candidates:
            right = records[j]
            similar = left["content_hash"] == right["content_hash"]
            if not similar and distance >= 0:
                similar = min((a ^ b).bit_count() for a in left["perceptual"] for b in right["perceptual"]) <= distance
            if similar:
                parents[find(i)] = find(j)
        exact[left["content_hash"]] = i
        if distance >= 0:
            for fingerprint in left["perceptual"]:
                for segment in segments(fingerprint):
                    buckets[segment].add(i)
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[find(index)].append(record)
    return list(groups.values())


def prepare(source, output, seed=42, val_fraction=0.15, test_fraction=0.15, min_images=6, hash_distance=4, exclude_conflicts=False, inventory_path=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir():
        raise ValueError(f"Image folder does not exist: {source}")
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("Validation and test fractions must be positive and sum to less than 1.")
    if min_images < 3 or not -1 <= hash_distance <= 64:
        raise ValueError("min-images must be >=3; hash-distance must be -1..64 (-1 disables near matching).")
    if (output / "manifest.csv").exists():
        raise ValueError("Prepared manifest already exists. Choose a new output folder to preserve the split.")
    original_names = {}
    if inventory_path:
        inventory = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
        original_names = {Path(r["path"]).as_posix(): Path(r["original_path"]).name for r in inventory}
    records, invalid, ignored = discover(source, original_names)
    if not records:
        raise ValueError("No images found under PLAYAS/<state>/<place>/ or PUEBLOS/<state>/<place>/.")
    by_hash, by_place, original_places = defaultdict(list), defaultdict(list), defaultdict(list)
    for record in records:
        by_hash[record["content_hash"]].append(record)
        original_places[record["key"]].append(record)
    conflicts = [[{"path": r["path"], "destination": r["key"]} for r in group]
                 for group in by_hash.values() if len({r["key"] for r in group}) > 1]
    families = duplicate_groups(records, hash_distance)
    conflicting_families = [group for group in families if len({r["key"] for r in group}) > 1]
    report = {"scanned_images": len(records), "unreadable_images": invalid, "ignored_images": ignored,
              "original_destinations": len(original_places), "image_families": len(families),
              "exact_content_label_conflicts": conflicts, "excluded_classes": [], "duplicate_groups": [],
              "cross_destination_duplicate_groups": [[{"path": r["path"], "destination": r["key"]} for r in group] for group in conflicting_families],
              "excluded_conflicting_images": sum(map(len, conflicting_families)) if exclude_conflicts else 0,
              "near_match_scope": "All destinations and both mirror orientations; visually similar groups require human review."}
    output.mkdir(parents=True, exist_ok=True)

    def save_report():
        (output / "preparation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if conflicting_families and not exclude_conflicts:
        save_report()
        raise ValueError("Matching image families have conflicting destination labels. Review preparation_report.json or explicitly quarantine them with --exclude-conflicts.")
    for group in families:
        if len({r["key"] for r in group}) == 1:
            by_place[group[0]["key"]].append(group)
    classes, rows = [], []
    for key in sorted(original_places):
        groups = by_place[key]
        place_records = [record for group in groups for record in group]
        for group in groups:
            if len(group) > 1:
                report["duplicate_groups"].append({"destination": key, "paths": [r["path"] for r in group]})
        if len(groups) < min_images:
            report["excluded_classes"].append({"destination": key, "files": len(original_places[key]), "unique_groups": len(groups)})
            continue
        class_id = len(classes)
        first = place_records[0]
        classes.append({"id": class_id, "key": key, "name": first["name"], "state": first["state"],
                        "types": sorted({r["type"] for r in original_places[key]})})
        # Keep one representative per group. Prefer original filenames, then larger images.
        representatives = [sorted(group, key=lambda r: ("ESPEJO" in normalize(r["original_name"]), -r["area"], r["path"]))[0] for group in groups]
        rng = random.Random(f"{seed}:{key}")
        rng.shuffle(representatives)
        n_val, n_test = max(1, round(len(groups) * val_fraction)), max(1, round(len(groups) * test_fraction))
        while n_val + n_test >= len(groups):
            if n_val >= n_test and n_val > 1:
                n_val -= 1
            else:
                n_test -= 1
        for i, record in enumerate(representatives):
            split = "val" if i < n_val else "test" if i < n_val + n_test else "train"
            rows.append({"path": record["path"], "class_id": class_id, "split": split, "group_id": record["content_hash"]})
    if len(classes) < 2:
        save_report()
        raise ValueError("Fewer than two destinations have enough independent photos. Review exclusions or add images.")
    manifest = output / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "class_id", "split", "group_id"])
        writer.writeheader()
        writer.writerows(rows)
    metadata = {"format_version": 1, "source_root": str(source), "classes": classes, "seed": seed,
                "val_fraction": val_fraction, "test_fraction": test_fraction, "min_images": min_images,
                "hash_distance": hash_distance, "exclude_conflicts": exclude_conflicts,
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    report.update({"included_destinations": len(classes), "included_independent_images": len(rows),
                   "split_images": {split: sum(r["split"] == split for r in rows) for split in ("train", "val", "test")}})
    save_report()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/prepared"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--min-images", type=int, default=6)
    parser.add_argument("--hash-distance", type=int, default=4)
    parser.add_argument("--exclude-conflicts", action="store_true", help="Quarantine every photo in cross-destination matching families; never guess which label is correct.")
    parser.add_argument("--inventory", type=Path, help="Optional local Drive inventory retaining original filenames when copies use file IDs.")
    args = parser.parse_args()
    try:
        report = prepare(args.source, args.output, args.seed, args.val_fraction, args.test_fraction, args.min_images, args.hash_distance, args.exclude_conflicts, args.inventory)
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps({k: report[k] for k in ("included_destinations", "included_independent_images", "split_images")}, indent=2))
    print(f"Review {args.output / 'preparation_report.json'} before training.")


if __name__ == "__main__":
    main()
