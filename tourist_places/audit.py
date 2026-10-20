"""Inspect the historical label files without changing them (no ML dependencies)."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames
        rows = list(reader)
    for row in rows:
        if "subcategoty" in row:
            row["subcategory"] = row["subcategoty"]
    return headers, rows


def audit_legacy(root):
    root = Path(root)
    _, labels = read_csv(root / "labels.csv")
    mapping = {(r["category"], r["subcategory"]): r for r in labels}
    destinations = defaultdict(list)
    for row in labels:
        destinations[(row["state"], row["name"])].append(
            [row["category"], row["subcategory"]]
        )
    report = {
        "label_entries": len(labels),
        "unique_destination_names_with_state": len(destinations),
        "duplicate_mapping_keys": len(labels) - len(mapping),
        "destinations_in_multiple_categories": [
            {"state": state, "name": name, "label_keys": keys}
            for (state, name), keys in destinations.items() if len(keys) > 1
        ],
        "splits": {},
    }
    paths = {}
    for name in ("train", "val"):
        headers, rows = read_csv(root / f"{name}_labels.csv")
        counts = Counter((r["category"], r["subcategory"]) for r in rows)
        paths[name] = {r["filename"] for r in rows}
        mismatches = []
        for row in rows:
            expected = mapping.get((row["category"], row["subcategory"]))
            if expected and any(row[k] != expected[k] for k in ("name", "state")):
                mismatches.append({"row": row, "mapping": expected})
        report["splits"][name] = {
            "images": len(rows), "classes": len(counts),
            "category_images": dict(Counter(r["category"] for r in rows)),
            "minimum_images_per_class": min(counts.values(), default=0),
            "median_images_per_class": median(counts.values()) if counts else 0,
            "classes_with_at_most_3_images": sum(n <= 3 for n in counts.values()),
            "class_size_distribution": dict(sorted(Counter(counts.values()).items())),
            "misspelled_subcategory_header": "subcategoty" in headers,
            "duplicate_filename_rows": len(rows) - len(paths[name]),
            "unknown_keys": [list(k) for k in sorted(set(counts) - set(mapping))],
            "missing_classes": [mapping[k] for k in sorted(set(mapping) - set(counts))],
            "name_state_mismatches": mismatches,
        }
    report["overlapping_filenames"] = sorted(paths["train"] & paths["val"])
    report["limitation"] = "Filename disjointness does not prove image-content disjointness."
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_legacy(args.root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {"label_entries": report["label_entries"], "splits": {}}
    for split, values in report["splits"].items():
        summary["splits"][split] = {
            key: values[key] for key in ("images", "classes", "median_images_per_class", "classes_with_at_most_3_images")
        }
        summary["splits"][split]["name_state_mismatches"] = len(values["name_state_mismatches"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
