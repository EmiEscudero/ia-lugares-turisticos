"""Evaluate a saved model on validation or on the final held-out test set."""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .learning import PlaceDataset, load_checkpoint, read_prepared, run_epoch, select_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    device = select_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    source, metadata, rows = read_prepared(args.prepared, args.data_root, checkpoint["manifest_sha256"])
    if metadata["classes"] != checkpoint["classes"]:
        parser.error("Class mapping differs from the checkpoint.")
    loader = DataLoader(PlaceDataset(source, rows, args.split, image_size=model.image_size, normalize_inputs=model.normalize_inputs), batch_size=args.batch_size, pin_memory=device.type == "cuda")
    metrics, confusion = run_epoch(model, loader, device, len(metadata["classes"]), torch.nn.CrossEntropyLoss(label_smoothing=0.05))
    destination = args.checkpoint.parent
    (destination / f"{args.split}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (destination / f"{args.split}_per_class.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["destination", "support", "correct", "recall"])
        for index, item in enumerate(metadata["classes"]):
            support, correct = confusion[index].sum().item(), confusion[index, index].item()
            writer.writerow([item["key"], support, correct, correct / support if support else 0])
    with (destination / f"{args.split}_confusions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["actual", "predicted", "images"])
        for actual, predicted in (confusion > 0).nonzero().tolist():
            writer.writerow([metadata["classes"][actual]["key"], metadata["classes"][predicted]["key"], confusion[actual, predicted].item()])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
