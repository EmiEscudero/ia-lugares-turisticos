"""Train an ImageNet-pretrained destination model; the held-out test set stays unused."""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .learning import PlaceDataset, build_model, read_prepared, run_epoch, select_device


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/efficientnet_b0"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true", help="Offline smoke tests only; use pretrained weights for real training.")
    parser.add_argument("--architecture", choices=("efficientnet_b0", "legacy_cnn"), default="efficientnet_b0")
    parser.add_argument("--cudnn-benchmark", action="store_true", help="Opt into convolution autotuning; may increase startup time on small datasets.")
    parser.add_argument("--update-batchnorm", action="store_true", help="Also adapt pretrained BatchNorm running statistics during fine-tuning.")
    args = parser.parse_args()
    if args.architecture == "legacy_cnn":
        args.warmup_epochs = 0
        args.backbone_learning_rate = args.learning_rate
    if min(args.epochs, args.batch_size, args.patience) < 1 or args.workers < 0 or args.warmup_epochs < 0:
        parser.error("Epochs, batch size, and patience must be positive; workers/warmup must be nonnegative.")
    if args.learning_rate <= 0 or args.backbone_learning_rate <= 0:
        parser.error("Learning rates must be positive.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output folder is not empty; choose a new run name.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    print(f"Device: {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}", flush=True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
    source, metadata, rows = read_prepared(args.prepared, args.data_root)
    args.output.mkdir(parents=True, exist_ok=True)
    # Keep downloaded weights in the project rather than the user's home folder.
    torch.hub.set_dir(str(Path(__file__).resolve().parents[1] / "artifacts" / "torch"))
    model = build_model(len(metadata["classes"]), pretrained=not args.no_pretrained, architecture=args.architecture).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    frozen = args.warmup_epochs > 0
    for parameter in model.features.parameters():
        parameter.requires_grad_(not frozen)
    optimizer = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": args.backbone_learning_rate},
        {"params": model.classifier.parameters(), "lr": args.learning_rate},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not args.no_amp)
    loaders = {}
    for split in ("train", "val"):
        loaders[split] = DataLoader(
            PlaceDataset(source, rows, split, training=split == "train", image_size=model.image_size, normalize_inputs=model.normalize_inputs),
            batch_size=args.batch_size, shuffle=split == "train", num_workers=args.workers,
            pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
            worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(args.seed),
        )
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.output / "run.json").write_text(json.dumps({"config": config, "data": metadata, "torch": str(torch.__version__)}, indent=2), encoding="utf-8")
    best_score, best_loss, stale = -1.0, float("inf"), 0
    history = []
    for epoch in range(args.epochs):
        if frozen and epoch >= args.warmup_epochs:
            for parameter in model.features.parameters():
                parameter.requires_grad_(True)
            frozen = False
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        train_metrics, _ = run_epoch(model, loaders["train"], device, len(metadata["classes"]), criterion, optimizer, scaler, not args.no_amp, frozen, freeze_batchnorm=not args.update_batchnorm)
        val_metrics, _ = run_epoch(model, loaders["val"], device, len(metadata["classes"]), criterion, amp=not args.no_amp)
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        record = {"epoch": epoch + 1, "seconds": time.perf_counter() - started,
                  "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0,
                  **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)
        with (args.output / "history.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(record))
            writer.writeheader()
            writer.writerows(history)
        improved = val_metrics["macro_f1"] > best_score + 1e-6 or (
            abs(val_metrics["macro_f1"] - best_score) <= 1e-6 and val_metrics["loss"] < best_loss)
        if improved:
            best_score, best_loss, stale = val_metrics["macro_f1"], val_metrics["loss"], 0
            checkpoint = {"format_version": 1, "architecture": args.architecture, "epoch": epoch + 1,
                          "model_state": model.state_dict(), "classes": metadata["classes"],
                          "manifest_sha256": metadata["manifest_sha256"], "val_metrics": val_metrics,
                          "config": config, "torch_version": str(torch.__version__)}
            temporary = args.output / "best.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(args.output / "best.pt")
        else:
            stale += 1
        print(f"Epoch {epoch+1}/{args.epochs}: val place top1={val_metrics['top1']:.3f}, top5={val_metrics['top5']:.3f}, macro-F1={val_metrics['macro_f1']:.3f}, {record['seconds']:.1f}s, {record['peak_vram_mb']:.0f} MB", flush=True)
        if stale >= args.patience:
            print("Early stopping: validation destination metrics stopped improving.")
            break
    print(f"Best checkpoint: {args.output / 'best.pt'}; test images have not been evaluated.")


if __name__ == "__main__":
    main()
