"""Shared model, data validation, and destination-level metrics."""
import csv
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


def select_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Install a CUDA PyTorch build and check the NVIDIA driver.")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


class LegacyCNN(torch.nn.Module):
    """Original 3-convolution/512-unit design adapted to one destination head."""
    def __init__(self, num_classes):
        super().__init__()
        layers = []
        for incoming, outgoing in ((3, 16), (16, 32), (32, 64)):
            layers.extend([torch.nn.Conv2d(incoming, outgoing, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2)])
        self.features = torch.nn.Sequential(*layers)
        self.classifier = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(64 * 16 * 16, 512),
                                              torch.nn.ReLU(), torch.nn.Linear(512, num_classes))

    def forward(self, images):
        return self.classifier(self.features(images))


def build_model(num_classes, pretrained=True, architecture="efficientnet_b0"):
    if architecture == "legacy_cnn":
        model = LegacyCNN(num_classes)
        model.image_size, model.normalize_inputs = 128, False
    elif architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        model.image_size, model.normalize_inputs = IMAGE_SIZE, True
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    return model


def image_transform(training=False, image_size=IMAGE_SIZE, normalize_inputs=True):
    if training:
        operations = [transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), interpolation=InterpolationMode.BICUBIC),
                      transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15)]
    else:
        operations = [transforms.Resize(round(image_size * 256 / 224), interpolation=InterpolationMode.BICUBIC), transforms.CenterCrop(image_size)]
    operations.append(transforms.ToTensor())
    if normalize_inputs:
        operations.append(transforms.Normalize(MEAN, STD))
    return transforms.Compose(operations)


def read_prepared(prepared, data_root=None, expected_hash=None):
    prepared = Path(prepared)
    metadata = json.loads((prepared / "metadata.json").read_text(encoding="utf-8"))
    manifest_path = prepared / "manifest.csv"
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if metadata.get("format_version") != 1:
        raise ValueError("Unsupported prepared data format.")
    if digest != metadata["manifest_sha256"] or (expected_hash is not None and digest != expected_hash):
        raise ValueError("Manifest has changed. Restore the saved split or prepare and train a new experiment.")
    classes = metadata["classes"]
    if len(classes) < 2 or [c["id"] for c in classes] != list(range(len(classes))):
        raise ValueError("Class IDs must be consecutive and include at least two destinations.")
    source = Path(data_root or metadata["source_root"]).resolve()
    with manifest_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    seen_paths, seen_groups = set(), set()
    represented = {split: set() for split in ("train", "val", "test")}
    for row in rows:
        row["class_id"] = int(row["class_id"])
        if row["split"] not in represented or not 0 <= row["class_id"] < len(classes):
            raise ValueError("Invalid split or class ID in the manifest.")
        path = (source / row["path"]).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError(f"Image is missing or outside the dataset root: {row['path']}")
        if path in seen_paths or row["group_id"] in seen_groups:
            raise ValueError("An image or duplicate group appears more than once in the manifest.")
        seen_paths.add(path)
        seen_groups.add(row["group_id"])
        represented[row["split"]].add(row["class_id"])
    if any(ids != set(range(len(classes))) for ids in represented.values()):
        raise ValueError("Every included destination must have train, validation, and test images.")
    return source, metadata, rows


class PlaceDataset(Dataset):
    def __init__(self, source, rows, split, training=False, image_size=IMAGE_SIZE, normalize_inputs=True):
        self.source = source
        self.rows = [r for r in rows if r["split"] == split]
        self.transform = image_transform(training, image_size, normalize_inputs)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.source / row["path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        return self.transform(image), row["class_id"]


def classification_metrics(confusion, top5_correct, total, loss_sum):
    confusion = confusion.to(dtype=torch.float64, device="cpu")
    true_positive = confusion.diag()
    support = confusion.sum(1)
    predicted = confusion.sum(0)
    recall = true_positive / support.clamp_min(1)
    precision = true_positive / predicted.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-15)
    present = support > 0
    return {"loss": loss_sum / total, "top1": true_positive.sum().item() / total,
            "top5": top5_correct / total, "macro_recall": recall[present].mean().item(),
            "macro_f1": f1[present].mean().item(), "images": total}


def run_epoch(model, loader, device, num_classes, criterion, optimizer=None, scaler=None, amp=True, frozen=False, freeze_batchnorm=True):
    training = optimizer is not None
    model.train(training)
    if training and frozen:
        # Keep frozen backbone BatchNorm statistics fixed after model.train().
        model.features.eval()
    elif training and freeze_batchnorm:
        for module in model.features.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)
    total = top5_correct = 0
    loss_sum = 0.0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            if not torch.isfinite(loss).item():
                raise ValueError("Non-finite loss. Try --no-amp or a smaller learning rate.")
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            batch_size = labels.size(0)
            total += batch_size
            loss_sum += loss.detach().item() * batch_size
            predictions = outputs.detach().argmax(1)
            top5_correct += (outputs.detach().topk(min(5, num_classes), dim=1).indices == labels[:, None]).any(1).sum().item()
            confusion += torch.bincount(labels * num_classes + predictions, minlength=num_classes**2).reshape(num_classes, num_classes)
    if total == 0:
        raise ValueError("Cannot evaluate an empty split.")
    return classification_metrics(confusion, top5_correct, total, loss_sum), confusion.cpu()


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("architecture") not in {"efficientnet_b0", "legacy_cnn"}:
        raise ValueError("Use a checkpoint produced by tourist_places.train, not the historical CNN.")
    model = build_model(len(checkpoint["classes"]), pretrained=False, architecture=checkpoint["architecture"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint
