"""Stable presentation of the four pilot variants and their recommended alias."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path


# This is the recorded validation order, never an ordering by test accuracy.
MODEL_ORDER = ("stable_transfer_pilot", "frozen_pilot", "transfer_pilot", "legacy_pilot")


@lru_cache(maxsize=32)
def _file_digest(path, size, modified):
    # Size/mtime are cache keys so a replaced checkpoint is checked again.
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_digest(path):
    path = Path(path)
    stat = path.stat()
    return _file_digest(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def model_run_name(path):
    """Resolve only a verified recommended copy; never hide a different model."""
    path = Path(path)
    if path.parent.name != "recommended":
        return path.parent.name
    try:
        selection = json.loads((path.parent / "selection.json").read_text(encoding="utf-8"))
        source = selection.get("source_run")
        if source not in MODEL_ORDER or selection.get("checkpoint_sha256") != checkpoint_digest(path):
            return "recommended"
        original = path.parent.parent / source / "best.pt"
        if original.is_file() and checkpoint_digest(original) != checkpoint_digest(path):
            return "recommended"
        return source
    except (OSError, ValueError, TypeError, AttributeError):
        return "recommended"


def ordered_checkpoints(root):
    paths = [path for path in Path(root).glob("*/best.pt") if "smoke" not in path.parent.name.lower()]
    recommended = next((path for path in paths if path.parent.name == "recommended"), None)
    if recommended:
        source = model_run_name(recommended)
        if source != "recommended":
            # Keep the selected copy, but offer its experiment just once.
            paths = [path for path in paths if path == recommended or path.parent.name != source]

    def order(path):
        name = model_run_name(path)
        rank = MODEL_ORDER.index(name) + 1 if name in MODEL_ORDER else 0 if name == "recommended" else 100
        return rank, name.casefold(), path.parent.name.casefold()

    return sorted(paths, key=order)
