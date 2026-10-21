"""Fetch only the four public, integrity-checked inference checkpoints."""
import hashlib
import json
import re
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

from .model_catalog import MODEL_ORDER

RELEASE_BASE = "https://github.com/EmiEscudero/ia-lugares-turisticos/releases/download"
_DOWNLOAD_LOCK = threading.Lock()
_CHUNK_BYTES = 1024 * 1024


def _reject_constant(value):
    raise ValueError("The model manifest must contain finite JSON values.")


def _validated_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_reject_constant)
    if not isinstance(manifest, dict):
        raise ValueError("The public model manifest must be an object.")
    release = manifest.get("release")
    if not isinstance(release, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", release):
        raise ValueError("The public model release tag is invalid.")
    models = manifest.get("models")
    if not isinstance(models, list) or len(models) != len(MODEL_ORDER):
        raise ValueError("The public manifest must contain all four pilot models.")
    seen = set()
    validated = []
    for item in models:
        if not isinstance(item, dict):
            raise ValueError("Each public model must be an object.")
        run = item.get("run")
        if not isinstance(run, str) or run not in MODEL_ORDER or run in seen:
            raise ValueError("The public model names must be the four unique pilot names.")
        seen.add(run)
        expected_url = f"{RELEASE_BASE}/{release}/{run}.pt"
        if item.get("url") != expected_url:
            raise ValueError("Public models must use their exact GitHub release asset URL.")
        digest = item.get("sha256")
        size = item.get("bytes")
        metrics = item.get("test_metrics")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError("Each public model requires a SHA256 digest.")
        if type(size) is not int or size <= 0:
            raise ValueError("Each public model requires a positive byte count.")
        if not isinstance(metrics, dict):
            raise ValueError("Each public model requires its recorded test metrics.")
        validated.append({**item, "sha256": digest.lower()})
    return sorted(validated, key=lambda item: MODEL_ORDER.index(item["run"]))


def _model_paths(root, run):
    directory = root / run
    if directory.is_symlink() or not directory.resolve().is_relative_to(root):
        raise ValueError("A hosted model directory must stay inside the model cache.")
    checkpoint = directory / "best.pt"
    metrics = directory / "test_metrics.json"
    if checkpoint.is_symlink() or metrics.is_symlink():
        raise ValueError("Hosted model cache files cannot be symbolic links.")
    return directory, checkpoint, metrics


def _matches(path, expected_size, expected_digest):
    if not path.is_file() or path.stat().st_size != expected_size:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest() == expected_digest


def _download(item, directory, checkpoint):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=directory, prefix=".best-", suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            digest = hashlib.sha256()
            received = 0
            request = urllib.request.Request(item["url"], headers={"User-Agent": "mexico-tourist-place-demo/1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                while True:
                    # The extra byte detects oversized responses without buffering them.
                    block = response.read(min(_CHUNK_BYTES, item["bytes"] - received + 1))
                    if not block:
                        break
                    received += len(block)
                    if received > item["bytes"]:
                        raise RuntimeError("The downloaded model exceeded its recorded size.")
                    output.write(block)
                    digest.update(block)
            if received != item["bytes"] or digest.hexdigest() != item["sha256"]:
                raise RuntimeError("The downloaded model failed its size or SHA256 verification.")
        temporary.replace(checkpoint)
    except (OSError, urllib.error.URLError) as error:
        raise RuntimeError("The public inference model could not be downloaded.") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_metrics(path, metrics):
    content = json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".metrics-", suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            output.write(content)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ensure_hosted_models(manifest_path, model_root):
    """Return a verified cache of the four public models, without any source photos.

    Validate the complete manifest before creating files or contacting the network.
    A process-wide lock serializes simultaneous first visits. Completed downloads
    survive restarts; invalid or incomplete downloads never replace a checkpoint.
    """
    models = _validated_manifest(manifest_path)
    root = Path(model_root).resolve()
    with _DOWNLOAD_LOCK:
        paths = [_model_paths(root, item["run"]) for item in models]
        for item, (directory, checkpoint, metrics) in zip(models, paths):
            directory.mkdir(parents=True, exist_ok=True)
            if not _matches(checkpoint, item["bytes"], item["sha256"]):
                _download(item, directory, checkpoint)
            _write_metrics(metrics, item["test_metrics"])
    return root
