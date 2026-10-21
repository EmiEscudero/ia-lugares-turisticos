"""Prediction helpers shared by the local interface and command line."""
import io

import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from .learning import image_transform
from .model_catalog import ordered_checkpoints

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_UPLOAD_PIXELS = 30_000_000


class UploadError(ValueError):
    """A stable code lets interfaces translate errors without parsing English."""

    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def decode_upload(data):
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise UploadError("upload_size", "Choose a photo no larger than 20 MB.")
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.format not in {"JPEG", "PNG", "WEBP"}:
                raise UploadError("upload_format", "Choose a JPEG, PNG, or WebP photo.")
            if source.width * source.height > MAX_UPLOAD_PIXELS:
                raise UploadError("upload_pixels", "Choose a photo with no more than 30 million pixels.")
            return ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise UploadError("upload_unreadable", "This file could not be read as a photo. Try another image.") from error


def predict_candidates(model, classes, image, device, top_k=5):
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    batch = image_transform(image_size=getattr(model, "image_size", 224), normalize_inputs=getattr(model, "normalize_inputs", True))(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(batch).float().softmax(dim=1)[0]
    values, indices = probabilities.topk(min(top_k, len(classes)))
    return [{**classes[index], "score": score}
            for score, index in zip(values.cpu().tolist(), indices.cpu().tolist())]


def available_checkpoints(root):
    return ordered_checkpoints(root)
