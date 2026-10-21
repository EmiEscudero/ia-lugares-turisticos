"""A small bilingual, memory-only Streamlit upload component."""
import base64
import binascii
from pathlib import Path

import streamlit.components.v1 as components

from .i18n import text
from .inference import MAX_UPLOAD_BYTES, UploadError

_component = components.declare_component("travel_photo", path=str(Path(__file__).parent / "upload_component"))


def travel_photo_uploader(language):
    labels = {key: text(key, language) for key in (
        "drop_photo", "choose_photo", "remove_photo", "upload_help", "uploading",
        "upload_size", "upload_format", "upload_unreadable",
    )}
    return _component(language=language, labels=labels, max_bytes=MAX_UPLOAD_BYTES, key="travel_photo", default=None)


def uploaded_bytes(value):
    """Validate the transport again on the server; never trust browser limits."""
    if not isinstance(value, dict) or value.get("error"):
        code = value.get("error") if isinstance(value, dict) else None
        raise UploadError(code if code in {"upload_size", "upload_format", "upload_unreadable"} else "upload_unreadable", "Invalid photo upload.")
    content = value.get("content")
    if not isinstance(content, str):
        raise UploadError("upload_unreadable", "Invalid photo upload.")
    if len(content) > 4 * ((MAX_UPLOAD_BYTES + 2) // 3):
        raise UploadError("upload_size", "Choose a photo no larger than 20 MB.")
    try:
        data = base64.b64decode(content, validate=True)
    except (ValueError, binascii.Error) as error:
        raise UploadError("upload_unreadable", "Invalid photo upload.") from error
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise UploadError("upload_size", "Choose a photo no larger than 20 MB.")
    return data
