"""Image loading, validation, conversion, and download helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_pil_image(file: BinaryIO, filename: str = "") -> Image.Image:
    """Validate and load an uploaded image as an RGB/RGBA Pillow image."""
    size = _file_size(file)
    if size is not None and size > MAX_UPLOAD_BYTES:
        raise ValueError("The image is larger than the 10 MB upload limit.")
    suffix = Path(filename).suffix.lower()
    if suffix and suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported format. Please upload a JPG, JPEG, or PNG image.")
    try:
        image = Image.open(file)
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc
    if image.mode not in {"L", "RGB", "RGBA"}:
        image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
    return image.copy()


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    """Return a Pillow image as an independent RGB uint8 NumPy array."""
    if image.mode == "L":
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def rgb_to_display_array(array: np.ndarray) -> np.ndarray:
    """Normalize a grayscale/RGB/RGBA result for Streamlit display."""
    result = _validate_array(array)
    if result.ndim == 2:
        return result
    if result.shape[2] == 4:
        return result[:, :, :3]
    return result


def encode_png(array: np.ndarray) -> bytes:
    """Encode an RGB, RGBA, or grayscale array as PNG bytes."""
    result = _validate_array(array)
    if result.ndim == 3 and result.shape[2] == 3:
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    elif result.ndim == 3 and result.shape[2] == 4:
        result = cv2.cvtColor(result, cv2.COLOR_RGBA2BGRA)
    success, encoded = cv2.imencode(".png", result)
    if not success:
        raise ValueError("The processed image could not be encoded as PNG.")
    return encoded.tobytes()


def _file_size(file: BinaryIO) -> int | None:
    try:
        position = file.tell()
        file.seek(0, 2)
        size = file.tell()
        file.seek(position)
        return size
    except (AttributeError, OSError):
        return None


def _validate_array(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array)
    if result.size == 0 or result.ndim not in {2, 3}:
        raise ValueError("Image data must be a non-empty 2D or 3D array.")
    if result.ndim == 3 and result.shape[2] not in {1, 3, 4}:
        raise ValueError("Image data must have 1, 3, or 4 channels.")
    if result.dtype != np.uint8:
        result = np.clip(result, 0, 255).astype(np.uint8)
    if result.ndim == 3 and result.shape[2] == 1:
        result = result[:, :, 0]
    return result

