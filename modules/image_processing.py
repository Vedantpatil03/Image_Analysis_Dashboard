"""OpenCV processing operations used by the dashboard."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _require_image(image: np.ndarray) -> np.ndarray:
    result = np.asarray(image)
    if result.size == 0 or result.ndim not in {2, 3}:
        raise ValueError("An image array is required.")
    if result.dtype != np.uint8:
        result = np.clip(result, 0, 255).astype(np.uint8)
    return result.copy()


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB/BGR or grayscale input to a grayscale uint8 copy."""
    result = _require_image(image)
    if result.ndim == 2:
        return result
    if result.shape[2] == 4:
        return cv2.cvtColor(result, cv2.COLOR_RGBA2GRAY)
    return cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)


def gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply Gaussian smoothing with an odd positive kernel."""
    result = _require_image(image)
    kernel_size = _odd_kernel(kernel_size)
    return cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)


def median_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply median smoothing with an odd positive kernel."""
    result = _require_image(image)
    kernel_size = _odd_kernel(kernel_size)
    return cv2.medianBlur(result, kernel_size)


def canny_edges(image: np.ndarray, lower: int = 100, upper: int = 200) -> np.ndarray:
    """Detect edges using Canny with validated thresholds."""
    if not 0 <= lower <= 255 or not 0 <= upper <= 255 or lower >= upper:
        raise ValueError("Canny thresholds must be 0-255 and lower must be below upper.")
    return cv2.Canny(to_grayscale(image), lower, upper)


def binary_threshold(image: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Create a binary image using a fixed intensity threshold."""
    if not 0 <= threshold <= 255:
        raise ValueError("Threshold must be between 0 and 255.")
    _, result = cv2.threshold(to_grayscale(image), threshold, 255, cv2.THRESH_BINARY)
    return result


def adaptive_threshold(
    image: np.ndarray, block_size: int = 11, constant: int = 2
) -> np.ndarray:
    """Create a binary image using adaptive Gaussian thresholding."""
    block_size = _odd_kernel(block_size)
    if not -50 <= constant <= 50:
        raise ValueError("Adaptive threshold constant must be between -50 and 50.")
    return cv2.adaptiveThreshold(
        to_grayscale(image),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        constant,
    )


def adjust_brightness_contrast(
    image: np.ndarray, brightness: int = 0, contrast: float = 1.0
) -> np.ndarray:
    """Adjust brightness in [-100, 100] and contrast in [0, 3]."""
    if not -100 <= brightness <= 100 or not 0 <= contrast <= 3:
        raise ValueError("Brightness or contrast is outside the supported range.")
    result = _require_image(image).astype(np.float32) * contrast + brightness
    return np.clip(result, 0, 255).astype(np.uint8)


def sharpen(image: np.ndarray) -> np.ndarray:
    """Enhance edges using a standard sharpening kernel."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(_require_image(image), -1, kernel)


def detect_faces(image: np.ndarray) -> tuple[np.ndarray, int, str | None]:
    """Draw Haar-cascade face boxes and return (image, count, error message)."""
    result = _require_image(image)
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        return result, 0, "OpenCV's Haar Cascade file is unavailable."
    faces = cascade.detectMultiScale(to_grayscale(result), 1.1, 5, minSize=(30, 30))
    output = result.copy()
    if output.ndim == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
    for x, y, width, height in faces:
        cv2.rectangle(output, (x, y), (x + width, y + height), (0, 220, 120), 2)
    return output, len(faces), None


def _odd_kernel(value: int) -> int:
    value = int(value)
    if value < 3 or value > 31 or value % 2 == 0:
        raise ValueError("Kernel size must be an odd number from 3 to 31.")
    return value
