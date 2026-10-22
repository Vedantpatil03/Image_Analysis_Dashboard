"""Matplotlib-based image statistics and histogram helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import cv2
import numpy as np

from .image_processing import to_grayscale


def intensity_statistics(image: np.ndarray) -> dict[str, float]:
    """Return min, max, mean, and standard deviation of image intensity."""
    values = to_grayscale(image).astype(np.float32)
    return {
        "Minimum": float(values.min()),
        "Maximum": float(values.max()),
        "Mean": float(values.mean()),
        "Standard deviation": float(values.std()),
    }


def pixel_information(image: np.ndarray) -> dict[str, int]:
    """Return dimensions, channels, and total pixel count."""
    array = np.asarray(image)
    height, width = array.shape[:2]
    channels = 1 if array.ndim == 2 else array.shape[2]
    return {
        "Width": int(width),
        "Height": int(height),
        "Channels": int(channels),
        "Total pixels": int(width * height),
    }


def grayscale_histogram(image: np.ndarray) -> plt.Figure:
    """Create a grayscale intensity-distribution figure."""
    values = to_grayscale(image)
    figure, axis = plt.subplots(figsize=(7, 3.5))
    axis.hist(values.ravel(), bins=256, range=(0, 256), color="#28b7a8")
    axis.set(title="Grayscale Histogram", xlabel="Pixel intensity", ylabel="Frequency")
    figure.tight_layout()
    return figure


def rgb_histogram(image: np.ndarray) -> plt.Figure:
    """Create separate red, green, and blue channel histograms."""
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    figure, axis = plt.subplots(figsize=(7, 3.5))
    for index, color in enumerate(("red", "green", "blue")):
        axis.hist(array[:, :, index].ravel(), bins=256, range=(0, 256), color=color,
                  alpha=0.45, label=color.upper())
    axis.set(title="RGB Histogram", xlabel="Pixel intensity", ylabel="Frequency")
    axis.legend()
    figure.tight_layout()
    return figure


def dominant_color_palette(image: np.ndarray, color_count: int = 5) -> list[dict[str, object]]:
    """Return dominant RGB colors with percentages and hexadecimal values."""
    if color_count < 2 or color_count > 10:
        raise ValueError("Color count must be between 2 and 10.")
    array = np.asarray(image)
    if array.size == 0:
        raise ValueError("An image is required for palette extraction.")
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.shape[2] == 4:
        array = array[:, :, :3]
    pixels = np.ascontiguousarray(array[:, :, :3].reshape(-1, 3).astype(np.float32))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, color_count, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    counts = np.bincount(labels.ravel(), minlength=color_count)
    order = np.argsort(counts)[::-1]
    palette = []
    for index in order:
        red, green, blue = np.clip(np.rint(centers[index]), 0, 255).astype(int)
        percentage = counts[index] / len(pixels) * 100
        palette.append({
            "RGB": (int(red), int(green), int(blue)),
            "HEX": f"#{red:02X}{green:02X}{blue:02X}",
            "Percentage": round(float(percentage), 2),
        })
    return palette


def image_quality_scorecard(image: np.ndarray) -> dict[str, object]:
    """Estimate brightness, contrast, sharpness, noise, and exposure status."""
    grayscale = to_grayscale(image).astype(np.float32)
    brightness = float(grayscale.mean())
    contrast = float(grayscale.std())
    sharpness = float(cv2.Laplacian(grayscale, cv2.CV_32F).var())
    noise = float(np.median(np.abs(grayscale - cv2.GaussianBlur(grayscale, (3, 3), 0))))
    if brightness < 70:
        exposure = "Underexposed"
    elif brightness > 190:
        exposure = "Overexposed"
    else:
        exposure = "Balanced"
    return {
        "Brightness": round(brightness, 2),
        "Contrast": round(contrast, 2),
        "Sharpness": round(sharpness, 2),
        "Noise estimate": round(noise, 2),
        "Exposure": exposure,
    }
