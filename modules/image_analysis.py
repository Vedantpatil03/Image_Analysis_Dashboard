"""Matplotlib-based image statistics and histogram helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
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
