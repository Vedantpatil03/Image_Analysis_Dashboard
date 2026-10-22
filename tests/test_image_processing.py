import numpy as np
import pytest

from modules.image_analysis import (
    dominant_color_palette,
    image_quality_scorecard,
    intensity_statistics,
    pixel_information,
)
from modules.image_processing import (
    adjust_brightness_contrast,
    binary_threshold,
    canny_edges,
    gaussian_blur,
    median_blur,
    sharpen,
    to_grayscale,
)
from modules.image_utils import comparison_preview


@pytest.fixture
def image():
    return np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)


def test_grayscale_shape_and_input_preservation(image):
    original = image.copy()
    result = to_grayscale(image)
    assert result.shape == (8, 8)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("operation", [gaussian_blur, median_blur, sharpen])
def test_filters_preserve_shape_and_dtype(image, operation):
    result = operation(image, 3) if operation != sharpen else operation(image)
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_canny_returns_binary_edges(image):
    result = canny_edges(image, 10, 100)
    assert result.shape == image.shape[:2]
    assert set(np.unique(result)).issubset({0, 255})


def test_binary_threshold_returns_binary_image(image):
    result = binary_threshold(image, 120)
    assert set(np.unique(result)).issubset({0, 255})


def test_brightness_and_contrast_are_clipped(image):
    result = adjust_brightness_contrast(image, brightness=100, contrast=2)
    assert result.max() == 255
    assert result.dtype == np.uint8


def test_analysis_statistics_and_pixel_information(image):
    statistics = intensity_statistics(image)
    information = pixel_information(image)
    assert statistics["Minimum"] >= 0
    assert statistics["Maximum"] <= 255
    assert information == {"Width": 8, "Height": 8, "Channels": 3, "Total pixels": 64}


def test_invalid_input_is_rejected():
    with pytest.raises(ValueError):
        to_grayscale(np.array([]))
    with pytest.raises(ValueError):
        canny_edges(np.ones((2, 2), dtype=np.uint8), 200, 100)


def test_palette_returns_sorted_colors_and_percentages():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:7] = (255, 0, 0)
    image[7:] = (0, 0, 255)
    palette = dominant_color_palette(image, 2)
    assert palette[0]["RGB"] in {(255, 0, 0), (0, 0, 255)}
    assert sum(item["Percentage"] for item in palette) == pytest.approx(100, abs=0.1)


def test_quality_scorecard_identifies_exposure_and_metrics():
    dark = np.zeros((8, 8, 3), dtype=np.uint8)
    quality = image_quality_scorecard(dark)
    assert quality["Exposure"] == "Underexposed"
    assert quality["Brightness"] == 0.0
    assert quality["Contrast"] == 0.0


def test_comparison_preview_has_original_and_processed_halves(image):
    processed = np.full_like(image, 255)
    preview = comparison_preview(image, processed, 50)
    assert preview.shape == image.shape
    np.testing.assert_array_equal(preview[:, 0], image[:, 0])
    np.testing.assert_array_equal(preview[:, -1], processed[:, -1])
