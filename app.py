"""Interactive Image Processing and Analysis Dashboard."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from modules.image_analysis import (
    grayscale_histogram,
    intensity_statistics,
    pixel_information,
    rgb_histogram,
)
from modules.image_processing import (
    adaptive_threshold,
    adjust_brightness_contrast,
    binary_threshold,
    canny_edges,
    detect_faces,
    gaussian_blur,
    median_blur,
    sharpen,
    to_grayscale,
)
from modules.image_utils import encode_png, load_pil_image, pil_to_rgb_array, rgb_to_display_array

st.set_page_config(page_title="PixelLab Dashboard", page_icon="🖼️", layout="wide")

OPERATIONS = {
    "Original Image": "View the unmodified uploaded image.",
    "Grayscale Conversion": "Convert color information into a single intensity channel.",
    "Gaussian Blur": "Smooth the image using a weighted neighborhood average.",
    "Median Blur": "Reduce impulse noise while preserving edges.",
    "Canny Edge Detection": "Highlight strong intensity transitions as edges.",
    "Binary Thresholding": "Separate pixels into black and white using one threshold.",
    "Adaptive Thresholding": "Choose a local threshold for changing lighting conditions.",
    "Brightness & Contrast": "Adjust image intensity and dynamic range.",
    "Sharpening": "Emphasize edges to make details appear crisper.",
    "Face Detection": "Locate faces with OpenCV Haar Cascade; this is not recognition.",
}


def main() -> None:
    _apply_style()
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    st.title("🖼️ PixelLab")
    st.caption("Interactive Image Processing and Analysis Dashboard")
    st.write(
        "Upload an image, explore classic computer-vision operations, compare results, "
        "and understand its pixel-intensity distribution."
    )
    with st.sidebar:
        st.header("Controls")
        if st.button("Clear image", use_container_width=True):
            st.session_state.uploader_key += 1
            st.rerun()
        uploaded = st.file_uploader(
            "Upload JPG, JPEG, or PNG",
            type=["jpg", "jpeg", "png"],
            key=f"image_uploader_{st.session_state.uploader_key}",
        )
        operation = st.selectbox("Processing operation", list(OPERATIONS))
        st.caption(OPERATIONS[operation])
        parameters = _parameters(operation)

    if uploaded is None:
        _empty_state()
        return
    try:
        image = load_pil_image(uploaded, uploaded.name)
        original = pil_to_rgb_array(image)
    except ValueError as exc:
        st.error(str(exc))
        return

    processed, notice = _process(original, operation, parameters)
    if notice:
        st.info(notice)
    _show_comparison(original, processed, operation, uploaded.name)
    _show_analysis(original, processed)


def _parameters(operation: str) -> dict:
    if operation in {"Gaussian Blur", "Median Blur"}:
        return {"kernel": st.slider("Kernel size", 3, 15, 5, step=2)}
    if operation == "Canny Edge Detection":
        lower = st.slider("Lower threshold", 0, 255, 100)
        upper = st.slider("Upper threshold", lower + 1, 255, min(200, lower + 1))
        return {"lower": lower, "upper": upper}
    if operation == "Binary Thresholding":
        return {"threshold": st.slider("Threshold", 0, 255, 127)}
    if operation == "Adaptive Thresholding":
        return {
            "block_size": st.slider("Block size", 3, 31, 11, step=2),
            "constant": st.slider("Constant (C)", -20, 20, 2),
        }
    if operation == "Brightness & Contrast":
        return {
            "brightness": st.slider("Brightness", -100, 100, 0),
            "contrast": st.slider("Contrast", 0.0, 3.0, 1.0, 0.1),
        }
    return {}


def _process(image, operation: str, parameters: dict):
    if operation == "Original Image":
        return image.copy(), None
    functions = {
        "Grayscale Conversion": lambda: to_grayscale(image),
        "Gaussian Blur": lambda: gaussian_blur(image, parameters["kernel"]),
        "Median Blur": lambda: median_blur(image, parameters["kernel"]),
        "Canny Edge Detection": lambda: canny_edges(image, parameters["lower"], parameters["upper"]),
        "Binary Thresholding": lambda: binary_threshold(image, parameters["threshold"]),
        "Adaptive Thresholding": lambda: adaptive_threshold(
            image, parameters["block_size"], parameters["constant"]
        ),
        "Brightness & Contrast": lambda: adjust_brightness_contrast(
            image, parameters["brightness"], parameters["contrast"]
        ),
        "Sharpening": lambda: sharpen(image),
    }
    if operation == "Face Detection":
        result, count, error = detect_faces(image)
        return result, error or f"{count} face(s) detected. Face detection does not identify people."
    try:
        return functions[operation](), None
    except ValueError as exc:
        return image.copy(), str(exc)


def _show_comparison(original, processed, operation: str, filename: str) -> None:
    st.subheader("Image comparison")
    left, right = st.columns(2)
    with left:
        st.markdown("**Original**")
        st.image(original, use_container_width=True)
        st.caption(f"{filename} · {original.shape[1]} × {original.shape[0]}")
    with right:
        st.markdown(f"**Processed — {operation}**")
        st.image(rgb_to_display_array(processed), use_container_width=True)
        st.caption(f"{processed.shape[1]} × {processed.shape[0]}")
        st.download_button(
            "Download processed PNG",
            encode_png(processed),
            file_name=f"{Path(filename).stem}_{operation.lower().replace(' ', '_')}.png",
            mime="image/png",
        )


def _show_analysis(original, processed) -> None:
    st.subheader("Image analysis")
    info = pixel_information(original)
    stats = intensity_statistics(processed)
    cards = st.columns(4)
    for card, (label, value) in zip(cards, info.items()):
        card.metric(label, value)
    st.caption("A histogram shows the distribution of pixel intensities; it is not a measure of image quality.")
    chart_left, chart_right = st.columns(2)
    with chart_left:
        figure = grayscale_histogram(processed)
        st.pyplot(figure, clear_figure=True)
        plt.close(figure)
    with chart_right:
        figure = rgb_histogram(original)
        st.pyplot(figure, clear_figure=True)
        plt.close(figure)
    st.write("Processed intensity statistics")
    st.json({key: round(value, 2) for key, value in stats.items()})


def _empty_state() -> None:
    st.info("Upload an image from the sidebar to begin.")
    columns = st.columns(3)
    for column, title, text in zip(
        columns,
        ("Process", "Compare", "Understand"),
        ("Apply filters, thresholds, and edge detection.", "See original and processed output side by side.", "Use histograms and statistics to inspect pixels."),
    ):
        column.markdown(f"### {title}")
        column.write(text)


def _apply_style() -> None:
    st.markdown(
        """<style>
        .stApp {
            background: #f5f8fa;
            color: #172033;
        }
        [data-testid="stAppViewContainer"] .main,
        [data-testid="stAppViewContainer"] .main h1,
        [data-testid="stAppViewContainer"] .main h2,
        [data-testid="stAppViewContainer"] .main h3,
        [data-testid="stAppViewContainer"] .main p,
        [data-testid="stAppViewContainer"] .main label,
        [data-testid="stAppViewContainer"] .main [data-testid="stCaptionContainer"] {
            color: #172033;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            color: #172033;
            padding: 12px;
            border-radius: 10px;
        }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"] {
            color: #172033 !important;
        }
        [data-testid="stSidebar"] {
            background: #242733;
        }
        [data-testid="stSidebar"] * {
            color: #f8fafc;
        }
        [data-testid="stSidebar"] [data-baseweb="select"],
        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: #ffffff;
            color: #172033;
        }
        [data-testid="stSidebar"] [data-baseweb="select"] *,
        [data-testid="stSidebar"] [role="option"] {
            color: #172033 !important;
        }
        [data-testid="stDownloadButton"] button,
        .stDownloadButton button {
            background: #ffffff;
            border: 1px solid #9aa5b1;
            color: #172033 !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
