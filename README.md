# PixelLab: Interactive Image Processing and Analysis Dashboard

## Abstract

PixelLab is a local Streamlit dashboard for exploring fundamental digital image
processing. Users upload JPG, JPEG, or PNG images, apply OpenCV operations, compare
the original and processed results, inspect Matplotlib histograms, and download a
processed PNG.

## Problem statement and objectives

Manual image-processing experiments often require separate scripts and do not make
the effect of parameters easy to compare. This project provides an approachable
visual interface for learning grayscale conversion, smoothing, edges, thresholding,
enhancement, and Haar-cascade face detection.

## Technology stack

- **Streamlit**: interactive layout, upload control, sidebar widgets, and downloads.
- **OpenCV**: conversion, filters, thresholds, Canny edges, and Haar-cascade detection.
- **NumPy**: safe pixel-array and numerical operations.
- **Pillow**: reliable uploaded-image decoding.
- **Matplotlib**: grayscale and RGB histogram visualization.

## Project structure

```text
app.py
modules/
  image_utils.py
  image_processing.py
  image_analysis.py
tests/test_image_processing.py
requirements.txt
```

## Requirements and installation (Windows / VS Code)

Install Python 3.10 or newer, open this folder in VS Code, and run PowerShell:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use `venv\Scripts\python.exe` directly.

## Run the application

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL shown by Streamlit, upload an image, choose an operation, and
change its controls. Download uses the current processed result.

## Workflow

1. Upload a supported image (maximum 10 MB).
2. Review dimensions and the original preview.
3. Select an operation and adjust its parameters in the sidebar.
4. Compare the output with the original.
5. Inspect histograms and intensity statistics.
6. Download the processed PNG.

## Testing

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

Tests use small synthetic arrays and cover conversion, filters, Canny, thresholding,
enhancement, statistics, validation, and original-image preservation.

## Known limitations and future enhancements

Face detection depends on OpenCV's bundled Haar Cascade and may miss faces at unusual
angles or lighting. It detects face-like regions but does not recognize identities.
Future work could add batch processing, configurable export formats, more detectors,
and an optional side-by-side slider.

## Viva notes and conclusion

Explain that RGB stores red, green, and blue channels while OpenCV commonly uses BGR
internally; this app keeps uploaded data in RGB and converts only at boundaries.
Grayscale stores intensity, blur reduces noise, Canny finds gradients, thresholding
separates intensity classes, and a histogram counts intensity values rather than
judging image quality. The project demonstrates these ideas without a database,
cloud service, or machine-learning training requirement.
