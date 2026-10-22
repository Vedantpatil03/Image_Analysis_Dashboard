# 🖼️ Interactive Image Processing and Analysis Dashboard Using Python

An interactive web-based application for image processing, computer vision, and graphical analysis, built using Python, Streamlit, OpenCV, and Matplotlib.

## 📌 Project Overview

The **Interactive Image Processing and Analysis Dashboard** allows users to upload images, apply different image-processing techniques, compare original and processed images, and visualize image characteristics through graphical analysis.

The application provides a simple, user-friendly dashboard that brings multiple image-processing and visualization operations together in one place.

This project is developed as a **Python for Interactive Media and Visualization (PIMV) mini-project**.

## 🎯 Objectives

- Develop an interactive image-processing dashboard using Python.
- Apply image-processing techniques using OpenCV.
- Visualize pixel-intensity distributions using Matplotlib.
- Provide an easy-to-use interface with Streamlit.
- Enable users to compare original and processed images.
- Display useful image statistics and graphical visualizations.
- Allow users to download processed images.

## ✨ Features

- 📤 **Image Upload:** Upload images in JPG, JPEG, and PNG formats.
- 🖼️ **Image Preview:** Display the original image and its properties.
- ⚫ **Grayscale Conversion:** Convert color images into grayscale.
- 🌫️ **Gaussian Blur:** Reduce image noise and smooth image details.
- 🔵 **Median Blur:** Apply median filtering to reduce certain types of noise.
- 🔍 **Edge Detection:** Identify edges using the Canny algorithm.
- 🎚️ **Thresholding:** Apply binary and adaptive thresholding techniques.
- ☀️ **Brightness and Contrast:** Adjust image brightness and contrast.
- ✨ **Image Sharpening:** Enhance visible edges and details.
- 👤 **Face Detection:** Detect faces using OpenCV Haar Cascade when the required classifier is available.
- 📊 **Histogram Visualization:** Analyze grayscale intensity and RGB channel distributions.
- 📈 **Image Statistics:** View dimensions, pixel count, and intensity statistics.
- ↔️ **Image Comparison:** Compare the original and processed images.
- 🎨 **Color Palette Extraction:** Identify dominant colors with HEX, RGB, and percentage values.
- ✅ **Image Quality Scorecard:** Review educational brightness, contrast, sharpness, noise, and exposure indicators.
- 💾 **Download Results:** Download processed images in PNG format.
- 🖥️ **Interactive Dashboard:** Control image-processing operations through an easy-to-use web interface.

*Note: Feature availability depends on the implemented application version. Face detection requires a compatible Haar Cascade classifier.*

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web interface and dashboard |
| OpenCV | Image processing and computer vision |
| Matplotlib | Histograms and graphical visualization |
| NumPy | Numerical operations and pixel-array processing |
| Pillow (PIL) | Image loading, conversion, and handling |
| Pytest | Automated testing of core functionality |

## 🏗️ System Architecture

```text
             ┌──────────────────────┐
             │        User          │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │  Streamlit Dashboard │
             │  Upload and Controls │
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │     Image Utils      │
             │ Validation/Conversion│
             └──────────┬───────────┘
                        │
                        ▼
             ┌──────────────────────┐
             │     OpenCV Engine    │
             │  Image Processing    │
             └──────────┬───────────┘
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
   ┌──────────────────┐  ┌──────────────────┐
   │ Matplotlib       │  │ Processed Image  │
   │ Graphical        │  │ Preview/Download │
   │ Analysis         │  │                  │
   └────────┬─────────┘  └────────┬─────────┘
            │                     │
            └──────────┬──────────┘
                       ▼
             ┌──────────────────────┐
             │ Display Results in   │
             │ Interactive Dashboard│
             └──────────────────────┘
```

## 🔄 Application Workflow

1. Launch the application.
2. Upload an image using the dashboard.
3. View the original image and its properties.
4. Select an image-processing operation.
5. Adjust the available parameters.
6. View the processed image.
7. Analyze image characteristics using histograms and statistics.
8. Compare the original and processed results.
9. Download the processed image.
10. Review the dominant color palette and image-quality scorecard.

## New analysis features

- **Color palette extraction:** OpenCV K-means clustering groups pixels into the
  selected number of dominant colors and displays each color's HEX value, RGB
  value, and image percentage.
- **Image quality scorecard:** Displays pixel-derived brightness, contrast,
  Laplacian sharpness, a small-noise estimate, and a simple exposure status.
  These values are educational indicators rather than a universal quality grade.

## 📂 Project Structure

```text
ImageProcessingDashboard/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── modules/
│   ├── __init__.py
│   ├── image_processing.py
│   ├── image_analysis.py
│   ├── image_utils.py
│
├── assets/
│   └── README.md
│
└── tests/
    ├── __init__.py
    └── test_image_processing.py
```

*The structure above represents the planned organization. Update it if the actual implementation differs.*

## ⚙️ Installation and Setup

### Prerequisites

Install the following before running the project:

- Python 3.10 or a compatible newer version supported by the selected dependencies.
- Visual Studio Code.
- Python extension for VS Code.
- Git (optional, for version control).

### Step 1: Clone the Repository

Replace `YOUR_USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR_USERNAME/ImageProcessingDashboard.git
```

Navigate to the project directory:

```bash
cd ImageProcessingDashboard
```

Alternatively, download the repository ZIP file and extract it.

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 3: Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```cmd
venv\Scripts\activate.bat
```

If PowerShell blocks activation, use Command Prompt or configure your environment according to your system's security policies.

### Step 4: Install Dependencies

```bash
python -m pip install --upgrade pip
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

If `requirements
