# Cell Detection

![Python](https://img.shields.io/badge/Python-3.x-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![NumPy](https://img.shields.io/badge/NumPy-Data%20Processing-orange)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Automatic detection of blood cells in microscopic images using classical computer vision techniques — no deep learning required.

---

## Overview

This project implements a classic image processing pipeline to detect circular cells (red blood cells) in microscopic blood smear images. It combines edge detection, binary segmentation, morphological cleanup, and Hough Circle detection to count and highlight cells in a given image.

The pipeline is designed to be lightweight and interpretable, making it a good starting point for understanding how traditional CV approaches work before moving to learned models.

---

## Pipeline

```
Input Image
    │
    ▼
Grayscale Conversion
    │
    ▼
Sobel Edge Detection (X and Y gradients)
    │
    ▼
Gradient Magnitude Computation
    │
    ▼
Binary Thresholding  (threshold = 50)
    │
    ▼
Morphological Opening  (removes noise)
    │
    ▼
Morphological Closing  (fills gaps)
    │
    ▼
Median Blur  (smooths for circle detection)
    │
    ▼
Hough Circle Transform
    │
    ▼
Annotated Output + Cell Count
```

---

## How It Works

| Step | Technique | Purpose |
|------|-----------|---------|
| Edge detection | Sobel operator (3×3 kernel) | Highlights cell boundaries |
| Magnitude | `cv2.magnitude` → normalized to [0, 255] | Combines X/Y gradients into a single map |
| Segmentation | Binary threshold at value 50 | Isolates strong edges |
| Noise removal | Morphological opening (ellipse kernel 3×3, 2 iterations) | Removes small artifacts |
| Gap filling | Morphological closing (ellipse kernel 3×3, 2 iterations) | Closes broken cell outlines |
| Smoothing | Median blur (kernel size 5) | Reduces noise before circle fitting |
| Detection | Hough Gradient (`dp=1.2`, `minDist=25`, `param1=50`, `param2=21`, radius 20–55 px) | Fits circles to cell shapes |

Detected cells are drawn with a red circle outline and a green center dot.

---

## Output

The script displays a 2×3 grid of intermediate results:

| Panel | Content |
|-------|---------|
| (0, 0) | X gradient |
| (0, 1) | Y gradient |
| (0, 2) | Gradient magnitude |
| (1, 0) | Binary threshold mask |
| (1, 1) | Blurred image |
| (1, 2) | Final detection with annotated circles |

The total cell count is printed to the console:

```
Cells detected: N
```

---

## Technologies

- **Python 3.x**
- **OpenCV** — image processing and Hough Circle detection
- **NumPy** — array operations and normalization
- **Matplotlib** — visualization grid

---

## Installation

```bash
git clone https://github.com/CarlosGl05/Detecci-n-de-c-lulas
cd Detecci-n-de-c-lulas
pip install opencv-python numpy matplotlib
```

---

## Usage

Place a blood smear image named `blood.jpeg` in the project directory, then run:

```bash
python detection.py
```

To use a different image, update the filename on line 4 of `detection.py`:

```python
img = cv2.imread("your_image.jpeg")
```

---

## Project Structure

```
.
├── detection.py      # Main detection script
├── blood.jpeg        # Default input image
├── blood2.png
├── blood3.png
├── blood4.png
└── Readme.md
```

---

## Limitations

- Sensitive to lighting variation and image noise — results may degrade with low-quality scans
- Hough parameters (`dp`, `minDist`, `param1`, `param2`, radius bounds) require manual tuning per image
- Overlapping or clustered cells may be missed or counted incorrectly
- Only works reliably on images where cells appear as well-defined circular shapes

---

## Potential Improvements

- **Canny edge detection** as an alternative to Sobel for sharper, thinner edges
- **Watershed segmentation** to separate touching cells
- **Automatic parameter tuning** using calibration images or optimization
- **Deep learning models** (e.g., U-Net, Mask R-CNN) for robust, training-based detection
- **Batch processing** to run detection across multiple images at once
