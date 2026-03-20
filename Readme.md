# 🧫 Cell Detection

![Python](https://img.shields.io/badge/Python-3.x-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Detección automática de células en imágenes de sangre usando técnicas de visión por computadora.

--------------------------------------------------

## 📌 Overview

Este proyecto implementa un pipeline clásico de procesamiento de imágenes para detectar células (formas circulares) en imágenes microscópicas.

Se combinan:
- Detección de bordes (Sobel)
- Segmentación (Threshold)
- Limpieza (Morfología)
- Detección de formas (Hough Circles)

--------------------------------------------------

## 🧠 Pipeline

Imagen → Grayscale → Sobel → Magnitud → Threshold → Morphology → Blur → Hough Circles

--------------------------------------------------

## ⚙️ Tecnologías

- Python
- OpenCV
- NumPy
- Matplotlib

--------------------------------------------------

## 🚀 Instalación

git clone https://github.com/AaronHero03/CellDetection.git
cd CellDetection
pip install opencv-python numpy matplotlib

--------------------------------------------------

## ▶️ Uso

python detection.py

Asegúrate de tener una imagen como "blood.jpeg" en el directorio.

--------------------------------------------------

## 📊 Output

El sistema muestra:

- Gradiente en X
- Gradiente en Y
- Magnitud del gradiente
- Imagen binaria (threshold)
- Imagen suavizada
- Detección final de células

Además imprime:

Células detectadas: N

--------------------------------------------------

## 📁 Estructura del proyecto

.
├── detection.py
├── blood.jpeg
├── blood2.png
├── blood3.png
├── blood4.png
└── README.md

--------------------------------------------------

## ⚠️ Limitaciones

- Sensible a iluminación y ruido
- Requiere ajuste manual de parámetros
- Puede fallar con células superpuestas

--------------------------------------------------

## 🔧 Mejoras futuras

- Canny Edge Detection
- Segmentación avanzada (Watershed)
- Ajuste automático de parámetros
- Modelos de Deep Learning

--------------------------------------------------
