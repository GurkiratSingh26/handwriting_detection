# Handwritten Medical Prescription Recognition

Applying Visual Transformers (ViT) to extract and recognize handwritten text from medical prescriptions — reducing transcription errors and improving patient safety through machine learning.

## Overview

Misread handwritten prescriptions are a leading source of medication errors in healthcare. This project tackles that problem using a Vision Transformer (ViT) model to detect and classify handwritten medicine names from prescription images converting unstructured handwritten data into structured, machine-readable text.

Unlike traditional CNN-based OCR approaches, this project leverages HuggingFace's transformer architecture applied to image data, demonstrating how attention mechanisms originally designed for NLP can be adapted for complex computer vision tasks.

## Key Features

- Vision Transformer (ViT) architecture for image-based text recognition, via HuggingFace transformers
- Structured Train / Test / Validation data pipeline with labeled CSVs mapping images to medicine names
- Image preprocessing pipeline using TensorFlow/Keras — resizing, normalization, and array conversion
- Data management with pandas and NumPy across 4,000+ labeled prescription word images
- Domain application in healthcare data — a high-stakes, real-world use case for ML

## Dataset

The dataset consists of labeled images of individual handwritten words extracted from medical prescriptions, organized across three splits:

```text
dataset/
├── Training/
│   ├── training_labels.csv       # Maps image index → MEDICINE_NAME
│   └── training_words/           # ~3,000 labeled word images
├── Testing/
│   ├── testing_labels.csv
│   └── testing_words/            # ~780 labeled word images
└── Validation/
    ├── validation_labels.csv
    └── validation_words/         # ~660 labeled word images
```

Each image is a cropped word-level region from a handwritten prescription, labeled with the corresponding medicine name. Images are resized to 64×64 pixels and normalized to [0, 1] before model input.

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| TensorFlow / Keras | Image loading, preprocessing, model training |
| HuggingFace Transformers | Vision Transformer (ViT) architecture |
| pandas | Label ingestion and CSV management |
| NumPy | Array operations and dataset construction |
| OpenCV | Image processing utilities |

## Data Pipeline

The `data.py` script handles the full preprocessing workflow:

- Label ingestion — reads MEDICINE_NAME column from training, testing, and validation CSVs using pandas  
- Image loading — iterates over image directories, loading each file with Keras load_img  
- Preprocessing — resizes all images to 64×64, converts to float32 arrays, normalizes pixel values to [0, 1]  
- Dataset construction — assembles x_train, x_test, x_val (image arrays) and y_train, y_test, y_val (label arrays) ready for model input  

## Installation

```bash
git clone https://github.com/Sukhpal25/handwriting_detection.git
cd handwriting_detection
pip install -r requirements.txt
```

Required libraries:

- pandas  
- numpy  
- tensorflow  
- transformers  
- opencv-python  

## Usage

Preprocess data:

```bash
python data.py
```

Train the model:

```bash
python model.py
```
