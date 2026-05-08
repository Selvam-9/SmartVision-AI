# SmartVision AI 🚀

An industry-ready image classification and object detection system built with TensorFlow, YOLOv8, and Streamlit. This platform provides a modular, multi-page web interface for real-time inference, model training, and performance analysis.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-blueviolet.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Modular Web Interface](#modular-web-interface)
- [Training](#training)
- [Inference](#inference)
- [Logging](#logging)

## ✨ Features

- **Multi-Model Classification**: Compare 4 state-of-the-art CNN backbones (MobileNetV2, VGG16, ResNet50, EfficientNetB0) simultaneously.
- **YOLOv8 Object Detection**: Real-time object detection with bounding boxes and confidence scores.
- **Live Webcam Integration**: Real-time detection directly from your browser using WebRTC.
- **Transfer Learning**: Optimized pipelines using pre-trained ImageNet weights.
- **Modular Architecture**: Clean separation between UI pages, source logic, and model weights.
- **Comprehensive Logging**: Centralized logging system for both training and application states.

## 📁 Project Structure

```text
SmartVision AI/
├── app/
│   ├── app.py                      # Main Streamlit application entry
│   └── app_pages/                  # Multi-page modular components
│       ├── page_home.py            # Dashboard landing page
│       ├── page_classification.py  # Image classification module
│       ├── page_detection.py       # Object detection module
│       ├── page_performance.py     # Model analytics and metrics
│       ├── page_webcam.py          # Live webcam detection
│       └── utils.py                # Shared UI utilities and loading logic
├── src/
│   ├── data_loader_classification.py # Data loading for classification
│   ├── model_classification.py       # Model architecture for classification
│   ├── train_classification.py       # Classification training pipeline
│   ├── train_detection.py            # YOLOv8 detection training pipeline
│   ├── inference_detection.py        # Object detection inference logic
│   ├── setup_yolo.py                 # YOLOv8 environment setup
│   ├── prepare_detection_data.py      # Detection dataset organization
│   └── logger.py                     # Centralized logging utility
├── models/                         # Trained weights (.keras and .pt)
├── notebooks/                      # EDA and experimental notebooks
├── logs/                           # Application logs
├── smartvision_data/               # Training and validation datasets
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🛠️ Installation

### 1. Clone and Setup
```bash
git clone https://github.com/yourusername/SmartVisionAI.git
cd SmartVisionAI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup YOLO Environment
Run the setup script to download base weights and verify the environment:
```bash
python -m src.setup_yolo
```

## 🚀 Quick Start

### Running the Web App
The platform uses a modular Streamlit setup. Launch the dashboard using:
```bash
streamlit run app/app.py
```

### Manual Inference (CLI)
You can run object detection directly from the command line:
```bash
python -m src.inference_detection --image path/to/your/image.jpg
```

## 🏋️ Training

### Image Classification
Train a classification model (e.g., VGG16):
```bash
python -m src.train_classification --backbone vgg16 --epochs 20
```

### Object Detection (YOLOv8)
Prepare your data and start fine-tuning:
```bash
python -m src.prepare_detection_data
python -m src.train_detection
```

## 🔮 Inference & Models

### Classification Backbones
- `classifier_mobilenetv2.keras`
- `classifier_vgg16.keras`
- `classifier_resnet50.keras`
- `classifier_efficientnetb0.keras`

### Detection Models
- `detector_yolov8_base.pt` (Pre-trained weights)
- `detector_yolov8_best.pt` (Your fine-tuned model)

## 📝 Logging
The system maintains a centralized log file at `logs/smartvision.log`. You can monitor training progress or application errors in real-time.

## 📊 Classes
### Classification & Detection (26 Core Classes)
airplane, bed, bench, bicycle, bird, bottle, bowl, bus, cake, car, cat, chair, couch, cow, cup, dog, elephant, horse, motorcycle, person, pizza, potted plant, stop sign, traffic light, train, truck

---

<p align="center">Made with ❤️ by SmartVision AI Team</p>