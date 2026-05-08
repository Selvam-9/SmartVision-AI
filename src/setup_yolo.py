import sys
import os
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
import torch
from src.logger import logger

def setup_yolo():
    logger.info("🚀 Setting up YOLOv8...")
    
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"💻 Using device: {device}")
    
    # Download pre-trained model weights (YOLOv8 nano)
    model_path = "models/detector_yolov8_base.pt"
    if not os.path.exists(model_path):
        logger.info(f"📥 Downloading pre-trained weights to {model_path}...")
        model = YOLO("models/detector_yolov8_base.pt")
        logger.info("✅ Download complete.")
    else:
        logger.info(f"📂 Found existing weights at {model_path}")
        model = YOLO(model_path)
    
    # Print model info
    logger.info(f"🔍 YOLOv8 version: {model.info()}")
    logger.info("✅ YOLOv8 setup successful!")

if __name__ == "__main__":
    setup_yolo()
