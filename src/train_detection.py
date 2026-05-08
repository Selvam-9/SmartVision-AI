import sys
import os
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO
from src.logger import logger

def train_detection():
    logger.info("🚀 Starting YOLOv8 training...")
    
    # Load pre-trained nano model
    model = YOLO("models/detector_yolov8_base.pt")
    
    # Path to data.yaml
    data_path = Path("smartvision_data/detection/data.yaml").resolve()
    
    # Training parameters
    # Note: Using small batch size and epochs because no GPU was detected
    # Increase these if running on a GPU-enabled environment
    results = model.train(
        data=str(data_path),
        epochs=200,
        imgsz=640,
        batch=16,
        project="models/detection",
        name="yolov8n_finetuned",
        patience=10, # Early stopping
        optimizer="Adam",
        lr0=0.01
    )
    
    logger.info("✅ Training complete!")
    logger.info(f"📁 Results saved in: {results.save_dir}")
    
    # Export the model to standard format
    # model.export(format="onnx")

if __name__ == "__main__":
    train_detection()
