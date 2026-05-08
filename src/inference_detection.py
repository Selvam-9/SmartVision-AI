import sys
import os
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
from src.logger import logger

def predict_detection(image, model_path='models/detector_yolov8_best.pt', conf=0.25):
    """
    Performs object detection using a YOLOv8 model.
    
    Args:
        image: Path to image or PIL Image object.
        model_path: Path to the .pt model file.
        conf: Confidence threshold.
        
    Returns:
        annotated_img: Image with bounding boxes (numpy array).
        results: Raw Ultralytics results object.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    # Load model
    model = YOLO(model_path)
    
    # Run inference
    results = model(image, conf=conf)
    
    # Get annotated image
    # results[0].plot() returns a BGR numpy array (OpenCV format)
    annotated_img = results[0].plot()
    
    # Convert BGR to RGB for PIL/Matplotlib
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    return annotated_img_rgb, results[0]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YOLOv8 Detection Inference")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--model", type=str, default="models/detector_yolov8_best.pt", help="Path to model file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    
    args = parser.parse_args()
    
    if args.image:
        try:
            logger.info(f"🚀 Running detection on {args.image}...")
            annotated, results = predict_detection(args.image, args.model, args.conf)
            
            # Show results
            plt.figure(figsize=(12, 8))
            plt.imshow(annotated)
            plt.axis('off')
            plt.title(f"Detections (Threshold: {args.conf})")
            plt.show()
            
            # Print summary
            logger.info("📊 Detection Summary:")
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = results.names[cls_id]
                conf_val = float(box.conf[0])
                logger.info(f" - {label}: {conf_val:.2f}")
                
        except Exception as e:
            logger.error(f"❌ Error: {e}")
    else:
        logger.info("Please provide an image path using --image argument.")
