import os
import shutil
import random
from pathlib import Path
from src.logger import logger

def prepare_detection_data():
    logger.info("Organizing detection dataset...")
    
    base_dir = Path("smartvision_data/detection")
    images_source = base_dir / "images"
    labels_source = base_dir / "labels"
    
    # Define target directories
    splits = ["train", "val"]
    subdirs = ["images", "labels"]
    
    for split in splits:
        for subdir in subdirs:
            (base_dir / split / subdir).mkdir(parents=True, exist_ok=True)
    
    # Get list of images
    image_files = [f for f in images_source.glob("*.jpg")]
    random.shuffle(image_files)
    
    # Split 80/20
    split_idx = int(len(image_files) * 0.8)
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]
    
    def move_files(files, split):
        logger.info(f"Moving {len(files)} files to {split} split...")
        for img_path in files:
            label_path = labels_source / (img_path.stem + ".txt")
            
            # Copy image
            shutil.copy(img_path, base_dir / split / "images" / img_path.name)
            
            # Copy label if exists
            if label_path.exists():
                shutil.copy(label_path, base_dir / split / "labels" / label_path.name)
            else:
                logger.warning(f"Warning: Label not found for {img_path.name}")

    move_files(train_images, "train")
    move_files(val_images, "val")
    
    # Update data.yaml with correct path and user's specific class mapping
    data_yaml_content = """
path: smartvision_data/detection
train: train/images
val: val/images

# Classes
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: traffic light
  9: stop sign
  10: bench
  11: bird
  12: cat
  13: dog
  14: horse
  15: cow
  16: elephant
  17: bottle
  18: cup
  19: bowl
  20: pizza
  21: cake
  22: chair
  23: couch
  24: potted plant
  25: bed

nc: 26
"""
    with open(base_dir / "data.yaml", "w") as f:
        f.write(data_yaml_content.strip())
        
    logger.info("Dataset preparation complete!")

if __name__ == "__main__":
    prepare_detection_data()
