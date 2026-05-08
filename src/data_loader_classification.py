import sys
import os
import tensorflow as tf
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_classification import get_preprocess_fn
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.logger import logger



class DataLoader:
    """
    Data loader class for image classification with proper path handling and generators.
    """
    def __init__(self, base_dir=None, batch_size=32, preprocess_fn=get_preprocess_fn()):
        if base_dir is None:
            self.base_dir = Path().resolve()
        else:
            self.base_dir = Path(base_dir)
            
        self.train_path = self.base_dir / 'smartvision_data' / 'classification' / 'train'
        self.test_path = self.base_dir / 'smartvision_data' / 'classification' / 'test'
        self.val_path = self.base_dir / 'smartvision_data' / 'classification' / 'val'
        
        self.image_size = (224, 224)
        self.batch_size = batch_size
        
        # Dynamically determine the number of classes
        if self.train_path.exists():
            self.num_classes = len([d for d in self.train_path.iterdir() if d.is_dir()])
            logger.info(f"📂 Detected {self.num_classes} classes.")
        else:
            self.num_classes = 0
            logger.warning(f"⚠️ Warning: Training path {self.train_path} not found.")

        self.preprocess_fn = preprocess_fn

    def get_generators(self):
        """Create and return ImageDataGenerators for training, validation, and testing."""
        train_datagen = ImageDataGenerator(
            preprocessing_function=self.preprocess_fn,
            rotation_range=10,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
            )

        val_datagen = ImageDataGenerator(preprocessing_function=self.preprocess_fn)
        test_datagen = ImageDataGenerator(preprocessing_function=self.preprocess_fn)
        
        return train_datagen, val_datagen, test_datagen

    def load_data(self):
        """Load and return training, validation, and test sets."""
        train_gen, val_gen, test_gen = self.get_generators()
        
        training_set = train_gen.flow_from_directory(
            str(self.train_path),
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )

        val_set = val_gen.flow_from_directory(
            str(self.val_path),
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )

        test_set = test_gen.flow_from_directory(
            str(self.test_path),
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        return training_set, val_set, test_set

# Standalone execution for testing
if __name__ == "__main__":
    try:
        loader = DataLoader()
        train_data, val_data, test_data = loader.load_data()
        logger.info(f"✅ Success! Loaded {train_data.samples} training images.")
        logger.info(f"✅ Loaded {val_data.samples} validation images.")
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")

