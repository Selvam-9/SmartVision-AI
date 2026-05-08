import sys
import os
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from src.data_loader_classification import DataLoader
from src.model_classification import build_model
from src.logger import logger


def train_network(backbone_name="mobilenetv2", epochs=20):
    """Main training function that ties data loading and model building together."""
    
    # 1. Load Data
    logger.info("📂 Loading data...")
    loader = DataLoader()
    training_set, val_set, test_set = loader.load_data()
    
    # Save class indices for the prediction app
    os.makedirs('models', exist_ok=True)
    import json
    class_indices = training_set.class_indices
    with open('models/classes.json', 'w') as f:
        json.dump(class_indices, f)
    logger.info(f"📄 Saved {len(class_indices)} class mappings to models/classes.json")
    
    # 2. Build Model
    logger.info(f"🏗️ Building model with {backbone_name} backbone for {loader.num_classes} classes...")
    model = build_model(
        backbone_name=backbone_name, 
        num_classes=loader.num_classes, 
        input_shape=(224, 224, 3)
    )

    
    # 3. Setup Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=7,
        min_lr=1e-7,
        verbose=1
    )
    
    os.makedirs('models', exist_ok=True)
    checkpoint = ModelCheckpoint(
        filepath=f'models/best_{backbone_name}_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # 4. Train
    logger.info("🚀 Starting training...")
    history = model.fit(
        training_set,
        validation_data=val_set,
        epochs=epochs,
        steps_per_epoch=len(training_set),
        validation_steps=len(val_set),
        callbacks=[early_stop, reduce_lr, checkpoint]

    )
    
    # 5. Save Final Model
    final_path = f'models/classifier_{backbone_name}.keras'
    model.save(final_path)
    logger.info(f"✅ Training complete! Final model saved to: {final_path}")
    
    return history

if __name__ == "__main__":
    train_network(backbone_name="vgg16", epochs=20)
