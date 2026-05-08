import sys
import os
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf
from tensorflow.keras.models import Model
# pyrefly: ignore [missing-import]
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, ResNet50, VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization  
from src.logger import logger

def get_preprocess_fn(name="mobilenetv2"):
    """Returns the correct preprocessing function for the chosen backbone."""
    if name.lower() == 'vgg16':
        from tensorflow.keras.applications.vgg16 import preprocess_input
    elif name.lower() == 'resnet50':
        from tensorflow.keras.applications.resnet50 import preprocess_input
    elif name.lower() == 'efficientnetb0':
        None # No preprocess_input needed
    else: # Default to MobileNetV2
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    return preprocess_input

def get_backbone(name="mobilenetv2", input_shape=(224, 224, 3)):
    """Returns the base model backbone."""
    backbones = {
        'efficientnetb0': EfficientNetB0,
        'mobilenetv2': MobileNetV2,
        'resnet50': ResNet50,
        'vgg16': VGG16
    }
    
    if name.lower() not in backbones:
        logger.warning(f"⚠️ Warning: {name} not found. Defaulting to MobileNetV2.")
        name = 'mobilenetv2'
        
    model_class = backbones[name.lower()]
    return model_class(weights='imagenet', include_top=False, input_shape=input_shape)

def build_model(backbone_name="mobilenetv2", num_classes=26, input_shape=(224, 224, 3)):
    """Builds the full model with a custom classification head."""
    # Load backbone
    base_model = get_backbone(backbone_name, input_shape=input_shape)
    base_model.trainable = False  # Freeze the base model by default
    
    # Custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Test model building
    try:
        my_model = build_model(backbone_name="mobilenetv2", num_classes=26)
        my_model.summary()
        logger.info("✅ Model built and compiled successfully!")
    except Exception as e:
        logger.error(f"❌ Error building model: {e}")
