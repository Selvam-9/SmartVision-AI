"""Shared utilities and model loading for SmartVision AI pages."""
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st

BASE_DIR = Path(__file__).parent.parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR  = BASE_DIR / "smartvision_data"
LOG_DIR   = BASE_DIR / "logs"

MODELS = {
    "MobileNetV2":    MODEL_DIR / "classifier_mobilenetv2.keras",
    "VGG16":          MODEL_DIR / "classifier_vgg16.keras",
    "ResNet50":       MODEL_DIR / "classifier_resnet50.keras",
    "EfficientNetB0": MODEL_DIR / "classifier_efficientnetb0.keras",
}

MODEL_COLORS = {
    "MobileNetV2":    "#00d4ff",
    "VGG16":          "#7b2ff7",
    "ResNet50":       "#ff6b6b",
    "EfficientNetB0": "#ffd700",
}

def load_classes():
    """Load class labels from classes.json."""
    classes_file = MODEL_DIR / "classes.json"
    if classes_file.exists():
        with open(classes_file, "r") as f:
            idx = json.load(f)
        return [k for k, _ in sorted(idx.items(), key=lambda x: x[1])]
    return [f"class_{i}" for i in range(26)]

@st.cache_resource
def load_single_model(model_path):
    """Load a keras model (cached)."""
    from tensorflow.keras.models import load_model
    path = Path(model_path)
    
    if not path.exists():
        st.warning(f"⚠️ Model file `{path.name}` not found in the repository.")
        if "vgg16" in path.name.lower():
            st.info("💡 **Note:** The VGG16 model (~136MB) exceeds GitHub's 100MB file limit and was not uploaded. Please use MobileNetV2, ResNet50, or EfficientNetB0 instead.")
        return None
        
    try:
        return load_model(str(path))
    except Exception as e:
        st.error(f"❌ Error loading `{path.name}`: {e}")
        return None

def get_preprocess_fn(model_name: str):
    """Return the right preprocessing function for a model."""
    name = model_name.lower()
    if "vgg16" in name:
        from tensorflow.keras.applications.vgg16 import preprocess_input
    elif "resnet" in name:
        from tensorflow.keras.applications.resnet50 import preprocess_input
    elif "efficient" in name:
        from tensorflow.keras.applications.efficientnet import preprocess_input
    else:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    return preprocess_input

def preprocess_image(img: Image.Image, model_name: str, size=(224, 224)):
    """Resize, preprocess and batch an image for inference."""
    import numpy as np
    img = img.convert("RGB").resize(size)
    arr = np.array(img, dtype="float32")
    preprocess = get_preprocess_fn(model_name)
    arr = preprocess(arr)
    return np.expand_dims(arr, axis=0)

def predict_top5(model, img_array, class_names):
    """Run inference and return top-5 (class_name, probability) tuples."""
    preds = model.predict(img_array, verbose=0)[0]
    top5_idx = np.argsort(preds)[::-1][:5]
    return [(class_names[i], float(preds[i])) for i in top5_idx]
