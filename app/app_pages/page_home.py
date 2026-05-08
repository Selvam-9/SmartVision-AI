"""Page 1 – Home"""
import streamlit as st
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "smartvision_data"

def show():
    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class='hero-title'>👁️ SmartVision AI</div>
    <div class='hero-subtitle'>
        Advanced Image Classification &amp; Object Detection Platform<br/>
        Powered by Transfer Learning with MobileNetV2 · VGG16 · ResNet50 · EfficientNetB0
    </div>
    """, unsafe_allow_html=True)

    # ── Key Stats ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        ("4", "CNN Models"),
        ("26", "Object Classes"),
        ("YOLO", "Detection Engine"),
        ("224×224", "Input Resolution"),
    ]
    for col, (val, label) in zip([c1,c2,c3,c4], stats):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    # ── Features Grid ─────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>✨ Key Features</div>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    features = [
        ("🖼️", "Image Classification", "Classify images into 26 COCO categories using 4 state-of-the-art CNN backbones with side-by-side model comparison."),
        ("🔍", "Object Detection", "Locate and identify multiple objects in a single image using YOLOv8 with bounding boxes and confidence scores."),
        ("📊", "Model Performance", "Compare accuracy, inference speed, and class-wise performance across all 4 CNN architectures."),
        ("📷", "Live Webcam", "Real-time object detection through your webcam with FPS monitoring using YOLO."),
        ("🔀", "Model Comparison", "Upload an image and instantly see how all 4 CNN models classify it — side by side."),
        ("💡", "Top-5 Predictions", "View confidence scores for the top 5 predicted classes for detailed insight."),
    ]
    cols = [f1, f2, f3, f1, f2, f3]
    for col, (icon, title, desc) in zip(cols, features):
        with col:
            st.markdown(f"""
            <div class='feature-card'>
                <div class='feature-icon'>{icon}</div>
                <div class='feature-title'>{title}</div>
                <div class='feature-desc'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
        st.write("")  # spacing

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    # ── How to Use ───────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🚀 Quick Start Guide</div>", unsafe_allow_html=True)
    steps = [
        ("Image Classification", "Navigate to **🖼️ Image Classification** in the sidebar. Upload a JPG or PNG image. The system will run all 4 CNN models and display top-5 predictions with confidence scores."),
        ("Object Detection",     "Navigate to **🔍 Object Detection**. Upload your image and adjust the confidence threshold. YOLO will detect and label all objects with bounding boxes."),
        ("Model Performance",    "Navigate to **📊 Model Performance** to explore accuracy metrics, confusion matrices, and inference speed comparisons between models."),
        ("Live Detection",       "Navigate to **📷 Live Webcam** and allow camera access for real-time YOLO object detection through your webcam."),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f"""
        <div style='display:flex; align-items:flex-start; gap:1rem; margin-bottom:1.2rem;
                    background:#0d1117; border:1px solid #1e2d45; border-radius:12px; padding:1.2rem;'>
            <div class='step-badge'>{i}</div>
            <div>
                <div style='color:#e6edf3; font-weight:600; font-size:1rem;'>{title}</div>
                <div style='color:#8b949e; font-size:0.9rem; margin-top:0.3rem;'>{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    # ── Sample Images from dataset ───────────────────────────────────────────
    st.markdown("<div class='section-header'>🖼️ Sample Training Images</div>", unsafe_allow_html=True)
    train_dir = DATA_DIR / "classification" / "train"
    if train_dir.exists():
        classes = sorted([d for d in train_dir.iterdir() if d.is_dir()])
        sample_classes = classes[:6]
        cols = st.columns(len(sample_classes))
        for col, cls_dir in zip(cols, sample_classes):
            img_files = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.jpeg")) + list(cls_dir.glob("*.png"))
            if img_files:
                with col:
                    try:
                        from PIL import Image
                        img = Image.open(img_files[0]).convert("RGB")
                        img.thumbnail((200, 200))
                        st.image(img, caption=cls_dir.name.title(), use_container_width=True)
                    except Exception:
                        st.caption(cls_dir.name.title())
    else:
        st.info("📂 Dataset not found. Ensure `smartvision_data/classification/train/` exists.")

    # ── Supported Classes ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>📦 Supported Classes</div>", unsafe_allow_html=True)
    classes_26 = [
        'airplane','bed','bench','bicycle','bird','bottle','bowl','bus',
        'cake','car','cat','chair','couch','cow','cup','dog','elephant',
        'horse','motorcycle','person','pizza','potted plant','stop sign',
        'traffic light','train','truck'
    ]
    badge_html = " ".join([
        f"<span style='display:inline-block; background:#1e2d45; color:#00d4ff; "
        f"border:1px solid #2d3748; border-radius:20px; padding:0.2rem 0.7rem; "
        f"font-size:0.8rem; margin:0.2rem;'>{c}</span>"
        for c in classes_26
    ])
    st.markdown(badge_html, unsafe_allow_html=True)
