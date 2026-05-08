"""Page 6 – About"""
import streamlit as st
from pathlib import Path

def show():
    st.markdown("<div class='hero-title' style='font-size:2.2rem;'>ℹ️ About SmartVision AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle' style='font-size:1rem;'>Project documentation, architecture details, and technical stack</div>", unsafe_allow_html=True)

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    # ── Project Overview ───────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🎯 Project Overview</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#0d1117; border:1px solid #2d3748; border-radius:12px; padding:1.5rem; line-height:1.8;'>
        <p style='color:#c9d1d9;'>
            <strong style='color:#00d4ff;'>SmartVision AI</strong> is an end-to-end computer vision platform that combines
            image classification and object detection in a production-ready Streamlit application.
            The project demonstrates how transfer learning with pre-trained CNN backbones can be
            applied to classify images across 26 COCO object categories.
        </p>
        <p style='color:#8b949e;'>
            The system employs four distinct neural network architectures — MobileNetV2, VGG16,
            ResNet50, and EfficientNetB0 — each pre-trained on ImageNet and fine-tuned on the
            custom SmartVision dataset. YOLO is used for real-time multi-object detection.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    # ── Dataset Info ───────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>📦 Dataset Information</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-icon'>🗂️</div>
            <div class='feature-title'>Classification Dataset</div>
            <div class='feature-desc'>
                <ul style='padding-left:1.2rem; margin:0;'>
                    <li>Source: COCO Dataset subset</li>
                    <li>26 object categories</li>
                    <li>Split: Train / Validation / Test</li>
                    <li>Image format: JPEG / PNG</li>
                    <li>Input resolution: 224 × 224 px</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='feature-card'>
            <div class='feature-icon'>🎯</div>
            <div class='feature-title'>Detection Dataset</div>
            <div class='feature-desc'>
                <ul style='padding-left:1.2rem; margin:0;'>
                    <li>Source: COCO Dataset annotations</li>
                    <li>Bounding box labels</li>
                    <li>YOLO format (.txt annotations)</li>
                    <li>Multiple objects per image</li>
                    <li>Same 26 object categories</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── 26 Classes ────────────────────────────────────────────────────────────
    st.markdown("#### 🏷️ Supported Classes")
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

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    # ── Model Architectures ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🏗️ Model Architectures</div>", unsafe_allow_html=True)
    archs = [
        ("MobileNetV2", "#00d4ff", "Lightweight depthwise separable convolutions designed for mobile and embedded applications. Achieves high accuracy with minimal computational cost. Ideal for real-time inference.", ["Inverted residuals", "Linear bottlenecks", "Depthwise separable convolutions", "3.4M parameters"]),
        ("VGG16",       "#7b2ff7", "Classic deep CNN with uniform 3×3 conv layers. Simple and reliable architecture with proven performance on large-scale image classification tasks.", ["16 weight layers", "Very deep uniform architecture", "3×3 convolution filters", "138M parameters"]),
        ("ResNet50",    "#ff6b6b", "Introduced skip connections (residual connections) to solve the vanishing gradient problem, enabling training of very deep networks.", ["50 layers with residual blocks", "Skip connections", "Batch normalization", "25.6M parameters"]),
        ("EfficientNetB0","#ffd700","Uses compound scaling to balance network depth, width, and resolution. Achieves state-of-the-art accuracy with significantly fewer parameters.", ["Compound scaling", "MBConv blocks", "Squeeze-and-excitation", "5.3M parameters"]),
    ]

    for name, color, desc, highlights in archs:
        with st.expander(f"**{name}**", expanded=False):
            cc1, cc2 = st.columns([2, 1])
            with cc1:
                st.markdown(f"<p style='color:#c9d1d9;'>{desc}</p>", unsafe_allow_html=True)
            with cc2:
                for h in highlights:
                    st.markdown(f"""
                    <div style='display:flex; align-items:center; gap:0.5rem; margin-bottom:0.4rem;'>
                        <span style='color:{color}; font-size:0.8rem;'>▶</span>
                        <span style='color:#8b949e; font-size:0.85rem;'>{h}</span>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    # ── Tech Stack ─────────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🛠️ Technical Stack</div>", unsafe_allow_html=True)
    stack = [
        ("🧠 Deep Learning",  "TensorFlow 2.x / Keras — Model building, training, and inference"),
        ("🔍 Detection",      "Ultralytics YOLOv8 — Real-time multi-object detection"),
        ("🌐 Web App",        "Streamlit — Interactive web application framework"),
        ("🖼️ Image Processing","Pillow (PIL) & OpenCV — Image loading, resizing, and preprocessing"),
        ("📊 Visualization",  "Matplotlib — Charts, confusion matrices, and performance plots"),
        ("📐 Numerics",       "NumPy — Array operations and numerical computations"),
        ("📋 Data Tables",    "Pandas — Tabular data display and analysis"),
        ("🐍 Language",       "Python 3.10+ — Primary development language"),
    ]
    sc1, sc2 = st.columns(2)
    for i, (tech, desc) in enumerate(stack):
        with (sc1 if i % 2 == 0 else sc2):
            st.markdown(f"""
            <div style='background:#0d1117; border:1px solid #2d3748; border-radius:10px;
                        padding:0.9rem 1rem; margin-bottom:0.6rem; display:flex; align-items:center; gap:0.8rem;'>
                <span style='font-size:1.2rem;'>{tech.split()[0]}</span>
                <div>
                    <div style='color:#e6edf3; font-weight:600; font-size:0.85rem;'>{tech.split(" ",1)[1]}</div>
                    <div style='color:#8b949e; font-size:0.8rem;'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    # ── Training Pipeline ──────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🔄 Training Pipeline</div>", unsafe_allow_html=True)
    pipeline_steps = [
        ("Data Loading",       "Images loaded from `smartvision_data/classification/{train,val,test}/` using `ImageDataGenerator` with backbone-specific preprocessing."),
        ("Augmentation",       "Training images augmented with random rotation, zoom, horizontal flip, and fill. Validation/test images receive only preprocessing."),
        ("Backbone Loading",   "Pre-trained ImageNet weights loaded with `include_top=False`. All backbone layers frozen initially."),
        ("Custom Head",        "GlobalAveragePooling2D → BatchNorm → Dense(256) → Dropout(0.5) → Dense(256) → Dropout(0.3) → Softmax output layer."),
        ("Training",           "Adam optimizer (lr=0.0001), categorical crossentropy loss. EarlyStopping, ReduceLROnPlateau, ModelCheckpoint callbacks."),
        ("Saving",             "Best model saved as `.keras` file. Class indices saved to `models/classes.json` for dynamic label loading."),
    ]
    for i, (title, desc) in enumerate(pipeline_steps, 1):
        st.markdown(f"""
        <div style='display:flex; align-items:flex-start; gap:1rem; margin-bottom:0.8rem;
                    background:#0d1117; border:1px solid #1e2d45; border-radius:10px; padding:1rem;'>
            <div class='step-badge'>{i}</div>
            <div>
                <div style='color:#00d4ff; font-weight:600; font-size:0.9rem;'>{title}</div>
                <div style='color:#8b949e; font-size:0.85rem; margin-top:0.2rem;'>{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    # ── Developer Info ─────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>👨‍💻 Developer Information</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:linear-gradient(135deg,#0d1117,#1a1f2e); border:1px solid #2d3748;
                border-radius:16px; padding:2rem; text-align:center;'>
        <div style='font-size:3rem; margin-bottom:0.8rem;'>🚀</div>
        <div style='color:#e6edf3; font-size:1.3rem; font-weight:700;'>SmartVision AI</div>
        <div style='color:#8b949e; margin:0.5rem 0;'>An End-to-End Computer Vision Project</div>
        <hr style='border-color:#2d3748; margin:1.2rem 0;'/>
        <div style='color:#8b949e; font-size:0.85rem;'>
            Framework: TensorFlow 2.x &nbsp;|&nbsp; Streamlit &nbsp;|&nbsp; YOLOv8<br/>
            © 2026 — All Rights Reserved
        </div>
    </div>
    """, unsafe_allow_html=True)
