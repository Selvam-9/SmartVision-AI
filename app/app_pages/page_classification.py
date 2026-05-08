"""Page 2 – Image Classification"""
import streamlit as st
import numpy as np
from PIL import Image
from .utils import MODELS, MODEL_COLORS, load_classes, load_single_model, preprocess_image, predict_top5

def show():
    st.markdown("<div class='hero-title' style='font-size:2.2rem;'>🖼️ Image Classification</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle' style='font-size:1rem;'>Upload an image to classify it using 4 CNN models simultaneously</div>", unsafe_allow_html=True)

    class_names = load_classes()

    # ── Upload ────────────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Drop your image here", type=["jpg","jpeg","png","bmp","webp"],
        help="Supported formats: JPG, PNG, BMP, WEBP"
    )

    if uploaded is None:
        st.markdown("""
        <div style='text-align:center; padding:4rem; background:#0d1117;
                    border:2px dashed #2d3748; border-radius:16px; margin-top:2rem;'>
            <div style='font-size:3rem;'>📤</div>
            <div style='color:#8b949e; font-size:1.1rem; margin-top:1rem;'>
                Upload an image to begin classification
            </div>
            <div style='color:#4a5568; font-size:0.85rem; margin-top:0.5rem;'>
                Supports JPG · PNG · BMP · WEBP
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    img = Image.open(uploaded).convert("RGB")

    # ── Sidebar model selector ────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🔧 Classification Options**")
    selected_models = st.sidebar.multiselect(
        "Models to run",
        list(MODELS.keys()),
        default=list(MODELS.keys()),
    )
    top_n = st.sidebar.slider("Top-N predictions", 1, 10, 5)
    show_comparison = st.sidebar.checkbox("Show model comparison table", value=True)

    if not selected_models:
        st.warning("Please select at least one model.")
        return

    # ── Display uploaded image ────────────────────────────────────────────────
    c_img, c_info = st.columns([1, 2])
    with c_img:
        st.markdown("#### 📷 Uploaded Image")
        st.image(img, use_container_width=True)
        w, h = img.size
        st.caption(f"Size: {w}×{h}px | Mode: {img.mode}")

    # ── Run all selected models ───────────────────────────────────────────────
    with c_info:
        st.markdown("#### 🏆 Classification Results")
        results = {}

        prog = st.progress(0, text="Loading models…")
        for i, model_name in enumerate(selected_models):
            prog.progress((i) / len(selected_models), text=f"Running {model_name}…")
            model_path = str(MODELS[model_name])
            model = load_single_model(model_path)
            if model is None:
                st.warning(f"⚠️ {model_name} could not be loaded.")
                continue
            arr = preprocess_image(img, model_name)
            top5 = predict_top5(model, arr, class_names)
            results[model_name] = top5
            prog.progress((i + 1) / len(selected_models), text=f"✅ {model_name} done")
        prog.empty()

        # Show top prediction per model
        for model_name, preds in results.items():
            color = MODEL_COLORS.get(model_name, "#00d4ff")
            top_class, top_conf = preds[0]
            st.markdown(f"""
            <div style='background:#0d1117; border:1px solid {color}44;
                        border-left: 4px solid {color}; border-radius:10px;
                        padding:1rem; margin-bottom:0.8rem;'>
                <span class='model-badge' style='color:{color}; border-color:{color}44;'>{model_name}</span>
                <span style='color:#e6edf3; font-size:1.1rem; font-weight:700; margin-left:0.8rem;'>{top_class.title()}</span>
                <span style='color:{color}; font-weight:700; font-size:1.1rem; float:right;'>{top_conf:.1%}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    # ── Top-N Predictions per Model ───────────────────────────────────────────
    st.markdown(f"<div class='section-header'>📋 Top-{top_n} Predictions by Model</div>", unsafe_allow_html=True)
    cols = st.columns(len(results))
    for col, (model_name, preds) in zip(cols, results.items()):
        color = MODEL_COLORS.get(model_name, "#00d4ff")
        with col:
            st.markdown(f"<div style='color:{color}; font-weight:700; font-size:1rem; margin-bottom:0.8rem;'>{model_name}</div>", unsafe_allow_html=True)
            for rank, (cls, conf) in enumerate(preds[:top_n], 1):
                bar_w = int(conf * 100)
                st.markdown(f"""
                <div class='pred-card'>
                    <div style='display:flex; justify-content:space-between;'>
                        <span class='pred-rank'>#{rank}</span>
                        <span class='pred-conf'>{conf:.1%}</span>
                    </div>
                    <div class='pred-label'>{cls.title()}</div>
                    <div style='background:#1e2d45; border-radius:4px; margin-top:0.5rem; height:6px;'>
                        <div style='background:{color}; width:{bar_w}%; height:6px; border-radius:4px;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Comparison Table ──────────────────────────────────────────────────────
    if show_comparison and results:
        st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>🔀 Model Comparison Table</div>", unsafe_allow_html=True)
        import pandas as pd
        rows = []
        for model_name, preds in results.items():
            for rank, (cls, conf) in enumerate(preds[:5], 1):
                rows.append({
                    "Model": model_name,
                    "Rank": rank,
                    "Predicted Class": cls.title(),
                    "Confidence": f"{conf:.2%}",
                })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
