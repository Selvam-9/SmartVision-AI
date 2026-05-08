"""Page 3 – Object Detection (YOLO placeholder - already built by user)"""
import streamlit as st

def show():
    st.markdown("<div class='hero-title' style='font-size:2.2rem;'>🔍 Object Detection</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle' style='font-size:1rem;'>YOLOv8-powered real-time object detection with bounding boxes</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background:linear-gradient(135deg,#0d1117,#1a1f2e); border:1px solid #7b2ff744;
                border-left:4px solid #7b2ff7; border-radius:12px; padding:1.5rem; margin:2rem 0;'>
        <div style='color:#7b2ff7; font-weight:700; font-size:1.1rem; margin-bottom:0.5rem;'>
            🔧 YOLO Detection Module
        </div>
        <div style='color:#8b949e;'>
            This page integrates with your existing YOLO model. Your custom detection 
            logic will be displayed here. The section below shows the integration 
            placeholder that you can replace with your YOLO detection code.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload + threshold control ────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload image for detection",
        type=["jpg","jpeg","png","bmp"],
    )

    conf_thresh = st.slider("Confidence threshold", 0.1, 1.0, 0.4, 0.05)

    if uploaded:
        from PIL import Image
        img = Image.open(uploaded).convert("RGB")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📷 Original Image")
            st.image(img, use_container_width=True)

        with col2:
            st.markdown("#### 🎯 Detection Results")
            try:
                from src.inference_detection import predict_detection
                
                with st.spinner("Detecting objects..."):
                    # The predict_detection function returns RGB annotated image and results object
                    annotated_img, results = predict_detection(img, conf=conf_thresh)
                    
                    st.image(annotated_img, use_container_width=True)
                    
                    # Count detections
                    detections = results.boxes
                    if len(detections) > 0:
                        st.success(f"✅ Found {len(detections)} objects!")
                    else:
                        st.warning("No objects detected with the current threshold.")
            except Exception as e:
                st.error(f"Error running detection: {e}")
                st.info("Make sure 'models/best_yolo_model.pt' exists and 'ultralytics' is installed.")

        st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)
        st.markdown("#### 📦 Detected Objects")
        
        if 'results' in locals() and len(results.boxes) > 0:
            # Create a summary table or list
            summary_data = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = results.names[cls_id]
                conf_val = float(box.conf[0])
                summary_data.append({"Object": label, "Confidence": f"{conf_val:.2%}"})
            
            st.table(summary_data)
        else:
            st.info("No objects to list.")
    else:
        st.markdown("""
        <div style='text-align:center; padding:4rem; background:#0d1117;
                    border:2px dashed #2d3748; border-radius:16px; margin-top:2rem;'>
            <div style='font-size:3rem;'>🎯</div>
            <div style='color:#8b949e; font-size:1.1rem; margin-top:1rem;'>
                Upload an image for YOLO object detection
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── YOLO Info ─────────────────────────────────────────────────────────────
    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>ℹ️ About YOLO Detection</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    info = [
        ("⚡ Speed",          "YOLOv8 processes images in real-time, typically under 50ms per frame on GPU."),
        ("🎯 Accuracy",       "Detects multiple objects simultaneously with class labels and bounding box coordinates."),
        ("🔧 Configuration",  f"Current confidence threshold: **{conf_thresh:.0%}**. Lower values detect more objects."),
    ]
    for col, (title, desc) in zip([c1,c2,c3], info):
        with col:
            st.markdown(f"""
            <div class='feature-card'>
                <div class='feature-title'>{title}</div>
                <div class='feature-desc'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
