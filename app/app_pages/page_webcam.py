"""Page 5 – Live Webcam Detection"""
import streamlit as st

def show():
    st.markdown("<div class='hero-title' style='font-size:2.2rem;'>📷 Live Webcam Detection</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-subtitle' style='font-size:1rem;'>Real-time object detection using your webcam</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='background:linear-gradient(135deg,#0d1117,#1a1f2e); border:1px solid #ffd70044;
                border-left:4px solid #ffd700; border-radius:12px; padding:1.5rem; margin:1.5rem 0;'>
        <div style='color:#ffd700; font-weight:700; font-size:1rem; margin-bottom:0.5rem;'>⚠️ Requirements</div>
        <ul style='color:#8b949e; margin:0; padding-left:1.2rem;'>
            <li>A webcam connected to your computer</li>
            <li>Your YOLO model available at <code style='color:#00d4ff;'>models/detector_yolov8_best.pt</code></li>
            <li>Browser camera permissions enabled</li>
            <li>The <code style='color:#00d4ff;'>streamlit-webrtc</code> package for live streaming</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # ── Try webrtc integration ─────────────────────────────────────────────────
    try:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
        import av
        import time
        from ultralytics import YOLO
        from pathlib import Path
        from src.logger import logger

        # Model detection and status
        yolo_path = Path(__file__).parent.parent.parent / "models" / "detector_yolov8_best.pt"
        model_exists = yolo_path.exists()
        
        if model_exists:
            st.success(f"✅ YOLO Model loaded from `{yolo_path.name}`")
        else:
            st.error(f"❌ YOLO Model not found at `{yolo_path}`")
            st.info("Please ensure your model is saved as `models/detector_yolov8_best.pt` after training.")

        conf_thresh = st.slider("Detection confidence threshold", 0.1, 1.0, 0.4, 0.05)

        class YOLOVideoProcessor(VideoProcessorBase):
            def __init__(self):
                self.confidence = conf_thresh
                self._model = None
                self._fps_start = time.time()
                self._frame_count = 0
                self.fps = 0.0

                # Load YOLO model
                if model_exists:
                    try:
                        self._model = YOLO(str(yolo_path))
                    except Exception as e:
                        logger.error(f"Error loading model in processor: {e}")

            def recv(self, frame):
                import cv2
                img = frame.to_ndarray(format="bgr24")
                
                # Update confidence from the current slider value
                # (Note: In streamlit-webrtc, the instance might persist, 
                # so we can use a callback or just access the value if we use processor_params)
                self.confidence = conf_thresh

                self._frame_count += 1
                elapsed = time.time() - self._fps_start
                if elapsed > 1.0:
                    self.fps = self._frame_count / elapsed
                    self._frame_count = 0
                    self._fps_start = time.time()

                if self._model is not None:
                    try:
                        results = self._model(img, conf=self.confidence, verbose=False)
                        if len(results) > 0:
                            img = results[0].plot()
                    except Exception as e:
                        # Fallback: just show raw frame if detection fails
                        pass

                # FPS overlay
                cv2.putText(img, f"FPS: {self.fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 212, 255), 2)
                
                # Confidence overlay
                cv2.putText(img, f"Conf: {self.confidence:.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                return av.VideoFrame.from_ndarray(img, format="bgr24")

        st.markdown("### 🎥 Live Feed")
        ctx = webrtc_streamer(
            key="yolo-live",
            video_processor_factory=YOLOVideoProcessor,
            rtc_configuration=RTCConfiguration({
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                    {"urls": ["stun:stun2.l.google.com:19302"]},
                    {"urls": ["stun:stun3.l.google.com:19302"]},
                    {"urls": ["stun:stun4.l.google.com:19302"]},
                ]
            }),
            media_stream_constraints={"video": True, "audio": False},
        )

        if ctx.video_processor:
            ctx.video_processor.confidence = conf_thresh

    except ImportError as e:
        st.warning(f"**Missing dependencies: `{e.name}`**")
        st.code("pip install streamlit-webrtc av ultralytics", language="bash")

    # ── Info cards ────────────────────────────────────────────────────────────
    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>💡 How Live Detection Works</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    info = [
        ("📡 Capture", "Your webcam feed is streamed frame by frame through WebRTC directly in the browser."),
        ("🎯 Detect",  "Each frame is passed to the YOLO model which detects objects and draws bounding boxes."),
        ("📈 Metrics", "FPS (frames per second) is calculated in real-time to show system performance."),
    ]
    for col, (title, desc) in zip([c1,c2,c3], info):
        with col:
            st.markdown(f"""
            <div class='feature-card'>
                <div class='feature-title'>{title}</div>
                <div class='feature-desc'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
